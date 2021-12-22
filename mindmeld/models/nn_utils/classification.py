# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Cisco Systems, Inc. and others.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Base for custom modules that are developed on top of nn layers that can do
sequence or token classification
"""
import json
import logging
import os
import shutil
import uuid
from abc import abstractmethod
from itertools import chain
from typing import Dict, Union, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .helpers import (
    BatchData,
    get_disk_space_of_model,
    get_num_weights_of_model,
    get_default_params,
    TokenizerType,
    EmbedderType,
    ValidationMetricType
)
from .input_encoders import InputEncoderFactory
from .._util import _get_module_or_attr
from ..containers import GloVeEmbeddingsContainer
from ...core import Bunch
from ...path import USER_CONFIG_DIR

try:
    import torch
    import torch.nn as nn

    nn_module = _get_module_or_attr("torch.nn", "Module")
except ImportError:
    nn_module = object
    pass

SEED = 6174
LABEL_PAD_TOKEN_IDX = -1  # value set based on default label padding idx in pytorch

logger = logging.getLogger(__name__)


class BaseClassification(nn_module):
    """
    A base class for sequence and token classifiation using deep neural nets. The trainable examples
    inputted to the methods of this class are generally in the form of strings or list of strings.
    """

    def __init__(self):
        super().__init__()

        self.name = self.__class__.__name__
        self.params = Bunch()
        self.params["name"] = self.name
        self.encoder = None

        self.ready = False  # True when .fit() is called or loaded from a checkpoint, else False
        self.dirty = False  # True when the model weights aren't saved to disk yet, else False

    def __repr__(self):
        return f"<{self.name}> ready:{self.ready} dirty:{self.dirty}"

    def get_default_params(self) -> Dict:
        return get_default_params(self.__class__.__name__)

    def print_and_return_model_info(self) -> str:
        msg = f"{self.name} " \
              f"ready:{self.ready} dirty:{self.dirty} device:{self.params.device}\n" \
              f"\tNumber of weights (trainable, all):{get_num_weights_of_model(self)} \n" \
              f"\tDisk Size (in MB): {get_disk_space_of_model(self):.4f}"
        logger.info(msg)
        return msg

    def to_device(self, batch_data: BatchData) -> BatchData:
        """method for gpu usage and data loading"""
        for k, v in batch_data.items():
            if v is not None and isinstance(v, torch.Tensor):
                batch_data[k] = v.to(self.params.device)
            elif isinstance(v, list):
                batch_data[k] = [
                    vv.to(self.params.device) if isinstance(vv, torch.Tensor) else vv
                    for vv in v
                ]
            elif isinstance(v, dict):
                batch_data[k] = self.to_device(batch_data[k])
        return batch_data

    def fit(self, examples, labels, **params):  # pylint: disable=too-many-locals
        """
        Trains the underlying neural model on the inputted data and finally retains the best scored
        model among all iterations.

        Because of possibly large sized neural models, instead of retaining a copy of best set of
        model weights on RAM, it is advisable to dump them in a temporary folder and upon training,
        load the best checkpointed weights.
        """

        if self.ready:
            msg = "The model is already fitted or is loaded from a file. Aborting re-fitting."
            logger.error(msg)

        # obtain and validate all parameters required to fit the model
        params = {
            **self.get_default_params(),
            **params  # overwrite keys of default params that are passed-in
        }
        params = self._validate_and_update_params(**params)

        # update params upon preparing encoder and embedder (eg. num_tokens, emb_dim, etc.)
        params = self._prepare_input_encoder(examples, **params)
        params = self._prepare_embedder(**params)

        # update params with label pad idx
        params["label_padding_idx"] = params.get("label_padding_idx", LABEL_PAD_TOKEN_IDX)

        # update number of labels in params upon identifying the unique labels
        try:
            # labels for sequence classification are of type List[str] whereas for token
            # classification they are List[List[str]] ; this try-except tries to obtain the number
            # of unique label strings for the purpose of classification
            num_labels = len(set(labels))
        except TypeError:
            # error will be raised in case of token classification as `set(labels)` attempts to
            # find a set of lists
            num_labels = len(set(chain.from_iterable(labels)))
        params.update({"num_labels": num_labels})

        # update self.params which will be used throughout the following modeling code
        self.params.update(params)

        # init the graph and move model to device, inputs are moved to device on-the-go
        self._init_graph()
        self.to(self.params.device)
        self.print_and_return_model_info()

        # dumping weights during training process into a temp folder instead of keeping in
        # memory to reduce memory usage
        temp_folder = os.path.join(USER_CONFIG_DIR, "tmp", "pytorch_models", str(uuid.uuid4()))
        temp_weights_save_path = os.path.join(temp_folder, "pytorch_model.bin")
        os.makedirs(temp_folder, exist_ok=True)

        # split input data into train & dev splits, and get data loaders
        train_examples, dev_examples, train_labels, dev_labels = train_test_split(
            examples, labels, test_size=self.params.dev_split_ratio, random_state=SEED
        )

        # create an optimizer and attach all model params to it
        num_training_steps = int(
            len(train_examples) / self.params.batch_size / self.params.gradient_accumulation_steps *
            self.params.number_of_epochs
        )
        optimizer, scheduler = self._create_optimizer_and_scheduler(num_training_steps)

        # training w/ validation
        best_dev_score, best_dev_epoch = -np.inf, -1
        msg = f"Beginning to train for {self.params.number_of_epochs} number of epochs"
        logger.info(msg)
        if self.params.number_of_epochs < 1:
            raise ValueError("Param 'number_of_epochs' must be a positive integer greater than 0")
        patience_counter = 0
        verbose = (
            logger.getEffectiveLevel() == logging.INFO or
            logger.getEffectiveLevel() == logging.DEBUG
        )
        for epoch in range(1, self.params.number_of_epochs + 1):
            # patience before terminating due to no dev score improvements
            if patience_counter >= self.params.patience:
                msg = f"Set patience of {self.params.patience} epochs reached"
                logger.info(msg)
                break
            # set modules to train phase, reset gradients, do forward-backward propagations
            self.train()
            optimizer.zero_grad()
            train_loss, train_batches = 0.0, 0.0
            t = tqdm(range(0, len(train_examples), self.params.batch_size), disable=not verbose)
            for start_idx in t:
                batch_examples = train_examples[start_idx:start_idx + self.params.batch_size]
                batch_labels = train_labels[start_idx:start_idx + self.params.batch_size]
                batch_data = self.encoder.batch_encode(
                    examples=batch_examples,
                    padding_length=self.params.padding_length,
                    add_terminals=self.params.add_terminals
                )
                batch_data.update({
                    "labels": self._prepare_labels(batch_labels, batch_data["split_lengths"])
                })
                batch_data = self.forward(batch_data)
                loss = batch_data["loss"]
                train_loss += loss.cpu().detach().numpy()
                train_batches += 1
                # find gradients
                loss = loss / self.params.gradient_accumulation_steps
                loss.backward()
                # optimizer and scheduler step
                batch_id = start_idx / self.params.batch_size
                if (
                    start_idx + self.params.batch_size >= len(train_examples) or
                    (batch_id + 1) % self.params.gradient_accumulation_steps == 0
                ):
                    # update weights when it is the last batch in the epoch or
                    # when specified step is reached or
                    if self.params.max_grad_norm:  # clip (accumulated) gradients if required
                        nn.utils.clip_grad_norm_(self.parameters(), self.params.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                # log progress
                progress_bar_msg = f"Epoch: {epoch} | Mean loss: {train_loss / train_batches:.4f}"
                t.set_description(progress_bar_msg, refresh=True)
            train_loss = train_loss / train_batches
            # dev evaluation
            predictions, targets = [], []
            t = tqdm(range(0, len(dev_examples), self.params.batch_size), disable=not verbose)
            for start_idx in t:
                batch_examples = dev_examples[start_idx:start_idx + self.params.batch_size]
                batch_labels_targetted = dev_labels[start_idx:start_idx + self.params.batch_size]
                batch_labels_predicted = self.predict(batch_examples)
                # validation
                if len(batch_labels_predicted) != len(batch_labels_targetted):
                    msg = f"Number of predictions ({len(batch_labels_predicted)}) " \
                          f"not equal to number of targets ({len(batch_labels_targetted)})"
                    logger.error(msg)
                    raise AssertionError(msg)
                # flatten if required
                try:
                    batch_labels_predicted = sum(batch_labels_predicted, [])
                    batch_labels_targetted = sum(batch_labels_targetted, [])
                except TypeError:
                    # raised in case of sequence classification; implies already flattened
                    pass
                # discard unwanted predictions using label_padding_idx
                batch_labels_predicted, batch_labels_targetted = zip(*[
                    (x, y) for x, y in zip(batch_labels_predicted, batch_labels_targetted)
                    if y != self.params.label_padding_idx
                ])
                predictions.extend(batch_labels_predicted)
                targets.extend(batch_labels_targetted)
                progress_bar_msg = f"Epoch: {epoch} | " \
                                   f"Validation Metric: {self.params.validation_metric} "
                t.set_description(progress_bar_msg, refresh=True)
            # compute score
            if ValidationMetricType(self.params.validation_metric) == ValidationMetricType.ACCURACY:
                dev_score = accuracy_score(targets, predictions, normalize=True)
            elif ValidationMetricType(self.params.validation_metric) == ValidationMetricType.F1:
                dev_score = f1_score(targets, predictions, average='weighted')
            else:
                msg = f"Invalid 'validation_metric' ({self.params.validation_metric}) provided " \
                      f"in params. Allowed values are only 'accuracy' and 'f1'"
                raise ValueError(msg)
            # save model weights in a temp folder; later move it to folder passed through dump()
            if dev_score >= best_dev_score:
                torch.save(self.state_dict(), temp_weights_save_path)
                phrase = (
                    f"improved from '{best_dev_score:.4f}' to" if dev_score > best_dev_score
                    else "remained at"
                )
                msg = f"Model weights saved after epoch: {epoch} when dev score {phrase} " \
                      f"'{dev_score:.4f}'\n"
                logger.info(msg)
                # update patience counter
                if dev_score == best_dev_score:
                    patience_counter += 1
                else:
                    patience_counter = 0
                best_dev_score, best_dev_epoch = dev_score, epoch
            else:
                patience_counter += 1
                msg = f"No weights saved after epoch: {epoch}. " \
                      f"The dev score last improved after epoch: {best_dev_epoch}"
                logger.info(msg)

        # load back the best model dumped in temporary path and delete the temp folder
        msg = f"Setting the model weights to checkpoint whose dev " \
              f"{self.params.validation_metric} score is {best_dev_score:.4f}"
        logger.info(msg)
        # because we are loading to same device, no `map_location` specified
        self.load_state_dict(torch.load(temp_weights_save_path))
        shutil.rmtree(temp_folder)

        self.ready = True
        self.dirty = True

    @staticmethod
    def _validate_and_update_params(**params):

        # populate few required keys
        params.update({
            "add_terminals": params.get("add_terminals"),
            "padding_length": params.get("padding_length"),
            "tokenizer_type": params.get(
                "tokenizer_type",
                TokenizerType.WHITESPACE_TOKENIZER.value
            ),
            "label_padding_idx": LABEL_PAD_TOKEN_IDX,
        })

        # validate tokenizer_type param
        allowed_tokenizer_types = {
            EmbedderType.GLOVE: [TokenizerType.WHITESPACE_TOKENIZER, ],
            EmbedderType.BERT: [TokenizerType.HUGGINGFACE_PRETRAINED_TOKENIZER, ]
        }
        if params.get("embedder_type") and params.get("tokenizer_type"):
            embedder_type = EmbedderType(params.get("embedder_type"))
            tokenizer_type = TokenizerType(params.get("tokenizer_type"))
            if embedder_type in allowed_tokenizer_types:
                if tokenizer_type not in allowed_tokenizer_types[embedder_type]:
                    msg = f"For the selected choice of embedder ({embedder_type.value}), only " \
                          f"the following tokenizer_type are allowed: " \
                          f"{[v.value for v in allowed_tokenizer_types[embedder_type]]}."
                    raise ValueError(msg)

        # validate validation metric
        validation_metric = params.get("validation_metric")
        try:
            validation_metric = ValidationMetricType(validation_metric)
        except ValueError as e:
            msg = f"Expected validation_metric amongst " \
                  f"{[v.value for v in ValidationMetricType.__members__.values()]} " \
                  f"but found '{validation_metric}'."
            raise ValueError(msg) from e

        return params

    def _prepare_input_encoder(self, examples, **params) -> Dict:

        # create and fit encoder
        self.encoder = InputEncoderFactory.get_encoder_cls(params.get("tokenizer_type"))(**params)
        self.encoder.prepare(examples=examples)
        params.update({
            "num_tokens": len(self.encoder.get_vocab()),
            "padding_idx": self.encoder.get_pad_token_idx(),
        })
        return params

    def _prepare_embedder(self, **params) -> Dict:

        # check: cannot specify any conflicting params as required by child class
        if self.encoder is None:
            raise ValueError("An encoder must be first fitted before calling _prepare_embedder()")

        # check: cannot specify any conflicting params as required by child class
        embedder_type = params.get("embedder_type")
        if embedder_type == "glove":
            # load glove embs
            glove_container = GloVeEmbeddingsContainer()
            token2emb = glove_container.get_pretrained_word_to_embeddings_dict()
            glove_emb_dim = glove_container.token_dimension
            # validate emb_dim
            emb_dim = params.get("emb_dim", glove_emb_dim)
            if emb_dim != glove_emb_dim:
                msg = f"Provided 'emb_dim':{emb_dim} cannot be used with the provided " \
                      f"'embedder_type':{embedder_type}. Consider not specifying any 'emb_dim' " \
                      f"with this embedder."
                raise ValueError(msg)
            params.update({
                "embedder_type": embedder_type,
                "emb_dim": emb_dim,
                "embedding_weights": {
                    i: token2emb[t] for t, i in self.encoder.get_vocab().items() if t in token2emb
                },
            })
        elif embedder_type == "bert":
            # the bert model is directly loaded in _init_core() itself
            params.update({
                "embedder_type": embedder_type,
                "emb_dim": self.encoder.config.hidden_size,
                "pretrained_model_name_or_path": params.get("pretrained_model_name_or_path"),
            })

        if not params.get("emb_dim"):
            msg = "Need a valid 'emb_dim' to initialize embedding layers. To specify a " \
                  "particular dimension, either pass-in the 'emb_dim' param or provide a " \
                  "valid 'embedder_type' param."
            raise ValueError(msg)

        return params

    def _create_optimizer_and_scheduler(self, num_training_steps):
        del num_training_steps

        # load a torch optimizer
        optimizer = getattr(torch.optim, self.params.optimizer)(
            self.parameters(), lr=self.params.learning_rate
        )
        # load a constant lr scheduler
        scheduler = getattr(torch.optim.lr_scheduler, "LambdaLR")(optimizer, lambda _: 1)
        return optimizer, scheduler

    @abstractmethod
    def _prepare_labels(
        self, labels: Union[List[int], List[List[int]]], split_lengths: "Tensor2d[int]"
    ):
        raise NotImplementedError

    @abstractmethod
    def _init_graph(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(self, batch_data: BatchData) -> BatchData:
        raise NotImplementedError

    @abstractmethod
    def predict(self, examples: List[str]) -> Union[List[int], List[List[int]]]:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, examples: List[str]) -> Union[List[List[int]], List[List[List[int]]]]:
        raise NotImplementedError

    def dump(self, path: str):
        """
        The following states are dumped into different files:
            - Pytorch model weights
            - Encoder state
            - Params (including params such as tokenizer_type and emb_dim that are used during
                loading to create encoder and forward graph)
        """
        # resolve path and create associated folder if required
        path = os.path.abspath(os.path.splitext(path)[0]) + ".pytorch_model"
        os.makedirs(path, exist_ok=True)

        # save weights
        torch.save(self.state_dict(), os.path.join(path, "model.bin"))

        # save encoder's state
        self.encoder.dump(path)

        # save all params
        with open(os.path.join(path, "params.json"), "w") as fp:
            json.dump(dict(self.params), fp, indent=4)
            fp.close()
        msg = f"{self.name} model weights are dumped successfully"
        logger.info(msg)

        self.dirty = False

    @classmethod
    def load(cls, path: str):
        # resolve path
        path = os.path.abspath(os.path.splitext(path)[0]) + ".pytorch_model"

        # load all params
        with open(os.path.join(path, "params.json"), "r") as fp:
            all_params = json.load(fp)
            fp.close()

        # create new instance
        module = cls()
        if module.name != all_params["name"]:
            msg = f"The name of the loaded model ({all_params['name']}) from the path '{path}' " \
                  f"is different from the name of the module instantiated ({module.name})"
            raise AssertionError(msg)

        # load encoder's state
        module.params.update(dict(all_params))
        module.encoder = InputEncoderFactory.get_encoder_cls(
            tokenizer_type=all_params["tokenizer_type"])(**all_params)
        module.encoder.load(path)

        # load weights
        module._init_graph()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != module.params.device:
            msg = f"Model was dumped when on the device:{module.params.device} " \
                  f"but is not being loaded on device:{device}"
            logger.warning(msg)
            module.params.device = device
        module.load_state_dict(  # load model weights from checkpoint
            torch.load(os.path.join(path, "model.bin"), map_location=torch.device(device))
        )
        module.to(device)
        msg = f"{module.name} model weights are loaded successfully on device:{device}"
        logger.info(msg)

        module.ready = True
        module.dirty = False
        return module
