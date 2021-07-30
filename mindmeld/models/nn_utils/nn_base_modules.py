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
from typing import Dict, Union, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from .layers import (
    get_disk_space_of_model,
    get_num_weights_of_model
)
from .._util import _get_module_or_attr
from ...core import Bunch
from ...path import USER_CONFIG_DIR

try:
    import torch
    import torch.nn as nn

    nn_module = _get_module_or_attr("torch.nn", "Module")
    is_cuda_available = _get_module_or_attr("torch.cuda", "is_available")()
except ImportError:
    nn_module = object
    is_cuda_available = False
    pass

SEED = 6174

logger = logging.getLogger(__name__)


class ClassificationBase(nn_module):
    DEFAULT_PARAMS = {
        "device": "cuda" if is_cuda_available else "cpu",
        "number_of_epochs": 100,
        "patience": 10,
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": None,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "validation_metric": "accuracy",  # or 'f1'
        "verbose": True,  # to print progress to stdout/log file
        "dev_split_ratio": 0.8
    }

    def __init__(self):
        super().__init__()

        self.name = self.__class__.__name__
        self.params = Bunch()
        self.params["name"] = self.name

        self.ready = False  # True when .fit() is called or loaded from a checkpoint
        self.dirty = False  # True when weights saved in a temp folder & to be moved to dump_folder

        self.encoder = None

    def __repr__(self):
        return f"<{self.name}> ready:{self.ready} dirty:{self.dirty}"

    def who_am_i(self) -> str:
        msg = f"Who Am I: <{self.name}> " \
              f"ready: {self.ready} dirty: {self.dirty} device:{self.params.device}\n" \
              f"\tNumber of weights (trainable, all): {get_num_weights_of_model(self)} \n" \
              f"\tDisk Size (in MB): {get_disk_space_of_model(self):.4f}"
        logger.info(msg)
        return msg

    # methods to load and dump resources, common across sub-classes

    def dump(self, path):
        # resolve path and create associated folder if required
        path = os.path.abspath(os.path.splitext(path)[0]) + ".pytorch_model"
        os.makedirs(path, exist_ok=True)

        # save weights
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))

        # save encoder
        self.encoder.dump(path)

        # save params
        with open(os.path.join(path, "params.json"), "w") as fp:
            json.dump(dict(self.params), fp, indent=4)
            fp.close()
        msg = f"{self.name} model weights are dumped successfully"
        logger.info(msg)

        self.dirty = False

    @classmethod
    def load(cls, path):
        # resolve path
        path = os.path.abspath(os.path.splitext(path)[0]) + ".pytorch_model"

        # create instance and populate with params
        module = cls()
        with open(os.path.join(path, "params.json"), "r") as fp:
            all_params = json.load(fp)
            fp.close()

        # validate name
        if module.name != all_params["name"]:
            msg = f"The name of the loaded model ({all_params['name']}) from the path '{path}' " \
                  f"is different from the name of the module instantiated ({module.name})"
            raise AssertionError(msg)

        # load resources
        module.encoder = module.encoder.load(path)  # .load() is a classmethod here
        module.params.update(dict(all_params))
        module._init_forward_graph()

        # load weights
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != module.params.device:
            msg = f"Model was trained on '{module.params.device}' but is being loaded on {device}"
            module.params.device = device
            logger.warning(msg)
        module.load_state_dict(torch.load(os.path.join(path, "pytorch_model.bin"),
                                          map_location=torch.device(device)))
        msg = f"{module.name} model weights are loaded successfully"
        logger.info(msg)

        module.ready = True
        module.dirty = False
        return module

    # methods for gpu usage and data loading

    def _to_device(self, batch_data_dict: Dict) -> Dict:
        for k, v in batch_data_dict.items():
            if v is not None and isinstance(v, torch.Tensor):
                batch_data_dict[k] = v.to(self.params.device)
        return batch_data_dict

    # methods for training

    def fit(self, examples, labels, **params):  # pylint: disable=too-many-locals

        if self.ready:
            msg = "The model is already fitted or is loaded from a file. Aborting re-fitting."
            logger.error(msg)

        # fit an encoder and update params
        params = self._init_encoder(examples, **params)

        # update number of labels
        try:
            num_labels = len(set(labels))
        except TypeError:  # raised in cased on token classification
            num_labels = len(set(sum(labels, [])))
        params.update({"num_labels": num_labels})

        # update label pad idx; used by token classifiers in crf masks and for evaluation
        try:
            self.label_padding_idx = params["label_padding_idx"]
            params.update({"label_padding_idx": params["label_padding_idx"]})
        except KeyError:  # raised in cased on sequence classification
            pass

        # use all default params ans inputted params to update self.params
        self.params.update({**self.get_default_params(), **params})

        # init the graph
        self._init_forward_graph()

        # move model to device
        self.to(self.params.device)

        # print model stats
        self.who_am_i()

        # create temp folder and a save path
        # dumping into a temp folder instead of keeping in memory to reduce memory usage
        temp_folder = os.path.join(USER_CONFIG_DIR, "pytorch_models", str(uuid.uuid4()))
        os.makedirs(temp_folder, exist_ok=True)
        temp_weights_save_path = os.path.join(temp_folder, "pytorch_model.bin")

        # split into train, dev splits and get data loaders
        indices = np.arange(len(examples))
        np.random.seed(SEED)
        np.random.shuffle(indices)
        train_examples, train_labels = zip(*[
            (examples[i], labels[i])
            for i in indices[:int(self.params.dev_split_ratio * len(indices))]
        ])
        dev_examples, dev_labels = zip(*[
            (examples[i], labels[i])
            for i in indices[int(self.params.dev_split_ratio * len(indices)):]
        ])

        # create an optimizer and attach all model params to it
        num_training_steps = int(
            len(train_examples) / self.params.batch_size / self.params.gradient_accumulation_steps *
            self.params.number_of_epochs
        )
        optimizer, scheduler = self._create_optimizer_and_scheduler(num_training_steps)

        # training and dev validation
        self.verbose = self.params.pop("verbose", True)
        best_dev_score, best_dev_epoch = -np.inf, -1
        msg = f"Beginning to train for {self.params.number_of_epochs} number of epochs"
        logger.info(msg)
        if self.params.number_of_epochs < 1:
            raise ValueError("Param 'number_of_epochs' must be a positive integer greater than 0")
        _patience_counter = 0
        for epoch in range(1, self.params.number_of_epochs + 1):
            # patience before terminating due to no dev score improvements
            if _patience_counter >= self.params.patience:
                msg = f"Set patience of {self.params.patience} epochs reached"
                logger.info(msg)
                break
            # set modules to train phase, reset gradients, do forward-backward propogations
            self.train()
            optimizer.zero_grad()
            train_loss, train_batches = 0.0, 0.0
            t = tqdm(range(0, len(train_examples), self.params.batch_size),
                     disable=not self.verbose)
            for start_idx in t:
                this_examples = train_examples[start_idx:start_idx + self.params.batch_size]
                this_labels = train_labels[start_idx:start_idx + self.params.batch_size]
                batch_data_dict = self.encoder.batch_encode(this_examples, this_labels)
                batch_data_dict = self.forward(batch_data_dict)
                loss = batch_data_dict["loss"]
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
                    if self.params.max_grad_norm:  # clip (accumlated) gradients if required
                        nn.utils.clip_grad_norm_(self.parameters(), self.params.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                progress_bar_msg = f"Epoch: {epoch} | Mean loss: " \
                                   f"{train_loss / (start_idx / self.params.batch_size + 1):.4f}"
                t.set_description(progress_bar_msg, refresh=True)
            train_loss = train_loss / train_batches
            # dev evaluation
            predictions, targets = [], []
            t = tqdm(range(0, len(dev_examples), self.params.batch_size), disable=not self.verbose)
            for start_idx in t:
                this_examples = dev_examples[start_idx:start_idx + self.params.batch_size]
                this_labels_predicted = self.predict(this_examples)
                this_labels_targetted = dev_labels[start_idx:start_idx + self.params.batch_size]
                # validation
                if len(this_labels_predicted) != len(this_labels_targetted):
                    msg = f"Number of predictions ({len(this_labels_predicted)}) " \
                          f"not equal to number of targets ({len(this_labels_targetted)})"
                    logger.error(msg)
                    raise AssertionError(msg)
                # flatten if required
                try:
                    this_labels_predicted = sum(this_labels_predicted, [])
                    this_labels_targetted = sum(this_labels_targetted, [])
                except TypeError:
                    # raised in case of sequence classification; implies already flattened
                    pass
                # discard unwanted predictions using label_padding_idx, if available
                if hasattr(self, "label_padding_idx"):
                    this_labels_predicted, this_labels_targetted = zip(*[
                        (x, y) for x, y in zip(this_labels_predicted, this_labels_targetted)
                        if y != self.label_padding_idx
                    ])
                predictions.extend(this_labels_predicted)
                targets.extend(this_labels_targetted)
                progress_bar_msg = f"Epoch: {epoch} | " \
                                   f"Validation Metric: {self.params.validation_metric} "
                t.set_description(progress_bar_msg, refresh=True)
            # compute score
            if self.params.validation_metric == "accuracy":
                dev_score = accuracy_score(targets, predictions, normalize=True)
            elif self.params.validation_metric == "f1":
                dev_score = f1_score(targets, predictions, average='weighted')
            else:
                msg = f"Invalid 'validation_metric' ({self.params.validation_metric}) provided " \
                      f"in params. Allowed values are only 'accuracy' and 'f1'"
            # update patience counter
            if dev_score < best_dev_score:
                _patience_counter += 1
                msg = f"No weights saved after epoch: {epoch}. " \
                      f"The dev score last improved after epoch: {best_dev_epoch}"
                logger.info(msg)
            # save model in a temp path if required
            if dev_score >= best_dev_score:
                # save model weights in a temp folder; later move it to folder passed through dump()
                torch.save(self.state_dict(), temp_weights_save_path)
                phrase = (
                    f"improved from '{best_dev_score:.4f}' to" if dev_score > best_dev_score
                    else "remained at"
                )
                msg = f"\nModel weights saved after epoch: {epoch} when dev score {phrase} " \
                      f"'{dev_score:.4f}'"
                logger.info(msg)
                # update patience counter
                if dev_score == best_dev_score:
                    _patience_counter += 1
                else:
                    _patience_counter = 0
                best_dev_score, best_dev_epoch = dev_score, epoch

        # load the best model, delete the temp folder and return
        msg = f"Setting the model weights to checkpoint whose dev score " \
              f"('{self.params.validation_metric}') was {best_dev_score:.4f}"
        logger.info(msg)
        # because we are loading to same device, no `map_location` specified
        self.load_state_dict(torch.load(temp_weights_save_path))
        shutil.rmtree(temp_folder)

        self.ready = True
        self.dirty = True

    def get_default_params(self) -> Dict:
        return {
            **ClassificationBase.DEFAULT_PARAMS,
            **self._get_subclasses_default_params(),
        }

    def _create_optimizer_and_scheduler(self, num_training_steps):
        del num_training_steps

        # load a torch optimizer
        optimizer = getattr(torch.optim, self.params.optimizer)(
            self.parameters(), lr=self.params.learning_rate
        )
        # load a constant lr scheduler
        scheduler = getattr(torch.optim.lr_scheduler, "LambdaLR")(optimizer, lambda _: 1)
        return optimizer, scheduler

    # abstract methods definitions, to be implemented by sub-classes

    @abstractmethod
    def _init_encoder(self, examples, **kwargs) -> Dict:
        # return updated kwargs dict
        raise NotImplementedError

    @abstractmethod
    def _get_subclasses_default_params(self) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def _init_forward_graph(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(self, batch_data_dict) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def predict(self, examples) -> Union[List[int], List[List[int]]]:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, examples) -> Union[List[List[int]], List[List[List[int]]]]:
        raise NotImplementedError
