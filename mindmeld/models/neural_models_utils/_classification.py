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

from ._layers import (
    get_disk_space_of_model,
    get_num_params_of_model
)
from ...path import USER_CONFIG_DIR

try:
    import torch
    import torch.nn as nn
except ImportError:
    pass

SEED = 6174

logger = logging.getLogger(__name__)


class ClassificationCore(nn.Module):  # pylint: disable=too-many-instance-attributes

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.params_keys = set(["name"])

        self.ready = False  # True when .fit() is called or loaded from a checkpoint
        self.dirty = False  # True when weights saved in a temp folder & to be moved to dump_folder

        self.encoder = None

    def __repr__(self):
        return f"<{self.name}> ready: {self.ready} dirty: {self.dirty}"

    def model_description(self, log_it=True, return_it=False):
        msg = f"Who Am I: <{self.name}> ready: {self.ready} dirty: {self.dirty} \n" \
              f"\tNumber of weights (trainable, all): {get_num_params_of_model(self)} \n" \
              f"\tDisk Size (in MB): {get_disk_space_of_model(self):.4f}"
        if log_it:
            logger.info(msg)
        if return_it:
            return msg

    # methods for training

    def fit(self, examples, labels, **params):  # pylint: disable=too-many-locals

        if self.ready:
            msg = "The model is already fitted or is loaded from a file. Aborting re-fitting."
            logger.error(msg)

        # fit an encoder and update params
        params = self._fit_encoder_and_update_params(examples, **params)

        # update number of labels and label pad idx
        try:
            num_labels = len(set(labels))
        except TypeError:  # raised in cased on token classification
            num_labels = len(set(sum(labels, [])))
        params.update({"num_labels": num_labels})
        try:
            # label pad idx used by token classifiers in crf masks and for evaluation
            self.label_padding_idx = params["label_padding_idx"]
            self.params_keys.update(["label_padding_idx"])
        except KeyError:  # raised in cased on sequence classification
            pass

        # init the graph
        self.init(**params)

        # load required vars from params for fitting
        self.device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.number_of_epochs = params.get("number_of_epochs", 100)
        self.patience = params.get("patience", 6)
        self.batch_size = params.get("batch_size", 32)
        self.gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = params.get("max_grad_norm", None)
        self.optimizer = params.get("optimizer", "Adam")
        self.learning_rate = params.get("learning_rate", 0.001)
        self.validation_metric = params.get("validation_metric", "accuracy").lower()  # or 'f1'
        self.verbose = params.get("verbose", True)  # to print progress to stdout/log file
        self.params_keys.update([
            "device", "number_of_epochs", "patience", "batch_size", "gradient_accumulation_steps",
            "max_grad_norm", "optimizer", "learning_rate", "validation_metric", "verbose"
        ])

        # create temp folder and a save path
        # dumping into a temp folder instead of keeping in memory to reduce memory usage
        temp_folder = os.path.join(USER_CONFIG_DIR, "pytorch_models", str(uuid.uuid4()))
        os.makedirs(temp_folder, exist_ok=True)
        temp_save_path = os.path.join(temp_folder, "pytorch_model.bin")

        # split into train, dev splits and get data loaders
        indices = np.arange(len(examples))
        np.random.seed(SEED)
        np.random.shuffle(indices)
        train_examples, train_labels = \
            zip(*[(examples[i], labels[i]) for i in indices[:int(0.8 * len(indices))]])
        dev_examples, dev_labels = \
            zip(*[(examples[i], labels[i]) for i in indices[int(0.8 * len(indices)):]])

        # move model to device
        self.to(self.device)

        # print model stats
        self.model_description()

        # create an optimizer and attach all model params to it
        num_training_steps = int(
            len(train_examples) / self.batch_size / self.gradient_accumulation_steps *
            self.number_of_epochs
        )
        optimizer, scheduler = self._create_optimizer_and_scheduler(num_training_steps)

        # training and dev validation
        best_dev_score, best_dev_epoch = -np.inf, -1
        msg = f"Beginning to train for {self.number_of_epochs} number of epochs"
        logger.info(msg)
        for epoch in range(1, self.number_of_epochs + 1):
            # patience before terminating due to no dev score improvements
            if epoch - best_dev_epoch > self.patience:
                msg = f"Set patience of {self.patience} epochs reached"
                logger.info(msg)
                break
            # set modules to train phase, reset gradients, do forward-backward propogations
            self.train()
            optimizer.zero_grad()
            train_loss, train_batches = 0.0, 0.0
            t = tqdm(range(0, len(train_examples), self.batch_size), disable=not self.verbose)
            for start_idx in t:
                this_examples = train_examples[start_idx:start_idx + self.batch_size]
                this_labels = train_labels[start_idx:start_idx + self.batch_size]
                batch_data_dict = self.encoder.batch_encode(this_examples, this_labels)
                batch_data_dict = self.forward(batch_data_dict)
                loss = batch_data_dict["loss"]
                train_loss += loss.cpu().detach().numpy()
                train_batches += 1
                # find gradients
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                # optimizer and scheduler step
                batch_id = start_idx / self.batch_size
                if (
                    start_idx + self.batch_size >= len(train_examples) or
                    (batch_id + 1) % self.gradient_accumulation_steps == 0
                ):
                    # update weights when it is the last batch in the epoch or
                    # when specified step is reached or
                    if self.max_grad_norm:  # clip (accumlated) gradients if required
                        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                progress_bar_msg = f"Epoch: {epoch} | Mean loss: " \
                                   f"{train_loss / (start_idx / self.batch_size + 1):.4f}"
                t.set_description(progress_bar_msg, refresh=True)
            train_loss = train_loss / train_batches
            # dev evaluation
            predictions, targets = [], []
            t = tqdm(range(0, len(dev_examples), self.batch_size), disable=not self.verbose)
            for start_idx in t:
                this_examples = dev_examples[start_idx:start_idx + self.batch_size]
                this_labels_predicted = self.predict(this_examples)
                this_labels_targetted = dev_labels[start_idx:start_idx + self.batch_size]
                # validation
                if len(this_labels_predicted) != len(this_labels_targetted):
                    msg = f"Number of predictions ({len(this_labels_predicted)}) " \
                          f"not equal to number of targets ({len(this_labels_targetted)})"
                    logger.error(msg)
                    raise AssertionError(msg)
                # flatten if required and discard unwanted predictions using label_padding_idx
                try:
                    this_labels_predicted = sum(this_labels_predicted, [])
                    this_labels_targetted = sum(this_labels_targetted, [])
                except TypeError:
                    # raised in cased on sequence classification, implies already flattened
                    pass
                if hasattr(self, "label_padding_idx"):
                    this_labels_predicted, this_labels_targetted = zip(*[
                        (x, y) for x, y in zip(this_labels_predicted, this_labels_targetted)
                        if y != self.label_padding_idx
                    ])
                predictions.extend(this_labels_predicted)
                targets.extend(this_labels_targetted)
                progress_bar_msg = f"Epoch: {epoch} | Validation Metric: {self.validation_metric} "
                t.set_description(progress_bar_msg, refresh=True)
            # compute score
            if self.validation_metric == "accuracy":
                dev_score = accuracy_score(targets, predictions, normalize=True)
            elif self.validation_metric == "f1":
                dev_score = f1_score(targets, predictions, average='weighted')
            else:
                msg = f"Invalid 'validation_metric' ({self.validation_metric}) provided in " \
                      f"params. Allowed values are only 'accuracy' and 'f1'"
            # save model in a temp path if required
            if dev_score >= best_dev_score:
                # save model weights in a temp folder; later move it to folder passed through dump()
                torch.save(self.state_dict(), temp_save_path)
                msg = f"\nModel weights saved after epoch: {epoch} when dev score improved " \
                      f"from '{best_dev_score:.4f}' to '{dev_score:.4f}'"
                logger.info(msg)
                best_dev_score, best_dev_epoch = dev_score, epoch

        # load the best model, delete the temp folder and return
        self.load_state_dict(torch.load(
            temp_save_path))  # because we are loading to same device, no `map_location` specified
        shutil.rmtree(temp_folder)

        self.ready = True
        self.dirty = True

    def _create_optimizer_and_scheduler(self, num_training_steps):
        del num_training_steps
        # load a torch optimizer
        optimizer = getattr(torch.optim, self.optimizer)(self.parameters(), lr=self.learning_rate)
        # load a constant lr scheduler
        scheduler = getattr(torch.optim.lr_scheduler, "LambdaLR")(optimizer, lambda _: 1)
        return optimizer, scheduler

    # methods to load and dump resources

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
            params_dict = {k: getattr(self, k) for k in self.params_keys}
            json.dump(params_dict, fp, indent=4)
            fp.close()
        msg = f"{self.name} model weights are dumped successfully"
        logger.info(msg)

        self.dirty = False

    @classmethod
    def load(cls, path):
        # resolve path
        path = os.path.abspath(os.path.splitext(path)[0]) + ".pytorch_model"
        # create instance and populate
        module = cls()
        # load params
        with open(os.path.join(path, "params.json"), "r") as fp:
            params = json.load(fp)
            for k, v in params.items():
                setattr(module, k, v)
            setattr(module, "params_keys", set(params.keys()))
            fp.close()
        module.init(**params)
        # load encoder
        module.encoder = module.encoder.load(path)  # .load() is a classmethod
        # load weights
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != params.get("device"):
            msg = f"Model trained on '{params.get('device')}' but is being loaded on to {device}"
            logger.warning(msg)
        module.load_state_dict(torch.load(
            os.path.join(path, "pytorch_model.bin"), map_location=torch.device(device)))
        msg = f"{module.name} model weights are loaded successfully"
        logger.info(msg)

        module.ready = True
        module.dirty = False
        return module

    # abstract methods definition, to be implemented by sub-classes

    @abstractmethod
    def init(self, **params) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(self, batch_data_dict) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def predict(self, examples) -> Union[List[int], List[List[int]]]:
        raise NotImplementedError

    @abstractmethod
    def _fit_encoder_and_update_params(self, examples, **kwargs) -> Dict:
        # return updated kwargs dict
        raise NotImplementedError
