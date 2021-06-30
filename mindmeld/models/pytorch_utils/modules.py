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

import json
import logging
import os
import shutil
import uuid
from abc import abstractmethod
from typing import Dict

import numpy as np
from tqdm import tqdm

from .encoders import SequenceClassificationEncoder
from ...path import USER_CONFIG_DIR

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
except ImportError:
    pass

SEED = 7246

logger = logging.getLogger(__name__)


# utils

def get_disk_space_of_model(nn_model: nn.Module):
    filename = "temp.bin"
    torch.save(nn_model.state_dict(), filename)
    size = os.path.getsize(filename) / 1e6
    os.remove(filename)
    return size


def get_num_params_of_model(nn_model: nn.Module):
    n_total = 0
    n_requires_grad = 0
    for param in list(nn_model.parameters()):
        t = 1
        for sz in list(param.size()):
            t *= sz
        n_total += t
        if param.requires_grad:
            n_requires_grad += t
    return n_total, n_requires_grad


# Various nn layers


class EmbeddingLayer(nn.Module):
    """A pytorch wrapper layer for embeddings that takes input as a batched sequence of ids
    and outputs embeddings correponding to those ids
    """

    def __init__(self, num_tokens, emb_dim, padding_idx=None,
                 embedding_weights=None, update_embeddings=True,
                 coefficients=None, update_coefficients=True):
        super().__init__()

        self.embeddings = nn.Embedding(num_tokens, emb_dim, padding_idx=padding_idx)
        if embedding_weights is not None:
            if isinstance(embedding_weights, dict):
                # when weights are passed as dict with keys as indices and values as embeddings
                for idx, emb in embedding_weights.items():
                    self.embeddings.weight.data[idx] = torch.as_tensor(emb)
                msg = f"Initialized {len(embedding_weights)} number of embedding weights " \
                      f"from the embedder model"
                logger.info(msg)
            else:
                # when weights are passed as an array or tensor
                self.embeddings.load_state_dict({'weight': torch.as_tensor(embedding_weights)})
        self.embeddings.weight.requires_grad = update_embeddings

        self.embedding_for_coefficients = None
        if coefficients is not None:
            if not len(coefficients) == num_tokens:
                msg = f"Length of coefficients ({len(coefficients)}) must match the number of " \
                      f"embeddings ({num_tokens})"
                raise ValueError(msg)
            self.embedding_for_coefficients = nn.Embedding(num_tokens, 1, padding_idx=padding_idx)
            self.embedding_for_coefficients.load_state_dict({'weight': coefficients})
            self.embedding_for_coefficients.weight.requires_grad = update_coefficients

    def forward(self, padded_token_ids):
        # padded_token_ids: dim: [BS, SEQ_LEN]

        # [BS, SEQ_LEN] -> [BS, SEQ_LEN, EMB_DIM]
        outputs = self.embeddings(padded_token_ids)

        if self.embedding_for_coefficients:
            # [BS, SEQ_LEN] -> [BS, SEQ_LEN, 1]
            coefficients = self.embedding_for_coefficients(padded_token_ids)
            # [BS, SEQ_LEN, EMB_DIM] -> [BS, SEQ_LEN, EMB_DIM]
            outputs = torch.mul(outputs, coefficients)

        return outputs


class CnnLayer(nn.Module):

    def __init__(self, emb_dim, kernel_sizes, num_kernels):
        super().__init__()

        if isinstance(num_kernels, list) and len(num_kernels) != len(kernel_sizes):
            # incorrect length of num_kernels list specified
            num_kernels = [num_kernels[0]] * len(kernel_sizes)
        elif isinstance(num_kernels, int) and num_kernels > 0:
            # num_kernels is a single integer value
            num_kernels = [num_kernels] * len(kernel_sizes)
        elif not isinstance(num_kernels, list):
            msg = f"Invalid value for num_kernels: {num_kernels}. " \
                  f"Expected a list of same length as emb_dim ({len(emb_dim)})"
            raise ValueError(msg)

        self.convs = nn.ModuleList()
        # Unsqueeze input dim [BS, SEQ_LEN, EMD_DIM] to [BS, 1, SEQ_LEN, EMDDIM] and send as input
        # Each conv module output's dimensions are [BS, n, SEQ_LEN, 1]
        for length, n in zip(kernel_sizes, num_kernels):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(1, n, (length, emb_dim), padding=(length - 1, 0),
                              dilation=1, bias=True, padding_mode='zeros'),
                    nn.ReLU(),
                )
            )

    def forward(self, padded_token_embs):
        # padded_token_ids: dim: [BS, SEQ_LEN, EMD_DIM]

        # [BS, SEQ_LEN, EMD_DIM] -> [BS, 1, SEQ_LEN, EMD_DIM]
        embs_unsqueezed = torch.unsqueeze(padded_token_embs, dim=1)

        # [BS, 1, SEQ_LEN, EMD_DIM] -> list([BS, n, SEQ_LEN])
        conv_outputs = [conv(embs_unsqueezed).squeeze(3) for conv in self.convs]

        # list([BS, n, SEQ_LEN]) -> list([BS, n])
        maxpool_conv_outputs = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outputs]

        # list([BS, n]) -> [BS, sum(n)]
        outputs = torch.cat(maxpool_conv_outputs, dim=1)

        return outputs


class LstmLayer(nn.Module):

    def __init__(self, emb_dim, hidden_dim, num_layers, lstm_dropout, bidirectional):
        super().__init__()

        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers,
                            dropout=lstm_dropout, bidirectional=bidirectional, batch_first=True)

    def forward(self, padded_token_embs, lengths):
        # padded_token_ids: dim: [BS, SEQ_LEN, EMD_DIM]
        # lengths:          dim: [BS]

        # [BS, SEQ_LEN, EMD_DIM] -> [BS, SEQ_LEN, EMD_DIM*(2 if bidirectional else 1)]
        packed = pack_padded_sequence(padded_token_embs, lengths,
                                      batch_first=True, enforce_sorted=False)
        lstm_outputs, _ = self.lstm(packed)
        outputs = pad_packed_sequence(lstm_outputs, batch_first=True)[0]

        return outputs


class PoolingLayer(nn.Module):

    def __init__(self, pooling_type):
        super().__init__()

        ALLOWED_TYPES = ["cls", "max", "mean", "mean_sqrt"]
        assert pooling_type in ALLOWED_TYPES

        self.pooling_type = pooling_type

    def forward(self, padded_token_embs, lengths):
        # padded_token_ids: dim: [BS, SEQ_LEN, EMD_DIM]
        # lengths:          dim: [BS]
        # outputs:          dim: [BS, EMD_DIM]

        if self.pooling_type.lower() == "cls":
            last_seq_idxs = torch.LongTensor([x - 1 for x in lengths])
            outputs = padded_token_embs[range(padded_token_embs.shape[0]), last_seq_idxs, :]
        else:
            mask = pad_sequence([torch.as_tensor([1] * length_) for length_ in lengths],
                                batch_first=True)
            mask = mask.unsqueeze(-1).expand(padded_token_embs.size()).float()
            if self.pooling_type.lower() == "max":
                padded_token_embs[mask == 0] = -1e9  # set to a large negative value
                outputs, _ = torch.max(padded_token_embs, dim=1)[0]
            elif self.pooling_type.lower() == "mean":
                summed_padded_token_embs = torch.sum(padded_token_embs, dim=1)
                expanded_lengths = lengths.unsqueeze(dim=1).expand(summed_padded_token_embs.size())
                outputs = torch.div(summed_padded_token_embs, expanded_lengths)
            elif self.pooling_type.lower() == "mean_sqrt":
                summed_padded_token_embs = torch.sum(padded_token_embs, dim=1)
                expanded_lengths = lengths.unsqueeze(dim=1).expand(summed_padded_token_embs.size())
                outputs = torch.div(summed_padded_token_embs, torch.sqrt(expanded_lengths))

        return outputs


# Custom modules built on top of above nn layers that can do sequence classification


class SequenceClassification(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Base Module class that defines all the necessary elements to succesfully train/infer,
     dump/load custom pytorch modules wrapped on top of this base class. Classes derived from
     this base can be trained for sequence classification. The output of a class derived from
     this base must contain `seq_embs` in its output dictionary.
    """

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.encoder = SequenceClassificationEncoder()  # have to either fit ot load to use it
        self.params_keys = set(["name"])

        self.ready = False  # True when .fit() is called or loaded from a checkpoint
        self.dirty = False  # True when weights saved in a temp folder & to be moved to dump_folder

    def __repr__(self):
        return f"<{self.name}> ready: {self.ready} dirty: {self.dirty}"

    def model_description(self, print_it=True, log_it=True, return_it=False):
        msg = f"Who Am I: <{self.name}> ready: {self.ready} dirty: {self.dirty} \n" \
              f"\tNumber of weights (all, trainable): {get_num_params_of_model(self)} \n" \
              f"\tDisk Size (in MB): {get_disk_space_of_model(self):.4f}"
        if print_it:
            print(msg)
        if log_it:
            logger.info(msg)
        if return_it:
            return msg

    ####################################
    # methods for training and inference
    ####################################

    def fit(self, examples, labels, **params):  # pylint: disable=too-many-locals

        if self.ready:
            msg = "The model is already fitted or loaded from a file. Aborting fitting again."
            logger.error(msg)

        # fit an encoder
        self.encoder.fit(examples=examples, **params)

        # update params
        params.update({
            "num_tokens": self.encoder.get_num_tokens(),
            "emb_dim": self.encoder.get_emb_dim(),
            "padding_idx": self.encoder.pad_token_idx,
            "embedding_weights": self.encoder.get_embedding_weights(),
            "num_labels": len(set(labels)),
        })

        # init the graph
        self.init_graph(**params)

        # load required vars from params for fitting
        self.batch_size = params.get("batch_size", 32)
        self.optimizer = params.get("optimizer", "Adam")
        self.device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.n_epochs = params.get("n_epochs", 100)
        self.patience = params.get("patience", 6)
        self.verbose = params.get("verbose", True)
        self.params_keys.update(["batch_size", "optimizer", "device", "n_epochs", "patience",
                                 "verbose"])

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

        # create an optimizer and attach all model params to it
        optimizer = getattr(torch.optim, self.optimizer)(self.parameters(), lr=0.001)

        # print model stats
        self.model_description(print_it=self.verbose)

        # training and dev validation
        best_dev_acc, best_dev_epoch = -np.inf, -1
        for epoch in range(1, self.n_epochs + 1):
            # patience before terminating due to no dev accuracy improvements
            if epoch - best_dev_epoch > self.patience:
                msg = f"Set patience of {self.patience} epochs reached"
                logger.info(msg)
                break
            # set modules to train phase, reset gradients, do forward-backward propogations
            self.train()
            self.zero_grad()
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
                loss.backward()
                optimizer.step()
                self.zero_grad()
                progress_bar_msg = f"Epoch: {epoch} | Mean loss: " \
                                   f"{train_loss / (start_idx / self.batch_size + 1):.4f}"
                t.set_description(progress_bar_msg, refresh=True)
            train_loss = train_loss / train_batches
            # dev evaluation
            dev_acc = 0.0
            t = tqdm(range(0, len(dev_examples), self.batch_size), disable=not self.verbose)
            for start_idx in t:
                this_examples = dev_examples[start_idx:start_idx + self.batch_size]
                this_labels = dev_labels[start_idx:start_idx + self.batch_size]
                this_labels_predicted = self.predict(this_examples)
                assert len(this_labels_predicted) == len(this_labels), \
                    print(len(this_labels_predicted), len(this_labels))
                dev_acc += sum([x == y for x, y in zip(this_labels_predicted, this_labels)])
                progress_bar_msg = f"Epoch: {epoch} | Mean Validation Accuracy: " \
                                   f"{dev_acc / max(start_idx, 1):.4f}"
                t.set_description(progress_bar_msg, refresh=True)
            dev_acc /= max(1, len(dev_examples))
            if dev_acc >= best_dev_acc:
                # save model weights in a temp folder; later move it to folder passed through dump()
                torch.save(self.state_dict(), temp_save_path)
                msg = f"Model weights saved in epoch: {epoch} when dev accuracy improved " \
                      f"from '{best_dev_acc:.4f}' to '{dev_acc:.4f}'"
                logger.info(msg)
                best_dev_acc, best_dev_epoch = dev_acc, epoch

        # load the best model, delete the temp folder and return
        self.load_state_dict(torch.load(temp_save_path))
        shutil.rmtree(temp_folder)

        self.ready = True
        self.dirty = True

    def predict(self, examples):
        logits = self.forward_with_batching_and_no_grad(examples)
        if self.num_labels == 2:
            preds = (logits >= 0.5).long()
        elif self.num_labels > 2:
            preds = torch.argmax(logits, dim=-1)
        return preds.cpu().detach().numpy().tolist()

    def predict_proba(self, examples):
        logits = self.forward_with_batching_and_no_grad(examples)
        if self.num_labels == 2:
            probs = F.sigmoid(logits)
            # extending the results from shape [N,1] to [N,2] to give out class probs distinctly
            probs = torch.cat((1 - probs, probs), dim=-1)
        elif self.num_labels > 2:
            probs = F.softmax(logits, dim=-1)
        return probs.cpu().detach().numpy().tolist()

    def forward_with_batching_and_no_grad(self, examples):
        logits = None
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for start_idx in range(0, len(examples), self.batch_size):
                this_examples = examples[start_idx:start_idx + self.batch_size]
                batch_data_dict = self.encoder.batch_encode(this_examples)
                _logits = self.forward(batch_data_dict)["logits"]
                logits = torch.cat((logits, _logits)) if logits is not None else _logits
        if was_training:
            self.train()
        return logits

    def forward(self, batch_data_dict):

        for k, v in batch_data_dict.items():
            if v is not None and isinstance(v, torch.Tensor):
                batch_data_dict[k] = v.to(self.device)

        batch_data_dict = self._forward_core(batch_data_dict)

        seq_embs = batch_data_dict["seq_embs"]
        seq_embs = self.dropout(seq_embs)
        logits = self.classifier_head(seq_embs)
        batch_data_dict.update({"logits": logits})

        targets = batch_data_dict.get("labels")
        if targets is not None:
            loss = self.criterion(logits, targets)
            batch_data_dict.update({"loss": loss})

        return batch_data_dict

    @abstractmethod
    def _forward_core(self, batch_data_dict: Dict) -> Dict:
        raise NotImplementedError

    ####################################
    # methods to load and dump resources
    ####################################

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
        module.init_graph(**params)
        # load encoder
        module.encoder = module.encoder.load(path)  # .load() is a classmethod
        # load weights
        module.load_state_dict(torch.load(os.path.join(path, "pytorch_model.bin")))
        msg = f"{module.name} model weights are loaded successfully"
        logger.info(msg)

        module.ready = True
        module.dirty = False
        return module

    ####################################
    # methods for creating pytorch graph
    ####################################

    def init_graph(self, **params):

        self._init_core(**params)

        # params
        self.num_labels = params.get("num_labels")
        self.hidden_dropout_prob = params.get("hidden_dropout_prob", 0.3)
        self.params_keys.update(["num_labels", "hidden_dropout_prob"])

        # init the underlying params and architectural components
        try:
            self.hidden_size = self.out_dim
            assert self.hidden_size
        except (AttributeError, AssertionError) as e:
            msg = f"Derived class '{self.name}' must indicate its hidden size for dense layer " \
                  f"classification by having an attribute 'self.out_dim', which must be a " \
                  f"positive integer"
            raise ValueError(msg) from e
        self.params_keys.update(["hidden_size"])

        # init the peripheral architecture params and architectural components
        if not self.num_labels:
            msg = f"Invalid number of labels ({self.num_labels}) inputted for '{self.name}' class"
            raise ValueError(msg)
        self.dropout = nn.Dropout(p=self.hidden_dropout_prob)
        self.classifier_head = nn.Linear(self.hidden_size, self.num_labels)

        # init the criterion to compute loss
        if self.num_labels == 2:
            # sigmoid criterion
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        elif self.num_labels > 2:
            # cross-entropy criterion
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        else:
            msg = f"Invalid number of labels specified: {self.num_labels}. " \
                  f"Valid number is equal to or greater than 2"
            raise ValueError(msg)

        print(f"{self.name} is initialized")

    @abstractmethod
    def _init_core(self, **params):
        raise NotImplementedError


class EmbeddingForSequenceClassification(SequenceClassification):
    """An embedder pooling module that operates on a batched sequence of token ids. The
    tokens could be characters or words or sub-words. This module finally outputs one 1D
    representation for each instance in the batch (i.e. [BS, EMB_DIM]).

    The `forward` method of this module expects padded token ids along with numer of tokens
    per instance in the batch.

    Additionally, one can set different coefficients for different tokens of the embedding
    matrix (e.g. tf-idf weights).
    """

    def _init_core(self, **params):
        # params
        self.num_tokens = params["num_tokens"]
        self.emb_dim = params["emb_dim"]
        self.padding_idx = params.get("padding_idx", None)
        embedding_weights = params.get("embedding_weights", None)
        self.update_embeddings = params.get("update_embeddings", True)
        self.embedder_output_pooling_type = params.get("embedder_output_pooling_type", "mean")
        self.params_keys.update([
            "num_tokens", "emb_dim", "padding_idx", "update_embeddings",
            "embedder_output_pooling_type"
        ])

        # core layers
        self.emb_layer = EmbeddingLayer(self.num_tokens, self.emb_dim, self.padding_idx,
                                        embedding_weights, self.update_embeddings)
        self.emb_layer_pooling = PoolingLayer(self.embedder_output_pooling_type)
        self.out_dim = self.emb_dim

    def _forward_core(self, batch_data_dict):
        seq_ids = batch_data_dict["seq_ids"]  # [BS, SEQ_LEN]
        seq_lengths = batch_data_dict["seq_lengths"]  # [BS]

        encodings = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, EMD_DIM]
        encodings = self.emb_layer_pooling(encodings, seq_lengths)  # [BS, self.out_dim]

        batch_data_dict.update({"seq_embs": encodings})

        return batch_data_dict


class SequenceCnnForSequenceClassification(SequenceClassification):
    """A CNN module that operates on a batched sequence of token ids. The tokens could be
    characters or words or sub-words. This module finally outputs one 1D representation
    for each instance in the batch (i.e. [BS, EMB_DIM]).

    The `forward` method of this module expects only padded token ids as input.
    """

    def _init_core(self, **params):
        # params
        self.num_tokens = params["num_tokens"]
        self.emb_dim = params["emb_dim"]
        self.padding_idx = params.get("padding_idx", None)
        embedding_weights = params.get("embedding_weights", None)
        self.update_embeddings = params.get("update_embeddings", True)
        self.cnn_kernel_sizes = params.get("cnn_kernel_sizes", [1, 3, 5])
        self.cnn_num_kernels = params.get("cnn_num_kernels", [100] * len(self.cnn_kernel_sizes))

        self.params_keys.update([
            "num_tokens", "emb_dim", "padding_idx", "update_embeddings",
            "cnn_kernel_sizes", "cnn_num_kernels"
        ])

        # core layers
        self.emb_layer = EmbeddingLayer(self.num_tokens, self.emb_dim, self.padding_idx,
                                        embedding_weights, self.update_embeddings)
        self.conv_layer = CnnLayer(self.emb_dim, self.cnn_kernel_sizes, self.cnn_num_kernels)
        self.out_dim = sum(self.cnn_num_kernels)

    def _forward_core(self, batch_data_dict):
        seq_ids = batch_data_dict["seq_ids"]  # [BS, SEQ_LEN]

        encodings = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, EMD_DIM]
        encodings = self.conv_layer(encodings)  # [BS, self.out_dim]

        batch_data_dict.update({"seq_embs": encodings})

        return batch_data_dict


class SequenceLstmForSequenceClassification(SequenceClassification):
    # pylint: disable=too-many-instance-attributes
    """A LSTM module that operates on a batched sequence of token ids. The tokens could be
    characters or words or sub-words. This module finally outputs one 1D representation
    for each instance in the batch (i.e. [BS, EMB_DIM]).

    The `forward` method of this module expects padded token ids along with numer of tokens
    per instance in the batch.
    """

    def _init_core(self, **params):
        # params
        self.num_tokens = params["num_tokens"]
        self.emb_dim = params["emb_dim"]
        self.padding_idx = params.get("padding_idx", None)
        embedding_weights = params.get("embedding_weights", None)
        self.update_embeddings = params.get("update_embeddings", True)
        self.lstm_hidden_dim = params.get("lstm_hidden_dim", 128)
        self.lstm_num_layers = params.get("lstm_num_layers", 2)
        self.lstm_dropout = params.get("lstm_dropout", 0.3)
        self.lstm_bidirectional = params.get("lstm_bidirectional", True)
        self.lstm_output_pooling_type = params.get("lstm_output_pooling_type", "cls")
        self.params_keys.update([
            "num_tokens", "emb_dim", "padding_idx", "update_embeddings",
            "lstm_hidden_dim", "lstm_num_layers", "lstm_dropout", "lstm_bidirectional"
        ])

        # core layers
        self.emb_layer = EmbeddingLayer(self.num_tokens, self.emb_dim, self.padding_idx,
                                        embedding_weights, self.update_embeddings)
        self.lstm_layer = LstmLayer(self.emb_dim, self.lstm_hidden_dim, self.lstm_num_layers,
                                    self.lstm_dropout, self.lstm_bidirectional)
        self.lstm_layer_pooling = PoolingLayer(self.lstm_output_pooling_type)
        self.out_dim = self.lstm_hidden_dim * 2 if self.lstm_bidirectional else self.lstm_hidden_dim

    def _forward_core(self, batch_data_dict):
        seq_ids = batch_data_dict["seq_ids"]  # [BS, SEQ_LEN]
        seq_lengths = batch_data_dict["seq_lengths"]  # [BS]

        encodings = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, EMD_DIM]
        encodings = self.lstm_layer(encodings, seq_lengths)  # [BS, SEQ_LEN, self.out_dim]
        encodings = self.lstm_layer_pooling(encodings, seq_lengths)  # [BS, self.out_dim]

        batch_data_dict.update({"seq_embs": encodings})

        return batch_data_dict


# Custom modules built on top of above nn layers that can do token classification


class BaseForTokenClassification(nn.Module):

    def __init__(self, **params):
        raise NotImplementedError


class EmbeddingForTokenClassification(BaseForTokenClassification):

    def __init__(self, **params):
        raise NotImplementedError


class SequenceLstmForTokenClassification(BaseForTokenClassification):
    """A LSTM module that operates on a batched sequence of token ids. The tokens could be
    characters or words or sub-words. This module uses an additional input that determines
    how the sequence of embeddings obtained after the LSTM layers for each instance in the
    batch, needs to be split. Once split, the sub-groups of embeddings (each sub-group
    corresponding to a word or a phrase) can be collapsed to 1D representation per sub-group
    through pooling operations. Finally, this module outputs a 2D representation for each
    instance in the batch (i.e. [BS, SEQ_LEN', EMB_DIM]).
    """

    def __init__(self, **params):
        raise NotImplementedError


class TokenLstmSequenceLstmForTokenClassification(BaseForTokenClassification):

    def __init__(self, **params):
        raise NotImplementedError


class TokenCnnSequenceLstmForTokenClassification(BaseForTokenClassification):

    def __init__(self, **params):
        raise NotImplementedError


# Custom modules built on top of above nn layers that can do joint classification

class BaseForJointSequenceTokenClassification(nn.Module):
    """This base class is wrapped around nn.Module and supports joint modeling, meaning
    multiple heads can be trained for models derived on top of this base class. Unlike classes
    derive on top of SequenceClassification or BaseForTokenClassification, the ones derived
    on this base output both `seq_embs` as well as `token_embs` in their output which facilitates
    multi-head training.
    """

    def __init__(self, **params):
        raise NotImplementedError
