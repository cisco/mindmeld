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
from abc import abstractmethod
from typing import Dict

import nn.functional as F
import torch
import torch.nn as nn
from nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logger = logging.getLogger(__name__)


# Different nn layers


class EmbeddingLayer(nn.Module):

    def __init__(self, num_embs, emb_dim, padding_idx=None,
                 embedding_weights=None, update_embeddings=True,
                 weightages=None, update_weightages=True):
        super().__init__()

        self.embeddings = nn.Embedding(num_embs, emb_dim, padding_idx=padding_idx)
        if embedding_weights:
            self.embeddings.load_state_dict({'weight': embedding_weights})
        self.embeddings.weight.requires_grad = update_embeddings

        self.embedding_weightages = None
        if weightages:
            if not len(weightages) == num_embs:
                msg = f"Length of weightages ({len(weightages)}) must match the number of " \
                      f"embeddings ({num_embs})"
                raise ValueError(msg)
            self.embedding_weightages = nn.Embedding(num_embs, 1, padding_idx=padding_idx)
            self.embedding_weightages.load_state_dict({'weight': weightages})
            self.embedding_weightages.weight.requires_grad = update_weightages

    def forward(self, padded_token_ids):
        # padded_token_ids: dim: [BS, SEQ_LEN]

        # [BS, SEQ_LEN] -> [BS, SEQ_LEN, EMB_DIM]
        outputs = self.embeddings(padded_token_ids)

        if self.embedding_weightages:
            # [BS, SEQ_LEN] -> [BS, SEQ_LEN, 1]
            weightages = self.embedding_weightages(padded_token_ids)
            # [BS, SEQ_LEN, EMB_DIM] -> [BS, SEQ_LEN, EMB_DIM]
            outputs = torch.mul(outputs, weightages)

        return outputs


class EmbeddingLayerPooling(nn.Module):

    def __init__(self, embedder_output_pooling_type):
        super().__init__()

        self.embedder_output_pooling_type = embedder_output_pooling_type

    def forward(self, padded_token_embs, lengths):
        # padded_token_ids: dim: [BS, SEQ_LEN, EMD_DIM]
        # lengths:          dim: [BS]

        # [BS, SEQ_LEN, EMD_DIM] -> [BS, EMD_DIM]
        if self.embedder_output_pooling_type.lower() == "max":
            outputs, _ = torch.max(padded_token_embs, dim=1)
        elif self.embedder_output_pooling_type.lower() == "mean":
            sum_ = torch.sum(padded_token_embs, dim=1)
            lens_ = lengths.unsqueeze(dim=1).expand(batch_size, self.lstm_model_outdim)
            assert sum_.size() == lens_.size()
            outputs = torch.div(sum_, lens_)
        else:
            raise NotImplementedError

        return outputs


class CnnLayer(nn.Module):

    def __init__(self, emb_dim, kernel_sizes, num_kernels):
        super().__init__()

        if not num_kernels:
            num_kernels = [50] * len(kernel_sizes)
        elif isinstance(num_kernels, list) and len(num_kernels) != len(kernel_sizes):
            num_kernels = [num_kernels[0]] * len(kernel_sizes)
        elif isinstance(num_kernels, int) and num_kernels > 0:
            num_kernels = [num_kernels] * len(kernel_sizes)
        else:
            raise ValueError(f"Invalid value for num_kernels: {num_kernels}")

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
        embs_unsqueezed = torch.unsqueeze(embs, dim=1)

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
        lstm_outputs, (last_hidden_states, last_cell_states) = self.lstm(packed)
        outputs = pad_packed_sequence(lstm_outputs, batch_first=True)[0]

        return outputs


class LstmLayerPooling(nn.Module):

    def __init__(self, lstm_output_pooling_type):
        super().__init__()

        self.lstm_output_pooling_type = lstm_output_pooling_type

    def forward(self, padded_token_embs, lengths):
        # padded_token_ids: dim: [BS, SEQ_LEN, EMD_DIM]
        # lengths:          dim: [BS]

        # [BS, SEQ_LEN, EMD_DIM] -> [BS, EMD_DIM]
        if self.lstm_output_pooling_type.lower() == "end":
            last_seq_idxs = torch.LongTensor([x - 1 for x in lengths])
            outputs = padded_token_embs[range(padded_token_embs.shape[0]), last_seq_idxs, :]
        elif self.lstm_output_pooling_type.lower() == "max":
            outputs, _ = torch.max(padded_token_embs, dim=1)
        elif self.lstm_output_pooling_type.lower() == "mean":
            sum_ = torch.sum(padded_token_embs, dim=1)
            lens_ = lengths.unsqueeze(dim=1).expand(batch_size, self.lstm_model_outdim)
            assert sum_.size() == lens_.size()
            outputs = torch.div(sum_, lens_)
        else:
            raise NotImplementedError

        return outputs


# Custom modules built on top of the nn layers that also hold params, dumps/loads models


class ModuleForIndividualClassification(nn.Module):
    """Base Module class that defines all the necessary elements to succesfully train/infer,
     dump/load custom pytorch modules wrapped on top of this base class. Classes derived from
     this base can either be trained for sequence classification or token classification, but
     not both, meaning this class does not entertain joint modeling. The output of a class
     derived from this base contains either `seq_embs` or `token_embs` in its output which
     facilitates single-head training only.
    """

    # TODO:
    #  1) See if all derived classes of nn.Module in ModuleForIndividualClassification are moved
    #       to GPU if we do ModuleForIndividualClassification.to("cuda")
    #  2) Similar to above, see if ModuleForIndividualClassification.state_dict of gives out
    #       state_dict of all involved nn.Module state_dicts

    def __init__(self, **params):
        super().__init__()

        # params
        self.name = self.__class__.__name__
        self.hidden_dropout_prob = params.get("hidden_dropout_prob", 0.3)
        self.num_labels = params.get("num_labels")
        self.label_pad_idx = params.get("label_pad_idx")

        self.params_keys = set(["name", "hidden_dropout_prob", "num_labels", "label_pad_idx"])

        # init derived class' layers
        self._init(**params)
        try:
            self.hidden_size = self.out_dim
            assert self.hidden_size
        except (AttributeError, AssertionError) as e:
            msg = f"Derived class '{self.name}' must indicate its hidden size for dense layer " \
                  f"classification by having an attribute 'self.out_dim' and must be a positive " \
                  f"integer"
            raise ValueError(msg) from e
        self.params_keys.update(["hidden_size"])

        # obtain number of labels and create output dense layer accordingly
        if not self.num_labels:
            msg = f"Invalid number of labels ({self.num_labels}) inputted for '{self.name}' class"
            raise ValueError(msg)
        self.dropout = nn.Dropout(p=self.hidden_dropout_prob)
        self.classifier_head = nn.Linear(self.hidden_size, self.num_labels)

        # Decide the criterion type based on the num_labels
        #   nn.CrossEntropy or nn.SigmoidWithLogits
        self.criterion = None  # with label_pad_idx?

    def forward(self, inputs):

        # TODO: should the labels be moved to device? or is it not required due to long type?
        for k, v in inputs.items():
            if "lengths" not in k and isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        outputs = self._forward(inputs)

        # validations
        if "seq_embs" in outputs and "token_embs" in outputs:
            msg = f"This class ({self.name}) can only perform single head training"
            raise ValueError(msg)
        elif "token_embs" in outputs and not self.label_pad_idx:
            msg = f"Invalid label_pad_idx ({label_pad_idx}) provided for token classification"
            raise ValueError(msg)

        # 'seq_embs' would be of shape [BS, self.hidden_size] whereas 'token_embs' would be
        # of shape [BS, SEQ_LEN, self.hidden_size]. The 'labels' in the input are however always
        # be a 1D shaped [BS/FLATTENED_BS] tensor.

        if not 'labels' in inputs:
            msg = f"If using '{self.name}' class for inference, must use .predict() method. " \
                  f"If using to train, must input labels for computing loss"
            raise ValueError(msg)
        labels = inputs["labels"]
        loss = self.criterion(flattened_outputs, labels)
        return loss

    def predict(self, inputs):

        # functionality similar to `forward` but does not input 'labels' key in the input
        # and additionaly computes the argmax-es
        # TODO: Implement
        raise NotImplementedError

    def predict_probal(self, inputs):
        # TODO: Implement
        raise NotImplementedError

    @abstractmethod
    def _init(self, **params) -> None:
        raise NotImplementedError

    @abstractmethod
    def _forward(self, inputs: Dict) -> Dict:
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def get_params_dict(self):
        return {k: getattr(self, k) for k in self.params_keys}

    @staticmethod
    def _get_model_path(path):
        return os.path.join(path, "pytorch_model.bin")

    @staticmethod
    def _get_params_path(path):
        return os.path.join(path, "config.json")

    @classmethod
    def load(cls, ckpt_folder):
        with open(self._get_params_path(ckpt_folder), "r") as fp:
            params_dict = json.load(fp)
            fp.close()
        model = cls(**params_dict)
        model.load_state_dict(torch.load(cls._get_model_path(ckpt_folder)))
        print(f"{cls.__name__} model loaded")
        return model

    def dump(self, ckpt_folder):
        os.makedirs(ckpt_folder, exist_ok=True)
        with open(self._get_params_path(ckpt_folder), "w") as fp:
            json.dump(self.get_params_dict, fp)
            fp.close()
        torch.save(self.state_dict(), self._get_model_path(ckpt_folder))
        return


class PooledEmbeddingForSequenceClassification(ModuleForIndividualClassification):
    """An embedder pooling module that operates on a batched sequence of token ids. The
    tokens could be characters or words or sub-words. This module finally outputs one 1D
    representation for each instance in the batch (i.e. [BS, EMB_DIM]).

    The `forward` method of this module expects padded token ids along with numer of tokens
    per instance in the batch.

    Additionally, one can set different weightages for different tokens of the embedding
    matrix (e.g. tf-idf weights).
    """

    def _init(self, **params):

        # params
        embedding_weights = params.get("embedding_weights")
        self.num_embs = params.get("num_embs")
        self.emb_dim = params.get("emb_dim")
        self.padding_idx = params.get("padding_idx")
        self.update_embeddings = params.get("update_embeddings", True)
        weightages = params.get("weightages")
        self.update_weightages = params.get("update_weightages", True)
        self.embedder_output_pooling_type = params.get("embedder_output_pooling_type", "mean")

        self.params_keys.update([
            "num_embs", "emb_dim", "padding_idx", "update_embeddings", "update_weightages"
        ])

        # embedding layer
        if embedding_weights:
            if self.num_embs or self.emb_dim:
                logger.warning("Cannot use both 'embedding_weights' and {'num_embs', 'emb_dim'}")
            self.num_embs, self.emb_dim = embedding_weights.shape
        else:
            if not (self.num_embs and self.emb_dim):
                logger.warning("Must input 'num_embs' and 'emb_dim' to initialize embeddings")
        self.emb_layer = EmbeddingLayer(self.num_embs, self.emb_dim, self.padding_idx,
                                        embedding_weights, self.update_embeddings,
                                        weightages, self.update_weightages)
        self.emb_layer_pooling = EmbeddingLayerPooling(self.embedder_output_pooling_type)
        self.out_dim = self.emb_dim

    def _forward(self, inputs):

        seq_ids = inputs["seq_ids"]  # [BS, SEQ_LEN]
        seq_lengths = inputs["seq_lengths"]  # [BS]

        encodings = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, EMD_DIM]
        encodings = self.emb_layer_pooling(encodings, seq_lengths)  # [BS, self.out_dim]

        inputs.update({"seq_embs": encodings})

        return inputs


class SequenceCnnForSequenceClassification(ModuleForIndividualClassification):
    """A CNN module that operates on a batched sequence of token ids. The tokens could be
    characters or words or sub-words. This module finally outputs one 1D representation
    for each instance in the batch (i.e. [BS, EMB_DIM]).

    The `forward` method of this module expects only padded token ids as input.
    """

    def _init(self, **params):

        # params
        embedding_weights = params.get("embedding_weights")
        self.num_embs = params.get("num_embs")
        self.emb_dim = params.get("emb_dim")
        self.padding_idx = params.get("padding_idx")
        self.update_embeddings = params.get("update_embeddings", True)
        self.cnn_kernel_sizes = params.get("cnn_kernel_sizes", [1, 3, 5])
        self.cnn_num_kernels = params.get("cnn_num_kernels", [50] * len(self.cnn_kernel_sizes))

        self.params_keys.update([
            "num_embs", "emb_dim", "padding_idx", "update_embeddings",
            "cnn_kernel_sizes", "cnn_num_kernels"
        ])

        # embedding layer
        if embedding_weights:
            if self.num_embs or self.emb_dim:
                logger.warning("Cannot use both 'embedding_weights' and {'num_embs', 'emb_dim'}")
            self.num_embs, self.emb_dim = embedding_weights.shape
        else:
            if not (self.num_embs and self.emb_dim):
                logger.warning("Must input 'num_embs' and 'emb_dim' to initialize embeddings")
        self.emb_layer = EmbeddingLayer(self.num_embs, self.emb_dim, self.padding_idx,
                                        embedding_weights, self.update_embeddings)

        # cnn layer
        self.conv_layer = CnnLayer(self.emb_dim, self.cnn_kernel_sizes, self.cnn_num_kernels)
        self.out_dim = sum(num_kernels)

        logger.info("SequenceCnnForSequenceClassification initialized")

    def _forward(self, inputs):

        seq_ids = inputs["seq_ids"]  # [BS, SEQ_LEN]

        encodings = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, EMD_DIM]
        encodings = self.conv_layer(encodings)  # [BS, self.out_dim]

        inputs.update({"seq_embs": encodings})

        return inputs


class SequenceLstmForSequenceClassification(ModuleForIndividualClassification):
    """A LSTM module that operates on a batched sequence of token ids. The tokens could be
    characters or words or sub-words. This module finally outputs one 1D representation
    for each instance in the batch (i.e. [BS, EMB_DIM]).

    The `forward` method of this module expects padded token ids along with numer of tokens
    per instance in the batch.
    """

    def _init(self, **params):

        # params
        embedding_weights = params.get("embedding_weights")
        self.num_embs = params.get("num_embs")
        self.emb_dim = params.get("emb_dim")
        self.padding_idx = params.get("padding_idx")
        self.update_embeddings = params.get("update_embeddings", True)
        self.lstm_hidden_dim = params.get("lstm_hidden_dim", 128)
        self.lstm_num_layers = params.get("lstm_num_layers", 2)
        self.lstm_dropout = params.get("lstm_dropout", 0.3)
        self.lstm_bidirectional = params.get("lstm_bidirectional", True)
        self.lstm_output_pooling_type = params.get("lstm_output_pooling_type", "end")

        self.params_keys.update([
            "num_embs", "emb_dim", "padding_idx", "update_embeddings",
            "lstm_hidden_dim", "lstm_num_layers", "lstm_dropout", "lstm_bidirectional"
        ])

        # embedding layer
        if embedding_weights:
            if self.num_embs or self.emb_dim:
                logger.warning("Cannot use both 'embedding_weights' and {'num_embs', 'emb_dim'}")
            self.num_embs, self.emb_dim = embedding_weights.shape
        else:
            if not (self.num_embs and self.emb_dim):
                logger.warning("Must input 'num_embs' and 'emb_dim' to initialize embeddings")
        self.emb_layer = EmbeddingLayer(self.num_embs, self.emb_dim, self.padding_idx,
                                        embedding_weights, self.update_embeddings)

        # lstm layer
        self.lstm_layer = LstmLayer(self.emb_dim, self.lstm_hidden_dim, self.lstm_num_layers,
                                    self.lstm_dropout, self.lstm_bidirectional)
        self.lstm_layer_pooling = LstmLayerPooling(self.lstm_output_pooling_type)
        self.out_dim = self.emb_dim * 2 if self.lstm_bidirectional else self.emb_dim

        logger.info("SequenceLstmForSequenceClassification initialized")

    def _forward(self, inputs):

        seq_ids = inputs["seq_ids"]  # [BS, SEQ_LEN]
        seq_lengths = inputs["seq_lengths"]  # [BS]

        encodings = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, EMD_DIM]
        encodings = self.lstm_layer(encodings, seq_lengths)  # [BS, SEQ_LEN, self.out_dim]
        encodings = self.lstm_layer_pooling(encodings, seq_lengths)  # [BS, self.out_dim]

        inputs.update({"seq_embs": encodings})

        return inputs


class SequenceSplittedCnnForTokenClassification(ModuleForIndividualClassification):
    """A CNN module that operates on a batched sequence of token ids. The tokens could be
    characters or words or sub-words. This module uses an additional input that determines
    how the sequence of embeddings obtained after the embedding layer for each instance
    in the batch, needs to be split. Once split, the sub-groups of embeddings (each sub-group
    corresponding to a word or a phrase) can be collapsed to 1D representation per sub-group
    through through CNN layers. Finally, this module outputs a 2D representation for each
    instance in the batch (i.e. [BS, SEQ_LEN, EMB_DIM]).
    """

    def _init(self, **params):
        # TODO: Implement
        raise NotImplementedError

    def _forward(self, inputs):
        # TODO: Implement
        raise NotImplementedError


class LstmSequenceSplittedForTokenClassification(ModuleForIndividualClassification):
    """A LSTM module that operates on a batched sequence of token ids. The tokens could be
    characters or words or sub-words. This module uses an additional input that determines
    how the sequence of embeddings obtained after the LSTM layers for each instance in the
    batch, needs to be split. Once split, the sub-groups of embeddings (each sub-group
    corresponding to a word or a phrase) can be collapsed to 1D representation per sub-group
    through pooling operations. Finally, this module outputs a 2D representation for each
    instance in the batch (i.e. [BS, SEQ_LEN, EMB_DIM]).
    """

    def _init(self, **params):
        # TODO: Implement
        raise NotImplementedError

    def _forward(self, inputs):
        # TODO: Implement
        raise NotImplementedError


class ModuleForJointClassification(ModuleForIndividualClassification):
    """This base class is wrapped around ModuleForIndividualClassification but entertains
    joint modeling, meaning multiple heads can be trained for models derived on top of this
    base class. Unlike classes derive on top of ModuleForIndividualClassification, the ones
    derived on this base output both `seq_embs` as well as `token_embs` in their output
    which facilitates multi-head training.
    """

    def __init__(self, **params):
        # TODO: Implement
        raise NotImplementedError
