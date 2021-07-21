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

import logging
import os

from .._util import _get_module_or_attr

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

    nn_module = _get_module_or_attr("torch.nn", "Module")
except ImportError:
    nn_module = object
    pass

SEED = 7246

logger = logging.getLogger(__name__)


# utils

def get_disk_space_of_model(pytorch_module):
    filename = "temp.bin"
    torch.save(pytorch_module.state_dict(), filename)
    size = os.path.getsize(filename) / 1e6
    os.remove(filename)
    return size


def get_num_params_of_model(pytorch_module):
    n_total = 0
    n_requires_grad = 0
    for param in list(pytorch_module.parameters()):
        t = 1
        for sz in list(param.size()):
            t *= sz
        n_total += t
        if param.requires_grad:
            n_requires_grad += t
    return n_requires_grad, n_total


# Various nn layers


class EmbeddingLayer(nn_module):
    """A pytorch wrapper layer for embeddings that takes input as a batched sequence of ids
    and outputs embeddings correponding to those ids
    """

    def __init__(self, num_tokens, emb_dim, padding_idx=None,
                 embedding_weights=None, update_embeddings=True,
                 embeddings_dropout=0.5, coefficients=None, update_coefficients=True):
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

        self.dropout = nn.Dropout(embeddings_dropout)

    def forward(self, padded_token_ids):
        # padded_token_ids: dim: [BS, SEQ_LEN]
        # returns:          dim: [BS, SEQ_LEN, EMB_DIM]

        # [BS, SEQ_LEN] -> [BS, SEQ_LEN, EMB_DIM]
        outputs = self.embeddings(padded_token_ids)

        if self.embedding_for_coefficients:
            # [BS, SEQ_LEN] -> [BS, SEQ_LEN, 1]
            coefficients = self.embedding_for_coefficients(padded_token_ids)
            # [BS, SEQ_LEN, EMB_DIM] -> [BS, SEQ_LEN, EMB_DIM]
            outputs = torch.mul(outputs, coefficients)

        outputs = self.dropout(outputs)

        return outputs


class CnnLayer(nn_module):

    def __init__(self, emb_dim, kernel_sizes, num_kernels, cnn_dropout):
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
        self.dropout = nn.Dropout(cnn_dropout)

    def forward(self, padded_token_embs):
        # padded_token_embs: dim: [BS, SEQ_LEN, EMD_DIM]
        # returns:          dim: [BS, EMB_DIM`]

        # [BS, SEQ_LEN, EMD_DIM] -> [BS, 1, SEQ_LEN, EMD_DIM]
        embs_unsqueezed = torch.unsqueeze(padded_token_embs, dim=1)

        # [BS, 1, SEQ_LEN, EMD_DIM] -> list([BS, n, SEQ_LEN])
        conv_outputs = [conv(embs_unsqueezed).squeeze(3) for conv in self.convs]

        # list([BS, n, SEQ_LEN]) -> list([BS, n])
        maxpool_conv_outputs = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outputs]

        # list([BS, n]) -> [BS, sum(n)]
        outputs = torch.cat(maxpool_conv_outputs, dim=1)
        outputs = self.dropout(outputs)

        return outputs


class LstmLayer(nn_module):

    def __init__(self, emb_dim, hidden_dim, num_layers, lstm_dropout, bidirectional):
        super().__init__()

        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, num_layers=num_layers, dropout=lstm_dropout,
            bidirectional=bidirectional, batch_first=True
        )
        self.dropout = nn.Dropout(lstm_dropout)

    def forward(self, padded_token_embs, lengths):
        # padded_token_embs: dim: [BS, SEQ_LEN, EMD_DIM]
        # lengths:          dim: [BS]
        # returns:          dim: [BS, SEQ_LEN, EMB_DIM]

        # [BS, SEQ_LEN, EMD_DIM] -> [BS, SEQ_LEN, EMD_DIM*(2 if bidirectional else 1)]
        packed = pack_padded_sequence(padded_token_embs, lengths,
                                      batch_first=True, enforce_sorted=False)
        lstm_outputs, _ = self.lstm(packed)
        outputs = pad_packed_sequence(lstm_outputs, batch_first=True)[0]
        outputs = self.dropout(outputs)

        return outputs


class PoolingLayer(nn_module):

    def __init__(self, pooling_type):
        super().__init__()

        pooling_type = pooling_type.lower()

        ALLOWED_TYPES = ["start", "end", "max", "mean", "mean_sqrt"]
        assert pooling_type in ALLOWED_TYPES

        self._requires_length = ["end", "max", "mean", "mean_sqrt"]

        self.pooling_type = pooling_type

    def forward(self, padded_token_embs, lengths=None):
        # padded_token_embs: dim: [BS, SEQ_LEN, EMD_DIM]
        # lengths:           dim: [BS]
        # returns:           dim: [BS, EMD_DIM]

        if self.pooling_type in self._requires_length and lengths is None:
            msg = f"Missing required value 'lengths' for pooling_type: {self.pooling_type}"
            raise ValueError(msg)

        if self.pooling_type == "start":
            outputs = padded_token_embs[:, 0, :]
        elif self.pooling_type == "end":
            last_seq_idxs = torch.LongTensor([x - 1 for x in lengths])
            outputs = padded_token_embs[range(padded_token_embs.shape[0]), last_seq_idxs, :]
        else:
            mask = pad_sequence(
                [torch.as_tensor([1] * length_) for length_ in lengths], batch_first=True
            ).unsqueeze(-1).expand(padded_token_embs.size()).float()
            if self.pooling_type == "max":
                padded_token_embs[mask == 0] = -1e9  # set to a large negative value
                outputs, _ = torch.max(padded_token_embs, dim=1)[0]
            elif self.pooling_type == "mean":
                summed_padded_token_embs = torch.sum(padded_token_embs, dim=1)
                expanded_lengths = lengths.unsqueeze(dim=1).expand(summed_padded_token_embs.size())
                outputs = torch.div(summed_padded_token_embs, expanded_lengths)
            elif self.pooling_type == "mean_sqrt":
                summed_padded_token_embs = torch.sum(padded_token_embs, dim=1)
                expanded_lengths = lengths.unsqueeze(dim=1).expand(summed_padded_token_embs.size())
                outputs = torch.div(summed_padded_token_embs, torch.sqrt(expanded_lengths))

        return outputs


class TokenSpanPoolingLayer(nn_module):

    def __init__(self, pooling_type):
        super().__init__()

        pooling_type = pooling_type.lower()

        ALLOWED_TYPES = ["start", "end", "max", "mean", "mean_sqrt"]
        assert pooling_type in ALLOWED_TYPES

        self.pooling_type = pooling_type

    def _split_and_pool(self, tensor_2d, list_of_chunk_lengths):
        # tensor_2d:         dim: [SEQ_LEN, EMD_DIM]
        # returns:           dim: [BS', EMD_DIM]

        splits = torch.split(tensor_2d[:sum(list_of_chunk_lengths)], list_of_chunk_lengths, dim=0)
        padded_token_embs = pad_sequence(splits, batch_first=True)  # [BS', SEQ_LEN', EMD_DIM]

        if self.pooling_type == "start":
            outputs = padded_token_embs[:, 0, :]
        elif self.pooling_type == "end":
            last_seq_idxs = torch.LongTensor([x - 1 for x in lengths])
            outputs = padded_token_embs[range(padded_token_embs.shape[0]), last_seq_idxs, :]
        else:
            mask = pad_sequence(
                [torch.as_tensor([1] * length_) for length_ in lengths], batch_first=True
            ).unsqueeze(-1).expand(padded_token_embs.size()).float()
            if self.pooling_type == "max":
                padded_token_embs[mask == 0] = -1e9  # set to a large negative value
                outputs, _ = torch.max(padded_token_embs, dim=1)[0]
            elif self.pooling_type == "mean":
                summed_padded_token_embs = torch.sum(padded_token_embs, dim=1)
                expanded_lengths = lengths.unsqueeze(dim=1).expand(summed_padded_token_embs.size())
                outputs = torch.div(summed_padded_token_embs, expanded_lengths)
            elif self.pooling_type == "mean_sqrt":
                summed_padded_token_embs = torch.sum(padded_token_embs, dim=1)
                expanded_lengths = lengths.unsqueeze(dim=1).expand(summed_padded_token_embs.size())
                outputs = torch.div(summed_padded_token_embs, torch.sqrt(expanded_lengths))

        return outputs

    def forward(self, padded_token_embs, span_lengths):
        # padded_token_embs: dim: [BS, SEQ_LEN, EMD_DIM]
        # span_lengths:      dim: List[List of Int summing up to SEQ_LEN' <= SEQ_LEN]
        # returns:           dim: [BS, SEQ_LEN', EMD_DIM]

        outputs = pad_sequence([
            self._split_and_pool(_padded_token_embs, _span_lengths)
            for _padded_token_embs, _span_lengths in zip(padded_token_embs, span_lengths)
        ], batch_first=True)

        return outputs
