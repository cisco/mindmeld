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
from typing import List, Union, Dict

import numpy as np

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

logger = logging.getLogger(__name__)


class EmbeddingLayer(nn_module):
    """A pytorch wrapper layer for embeddings that takes input a batched sequence of ids
    and outputs embeddings corresponding to those ids
    """

    def __init__(
        self,
        num_tokens: int,
        emb_dim: int,
        padding_idx: int = None,
        embedding_weights: Dict[int, Union[List, np.ndarray]] = None,
        update_embeddings: bool = True,
        embeddings_dropout: float = 0.5,
        coefficients: List[float] = None,
        update_coefficients: bool = True
    ):
        """
        Args:
            num_tokens (int): size of the dictionary of embeddings
            emb_dim (int): the size of each embedding vector
            padding_idx (int, Optional): If given, pads the output with the embedding vector at
                `padding_idx` (initialized to zeros) whenever it encounters the index.
            embedding_weights (Dict[int, Union[List, np.ndarray]], Optional): weights to overwrite
                the already initialized embedding weights
            update_embeddings (bool, Optional): whether to freeze or train the embedding weights
            embeddings_dropout (float, Optional): dropout rate to apply on the forward call
            coefficients (List[float], Optional): weight coefficients for the dictionary of
                embeddings
            update_coefficients (bool, Optional): whether to freeze or train the coefficients
        """
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
            self.embedding_for_coefficients.load_state_dict(
                {'weight': torch.as_tensor(coefficients).view(-1, 1)}
            )
            self.embedding_for_coefficients.weight.requires_grad = update_coefficients

        self.dropout = nn.Dropout(embeddings_dropout)

    def forward(self, padded_token_ids: "Tensor2d[int]") -> "Tensor3d[float]":
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
    """A pytorch wrapper layer for 2D Convolutions
    """

    def __init__(self, emb_dim: int, kernel_sizes: List[int], num_kernels: List[int]):
        """
        Args:
            emb_dim (int): the size of embedding vectors or last dimension of hidden state prior
                to this CNN layer (i.e. width for convolution filters)
            kernel_sizes (List[int]): the length of each kernel provided as a list of lengths
                (i.e. length for convolution filters)
            num_kernels (List[int]): the number of kernels for each kernel size provided as a list
                of numbers (one number per provided size of kernel)
        """
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
        for kernel_size, num_kernel in zip(kernel_sizes, num_kernels):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(1, num_kernel, (kernel_size, emb_dim), padding=(kernel_size - 1, 0),
                              dilation=1, bias=True, padding_mode='zeros'),
                    nn.ReLU(),
                )
            )

    def forward(self, padded_token_embs: "Tensor3d[float]") -> "Tensor2d[float]":
        # padded_token_embs: dim: [BS, SEQ_LEN, EMD_DIM]
        # returns:           dim: [BS, EMB_DIM`]

        # [BS, SEQ_LEN, EMD_DIM] -> [BS, 1, SEQ_LEN, EMD_DIM]
        embs_unsqueezed = torch.unsqueeze(padded_token_embs, dim=1)

        # [BS, 1, SEQ_LEN, EMD_DIM] -> list([BS, n, SEQ_LEN])
        conv_outputs = [conv(embs_unsqueezed).squeeze(3) for conv in self.convs]

        # list([BS, n, SEQ_LEN]) -> list([BS, n])
        maxpool_conv_outputs = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outputs]

        # list([BS, n]) -> [BS, sum(n)]
        outputs = torch.cat(maxpool_conv_outputs, dim=1)

        return outputs


class LstmLayer(nn_module):
    """A pytorch wrapper layer for BiLSTMs
    """

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        lstm_dropout: float,
        bidirectional: bool
    ):
        """
        Args:
            emb_dim (int): the size of embedding vectors or last dimension of hidden state prior
                to this LSTM layer
            hidden_dim (int): the hidden dimension for nn.LSTM
            num_layers (int): the number of nn.LSTM layers to stack
            lstm_dropout (float): the dropout rate for nn.LSTM
            bidirectional (bool): whether LSTMs should be applied on both forward and
                backward sequences of the input or not
        """
        super().__init__()

        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, num_layers=num_layers, dropout=lstm_dropout,
            bidirectional=bidirectional, batch_first=True
        )

    def forward(
        self,
        padded_token_embs: "Tensor3d[float]",
        lengths: "Tensor1d[int]",
    ) -> "Tensor3d[float]":
        # padded_token_embs: dim: [BS, SEQ_LEN, EMD_DIM]
        # lengths:           dim: [BS]
        # returns:           dim: [BS, SEQ_LEN, EMB_DIM]

        # [BS, SEQ_LEN, EMD_DIM] -> [BS, SEQ_LEN, EMD_DIM*(2 if bidirectional else 1)]
        lengths = lengths.to(torch.device("cpu"))
        packed = pack_padded_sequence(padded_token_embs, lengths,
                                      batch_first=True, enforce_sorted=False)
        lstm_outputs, _ = self.lstm(packed)
        outputs = pad_packed_sequence(lstm_outputs, batch_first=True)[0]

        return outputs


class PoolingLayer(nn_module):
    """A pooling layer for Tensor3d objects that pools along the last dimension. Assumes that
    padding if any exists on the right side of inputs (i.e. not in the beginning of inputs)
    """

    def __init__(self, pooling_type: str):
        """
        Args:
            pooling_type (str): the choice of pooling; to be amongst following:
                first: the first index of each sequence will be the pooled output (similar to CLS
                    token in BERT models)
                last: the last index of each sequence will be the pooled output (useful for pooling
                    outputs from nn.LSTM)
                max: max pool across last dimension will be the pooled output
                mean: mean pool across last dimension will be the pooled output
                mean_sqrt: similar to 'mean' but slashed by the square root of sequence length
        """
        super().__init__()

        pooling_type = pooling_type.lower()

        allowed_pooling_types = ["first", "last", "max", "mean", "mean_sqrt"]
        if pooling_type not in allowed_pooling_types:
            msg = f"Expected pooling_type amongst {allowed_pooling_types} " \
                  f"but found '{pooling_type}'"
            raise ValueError(msg)

        # assumption: first token is never a pad token for the passed inputs
        self._requires_length = ["last", "max", "mean", "mean_sqrt"]

        self.pooling_type = pooling_type

    def forward(
        self,
        padded_token_embs: "Tensor3d[float]",
        lengths: "Tensor1d[int]" = None,
    ) -> "Tensor2d[float]":
        # padded_token_embs: dim: [BS, SEQ_LEN, EMD_DIM]
        # lengths:           dim: [BS]
        # returns:           dim: [BS, EMD_DIM]

        if self.pooling_type in self._requires_length and lengths is None:
            msg = f"Missing required value 'lengths' for pooling_type: {self.pooling_type}"
            raise ValueError(msg)

        if self.pooling_type == "first":
            outputs = padded_token_embs[:, 0, :]
        elif self.pooling_type == "last":
            last_seq_idxs = torch.LongTensor([x - 1 for x in lengths])
            outputs = padded_token_embs[range(padded_token_embs.shape[0]), last_seq_idxs, :]
        else:
            try:
                target_device = padded_token_embs.device
                mask = pad_sequence(
                    [torch.as_tensor([1] * length_) for length_ in lengths],
                    batch_first=True,
                    padding_value=0.0,
                ).unsqueeze(-1).expand(padded_token_embs.size()).float().to(target_device)
            except RuntimeError as e:
                msg = f"Unable to create a mask for '{self.pooling_type}' pooling operation in " \
                      f"{self.__class__.__name__}. It is possible that your choice of tokenizer " \
                      f"does not split input text at whitespace (eg. robert-base tokenizer), due " \
                      f"to which tokenization of a word is different between with and without " \
                      f"context. If working with a transformers model, consider changing the " \
                      f"pretrained model name and restart training."
                raise ValueError(msg) from e
            if self.pooling_type == "max":
                padded_token_embs[mask == 0] = -1e9  # set to a large negative value
                outputs, _ = torch.max(padded_token_embs, dim=1)
            elif self.pooling_type == "mean":
                summed_padded_token_embs = torch.sum(padded_token_embs * mask, dim=1)
                outputs = summed_padded_token_embs / mask.sum(1)
            elif self.pooling_type == "mean_sqrt":
                summed_padded_token_embs = torch.sum(padded_token_embs * mask, dim=1)
                expanded_lengths = lengths.unsqueeze(dim=1).expand(summed_padded_token_embs.size())
                outputs = torch.div(summed_padded_token_embs, torch.sqrt(expanded_lengths.float()))

        return outputs


class SplittingAndPoolingLayer(nn_module):
    """Pooling class that first splits a sequence of representations into subgroups of
    representations based on lengths of subgroups inputted, and pools each subgroup separately.
    """

    def __init__(self, pooling_type: str, number_of_terminal_tokens: int):
        """
        Args:
            pooling_type (str): the choice of pooling; to be amongst following:
                first: the first index of each subsequence will be the pooled output of that
                    subgroup(e.g. token classification using BERT models with sub-word tokenization)
                last: the last index of each subsequence will be the pooled output of that subgroup
                    (e.g. for word level representations when using a character BiLSTM)
                max: max pool across subsequence will be the pooled output of that subgroup
                mean: mean pool across subsequence will be the pooled output of that subgroup
                    (e.g. token classification using BERT models with sub-word tokenization)
                mean_sqrt: similar to 'mean' but slashed by the square root of subsequence length
            number_of_terminal_tokens (int): the number of terminal tokens that will be discarded
                if discard terminals is set to True in the forward method.
        """
        super().__init__()

        self.pooling_type = pooling_type.lower()
        self.number_of_terminal_tokens = number_of_terminal_tokens

        self.pooling_layer = PoolingLayer(pooling_type=self.pooling_type)

    def _split_and_pool(
        self,
        tensor_2d: "Tensor2d[float]",
        list_of_subgroup_lengths: "Tensor1d[int]",
        discard_terminals: bool
    ):
        # tensor_2d:                 dim: [SEQ_LEN, EMD_DIM]
        # list_of_subgroup_lengths:  dim: List of int summing up to SEQ_LEN' <= SEQ_LEN
        # discard_terminals:         bool
        # returns:                   dim: [SEQ_LEN``, EMD_DIM]

        if discard_terminals:
            # TODO: Number of terminals can also be 1 (maybe just left or just right) in some models
            if self.number_of_terminal_tokens != 2:
                msg = f"Unable to combine sub-tokens' representations for each word into one in " \
                      f"{self.__class__.__name__}. It is possible that your choice of tokenizer " \
                      f"has {self.number_of_terminal_tokens} terminal token instead of assumed " \
                      f"2 terminals."  # (eg. t5-base tokenizer)
                raise NotImplementedError(msg)

            # since list_of_subgroup_lengths consists of lengths of only non-terminal subgroups but
            # the inputted tensor_2d consists of terminals
            seq_len_required = sum(list_of_subgroup_lengths) + self.number_of_terminal_tokens
            tensor_2d_with_terminals = tensor_2d[:seq_len_required]
            tensor_2d = tensor_2d_with_terminals[1:-1]  # discard terminal representations
        else:
            seq_len_required = sum(list_of_subgroup_lengths)
            tensor_2d = tensor_2d[:seq_len_required]

        try:
            # argument 'split_sizes' (position 1) must be tuple of ints, not Tensor
            splits = torch.split(tensor_2d, list_of_subgroup_lengths.tolist(), dim=0)
        except RuntimeError as e:
            msg = f"Unable to combine sub-tokens' representations for each word into one in " \
                  f"{self.__class__.__name__}. It is possible that your choice of tokenizer " \
                  f"does not split input text at whitespace (eg. robert-base tokenizer), due " \
                  f"to which one-representation-per-word cannot be obtained to do tagging at " \
                  f"word-level for token classification."
            raise ValueError(msg) from e
        padded_token_embs = pad_sequence(splits, batch_first=True)  # [BS', SEQ_LEN', EMD_DIM]

        # return dims: [len(list_of_subgroup_lengths), EMD_DIM]
        pooled_repr_for_each_subgroup = self.pooling_layer(
            padded_token_embs=padded_token_embs,
            lengths=list_of_subgroup_lengths
        )

        return pooled_repr_for_each_subgroup

    def forward(
        self,
        padded_token_embs: "Tensor3d[float]",
        span_lengths: "List[Tensor1d[int]]",
        discard_terminals: bool = None
    ):
        # padded_token_embs: dim: [BS, SEQ_LEN, EMD_DIM]
        # span_lengths:      dim: List[List of int summing up to SEQ_LEN' <= SEQ_LEN]
        # discard_terminals: bool
        # returns:           dim: [BS, SEQ_LEN', EMD_DIM]

        outputs = pad_sequence([
            self._split_and_pool(_padded_token_embs, _span_lengths, discard_terminals)
            for _padded_token_embs, _span_lengths in zip(padded_token_embs, span_lengths)
        ], batch_first=True)

        return outputs
