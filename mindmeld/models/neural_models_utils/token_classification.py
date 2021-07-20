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
Custom modules built on top of nn layers that can do sequence classification
"""

import logging
from abc import abstractmethod
from typing import Dict

from ._classification import ClassificationCore
from ._encoders import (
    TokenClassificationEncoderWithStaticEmbeddings,
    TokenClassificationDualEncoderWithStaticEmbeddings
)
from ._layers import (
    EmbeddingLayer,
    LstmLayer,
    PoolingLayer
)

try:
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pad_sequence
    from torchcrf import CRF
except ImportError:
    pass

logger = logging.getLogger(__name__)


class TokenClassificationCore(ClassificationCore):

    def __init__(self):
        super().__init__()

        # default encoder; have to either fit ot load to use it
        self.encoder = TokenClassificationEncoderWithStaticEmbeddings()

    # methods for training

    def fit_encoder_and_update_params(self, examples, **params):
        self.encoder.fit(examples=examples, **params)
        params.update({
            "num_tokens": self.encoder.get_num_tokens(),
            "emb_dim": self.encoder.get_emb_dim(),
            "padding_idx": self.encoder.get_pad_token_idx(),
            "embedding_weights": self.encoder.get_embedding_weights(),
            "label_padding_idx": self.encoder.get_pad_label_idx()
        })
        return params

    def init(self, **params):

        self._init(**params)

        # params
        self.num_labels = params.get("num_labels")
        self.dense_keep_prob = params.get("dense_keep_prob", 0.5)
        self.params_keys.update(["num_labels", "dense_keep_prob"])
        add_terminals = params.pop("add_terminals", None)
        if add_terminals:
            msg = "Setting param 'add_terminals' to True is not supported with token classification"
            raise NotImplementedError(msg)

        # init the underlying params and architectural components
        try:
            self.hidden_size = self.out_dim
            assert self.hidden_size
        except (AttributeError, AssertionError) as e:
            msg = f"Derived class '{self.name}' must indicate its hidden size for dense layer " \
                  f"classification by having an attribute 'self.out_dim', which must be a " \
                  f"positive integer greater than 1"
            raise ValueError(msg) from e
        self.params_keys.update(["hidden_size"])

        # init the peripheral architecture params and architectural components
        if not self.num_labels:
            msg = f"Invalid number of labels ({self.num_labels}) inputted for '{self.name}' class"
            raise ValueError(msg)

        # more params
        self.use_crf_layer = params.get("use_crf_layer")
        self.params_keys.update(["use_crf_layer"])

        # init the peripheral architectural components and the criterion to compute loss
        self.dense_layer_dropout = nn.Dropout(p=1 - self.dense_keep_prob)
        if self.use_crf_layer:
            self.classifier_head = nn.Linear(self.hidden_size, self.num_labels)
            self.crf_layer = CRF(self.num_labels, batch_first=True)
        else:
            if self.num_labels >= 2:
                # cross-entropy criterion
                self.classifier_head = nn.Linear(self.hidden_size, self.num_labels)
                self.criterion = nn.CrossEntropyLoss(reduction='mean',
                                                     ignore_index=self.label_padding_idx)
            else:
                msg = f"Invalid number of labels specified: {self.num_labels}. " \
                      f"A valid number is equal to or greater than 2"
                raise ValueError(msg)

        print(f"{self.name} is initialized")

    def forward(self, batch_data_dict):

        for k, v in batch_data_dict.items():
            if v is not None and isinstance(v, torch.Tensor):
                batch_data_dict[k] = v.to(self.device)

        batch_data_dict = self._forward(batch_data_dict)

        token_embs = batch_data_dict["token_embs"]
        token_embs = self.dense_layer_dropout(token_embs)
        logits = self.classifier_head(token_embs)
        batch_data_dict.update({"logits": logits})

        targets = batch_data_dict.get("labels")
        if targets is not None:
            if self.use_crf_layer:
                # create a mask to ignore token positions that are padded
                mask = torch.as_tensor(targets != self.label_padding_idx, dtype=torch.uint8)
                loss = - self.crf_layer(logits, targets, mask=mask)  # negative log likelihood
            else:
                loss = self.criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))
            batch_data_dict.update({"loss": loss})

        return batch_data_dict

    # methods for inference

    def predict(self, examples):
        preds = []
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for start_idx in range(0, len(examples), self.batch_size):
                this_examples = examples[start_idx:start_idx + self.batch_size]
                batch_data_dict = self.encoder.batch_encode(this_examples)
                this_logits = self.forward(batch_data_dict)["logits"]
                # find predictions
                if self.use_crf_layer:
                    # create a mask to ignore token positions that are padded
                    mask = torch.as_tensor(
                        pad_sequence([torch.as_tensor([1] * length_) for length_ in
                                      batch_data_dict["seq_lengths"]], batch_first=True),
                        dtype=torch.uint8
                    )
                    this_preds = self.crf_layer.decode(this_logits, mask=mask)  # -> List[List[int]]
                else:
                    this_preds = torch.argmax(this_logits, dim=-1).tolist()
                # trim predictions as per sequence length
                this_preds = [
                    pred_list[:list_len] for pred_list, list_len in
                    zip(this_preds, batch_data_dict["seq_lengths"])
                ]
                preds.extend(this_preds)
        if was_training:
            self.train()
        return preds

    def predict_proba(self, examples):
        raise NotImplementedError

    # abstract methods definition, to be implemented by sub-classes

    @abstractmethod
    def _init(self, **params) -> None:
        raise NotImplementedError

    @abstractmethod
    def _forward(self, batch_data_dict: Dict) -> Dict:
        raise NotImplementedError


class EmbedderForTokenClassification(TokenClassificationCore):

    def _init(self, **params):
        # params
        self.num_tokens = params["num_tokens"]
        self.emb_dim = params["emb_dim"]
        self.padding_idx = params.get("padding_idx", None)
        self.update_embeddings = params.get("update_embeddings", True)
        self.params_keys.update([
            "num_tokens", "emb_dim", "padding_idx", "update_embeddings",
        ])

        # core layers
        self.emb_layer = EmbeddingLayer(self.num_tokens, self.emb_dim, self.padding_idx,
                                        params.get("embedding_weights", None),
                                        self.update_embeddings)
        self.out_dim = self.emb_dim

    def _forward(self, batch_data_dict):
        seq_ids = batch_data_dict["seq_ids"]  # [BS, SEQ_LEN]

        encodings = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, self.out_dim]

        batch_data_dict.update({"token_embs": encodings})

        return batch_data_dict


class SequenceLstmForTokenClassification(TokenClassificationCore):
    """A LSTM module that operates on a batched sequence of token ids. The tokens could be
    characters or words or sub-words. This module uses an additional input that determines
    how the sequence of embeddings obtained after the LSTM layers for each instance in the
    batch, needs to be split. Once split, the sub-groups of embeddings (each sub-group
    corresponding to a word or a phrase) can be collapsed to 1D representation per sub-group
    through pooling operations. Finally, this module outputs a 2D representation for each
    instance in the batch (i.e. [BS, SEQ_LEN', EMB_DIM]).
    """

    def _init(self, **params):
        # params
        self.num_tokens = params["num_tokens"]
        self.emb_dim = params["emb_dim"]
        self.padding_idx = params.get("padding_idx", None)
        self.update_embeddings = params.get("update_embeddings", True)
        self.lstm_hidden_dim = params.get("lstm_hidden_dim", 128)
        self.lstm_num_layers = params.get("lstm_num_layers", 2)
        self.lstm_output_keep_prob = params.get("lstm_output_keep_prob", 0.5)
        self.lstm_bidirectional = params.get("lstm_bidirectional", True)
        self.params_keys.update([
            "num_tokens", "emb_dim", "padding_idx", "update_embeddings",
            "lstm_hidden_dim", "lstm_num_layers", "lstm_output_keep_prob", "lstm_bidirectional"
        ])

        # core layers
        self.emb_layer = EmbeddingLayer(self.num_tokens, self.emb_dim, self.padding_idx,
                                        params.get("embedding_weights", None),
                                        self.update_embeddings)
        self.lstm_layer = LstmLayer(self.emb_dim, self.lstm_hidden_dim, self.lstm_num_layers,
                                    1 - self.lstm_output_keep_prob, self.lstm_bidirectional)
        self.out_dim = self.lstm_hidden_dim * 2 if self.lstm_bidirectional else self.lstm_hidden_dim

    def _forward(self, batch_data_dict):
        seq_ids = batch_data_dict["seq_ids"]  # [BS, SEQ_LEN]
        seq_lengths = batch_data_dict["seq_lengths"]  # [BS]

        encodings = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, EMD_DIM]
        encodings = self.lstm_layer(encodings, seq_lengths)  # [BS, SEQ_LEN, self.out_dim]

        batch_data_dict.update({"token_embs": encodings})

        return batch_data_dict


class TokenLstmSequenceLstmForTokenClassification(TokenClassificationCore):
    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        super().__init__()

        # default encoder; have to either fit ot load to use it
        self.encoder = TokenClassificationDualEncoderWithStaticEmbeddings()

    # methods for training

    def fit_encoder_and_update_params(self, examples, **params):
        params = super().fit_encoder_and_update_params(examples, **params)
        params.update({
            "char_num_tokens": self.encoder.get_char_num_tokens(),
            "char_emb_dim": self.encoder.get_char_emb_dim(),
            "char_padding_idx": self.encoder.get_char_pad_token_idx(),
        })
        return params

    def _init(self, **params):
        # params
        self.num_tokens = params["num_tokens"]
        self.emb_dim = params["emb_dim"]
        self.padding_idx = params.get("padding_idx", None)
        self.update_embeddings = params.get("update_embeddings", True)
        self.lstm_hidden_dim = params.get("lstm_hidden_dim", 128)
        self.lstm_num_layers = params.get("lstm_num_layers", 2)
        self.lstm_output_keep_prob = params.get("lstm_output_keep_prob", 0.5)
        self.lstm_bidirectional = params.get("lstm_bidirectional", True)
        self.char_num_tokens = params["char_num_tokens"]
        self.char_emb_dim = params["char_emb_dim"]
        self.char_padding_idx = params.get("char_padding_idx", None)
        self.char_lstm_hidden_dim = params.get("char_lstm_hidden_dim", 128)
        self.char_lstm_num_layers = params.get("char_lstm_num_layers", 2)
        self.char_lstm_output_keep_prob = params.get("char_lstm_output_keep_prob", 0.5)
        self.char_lstm_bidirectional = params.get("char_lstm_bidirectional", True)
        self.word_level_character_embedding_size = params.get(
            "word_level_character_embedding_size", self.emb_dim)
        self.char_lstm_output_pooling_type = params.get("char_lstm_output_pooling_type", "end")
        self.params_keys.update([
            "num_tokens", "emb_dim", "padding_idx", "update_embeddings",
            "lstm_hidden_dim", "lstm_num_layers", "lstm_output_keep_prob", "lstm_bidirectional",
            "char_num_tokens", "char_emb_dim", "char_padding_idx",
            "char_lstm_hidden_dim", "char_lstm_num_layers", "char_lstm_output_keep_prob",
            "char_lstm_bidirectional",
            "word_level_character_embedding_size", "char_lstm_output_pooling_type"
        ])

        # core layers
        self.char_emb_layer = EmbeddingLayer(
            self.char_num_tokens, self.char_emb_dim, self.char_padding_idx,
            params.get("char_embedding_weights", None), self.update_embeddings
        )
        self.char_lstm_layer = LstmLayer(
            self.char_emb_dim, self.char_lstm_hidden_dim, self.char_lstm_num_layers,
            1 - self.char_lstm_output_keep_prob, self.char_lstm_bidirectional
        )
        char_out_dim = (
            self.char_lstm_hidden_dim * 2 if self.char_lstm_bidirectional
            else self.char_lstm_hidden_dim
        )
        self.char_lstm_output_transform = nn.Linear(
            char_out_dim, self.word_level_character_embedding_size
        )
        self.char_lstm_layer_pooling = PoolingLayer(self.char_lstm_output_pooling_type)
        self.emb_layer = EmbeddingLayer(
            self.num_tokens, self.emb_dim, self.padding_idx, params.get("embedding_weights", None),
            self.update_embeddings
        )
        self.lstm_layer = LstmLayer(
            self.emb_dim + self.word_level_character_embedding_size,
            self.lstm_hidden_dim, self.lstm_num_layers, 1 - self.lstm_output_keep_prob,
            self.lstm_bidirectional
        )
        self.out_dim = self.lstm_hidden_dim * 2 if self.lstm_bidirectional else self.lstm_hidden_dim

    def _forward(self, batch_data_dict):
        char_seq_ids = batch_data_dict["char_seq_ids"]  # List of [BS, SEQ_LEN]
        char_seq_lengths = batch_data_dict["char_seq_lengths"]  # List of [BS]

        encs = [self.char_emb_layer(_seq_ids) for _seq_ids in char_seq_ids]
        encs = [self.char_lstm_layer(enc, seq_len) for enc, seq_len in zip(encs, char_seq_lengths)]
        encs = [self.char_lstm_output_transform(enc) for enc in encs]
        char_encs = pad_sequence(
            [self.char_lstm_layer_pooling(enc, seq_len) for enc, seq_len in
             zip(encs, char_seq_lengths)], batch_first=True
        )  # [BS, SEQ_LEN, self.word_level_character_embedding_size]

        seq_ids = batch_data_dict["seq_ids"]  # [BS, SEQ_LEN]
        seq_lengths = batch_data_dict["seq_lengths"]  # [BS]

        word_encs = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, self.emb_dim]
        char_plus_word_encs = torch.cat((char_encs, word_encs), dim=-1)  # [BS, SEQ_LEN, sum(lens)]
        encodings = self.lstm_layer(char_plus_word_encs, seq_lengths)  # [BS, SEQ_LEN, self.out_dim]

        batch_data_dict.update({"token_embs": encodings})

        return batch_data_dict


class TokenCnnSequenceLstmForTokenClassification(TokenClassificationCore):

    def __init__(self, **params):
        raise NotImplementedError
