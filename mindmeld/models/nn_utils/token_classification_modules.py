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

from .input_encoders import (
    TokenClsEncoderForEmbLayer,
    TokenClsDualEncoderForEmbLayers,
    TokenClsEncoderWithPlmLayer
)
from .layers import (
    EmbeddingLayer,
    CnnLayer,
    LstmLayer,
    PoolingLayer,
    SplittingAndPoolingLayer
)
from .nn_base_modules import ClassificationBase

try:
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pad_sequence
    from torchcrf import CRF
except ImportError:
    pass

logger = logging.getLogger(__name__)


class TokenClassificationBase(ClassificationBase):
    """Base class that defines all the necessary elements to succesfully train/infer
     custom pytorch modules wrapped on top of this base class. Classes derived from
     this base can be trained for sequence tagging aka. token classification.
    """

    DEFAULT_PARAMS = {
        "output_keep_prob": 0.7,
        "use_crf_layer": True,
    }

    def __init__(self):
        super().__init__()

        # default encoder; have to either fit ot load to use it
        self.encoder = TokenClsEncoderForEmbLayer()

    def _get_subclasses_default_params(self):
        return {
            **TokenClassificationBase.DEFAULT_PARAMS,
            **self.__class__.DEFAULT_PARAMS
        }

    def _init_encoder(self, examples, **params):
        self.encoder.fit(examples=examples, **params)
        params.update({
            "num_tokens": self.encoder.get_num_tokens(),
            "emb_dim": self.encoder.get_emb_dim(),
            "padding_idx": self.encoder.get_pad_token_idx(),
            "embedding_weights": self.encoder.get_embedding_weights(),
            "label_padding_idx": self.encoder.get_pad_label_idx()
        })
        return params

    def _init_forward_graph(self):

        self._init_core()

        add_terminals = self.params.pop("add_terminals", False)
        if add_terminals:
            msg = "Setting param 'add_terminals' to True is not supported with token classification"
            raise NotImplementedError(msg)

        # init the underlying params and architectural components
        try:
            assert self.out_dim > 0
            self.params.update({"out_dim": self.out_dim})
        except (AttributeError, AssertionError) as e:
            msg = f"Derived class '{self.name}' must indicate its hidden size for dense layer " \
                  f"classification by having an attribute 'self.out_dim', which must be a " \
                  f"positive integer greater than 1"
            raise ValueError(msg) from e

        # init the peripheral architecture params and architectural components
        if not self.params.num_labels:
            msg = f"Invalid number of labels ({self.params.num_labels}) inputted to '{self.name}'"
            raise ValueError(msg)

        # init the peripheral architectural components and the criterion to compute loss
        self.dense_layer_dropout = nn.Dropout(
            p=1 - self.params.output_keep_prob
        )
        if self.params.use_crf_layer:
            self.classifier_head = nn.Linear(self.out_dim, self.params.num_labels)
            self.crf_layer = CRF(self.params.num_labels, batch_first=True)
        else:
            if self.params.num_labels >= 2:
                # cross-entropy criterion
                self.classifier_head = nn.Linear(self.out_dim, self.params.num_labels)
                self.criterion = nn.CrossEntropyLoss(reduction='mean',
                                                     ignore_index=self.label_padding_idx)
            else:
                msg = f"Invalid number of labels specified: {self.params.num_labels}. " \
                      f"A valid number is equal to or greater than 2"
                raise ValueError(msg)

        print(f"{self.name} is initialized")

    def forward(self, batch_data_dict):

        batch_data_dict = self._to_device(batch_data_dict)
        batch_data_dict = self._forward_core(batch_data_dict)

        token_embs = batch_data_dict["token_embs"]
        token_embs = self.dense_layer_dropout(token_embs)
        logits = self.classifier_head(token_embs)
        batch_data_dict.update({"logits": logits})

        targets = batch_data_dict.get("labels")
        if targets is not None:
            if self.params.use_crf_layer:
                # create a mask to ignore token positions that are padded
                mask = torch.as_tensor(targets != self.label_padding_idx, dtype=torch.uint8)
                loss = - self.crf_layer(logits, targets, mask=mask)  # negative log likelihood
            else:
                loss = self.criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))
            batch_data_dict.update({"loss": loss})

        return batch_data_dict

    def predict(self, examples):
        preds = []
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for start_idx in range(0, len(examples), self.params.batch_size):
                this_examples = examples[start_idx:start_idx + self.params.batch_size]
                batch_data_dict = self.encoder.batch_encode(this_examples)
                this_logits = self.forward(batch_data_dict)["logits"]
                # find predictions
                if self.params.use_crf_layer:
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
        preds = []
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for start_idx in range(0, len(examples), self.params.batch_size):
                this_examples = examples[start_idx:start_idx + self.params.batch_size]
                batch_data_dict = self.encoder.batch_encode(this_examples)
                this_logits = self.forward(batch_data_dict)["logits"]
                # find predictions
                if self.params.use_crf_layer:
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

    @abstractmethod
    def _init_core(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _forward_core(self, batch_data_dict: Dict) -> Dict:
        raise NotImplementedError


class EmbedderForTokenClassification(TokenClassificationBase):
    DEFAULT_PARAMS = {
        "padding_idx": None,
        "update_embeddings": True,
        "embedder_output_keep_prob": 0.7,
        "output_keep_prob": 1.0,
    }

    def _init_core(self):
        self.emb_layer = EmbeddingLayer(
            self.params.num_tokens,
            self.params.emb_dim,
            self.params.padding_idx,
            self.params.pop("embedding_weights", None),
            self.params.update_embeddings,
            1 - self.params.embedder_output_keep_prob
        )
        self.out_dim = self.params.emb_dim

    def _forward_core(self, batch_data_dict):
        seq_ids = batch_data_dict["seq_ids"]  # [BS, SEQ_LEN]

        encodings = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, self.out_dim]

        batch_data_dict.update({"token_embs": encodings})

        return batch_data_dict


class LstmForTokenClassification(TokenClassificationBase):
    """A LSTM module that operates on a batched sequence of token ids. The tokens could be
    characters or words or sub-words. This module uses an additional input that determines
    how the sequence of embeddings obtained after the LSTM layers for each instance in the
    batch, needs to be split. Once split, the sub-groups of embeddings (each sub-group
    corresponding to a word or a phrase) can be collapsed to 1D representation per sub-group
    through pooling operations. Finally, this module outputs a 2D representation for each
    instance in the batch (i.e. [BS, SEQ_LEN', EMB_DIM]).
    """

    DEFAULT_PARAMS = {
        "padding_idx": None,
        "update_embeddings": True,
        "embedder_output_keep_prob": 0.7,
        "lstm_hidden_dim": 128,
        "lstm_num_layers": 2,
        "lstm_keep_prob": 0.7,
        "lstm_bidirectional": True,
    }

    def _init_core(self):
        self.emb_layer = EmbeddingLayer(
            self.params.num_tokens,
            self.params.emb_dim,
            self.params.padding_idx,
            self.params.pop("embedding_weights", None),
            self.params.update_embeddings,
            1 - self.params.embedder_output_keep_prob
        )
        self.lstm_layer = LstmLayer(
            self.params.emb_dim,
            self.params.lstm_hidden_dim,
            self.params.lstm_num_layers,
            1 - self.params.lstm_keep_prob,
            self.params.lstm_bidirectional
        )
        self.out_dim = (
            self.params.lstm_hidden_dim * 2 if self.params.lstm_bidirectional
            else self.params.lstm_hidden_dim
        )

    def _forward_core(self, batch_data_dict):
        seq_ids = batch_data_dict["seq_ids"]  # [BS, SEQ_LEN]
        seq_lengths = batch_data_dict["seq_lengths"]  # [BS]

        encodings = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, EMD_DIM]
        encodings = self.lstm_layer(encodings, seq_lengths)  # [BS, SEQ_LEN, self.out_dim]

        batch_data_dict.update({"token_embs": encodings})

        return batch_data_dict


class CharLstmWithWordLstmForTokenClassification(TokenClassificationBase):
    DEFAULT_PARAMS = {
        "padding_idx": None,
        "update_embeddings": True,
        "embedder_output_keep_prob": 0.7,
        "lstm_hidden_dim": 128,
        "lstm_num_layers": 2,
        "lstm_keep_prob": 0.7,
        "lstm_bidirectional": True,
        "char_lstm_hidden_dim": 128,
        "char_lstm_num_layers": 2,
        "char_lstm_keep_prob": 0.7,
        "char_lstm_bidirectional": True,
        "char_lstm_output_pooling_type": "last",
        "word_level_character_embedding_size": None
    }

    def __init__(self):
        super().__init__()

        # default encoder; have to either fit ot load to use it
        self.encoder = TokenClsDualEncoderForEmbLayers()

    def _init_encoder(self, examples, **params):
        params = super()._init_encoder(examples, **params)
        params.update({
            "char_num_tokens": self.encoder.get_char_num_tokens(),
            "char_emb_dim": self.encoder.get_char_emb_dim(),
            "char_padding_idx": self.encoder.get_char_pad_token_idx(),
        })
        return params

    def _init_core(self):
        self.word_level_character_embedding_size = self.params.get(
            "word_level_character_embedding_size") or self.params.emb_dim  # former can be None
        self.params.update({
            "word_level_character_embedding_size": self.word_level_character_embedding_size})

        self.char_emb_layer = EmbeddingLayer(
            self.params.char_num_tokens,
            self.params.char_emb_dim,
            self.params.char_padding_idx,
            self.params.pop("char_embedding_weights", None),
            self.params.update_embeddings
        )
        self.char_lstm_layer = LstmLayer(
            self.params.char_emb_dim,
            self.params.char_lstm_hidden_dim,
            self.params.char_lstm_num_layers,
            1 - self.params.char_lstm_keep_prob,
            self.params.char_lstm_bidirectional
        )
        self.char_dropout = nn.Dropout(
            p=1 - self.params.char_lstm_keep_prob
        )
        self.char_lstm_output_pooling = PoolingLayer(
            self.params.char_lstm_output_pooling_type
        )
        char_out_dim = (
            self.params.char_lstm_hidden_dim * 2 if self.params.char_lstm_bidirectional
            else self.params.char_lstm_hidden_dim
        )
        self.char_lstm_output_transform = nn.Linear(
            char_out_dim, self.params.word_level_character_embedding_size
        )
        self.emb_layer = EmbeddingLayer(
            self.params.num_tokens,
            self.params.emb_dim,
            self.params.padding_idx,
            self.params.pop("embedding_weights", None),
            self.params.update_embeddings,
            1 - self.params.embedder_output_keep_prob
        )
        self.lstm_layer = LstmLayer(
            self.params.emb_dim + self.params.word_level_character_embedding_size,
            self.params.lstm_hidden_dim,
            self.params.lstm_num_layers,
            1 - self.params.lstm_keep_prob,
            self.params.lstm_bidirectional
        )
        self.out_dim = (
            self.params.lstm_hidden_dim * 2 if self.params.lstm_bidirectional
            else self.params.lstm_hidden_dim
        )

    def _forward_core(self, batch_data_dict):
        char_seq_ids = batch_data_dict["char_seq_ids"]  # List of [BS, SEQ_LEN]
        char_seq_lengths = batch_data_dict["char_seq_lengths"]  # List of [BS]

        encs = [self.char_emb_layer(_seq_ids) for _seq_ids in char_seq_ids]
        encs = [self.char_lstm_layer(enc, seq_len) for enc, seq_len in zip(encs, char_seq_lengths)]
        encs = [self.char_dropout(enc) for enc in encs]
        encs = [self.char_lstm_output_pooling(enc, seq_len)
                for enc, seq_len in zip(encs, char_seq_lengths)]
        encs = pad_sequence(encs, batch_first=True)  # [BS, SEQ_LEN, char_out_dim]
        char_encs = self.char_lstm_output_transform(
            encs)  # [BS, SEQ_LEN, self.word_level_character_embedding_size]

        seq_ids = batch_data_dict["seq_ids"]  # [BS, SEQ_LEN]
        seq_lengths = batch_data_dict["seq_lengths"]  # [BS]

        word_encs = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, self.emb_dim]
        char_plus_word_encs = torch.cat((char_encs, word_encs), dim=-1)  # [BS, SEQ_LEN, sum(both)]
        encodings = self.lstm_layer(char_plus_word_encs, seq_lengths)  # [BS, SEQ_LEN, self.out_dim]

        batch_data_dict.update({"token_embs": encodings})

        return batch_data_dict


class CharCnnWithWordLstmForTokenClassification(TokenClassificationBase):
    DEFAULT_PARAMS = {
        "padding_idx": None,
        "update_embeddings": True,
        "embedder_output_keep_prob": 0.7,
        "lstm_hidden_dim": 128,
        "lstm_num_layers": 2,
        "lstm_keep_prob": 0.7,
        "lstm_bidirectional": True,
        "char_window_sizes": [3, 4, 5],
        "char_number_of_windows": [100, 100, 100],
        "char_cnn_output_keep_prob": 0.7,
        "word_level_character_embedding_size": None
    }

    def __init__(self):
        super().__init__()

        # default encoder; have to either fit ot load to use it
        self.encoder = TokenClsDualEncoderForEmbLayers()

    def _init_encoder(self, examples, **params):
        params = super()._init_encoder(examples, **params)
        params.update({
            "char_num_tokens": self.encoder.get_char_num_tokens(),
            "char_emb_dim": self.encoder.get_char_emb_dim(),
            "char_padding_idx": self.encoder.get_char_pad_token_idx(),
        })
        return params

    def _init_core(self):
        self.word_level_character_embedding_size = self.params.get(
            "word_level_character_embedding_size") or self.params.emb_dim
        self.params.update({
            "word_level_character_embedding_size": self.word_level_character_embedding_size})

        self.char_emb_layer = EmbeddingLayer(
            self.params.char_num_tokens,
            self.params.char_emb_dim,
            self.params.char_padding_idx,
            self.params.pop("char_embedding_weights", None),
            self.params.update_embeddings
        )
        self.char_conv_layer = CnnLayer(
            self.params.char_emb_dim,
            self.params.char_window_sizes,
            self.params.char_number_of_windows
        )
        self.char_dropout = nn.Dropout(
            p=1 - self.params.char_cnn_output_keep_prob
        )
        char_out_dim = sum(self.params.char_number_of_windows)
        self.char_cnn_output_transform = nn.Linear(
            char_out_dim, self.params.word_level_character_embedding_size
        )
        self.emb_layer = EmbeddingLayer(
            self.params.num_tokens,
            self.params.emb_dim,
            self.params.padding_idx,
            self.params.pop("embedding_weights", None),
            self.params.update_embeddings,
            1 - self.params.embedder_output_keep_prob
        )
        self.lstm_layer = LstmLayer(
            self.params.emb_dim + self.params.word_level_character_embedding_size,
            self.params.lstm_hidden_dim,
            self.params.lstm_num_layers,
            1 - self.params.lstm_keep_prob,
            self.params.lstm_bidirectional
        )
        self.out_dim = (
            self.params.lstm_hidden_dim * 2 if self.params.lstm_bidirectional
            else self.params.lstm_hidden_dim
        )

    def _forward_core(self, batch_data_dict):
        char_seq_ids = batch_data_dict["char_seq_ids"]  # List of [BS, SEQ_LEN]
        # char_seq_lengths = batch_data_dict["char_seq_lengths"]  # List of [BS]

        encs = [self.char_emb_layer(_seq_ids) for _seq_ids in char_seq_ids]
        encs = [self.char_conv_layer(enc) for enc in encs]
        encs = [self.char_dropout(enc) for enc in encs]
        encs = pad_sequence(encs, batch_first=True)  # [BS, SEQ_LEN, sum(self.number_of_windows)]
        char_encs = self.char_cnn_output_transform(
            encs)  # [BS, SEQ_LEN, self.word_level_character_embedding_size]

        seq_ids = batch_data_dict["seq_ids"]  # [BS, SEQ_LEN]
        seq_lengths = batch_data_dict["seq_lengths"]  # [BS]

        word_encs = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, self.emb_dim]
        char_plus_word_encs = torch.cat((char_encs, word_encs), dim=-1)  # [BS, SEQ_LEN, sum(both)]
        encodings = self.lstm_layer(char_plus_word_encs, seq_lengths)  # [BS, SEQ_LEN, self.out_dim]

        batch_data_dict.update({"token_embs": encodings})

        return batch_data_dict


class BertForTokenClassification(TokenClassificationBase):
    DEFAULT_PARAMS = {
        "embedder_output_keep_prob": 0.7,
        "token_spans_pooling_type": "first",
        "output_keep_prob": 1.0,
        "use_crf_layer": False
    }

    def __init__(self):
        super().__init__()

        # overwrite default encoder
        self.encoder = TokenClsEncoderWithPlmLayer()

    def _init_core(self):

        self.dropout = nn.Dropout(
            p=1 - self.params.embedder_output_keep_prob
        )
        self.span_pooling_layer = SplittingAndPoolingLayer(
            self.params.token_spans_pooling_type
        )
        self.out_dim = self.params.emb_dim

    def _forward_core(self, batch_data_dict):
        split_lengths = batch_data_dict["split_lengths"]  # List[List[Int]]
        last_hidden_state = batch_data_dict["last_hidden_state"]  # [BS, SEQ_LEN, EMD_DIM]

        last_hidden_state = self.dropout(last_hidden_state)

        encodings = self.span_pooling_layer(
            last_hidden_state, split_lengths
        )  # [BS, SEQ_LEN`, EMD_DIM]
        batch_data_dict.update({"token_embs": encodings})

        return batch_data_dict

    def _create_optimizer(self):
        params = list(self.named_parameters())
        no_decay = ["bias", 'LayerNorm.bias', "LayerNorm.weight",
                    'layer_norm.bias', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = getattr(torch.optim, self.params.optimizer)(
            optimizer_grouped_parameters,
            lr=self.params.learning_rate,
            eps=1e-08,
            weight_decay=0.01
        )
        return optimizer

    def _create_optimizer_and_scheduler(self, num_training_steps):

        num_warmup_steps = min(0.1 * num_training_steps, self.params.num_warmup_steps)
        self.params.update({"num_warmup_steps": num_warmup_steps})

        # https://github.com/huggingface/transformers/blob/master/src/transformers/optimization.py
        # refer `get_linear_schedule_with_warmup` method
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(
                    max(1, num_training_steps - num_warmup_steps))
            )

        # load a torch optimizer
        optimizer = self._create_optimizer()
        # load a lr scheduler
        scheduler = getattr(torch.optim.lr_scheduler, "LambdaLR")(optimizer, lr_lambda)
        return optimizer, scheduler

    def fit(self, examples, labels, **params):
        # overriding base class' method to set params, and then calling base class' .fit()

        embedder_type = params.get("embedder_type", "bert")
        if embedder_type != "bert":
            msg = f"{self.name} can only be used with 'embedder_type': 'bert'. " \
                  f"Other values passed through config params are not allowed"
            raise ValueError(msg)

        safe_values = {
            "num_warmup_steps": 50,
            "learning_rate": 2e-5,
            "optimizer": "AdamW",
            "number_of_epochs": 20,
            "patience": 4,
            "batch_size": 16,
            "gradient_accumulation_steps": 2,
            "max_grad_norm": 1.0
        }

        for k, v in safe_values.items():
            v_inputted = params.get(k, v)
            if v != v_inputted:
                msg = f"{self.name} can be best used with '{k}' equal to '{v}'. Other " \
                      f"values passed through config params might lead to unexpected results."
                logger.warning(msg)
            else:
                params.update({k: v})

        params.update({"embedder_type": "bert"})

        super().fit(examples, labels, **params)
