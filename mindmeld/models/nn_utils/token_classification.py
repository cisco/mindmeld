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
Custom modules built on top of nn layers that can do token classification
"""

import logging
from abc import abstractmethod
from collections import OrderedDict
from typing import List

from .classification import BaseClassification
from .helpers import TokenClassificationType
from .helpers import TokenizerType, EmbedderType
from .layers import (
    EmbeddingLayer,
    CnnLayer,
    LstmLayer,
    PoolingLayer,
    SplittingAndPoolingLayer
)
from ..containers import HuggingfaceTransformersContainer

try:
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pad_sequence
except ImportError:
    pass

try:
    from torchcrf import CRF

    TORCHCRF_AVAILABLE = True
except ImportError:
    TORCHCRF_AVAILABLE = False
    pass

logger = logging.getLogger(__name__)


class BaseTokenClassification(BaseClassification):
    """Base class that defines all the necessary elements to successfully train/infer
     custom pytorch modules wrapped on top of this base class. Classes derived from
     this base can be trained for sequence tagging aka. token classification.
    """

    @property
    def classification_type(self):
        return "tagger"

    def _prepare_labels(self, labels: List[List[int]], max_length: int):
        # for token classification, the length of an example matters (i.e. the number of
        # (sub-)words in it) as the labels/targets are one-to-one mapped per (sub-)word.
        # hence, we need to do padding with a _label_padding_idx

        def _trim_or_pad(label_sequence):
            if len(label_sequence) > max_length:
                return label_sequence[:max_length]
            else:
                return (
                    label_sequence + [self.params._label_padding_idx] *
                    (max_length - len(label_sequence))
                )

        return torch.as_tensor(
            [_trim_or_pad(_label_sequence) for _label_sequence in labels],
            dtype=torch.long
        )

    def _init_graph(self):

        self._init_core()

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
        self.span_pooling_layer = SplittingAndPoolingLayer(
            self.params.token_spans_pooling_type,
            self.encoder.number_of_terminal_tokens,  # arg max number of terminal tokens
        )
        if not self.params.num_labels:
            msg = f"Invalid number of labels ({self.params.num_labels}) inputted to '{self.name}'"
            raise ValueError(msg)

        # init the peripheral architectural components and the criterion to compute loss
        self.dense_layer_dropout = nn.Dropout(
            p=1 - self.params.output_keep_prob
        )
        if self.params.use_crf_layer:
            self.classifier_head = nn.Linear(self.out_dim, self.params.num_labels)
            if not TORCHCRF_AVAILABLE:
                raise ImportError("pip install torchcrf")
            self.crf_layer = CRF(self.params.num_labels, batch_first=True)
        else:
            if self.params.num_labels >= 2:
                # cross-entropy criterion
                self.classifier_head = nn.Linear(self.out_dim, self.params.num_labels)
                self.criterion = nn.CrossEntropyLoss(
                    reduction='mean', ignore_index=self.params._label_padding_idx)
            else:
                msg = f"Invalid number of labels specified: {self.params.num_labels}. " \
                      f"A valid number is equal to or greater than 2"
                raise ValueError(msg)

        msg = f"{self.name} is initialized"
        logger.info(msg)

    def forward(self, batch_data):

        batch_data = self.to_device(batch_data)
        batch_data = self._forward_core(batch_data)

        token_embs = batch_data["token_embs"]

        # strip the terminals if required; note they are not accounted in the label padding process
        token_embs = self.span_pooling_layer(
            token_embs,
            batch_data["split_lengths"],  # List[Tensor1d[int]],
            discard_terminals=self.params.add_terminals
        )  # [BS, SEQ_LEN`, EMD_DIM], and the SEQ_LEN` matches the width of labels upon padding

        token_embs = self.dense_layer_dropout(token_embs)
        logits = self.classifier_head(token_embs)
        batch_data.update({"logits": logits})

        targets = batch_data.pop("_labels", None)
        if targets is not None:
            if self.params.use_crf_layer:
                # create a mask to ignore token positions that are padded
                mask = torch.as_tensor(targets != self.params._label_padding_idx,
                                       dtype=torch.uint8).to(self.params.device)
                loss = - self.crf_layer(logits, targets, mask=mask)  # negative log likelihood
            else:
                loss = self.criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))
            batch_data.update({"loss": loss})

        return batch_data

    def predict(self, examples):
        predictions = []
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for start_idx in range(0, len(examples), self.params.batch_size):
                batch_examples = examples[start_idx:start_idx + self.params.batch_size]
                batch_data = self.encoder.batch_encode(
                    batch_examples,
                    padding_length=self.params.padding_length,
                    **({'add_terminals': self.params.add_terminals}
                       if self.params.add_terminals is not None else {})
                )
                batch_logits = self.forward(batch_data)["logits"]
                # find predictions
                if self.params.use_crf_layer:
                    # create a mask to ignore token positions that are padded
                    mask = pad_sequence(
                        [torch.as_tensor([1] * len(_split_lengths))
                         for _split_lengths in batch_data["split_lengths"]],
                        batch_first=True
                    )
                    # [BS] -> [BS, SEQ_LEN]
                    mask = torch.as_tensor(mask, dtype=torch.uint8).to(self.params.device)
                    batch_predictions = self.crf_layer.decode(batch_logits, mask=mask)
                else:
                    batch_predictions = torch.argmax(batch_logits, dim=-1).tolist()
                # trim predictions as per sequence length
                batch_predictions = [
                    predictions[:len(_split_lengths)] for predictions, _split_lengths in
                    zip(batch_predictions, batch_data["split_lengths"])
                ]
                predictions.extend(batch_predictions)
        if was_training:
            self.train()
        return predictions

    def predict_proba(self, examples):
        prediction_tuples = []
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for start_idx in range(0, len(examples), self.params.batch_size):
                batch_examples = examples[start_idx:start_idx + self.params.batch_size]
                batch_data = self.encoder.batch_encode(
                    batch_examples,
                    padding_length=self.params.padding_length,
                    **({'add_terminals': self.params.add_terminals}
                       if self.params.add_terminals is not None else {})
                )
                batch_logits = self.forward(batch_data)["logits"]
                # find prediction probabilities
                if self.params.use_crf_layer:
                    msg = f"Prediction probabilities cannot be computed when the param " \
                          f"use_crf_layer is set to True in {self.__class__.__name__}"
                    raise NotImplementedError(msg)
                else:
                    batch_maxes = torch.max(batch_logits, dim=-1)
                    batch_predictions = batch_maxes.indices.tolist()
                    batch_probabilities = batch_maxes.values.tolist()
                for preds, probs, _split_lengths in zip(
                    batch_predictions, batch_probabilities, batch_data["split_lengths"]
                ):
                    prediction_tuples.append(list(zip(preds, probs))[:len(_split_lengths)])
        if was_training:
            self.train()
        return prediction_tuples

    @abstractmethod
    def _init_core(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _forward_core(self, batch_data):
        raise NotImplementedError


class EmbedderForTokenClassification(BaseTokenClassification):

    def _init_core(self):
        self.emb_layer = EmbeddingLayer(
            self.params._num_tokens,
            self.params.emb_dim,
            self.params._padding_idx,
            self.params.pop("_embedding_weights", None),
            self.params.update_embeddings,
            1 - self.params.embedder_output_keep_prob
        )
        self.out_dim = self.params.emb_dim

    def _forward_core(self, batch_data):
        seq_ids = batch_data["seq_ids"]  # [BS, SEQ_LEN]

        token_embs = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, self.out_dim]

        batch_data.update({"token_embs": token_embs})
        return batch_data


class LstmForTokenClassification(BaseTokenClassification):
    """A LSTM module that operates on a batched sequence of token ids. The tokens could be
    characters or words or sub-words. This module uses an additional input that determines
    how the sequence of embeddings obtained after the LSTM layers for each instance in the
    batch, needs to be split. Once split, the sub-groups of embeddings (each sub-group
    corresponding to a word or a phrase) can be collapsed to 1D representation per sub-group
    through pooling operations. Finally, this module outputs a 2D representation for each
    instance in the batch (i.e. [BS, SEQ_LEN', EMB_DIM]).
    """

    def _init_core(self):
        self.emb_layer = EmbeddingLayer(
            self.params._num_tokens,
            self.params.emb_dim,
            self.params._padding_idx,
            self.params.pop("_embedding_weights", None),
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

    def _forward_core(self, batch_data):
        seq_ids = batch_data["seq_ids"]  # [BS, SEQ_LEN]

        summed_split_lengths = [
            sum(_split_lengths) +
            (self.encoder.number_of_terminal_tokens if self.params.add_terminals else 0)
            for _split_lengths in batch_data["split_lengths"]
        ]
        summed_split_lengths = torch.as_tensor(summed_split_lengths, dtype=torch.long)  # [BS]

        token_embs = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, EMD_DIM]
        token_embs = self.lstm_layer(
            token_embs, summed_split_lengths)  # [BS, SEQ_LEN, self.out_dim]

        batch_data.update({"token_embs": token_embs})

        return batch_data


class CharLstmWithWordLstmForTokenClassification(BaseTokenClassification):

    def _prepare_input_encoder(self, examples, **params):
        if (params.get("tokenizer_type") and
            TokenizerType(params.get("tokenizer_type")) !=
            TokenizerType.WHITESPACE_AND_CHAR_DUAL_TOKENIZER
        ):
            msg = f"Param 'tokenizer_type' must be " \
                  f"{TokenizerType.WHITESPACE_AND_CHAR_DUAL_TOKENIZER.value} for " \
                  f"{self.__class__.__name__}."
            raise ValueError(msg)

        params = super()._prepare_input_encoder(examples, **params)

        # update params without which can become ambiguous when loading a model
        params.update({
            "_char_num_tokens": len(self.encoder.get_char_vocab()),
            "_char_padding_idx": self.encoder.get_char_pad_token_idx(),
            "char_add_terminals": params.get("char_add_terminals"),
            "char_padding_length": params.get("char_padding_length"),
            "char_emb_dim": params.get("char_emb_dim"),
        })

        return params

    def _init_core(self):
        self.char_proj_dim = self.params.get("char_proj_dim") or self.params.emb_dim
        self.params.update({"char_proj_dim": self.char_proj_dim})

        self.char_emb_layer = EmbeddingLayer(
            self.params._char_num_tokens,
            self.params.char_emb_dim,
            self.params._char_padding_idx,
            self.params.pop("_char_embedding_weights", None),
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
        self.char_lstm_output_transform = nn.Linear(char_out_dim, self.params.char_proj_dim)
        self.emb_layer = EmbeddingLayer(
            self.params._num_tokens,
            self.params.emb_dim,
            self.params._padding_idx,
            self.params.pop("_embedding_weights", None),
            self.params.update_embeddings,
            1 - self.params.embedder_output_keep_prob
        )
        self.lstm_layer = LstmLayer(
            self.params.emb_dim + self.params.char_proj_dim,
            self.params.lstm_hidden_dim,
            self.params.lstm_num_layers,
            1 - self.params.lstm_keep_prob,
            self.params.lstm_bidirectional
        )
        self.out_dim = (
            self.params.lstm_hidden_dim * 2 if self.params.lstm_bidirectional
            else self.params.lstm_hidden_dim
        )

    def _forward_core(self, batch_data):
        char_seq_ids = batch_data["char_seq_ids"]  # List of [BS, SEQ_LEN]
        char_seq_lengths = batch_data["char_seq_lengths"]  # List of [BS]

        encs = [self.char_emb_layer(_seq_ids) for _seq_ids in char_seq_ids]
        encs = [self.char_lstm_layer(enc, seq_len) for enc, seq_len in zip(encs, char_seq_lengths)]
        encs = [self.char_dropout(enc) for enc in encs]
        encs = [self.char_lstm_output_pooling(enc, seq_len)
                for enc, seq_len in zip(encs, char_seq_lengths)]
        encs = pad_sequence(encs, batch_first=True)  # [BS, SEQ_LEN, char_out_dim]
        char_encs = self.char_lstm_output_transform(encs)  # [BS, SEQ_LEN, self.char_proj_dim]

        summed_split_lengths = [
            sum(_split_lengths) +
            (self.encoder.number_of_terminal_tokens if self.params.add_terminals else 0)
            for _split_lengths in batch_data["split_lengths"]
        ]
        summed_split_lengths = torch.as_tensor(summed_split_lengths, dtype=torch.long)  # [BS]

        seq_ids = batch_data["seq_ids"]  # [BS, SEQ_LEN]
        word_encs = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, self.emb_dim]
        char_plus_word_encs = torch.cat((char_encs, word_encs), dim=-1)  # [BS, SEQ_LEN, sum(both)]
        token_embs = self.lstm_layer(
            char_plus_word_encs, summed_split_lengths)  # [BS, SEQ_LEN, self.out_dim]

        batch_data.update({"token_embs": token_embs})

        return batch_data


class CharCnnWithWordLstmForTokenClassification(BaseTokenClassification):

    def _prepare_input_encoder(self, examples, **params):
        if (params.get("tokenizer_type") and
            TokenizerType(params.get("tokenizer_type")) !=
            TokenizerType.WHITESPACE_AND_CHAR_DUAL_TOKENIZER
        ):
            msg = f"Param 'tokenizer_type' must be " \
                  f"{TokenizerType.WHITESPACE_AND_CHAR_DUAL_TOKENIZER.value} for " \
                  f"{self.__class__.__name__}."
            raise ValueError(msg)

        params = super()._prepare_input_encoder(examples, **params)

        # update params without which can become ambiguous when loading a model
        params.update({
            "_char_num_tokens": len(self.encoder.get_char_vocab()),
            "_char_padding_idx": self.encoder.get_char_pad_token_idx(),
            "char_add_terminals": params.get("char_add_terminals"),
            "char_padding_length": params.get("char_padding_length"),
            "char_emb_dim": params.get("char_emb_dim"),
        })

        return params

    def _init_core(self):
        self.char_proj_dim = self.params.get("char_proj_dim") or self.params.emb_dim
        self.params.update({"char_proj_dim": self.char_proj_dim})

        self.char_emb_layer = EmbeddingLayer(
            self.params._char_num_tokens,
            self.params.char_emb_dim,
            self.params._char_padding_idx,
            self.params.pop("_char_embedding_weights", None),
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
        self.char_cnn_output_transform = nn.Linear(char_out_dim, self.params.char_proj_dim)
        self.emb_layer = EmbeddingLayer(
            self.params._num_tokens,
            self.params.emb_dim,
            self.params._padding_idx,
            self.params.pop("_embedding_weights", None),
            self.params.update_embeddings,
            1 - self.params.embedder_output_keep_prob
        )
        self.lstm_layer = LstmLayer(
            self.params.emb_dim + self.params.char_proj_dim,
            self.params.lstm_hidden_dim,
            self.params.lstm_num_layers,
            1 - self.params.lstm_keep_prob,
            self.params.lstm_bidirectional
        )
        self.out_dim = (
            self.params.lstm_hidden_dim * 2 if self.params.lstm_bidirectional
            else self.params.lstm_hidden_dim
        )

    def _forward_core(self, batch_data):
        char_seq_ids = batch_data["char_seq_ids"]  # List of [BS, SEQ_LEN]

        encs = [self.char_emb_layer(_seq_ids) for _seq_ids in char_seq_ids]
        encs = [self.char_conv_layer(enc) for enc in encs]
        encs = [self.char_dropout(enc) for enc in encs]
        encs = pad_sequence(encs, batch_first=True)  # [BS, SEQ_LEN, sum(self.number_of_windows)]
        char_encs = self.char_cnn_output_transform(encs)  # [BS, SEQ_LEN, self.char_proj_dim]

        summed_split_lengths = [
            sum(_split_lengths) +
            (self.encoder.number_of_terminal_tokens if self.params.add_terminals else 0)
            for _split_lengths in batch_data["split_lengths"]
        ]
        summed_split_lengths = torch.as_tensor(summed_split_lengths, dtype=torch.long)  # [BS]

        seq_ids = batch_data["seq_ids"]  # [BS, SEQ_LEN]
        word_encs = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, self.emb_dim]
        char_plus_word_encs = torch.cat((char_encs, word_encs), dim=-1)  # [BS, SEQ_LEN, sum(both)]
        token_embs = self.lstm_layer(
            char_plus_word_encs, summed_split_lengths)  # [BS, SEQ_LEN, self.out_dim]

        batch_data.update({"token_embs": token_embs})

        return batch_data


class BertForTokenClassification(BaseTokenClassification):

    def fit(self, examples, labels, **params):
        # overriding base class' method to set params, and then calling base class' .fit()

        embedder_type = params.get("embedder_type", EmbedderType.BERT.value)
        if EmbedderType(embedder_type) != EmbedderType.BERT:
            msg = f"{self.name} can only be used with 'embedder_type': " \
                  f"'{EmbedderType.BERT.value}'. " \
                  f"Other values passed through config params are not allowed."
            raise ValueError(msg)

        safe_values = {
            "num_warmup_steps": 50,
            "learning_rate": 2e-5,
            "optimizer": "AdamW",
            "max_grad_norm": 1.0
        }

        for k, v in safe_values.items():
            v_inputted = params.get(k, v)
            if v != v_inputted:
                msg = f"{self.name} can be best used with '{k}' equal to '{v}' but found " \
                      f"the value '{v_inputted}'. Use the non-default value with caution as it " \
                      f"may lead to unexpected results and longer training times depending on " \
                      f"the choice of pretrained model."
                logger.warning(msg)
            else:
                params.update({k: v})

        params.update({
            "embedder_type": embedder_type,
            "save_frozen_embedder": params.get("save_frozen_embedder", False)  # if True,
            # frozen set of bert weights are also dumped, else they are skipped as they are not
            # tuned and anyway frozen during training.
        })

        super().fit(examples, labels, **params)

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

        num_warmup_steps = min(int(0.1 * num_training_steps), self.params.num_warmup_steps)
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

    def _get_dumpable_state_dict(self):
        if not self.params.update_embeddings and not self.params.save_frozen_embedder:
            state_dict = OrderedDict(
                {k: v for k, v in self.state_dict().items() if not k.startswith("bert_model")}
            )
            return state_dict
        return self.state_dict()

    def _init_core(self):

        self.bert_model = HuggingfaceTransformersContainer(
            self.params.pretrained_model_name_or_path,
            cache_lookup=False
        ).get_transformer_model()
        if not self.params.update_embeddings:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(
            p=1 - self.params.embedder_output_keep_prob
        )
        self.out_dim = self.params.emb_dim

    def _forward_core(self, batch_data):

        # refer to https://huggingface.co/docs/transformers/master/en/main_classes/output
        # for more details on huggingface's bert outputs

        bert_outputs = self.bert_model(**batch_data["hgf_encodings"], return_dict=True)

        # 'last_hidden_state' refers to the tensor output of the final transformer layer (i.e.
        # before the logit layer) and hence its dimension  [BS, SEQ_LEN, EMD_DIM]
        last_hidden_state = bert_outputs.get("last_hidden_state")  # [BS, SEQ_LEN, EMD_DIM]
        if last_hidden_state is None:
            msg = f"The choice of pretrained bert model " \
                  f"({self.params.pretrained_model_name_or_path}) " \
                  f"has no key 'last_hidden_state' in its output dictionary."
            raise ValueError(msg)

        last_hidden_state = self.dropout(last_hidden_state)
        batch_data.update({"token_embs": last_hidden_state})
        return batch_data


def get_token_classifier_cls(classifier_type: str, embedder_type: str = None):
    try:
        classifier_type = TokenClassificationType(classifier_type)
    except ValueError as e:
        msg = f"Neural Nets' token classification module expects classifier_type to be amongst" \
              f" {[v.value for v in TokenClassificationType.__members__.values()]}" \
              f" but found '{classifier_type}'."
        raise ValueError(msg) from e

    try:
        embedder_type = EmbedderType(embedder_type)
    except ValueError as e:
        msg = f"Neural Nets' token classification module expects embedder_type to be amongst" \
              f" {[v.value for v in EmbedderType.__members__.values()]} " \
              f" but found '{embedder_type}'."
        raise ValueError(msg) from e

    if (
        embedder_type == EmbedderType.BERT and
        classifier_type not in [TokenClassificationType.EMBEDDER]
    ):
        msg = f"To use the embedder_type '{EmbedderType.BERT.value}', " \
              f"classifier_type must be '{TokenClassificationType.EMBEDDER.value}'."
        raise ValueError(msg)

    # disambiguation between glove, bert and non-pretrained embedders
    def _resolve_and_return_embedder_class(_embedder_type):
        return {
            EmbedderType.NONE: EmbedderForTokenClassification,
            EmbedderType.GLOVE: EmbedderForTokenClassification,
            EmbedderType.BERT: BertForTokenClassification
        }[_embedder_type]

    return {
        TokenClassificationType.EMBEDDER: _resolve_and_return_embedder_class(embedder_type),
        TokenClassificationType.LSTM: LstmForTokenClassification,
        TokenClassificationType.CNN_LSTM: CharCnnWithWordLstmForTokenClassification,
        TokenClassificationType.LSTM_LSTM: CharLstmWithWordLstmForTokenClassification,
    }[classifier_type]
