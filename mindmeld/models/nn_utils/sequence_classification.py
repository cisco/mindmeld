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
from collections import OrderedDict
from typing import List

from .classification import BaseClassification
from .helpers import BatchData, EmbedderType, SequenceClassificationType
from .layers import (
    EmbeddingLayer,
    CnnLayer,
    LstmLayer,
    PoolingLayer
)
from ..containers import HuggingfaceTransformersContainer

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    pass

logger = logging.getLogger(__name__)


class BaseSequenceClassification(BaseClassification):
    """Base class that defines all the necessary elements to successfully train/infer
     custom pytorch modules wrapped on top of this base class. Classes derived from
     this base can be trained for sequence classification.
    """

    @property
    def classification_type(self):
        return "text"

    def _prepare_labels(self, labels: List[int], max_length: int = None):
        # for sequence classification, the length of an example doesn't matter as we have only one
        # label for each example. hence, no need to do any padding or validation checks.
        del max_length
        return torch.as_tensor(labels, dtype=torch.long)

    def _init_graph(self):

        # initialize the beginning layers of the graph
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

        # init the peripheral architecture params
        if not self.params.num_labels:
            msg = f"Invalid number of labels ({self.params.num_labels}) inputted to '{self.name}'"
            raise ValueError(msg)

        # init the peripheral architectural components and the criterion to compute loss
        self.dense_layer_dropout = nn.Dropout(
            p=1 - self.params.output_keep_prob
        )
        if self.params.num_labels == 2:
            # sigmoid criterion
            self.classifier_head = nn.Linear(self.out_dim, 1)
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        elif self.params.num_labels > 2:
            # cross-entropy criterion
            self.classifier_head = nn.Linear(self.out_dim, self.params.num_labels)
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        else:
            msg = f"Invalid number of labels specified: {self.params.num_labels}. " \
                  f"A valid number is equal to or greater than 2"
            raise ValueError(msg)

        msg = f"{self.name} is initialized"
        logger.info(msg)

    def forward(self, batch_data):

        batch_data = self.to_device(batch_data)
        batch_data = self._forward_core(batch_data)

        seq_embs = batch_data["seq_embs"]
        seq_embs = self.dense_layer_dropout(seq_embs)
        logits = self.classifier_head(seq_embs)
        batch_data.update({"logits": logits})

        targets = batch_data.pop("_labels", None)
        if targets is not None:
            if self.params.num_labels == 2:
                loss = self.criterion(logits.view(-1), targets.float())
            else:  # self.params.num_labels > 2:
                loss = self.criterion(logits, targets)
            batch_data.update({"loss": loss})

        return batch_data

    def predict(self, examples):
        logits = self._forward_with_batching_and_no_grad(examples)
        if self.params.num_labels == 2:
            predictions = (logits >= 0.5).long().view(-1)
        else:  # self.params.num_labels > 2:
            predictions = torch.argmax(logits, dim=-1)
        return predictions.tolist()

    def predict_proba(self, examples):
        logits = self._forward_with_batching_and_no_grad(examples)
        if self.params.num_labels == 2:
            probs = F.sigmoid(logits)
            # extending the results from shape [N,1] to [N,2] to give out class probs distinctly
            probs = torch.cat((1 - probs, probs), dim=-1)
        else:  # self.params.num_labels > 2:
            probs = F.softmax(logits, dim=-1)
        return probs.tolist()

    def _forward_with_batching_and_no_grad(self, examples):
        logits = None
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
                logits = torch.cat((logits, batch_logits)) if logits is not None else batch_logits
        if was_training:
            self.train()
        return logits

    @abstractmethod
    def _init_core(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _forward_core(self, batch_data: BatchData) -> BatchData:
        raise NotImplementedError


class EmbedderForSequenceClassification(BaseSequenceClassification):
    """An embedder pooling module that operates on a batched sequence of token ids. The
    tokens could be characters or words or sub-words. This module finally outputs one 1D
    representation for each instance in the batch (i.e. [BS, EMB_DIM]).

    The `forward` method of this module expects padded token ids along with numer of tokens
    per instance in the batch.

    Additionally, one can set different coefficients for different tokens of the embedding
    matrix (e.g. tf-idf weights).
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
        self.emb_layer_pooling = PoolingLayer(
            self.params.embedder_output_pooling_type
        )
        self.out_dim = self.params.emb_dim

    def _forward_core(self, batch_data):
        seq_ids = batch_data["seq_ids"]  # [BS, SEQ_LEN]

        summed_split_lengths = [
            sum(_split_lengths) +
            (self.encoder.number_of_terminal_tokens if self.params.add_terminals else 0)
            for _split_lengths in batch_data["split_lengths"]
        ]
        summed_split_lengths = torch.as_tensor(summed_split_lengths, dtype=torch.long)  # [BS]

        encodings = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, EMD_DIM]
        encodings = self.emb_layer_pooling(encodings, summed_split_lengths)  # [BS, self.out_dim]

        batch_data.update({"seq_embs": encodings})

        return batch_data


class CnnForSequenceClassification(BaseSequenceClassification):
    """A CNN module that operates on a batched sequence of token ids. The tokens could be
    characters or words or sub-words. This module finally outputs one 1D representation
    for each instance in the batch (i.e. [BS, EMB_DIM]).

    The `forward` method of this module expects only padded token ids as input.
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
        self.conv_layer = CnnLayer(
            self.params.emb_dim,
            self.params.window_sizes,
            self.params.number_of_windows
        )
        self.out_dim = sum(self.params.number_of_windows)

    def _forward_core(self, batch_data):
        seq_ids = batch_data["seq_ids"]  # [BS, SEQ_LEN]

        encodings = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, EMD_DIM]
        encodings = self.conv_layer(encodings)  # [BS, self.out_dim]

        batch_data.update({"seq_embs": encodings})

        return batch_data


class LstmForSequenceClassification(BaseSequenceClassification):
    """A LSTM module that operates on a batched sequence of token ids. The tokens could be
    characters or words or sub-words. This module finally outputs one 1D representation
    for each instance in the batch (i.e. [BS, EMB_DIM]).

    The `forward` method of this module expects padded token ids along with numer of tokens
    per instance in the batch.
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
        self.lstm_layer_pooling = PoolingLayer(
            self.params.lstm_output_pooling_type
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

        encodings = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, EMD_DIM]
        encodings = self.lstm_layer(encodings, summed_split_lengths)  # [BS,SEQ_LEN,self.out_dim]
        encodings = self.lstm_layer_pooling(encodings, summed_split_lengths)  # [BS,self.out_dim]

        batch_data.update({"seq_embs": encodings})

        return batch_data


class BertForSequenceClassification(BaseSequenceClassification):

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
            "max_grad_norm": 1.0,
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
        if self.params.embedder_output_pooling_type != "first":
            self.emb_layer_pooling = PoolingLayer(
                self.params.embedder_output_pooling_type
            )
        self.out_dim = self.params.emb_dim

        self.no_pooler_output_exists = None  # relevant in forward() if the pooling type is "first"

    def _forward_core(self, batch_data):

        # refer to https://huggingface.co/docs/transformers/master/en/main_classes/output
        # for more details on huggingface's bert outputs

        bert_outputs = self.bert_model(**batch_data["hgf_encodings"], return_dict=True)

        if self.params.embedder_output_pooling_type == "first":
            # 'pooler_output' refers to the first token's (aka. CLS) representation obtained form
            # the last hidden layer, and hence its dimension [BS, self.out_dim]
            if not self.no_pooler_output_exists:
                pooler_output = bert_outputs.get("pooler_output")  # [BS, self.out_dim]
                if pooler_output is None:
                    available_keys = bert_outputs.keys()
                    msg = f"The transformer model ({self.params.pretrained_model_name_or_path}) " \
                          f"has no key 'pooler_output' in its output dictionary (maybe the " \
                          f"selected choice of model has no specific pooler layer on top of the " \
                          f"CLS kind-of token?). The available keys are: {available_keys}.\n"
                    msg += f"Continuing to obtain the first index of the last hidden state due " \
                           f"to the passed-in value for embedder output pooling type param: " \
                           f"'{self.params.embedder_output_pooling_type}'. If you wish to pool " \
                           f"differently (eg. 'mean' or 'max' pooling), change the " \
                           f"'embedder_output_pooling_type' param value in configs and restart " \
                           f"training."
                    logger.error(msg)
                    self.no_pooler_output_exists = True  # avoids logging same error multiple times
            if self.no_pooler_output_exists:  # not just if-else cond. as its initial value is None
                last_hidden_state = bert_outputs.get("last_hidden_state")  # [BS, SEQ_LEN, EMD_DIM]
                if last_hidden_state is None:
                    available_keys = bert_outputs.keys()
                    msg = f"The choice of pretrained transformer model " \
                          f"({self.params.pretrained_model_name_or_path}) " \
                          f"has no key 'last_hidden_state' in its output dictionary. " \
                          f"The available keys are: {available_keys}."
                    raise ValueError(msg)
                pooler_output = last_hidden_state[:, 0, :]  # [BS, self.out_dim]
            encodings = self.dropout(pooler_output)
        else:
            # 'last_hidden_state' refers to the tensor output of the final transformer layer (i.e.
            # before the logit layer) and hence its dimension  [BS, SEQ_LEN, EMD_DIM]
            last_hidden_state = bert_outputs.get("last_hidden_state")  # [BS, SEQ_LEN, EMD_DIM]
            if last_hidden_state is None:
                # TODO: Do all huggingface models have this key? Are there any alternatives for this
                #  key? If so, we can enumerate the set of key names instead of just one key name?
                msg = f"The choice of pretrained bert model " \
                      f"({self.params.pretrained_model_name_or_path}) " \
                      f"has no key 'last_hidden_state' in its output dictionary."
                raise ValueError(msg)
            last_hidden_state = self.dropout(last_hidden_state)
            summed_split_lengths = [
                sum(_split_lengths) +
                (self.encoder.number_of_terminal_tokens if self.params.add_terminals else 0)
                for _split_lengths in batch_data["split_lengths"]
            ]
            summed_split_lengths = torch.as_tensor(summed_split_lengths, dtype=torch.long)
            encodings = self.emb_layer_pooling(
                last_hidden_state, summed_split_lengths)  # [BS, self.out_dim]

        batch_data.update({"seq_embs": encodings})

        return batch_data


def get_sequence_classifier_cls(classifier_type: str, embedder_type: str = None):
    try:
        classifier_type = SequenceClassificationType(classifier_type)
    except ValueError as e:
        msg = f"Neural Nets' sequence classification module expects classifier_type to be amongst" \
              f" {[v.value for v in SequenceClassificationType.__members__.values()]}" \
              f" but found '{classifier_type}'."
        raise ValueError(msg) from e

    try:
        embedder_type = EmbedderType(embedder_type)
    except ValueError as e:
        msg = f"Neural Nets' sequence classification module expects embedder_type to be amongst " \
              f" {[v.value for v in EmbedderType.__members__.values()]} " \
              f" but found '{embedder_type}'."
        raise ValueError(msg) from e

    if (
        embedder_type == EmbedderType.BERT and
        classifier_type not in [SequenceClassificationType.EMBEDDER]
    ):
        msg = f"To use the embedder_type '{EmbedderType.BERT.value}', " \
              f"classifier_type must be '{SequenceClassificationType.EMBEDDER.value}'."
        raise ValueError(msg)

    # disambiguation between glove, bert and non-pretrained embedders
    def _resolve_and_return_embedder_class(_embedder_type):
        return {
            EmbedderType.NONE: EmbedderForSequenceClassification,
            EmbedderType.GLOVE: EmbedderForSequenceClassification,
            EmbedderType.BERT: BertForSequenceClassification
        }[_embedder_type]

    return {
        SequenceClassificationType.EMBEDDER: _resolve_and_return_embedder_class(embedder_type),
        SequenceClassificationType.CNN: CnnForSequenceClassification,
        SequenceClassificationType.LSTM: LstmForSequenceClassification,
    }[classifier_type]
