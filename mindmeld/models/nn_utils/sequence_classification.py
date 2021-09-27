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

from .classification import BaseClassification
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
    """Base class that defines all the necessary elements to succesfully train/infer
     custom pytorch modules wrapped on top of this base class. Classes derived from
     this base can be trained for sequence classification.
    """

    def _prepare_labels(self, labels, split_lengths):
        # for sequence classification, the length of an example doesn't matter as we have only one
        # label for each example. hence, no need to do any padding or validation checks.
        del split_lengths
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

        print(f"{self.name} is initialized")

    def forward(self, batch_data):

        batch_data = self._to_device(batch_data)
        batch_data = self._forward_core(batch_data)

        seq_embs = batch_data["seq_embs"]
        seq_embs = self.dense_layer_dropout(seq_embs)
        logits = self.classifier_head(seq_embs)
        batch_data.update({"logits": logits})

        targets = batch_data.get("labels")
        if targets is not None:
            if self.params.num_labels == 2:
                loss = self.criterion(logits.view(-1), targets.float())
            elif self.params.num_labels > 2:
                loss = self.criterion(logits, targets)
            batch_data.update({"loss": loss})

        return batch_data

    def predict(self, examples):
        logits = self._forward_with_batching_and_no_grad(examples)
        if self.params.num_labels == 2:
            preds = (logits >= 0.5).long().view(-1)
        elif self.params.num_labels > 2:
            preds = torch.argmax(logits, dim=-1)
        return preds.tolist()

    def predict_proba(self, examples):
        logits = self._forward_with_batching_and_no_grad(examples)
        if self.params.num_labels == 2:
            probs = F.sigmoid(logits)
            # extending the results from shape [N,1] to [N,2] to give out class probs distinctly
            probs = torch.cat((1 - probs, probs), dim=-1)
        elif self.params.num_labels > 2:
            probs = F.softmax(logits, dim=-1)
        return probs.tolist()

    def _forward_with_batching_and_no_grad(self, examples):
        logits = None
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for start_idx in range(0, len(examples), self.params.batch_size):
                this_examples = examples[start_idx:start_idx + self.params.batch_size]
                batch_data = self.encoder.batch_encode(this_examples)
                this_logits = self.forward(batch_data)["logits"]
                logits = torch.cat((logits, this_logits)) if logits is not None else this_logits
        if was_training:
            self.train()
        return logits

    @abstractmethod
    def _init_core(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _forward_core(self, batch_data):
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
            self.params.num_tokens,
            self.params.emb_dim,
            self.params.padding_idx,
            self.params.pop("embedding_weights", None),
            self.params.update_embeddings,
            1 - self.params.embedder_output_keep_prob
        )
        self.emb_layer_pooling = PoolingLayer(
            self.params.embedder_output_pooling_type
        )
        self.out_dim = self.params.emb_dim

    def _forward_core(self, batch_data):
        seq_ids = batch_data["seq_ids"]  # [BS, SEQ_LEN]
        seq_lengths = batch_data["seq_lengths"]  # [BS]

        encodings = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, EMD_DIM]
        encodings = self.emb_layer_pooling(encodings, seq_lengths)  # [BS, self.out_dim]

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
            self.params.num_tokens,
            self.params.emb_dim,
            self.params.padding_idx,
            self.params.pop("embedding_weights", None),
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
        self.lstm_layer_pooling = PoolingLayer(
            self.params.lstm_output_pooling_type
        )
        self.out_dim = (
            self.params.lstm_hidden_dim * 2 if self.params.lstm_bidirectional
            else self.params.lstm_hidden_dim
        )

    def _forward_core(self, batch_data):
        seq_ids = batch_data["seq_ids"]  # [BS, SEQ_LEN]
        seq_lengths = batch_data["seq_lengths"]  # [BS]

        encodings = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, EMD_DIM]
        encodings = self.lstm_layer(encodings, seq_lengths)  # [BS, SEQ_LEN, self.out_dim]
        encodings = self.lstm_layer_pooling(encodings, seq_lengths)  # [BS, self.out_dim]

        batch_data.update({"seq_embs": encodings})

        return batch_data


class BertForSequenceClassification(BaseSequenceClassification):

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

    def _init_core(self):

        self.bert_model = HuggingfaceTransformersContainer(
            self.params.pretrained_model_name_or_path,
            cache_lookup=False
        ).get_transformer_model()
        self.dropout = nn.Dropout(
            p=1 - self.params.embedder_output_keep_prob
        )
        if self.params.embedder_output_pooling_type != "first":
            self.emb_layer_pooling = PoolingLayer(
                self.params.embedder_output_pooling_type
            )
        self.out_dim = self.params.emb_dim

    def _forward_core(self, batch_data):

        bert_outputs = self.bert_model(**batch_data["hgf_encodings"], return_dict=True)

        if self.params.embedder_output_pooling_type != "first":
            last_hidden_state = bert_outputs.get("last_hidden_state")  # [BS, SEQ_LEN, EMD_DIM]
            if last_hidden_state is None:
                msg = f"The choice of pretrained bert model " \
                      f"({self.params.pretrained_model_name_or_path}) " \
                      f"has no key 'last_hidden_state' in its output dictionary"
                raise ValueError(msg)
            last_hidden_state = self.dropout(last_hidden_state)
            seq_lengths = batch_data["seq_lengths"]  # [BS]
            encodings = self.emb_layer_pooling(last_hidden_state, seq_lengths)  # [BS, self.out_dim]
        else:
            pooler_output = bert_outputs.get("pooler_output")  # [BS, self.out_dim]
            if pooler_output is None:
                msg = f"The choice of pretrained bert model " \
                      f"({self.params.pretrained_model_name_or_path}) " \
                      "has no key 'pooler_output' in its output dictionary. " \
                      "Considering to obtain " \
                      "the first embedding of last_hidden_state."
                logger.error(msg)
                last_hidden_state = bert_outputs.get("last_hidden_state")  # [BS, SEQ_LEN, EMD_DIM]
                if last_hidden_state is None:
                    msg = f"The choice of pretrained bert model " \
                          f"({self.params.pretrained_model_name_or_path}) " \
                          f"has no key 'last_hidden_state' in its output dictionary"
                    raise ValueError(msg)
                pooler_output = last_hidden_state[:, 0, :]  # [BS, self.out_dim]
            encodings = self.dropout(pooler_output)

        batch_data.update({"seq_embs": encodings})

        return batch_data
