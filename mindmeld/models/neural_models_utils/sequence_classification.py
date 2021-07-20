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
    SequenceClassificationEncoderWithStaticEmbeddings,
    SequenceClassificationEncoderWithBert
)
from ._layers import (
    EmbeddingLayer,
    CnnLayer,
    LstmLayer,
    PoolingLayer
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    pass

logger = logging.getLogger(__name__)


class SequenceClassificationCore(ClassificationCore):
    """Base Module class that defines all the necessary elements to succesfully train/infer,
     dump/load custom pytorch modules wrapped on top of this base class. Classes derived from
     this base can be trained for sequence classification. The output of a class derived from
     this base must contain `seq_embs` in its output dictionary.
    """

    def __init__(self):
        super().__init__()

        # default encoder; have to either fit ot load to use it
        self.encoder = SequenceClassificationEncoderWithStaticEmbeddings()

    # methods for training

    def fit_encoder_and_update_params(self, examples, **params):
        use_character_embeddings = params.pop("use_character_embeddings", False)
        if use_character_embeddings:
            tokenizer_type = params.get("tokenizer_type", "char-tokenizer")
            # Ensure that the params do not contain both
            # `use_character_embeddings` as well as `tokenizer_type` in a contradicting way
            if tokenizer_type != "char-tokenizer":
                msg = "To use character embeddings, 'tokenizer_type' must be 'char-tokenizer'. " \
                      "Other values passed thorugh params are not allowed."
                raise ValueError(msg)
            params.update({"tokenizer_type": tokenizer_type})
        self.encoder.fit(examples=examples, **params)
        params.update({
            "num_tokens": self.encoder.get_num_tokens(),
            "emb_dim": self.encoder.get_emb_dim(),
            "padding_idx": self.encoder.get_pad_token_idx(),
            "embedding_weights": self.encoder.get_embedding_weights()
        })
        return params

    def init(self, **params):

        self._init(**params)

        # params
        self.num_labels = params.get("num_labels")
        self.dense_keep_prob = params.get("dense_keep_prob", 0.3)
        self.params_keys.update(["num_labels", "dense_keep_prob"])

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

        # init the peripheral architecture params
        if not self.num_labels:
            msg = f"Invalid number of labels ({self.num_labels}) inputted for '{self.name}' class"
            raise ValueError(msg)

        # init the peripheral architectural components and the criterion to compute loss
        self.dense_layer_dropout = nn.Dropout(p=1 - self.dense_keep_prob)
        if self.num_labels == 2:
            # sigmoid criterion
            self.classifier_head = nn.Linear(self.hidden_size, 1)
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        elif self.num_labels > 2:
            # cross-entropy criterion
            self.classifier_head = nn.Linear(self.hidden_size, self.num_labels)
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
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

        seq_embs = batch_data_dict["seq_embs"]
        seq_embs = self.dense_layer_dropout(seq_embs)
        logits = self.classifier_head(seq_embs)
        batch_data_dict.update({"logits": logits})

        targets = batch_data_dict.get("labels")
        if targets is not None:
            if self.num_labels == 2:
                loss = self.criterion(logits.view(-1), targets.float())
            elif self.num_labels > 2:
                loss = self.criterion(logits, targets)
            batch_data_dict.update({"loss": loss})

        return batch_data_dict

    # methods for inference

    def predict(self, examples):
        logits = self._forward_with_batching_and_no_grad(examples)
        if self.num_labels == 2:
            preds = (logits >= 0.5).long().view(-1)
        elif self.num_labels > 2:
            preds = torch.argmax(logits, dim=-1)
        return preds.tolist()

    def predict_proba(self, examples):
        logits = self._forward_with_batching_and_no_grad(examples)
        if self.num_labels == 2:
            probs = F.sigmoid(logits)
            # extending the results from shape [N,1] to [N,2] to give out class probs distinctly
            probs = torch.cat((1 - probs, probs), dim=-1)
        elif self.num_labels > 2:
            probs = F.softmax(logits, dim=-1)
        return probs.tolist()

    def _forward_with_batching_and_no_grad(self, examples):
        logits = None
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for start_idx in range(0, len(examples), self.batch_size):
                this_examples = examples[start_idx:start_idx + self.batch_size]
                batch_data_dict = self.encoder.batch_encode(this_examples)
                this_logits = self.forward(batch_data_dict)["logits"]
                logits = torch.cat((logits, this_logits)) if logits is not None else this_logits
        if was_training:
            self.train()
        return logits

    # abstract methods definition, to be implemented by sub-classes

    @abstractmethod
    def _init(self, **params) -> None:
        raise NotImplementedError

    @abstractmethod
    def _forward(self, batch_data_dict: Dict) -> Dict:
        raise NotImplementedError


class EmbedderForSequenceClassification(SequenceClassificationCore):
    """An embedder pooling module that operates on a batched sequence of token ids. The
    tokens could be characters or words or sub-words. This module finally outputs one 1D
    representation for each instance in the batch (i.e. [BS, EMB_DIM]).

    The `forward` method of this module expects padded token ids along with numer of tokens
    per instance in the batch.

    Additionally, one can set different coefficients for different tokens of the embedding
    matrix (e.g. tf-idf weights).
    """

    def _init(self, **params):
        # params
        self.num_tokens = params["num_tokens"]
        self.emb_dim = params["emb_dim"]
        self.padding_idx = params.get("padding_idx", None)
        self.update_embeddings = params.get("update_embeddings", True)
        self.embedder_output_pooling_type = params.get("embedder_output_pooling_type", "mean")
        self.params_keys.update([
            "num_tokens", "emb_dim", "padding_idx", "update_embeddings",
            "embedder_output_pooling_type"
        ])

        # core layers
        self.emb_layer = EmbeddingLayer(self.num_tokens, self.emb_dim, self.padding_idx,
                                        params.get("embedding_weights", None),
                                        self.update_embeddings)
        self.emb_layer_pooling = PoolingLayer(self.embedder_output_pooling_type)
        self.out_dim = self.emb_dim

    def _forward(self, batch_data_dict):
        seq_ids = batch_data_dict["seq_ids"]  # [BS, SEQ_LEN]
        seq_lengths = batch_data_dict["seq_lengths"]  # [BS]

        encodings = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, EMD_DIM]
        encodings = self.emb_layer_pooling(encodings, seq_lengths)  # [BS, self.out_dim]

        batch_data_dict.update({"seq_embs": encodings})

        return batch_data_dict


class CnnForSequenceClassification(SequenceClassificationCore):
    """A CNN module that operates on a batched sequence of token ids. The tokens could be
    characters or words or sub-words. This module finally outputs one 1D representation
    for each instance in the batch (i.e. [BS, EMB_DIM]).

    The `forward` method of this module expects only padded token ids as input.
    """

    def _init(self, **params):
        # params
        self.num_tokens = params["num_tokens"]
        self.emb_dim = params["emb_dim"]
        self.padding_idx = params.get("padding_idx", None)
        self.update_embeddings = params.get("update_embeddings", True)
        self.cnn_kernel_sizes = params.get("cnn_kernel_sizes", [1, 3, 5])
        self.cnn_num_kernels = params.get("cnn_num_kernels", [100] * len(self.cnn_kernel_sizes))

        self.params_keys.update([
            "num_tokens", "emb_dim", "padding_idx", "update_embeddings",
            "cnn_kernel_sizes", "cnn_num_kernels"
        ])

        # core layers
        self.emb_layer = EmbeddingLayer(self.num_tokens, self.emb_dim, self.padding_idx,
                                        params.get("embedding_weights", None),
                                        self.update_embeddings)
        self.conv_layer = CnnLayer(self.emb_dim, self.cnn_kernel_sizes, self.cnn_num_kernels)
        self.out_dim = sum(self.cnn_num_kernels)

    def _forward(self, batch_data_dict):
        seq_ids = batch_data_dict["seq_ids"]  # [BS, SEQ_LEN]

        encodings = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, EMD_DIM]
        encodings = self.conv_layer(encodings)  # [BS, self.out_dim]

        batch_data_dict.update({"seq_embs": encodings})

        return batch_data_dict


class LstmForSequenceClassification(SequenceClassificationCore):
    # pylint: disable=too-many-instance-attributes
    """A LSTM module that operates on a batched sequence of token ids. The tokens could be
    characters or words or sub-words. This module finally outputs one 1D representation
    for each instance in the batch (i.e. [BS, EMB_DIM]).

    The `forward` method of this module expects padded token ids along with numer of tokens
    per instance in the batch.
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
        self.lstm_output_pooling_type = params.get("lstm_output_pooling_type", "end")
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
        self.lstm_layer_pooling = PoolingLayer(self.lstm_output_pooling_type)
        self.out_dim = self.lstm_hidden_dim * 2 if self.lstm_bidirectional else self.lstm_hidden_dim

    def _forward(self, batch_data_dict):
        seq_ids = batch_data_dict["seq_ids"]  # [BS, SEQ_LEN]
        seq_lengths = batch_data_dict["seq_lengths"]  # [BS]

        encodings = self.emb_layer(seq_ids)  # [BS, SEQ_LEN, EMD_DIM]
        encodings = self.lstm_layer(encodings, seq_lengths)  # [BS, SEQ_LEN, self.out_dim]
        encodings = self.lstm_layer_pooling(encodings, seq_lengths)  # [BS, self.out_dim]

        batch_data_dict.update({"seq_embs": encodings})

        return batch_data_dict


class BertForSequenceClassification(SequenceClassificationCore):

    def __init__(self):
        super().__init__()

        # overwrite default encoder
        self.encoder = SequenceClassificationEncoderWithBert()

    def fit(self, examples, labels, **params):  # overriding base class' method to set params
        # this class is based only on bert embedder and
        # hence no embedder info from params is expected in the inputted params

        number_of_epochs = params.get("number_of_epochs", 5)
        patience = params.get("patience", 2)
        embedder_type = params.get("embedder_type", "bert")
        if embedder_type != "bert":
            msg = f"{self.__class__.__name__} can only be used with 'embedder_type': 'bert'. " \
                  f"Other values passed through config params are not allowed"
            raise ValueError(msg)
        optimizer = params.get("optimizer", "AdamW")
        if optimizer != "AdamW":
            msg = f"{self.__class__.__name__} can only be used with 'optimizer': 'AdamW'. " \
                  f"Other values passed through config params are not allowed"
            raise ValueError(msg)
        lr = params.get("learning_rate", 2e-5)
        if lr > 2e-5:
            msg = f"{self.__class__.__name__} can only be used with 'lr' less than or equal to " \
                  f"'2e-5'. Other values passed through config params are not allowed"
            raise ValueError(msg)

        # update params
        params.update({
            "number_of_epochs": number_of_epochs,
            "patience": patience,
            "embedder_type": embedder_type,
            "optimizer": optimizer,
            "learning_rate": lr,
            "batch_size": 16,  # TODO: add ValueError message
            "gradient_accumulation_steps": 2,  # TODO: add ValueError message
            "max_grad_norm": 1.0,  # TODO: add ValueError message
        })

        super().fit(examples, labels, **params)

    def _create_optimizer(self):

        # references:
        #   https://arxiv.org/pdf/2006.05987.pdf#page=3
        #   https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html,
        #   https://huggingface.co/transformers/custom_datasets.html,
        #   https://huggingface.co/transformers/migration.html

        params = list(self.named_parameters())
        no_decay = ["bias", 'LayerNorm.bias', "LayerNorm.weight",
                    'layer_norm.bias', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = getattr(torch.optim, self.optimizer)(
            optimizer_grouped_parameters, lr=self.learning_rate, eps=1e-08, weight_decay=0.01)
        return optimizer

    def _create_optimizer_and_scheduler(self, num_training_steps):
        # load a torch optimizer
        optimizer = self._create_optimizer()

        # load a lr scheduler
        num_warmup_steps = 0.1 * num_training_steps

        # https://github.com/huggingface/transformers/blob/master/src/transformers/optimization.py
        # refer `get_linear_schedule_with_warmup` method
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(
                    max(1, num_training_steps - num_warmup_steps))
            )

        scheduler = getattr(torch.optim.lr_scheduler, "LambdaLR")(optimizer, lr_lambda)
        return optimizer, scheduler

    def _init(self, **params):
        # params
        self.emb_dim = params["emb_dim"]
        self.embedder_output_pooling_type = params.get("embedder_output_pooling_type", "start")
        self.params_keys.update(["emb_dim", "embedder_output_pooling_type"])

        # core layers
        if self.embedder_output_pooling_type != "start":
            self.emb_layer_pooling = PoolingLayer(self.embedder_output_pooling_type)
        self.out_dim = self.emb_dim

    def _forward(self, batch_data_dict):

        if self.embedder_output_pooling_type != "start":
            seq_lengths = batch_data_dict["seq_lengths"]  # [BS]
            last_hidden_state = batch_data_dict["last_hidden_state"]  # [BS, SEQ_LEN, EMD_DIM]
            encodings = self.emb_layer_pooling(last_hidden_state, seq_lengths)  # [BS, self.out_dim]
        else:
            encodings = batch_data_dict["pooler_output"]  # [BS, self.out_dim]

        batch_data_dict.update({"seq_embs": encodings})

        return batch_data_dict
