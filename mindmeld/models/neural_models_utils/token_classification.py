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
    TokenClassificationEncoderWithStaticEmbeddings
)
from ._layers import (
    EmbeddingLayer
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

    def _fit_encoder_and_update_params(self, examples, **params):
        self.encoder.fit(examples=examples, **params)
        params.update({
            "num_tokens": self.encoder.get_num_tokens(),
            "emb_dim": self.encoder.get_emb_dim(),
            "padding_idx": self.encoder.get_pad_token_idx(),
            "embedding_weights": self.encoder.get_embedding_weights(),
            "label_padding_idx": self.encoder.get_pad_label_idx()
        })
        return params

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
                    mask = pad_sequence(
                        [torch.as_tensor([1] * length_) for length_ in
                         batch_data_dict["seq_lengths"]], batch_first=True
                    ).long()
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

    # methods for forward pass

    def init(self, **params):

        self._init(**params)

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
        if self.use_crf_layer and params.get("add_terminals"):
            msg = "Using CRF layer while specifying 'add_terminals' is not supported"
            raise NotImplementedError(msg)

        # init the peripheral architectural components and the criterion to compute loss
        self.dropout = nn.Dropout(p=self.hidden_dropout_prob)
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
        token_embs = self.dropout(token_embs)
        logits = self.classifier_head(token_embs)
        batch_data_dict.update({"logits": logits})

        targets = batch_data_dict.get("labels")
        if targets is not None:
            if self.use_crf_layer:
                # create a mask to ignore token positions that are padded
                mask = torch.as_tensor(targets != self.label_padding_idx).long()
                loss = self.crf_layer(logits, targets, mask=mask)
            else:
                loss = self.criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))
            batch_data_dict.update({"loss": loss})

        return batch_data_dict

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

    def __init__(self, **params):
        raise NotImplementedError


class TokenLstmSequenceLstmForTokenClassification(TokenClassificationCore):

    def __init__(self, **params):
        raise NotImplementedError


class TokenCnnSequenceLstmForTokenClassification(TokenClassificationCore):

    def __init__(self, **params):
        raise NotImplementedError


# Custom modules built on top of above nn layers that can do joint classification

class JointSequenceTokenClassificationCore(nn.Module):
    """This base class is wrapped around nn.Module and supports joint modeling, meaning
    multiple heads can be trained for models derived on top of this base class. Unlike classes
    derive on top of SequenceClassification or TokenClassification, the ones derived
    on this base output both `seq_embs` as well as `token_embs` in their output which facilitates
    multi-head training.
    """

    def __init__(self, **params):
        raise NotImplementedError
