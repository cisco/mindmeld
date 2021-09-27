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
Default params used by various sequence and token classification classes
"""

import logging

from .._util import _get_module_or_attr

try:
    is_cuda_available = _get_module_or_attr("torch.cuda", "is_available")()
except ImportError:
    is_cuda_available = False
    pass

logger = logging.getLogger(__name__)

DEFAULT_TRAINING_INFERENCE_PARAMS = {
    "device": "cuda" if is_cuda_available else "cpu",
    "number_of_epochs": 100,
    "patience": 7,
    "batch_size": 32,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": None,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "validation_metric": "accuracy",  # or 'f1'
    "dev_split_ratio": 0.2
}

DEFAULT_TOKEN_CLASSIFICATION_PARAMS = {
    "output_keep_prob": 0.7,
    "use_crf_layer": True,
    "patience": 10,  # observed in benchmarking that more patience is better when using crf
    "token_spans_pooling_type":
        "first",  # if words split in subgroups, tells which subword representation to consider
}

DEFAULT_FORWARD_PASS_PARAMS = {
    "EmbedderForSequenceClassification": {
        "padding_idx": None,
        "update_embeddings": True,
        "embedder_output_keep_prob": 0.7,
        "embedder_output_pooling_type": "mean",
        "output_keep_prob": 1.0,  # set to 1.0 due to the shallowness of the architecture
    },
    "CnnForSequenceClassification": {
        "padding_idx": None,
        "update_embeddings": True,
        "embedder_output_keep_prob": 0.7,
        "window_sizes": [3, 4, 5],
        "number_of_windows": [100, 100, 100],
        "output_keep_prob": 0.7
    },
    "LstmForSequenceClassification": {
        "padding_idx": None,
        "update_embeddings": True,
        "embedder_output_keep_prob": 0.7,
        "lstm_hidden_dim": 128,
        "lstm_num_layers": 2,
        "lstm_keep_prob": 0.7,
        "lstm_bidirectional": True,
        "lstm_output_pooling_type": "last",
        "output_keep_prob": 0.7
    },
    "BertForSequenceClassification": {
        "tokenizer_type": "huggingface_pretrained-tokenizer",
        "embedder_output_keep_prob": 0.7,
        "embedder_output_pooling_type": "first",
        "output_keep_prob": 1.0,  # unnecessary upon using `embedder_output_keep_prob`
    },
    "EmbedderForTokenClassification": {
        **DEFAULT_TOKEN_CLASSIFICATION_PARAMS,
        "padding_idx": None,
        "update_embeddings": True,
        "embedder_output_keep_prob": 0.7,
        "output_keep_prob": 1.0,  # set to 1.0 due to the shallowness of the architecture,
    },
    "LstmForTokenClassification": {
        **DEFAULT_TOKEN_CLASSIFICATION_PARAMS,
        "padding_idx": None,
        "update_embeddings": True,
        "embedder_output_keep_prob": 0.7,
        "lstm_hidden_dim": 128,
        "lstm_num_layers": 2,
        "lstm_keep_prob": 0.7,
        "lstm_bidirectional": True,
    },
    "CharLstmWithWordLstmForTokenClassification": {
        **DEFAULT_TOKEN_CLASSIFICATION_PARAMS,
        "tokenizer_type": "whitespace_and_char-tokenizer",
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
        "word_level_character_embedding_size": None,
    },
    "CharCnnWithWordLstmForTokenClassification": {
        **DEFAULT_TOKEN_CLASSIFICATION_PARAMS,
        "tokenizer_type": "whitespace_and_char-tokenizer",
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
        "word_level_character_embedding_size": None,

    },
    "BertForTokenClassification": {
        **DEFAULT_TOKEN_CLASSIFICATION_PARAMS,
        "tokenizer_type": "huggingface_pretrained-tokenizer",
        "embedder_output_keep_prob": 0.7,
        "output_keep_prob": 1.0,  # unnecessary upon using `embedder_output_keep_prob`
        "use_crf_layer": False,  # Following BERT paper's best results,
    }
}


def get_default_params(module_name: str):
    try:
        return {
            **DEFAULT_TRAINING_INFERENCE_PARAMS,
            **DEFAULT_FORWARD_PASS_PARAMS[module_name]
        }
    except KeyError as e:
        msg = f"Cannot find module name {module_name} when looking for default params."
        logger.error(msg)
        raise KeyError(msg) from e
