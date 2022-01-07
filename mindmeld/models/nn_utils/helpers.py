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

import enum
import logging
import os

from .._util import _get_module_or_attr
from ...core import Bunch

try:
    is_cuda_available = _get_module_or_attr("torch.cuda", "is_available")()
except ImportError:
    is_cuda_available = False
    pass

logger = logging.getLogger(__name__)


class BatchData(Bunch):
    pass


class ValidationMetricType(enum.Enum):
    ACCURACY = "accuracy"
    F1 = "f1"


class TokenizerType(enum.Enum):
    WHITESPACE_TOKENIZER = "whitespace-tokenizer"
    CHAR_TOKENIZER = "char-tokenizer"
    WHITESPACE_AND_CHAR_DUAL_TOKENIZER = "whitespace_and_char-tokenizer"
    BPE_TOKENIZER = "bpe-tokenizer"
    WORDPIECE_TOKENIZER = "wordpiece-tokenizer"
    HUGGINGFACE_PRETRAINED_TOKENIZER = "huggingface_pretrained-tokenizer"


class EmbedderType(enum.Enum):
    NONE = None
    GLOVE = "glove"
    BERT = "bert"


class SequenceClassificationType(enum.Enum):
    EMBEDDER = "embedder"
    CNN = "cnn"
    LSTM = "lstm"


class TokenClassificationType(enum.Enum):
    EMBEDDER = "embedder"
    LSTM = "lstm-pytorch"
    CNN_LSTM = "cnn-lstm"
    LSTM_LSTM = "lstm-lstm"


DEFAULT_TRAINING_INFERENCE_PARAMS = {
    "device": "cuda" if is_cuda_available else "cpu",
    "number_of_epochs": 100,
    "patience": 7,
    "batch_size": 32,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": None,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "validation_metric": "accuracy",
    "dev_split_ratio": 0.2
}

DEFAULT_TOKEN_CLASSIFICATION_PARAMS = {
    "output_keep_prob": 0.7,
    "use_crf_layer": True,
    "patience": 10,  # observed in benchmarking that more patience is better when using crf
    "token_spans_pooling_type":
        "first",  # if words split in subgroups, tells which subword representation to consider
    "validation_metric": "f1",
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
        "tokenizer_type": TokenizerType.HUGGINGFACE_PRETRAINED_TOKENIZER.value,
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
        "tokenizer_type": TokenizerType.WHITESPACE_AND_CHAR_DUAL_TOKENIZER.value,
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
        "tokenizer_type": TokenizerType.WHITESPACE_AND_CHAR_DUAL_TOKENIZER.value,
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
        "tokenizer_type": TokenizerType.HUGGINGFACE_PRETRAINED_TOKENIZER.value,
        "embedder_output_keep_prob": 0.7,
        "output_keep_prob": 1.0,  # unnecessary upon using `embedder_output_keep_prob`
        "use_crf_layer": False,  # Following BERT paper's best results,
    }
}


def get_default_params(class_name: str):
    """
    Returns all the default params based on the inputted class name

    Args:
        class_name (str): A (child) class name from sequence_classification.py or
            token_classification.py
    """
    try:
        return {
            **DEFAULT_TRAINING_INFERENCE_PARAMS,
            **DEFAULT_FORWARD_PASS_PARAMS[class_name]
        }
    except KeyError as e:
        msg = f"Cannot find module name {class_name} when looking for default params."
        logger.error(msg)
        raise KeyError(msg) from e


def get_disk_space_of_model(pytorch_module):
    """
    Returns the disk space of a pytorch module in MB units

    Args:
        pytorch_module: a pytorch neural network module derived from torch.nn.Module

    Returns:
        size (float): The size of model when dumped
    """
    filename = "temp.bin"
    _get_module_or_attr("torch").save(pytorch_module.state_dict(), filename)
    size = os.path.getsize(filename) / 1e6
    os.remove(filename)
    msg = f"Pytorch module will be dumped temporarily at {filename} in order to " \
          f"calculate its disk size."
    logger.debug(msg)
    return size


def get_num_weights_of_model(pytorch_module):
    """
    Returns the number of trainable and the total parameters in a pytorch module. Returning both
    helps to do a sanity check if any layers which are meant to be frozen are being trained or not.

    Args:
        pytorch_module: a pytorch neural network module derived from torch.nn.Module

    Returns:
        number_of_params (tuple): A tuple of number of params that are trainable and total number
            of params of the pytorch module
    """
    n_total = 0
    n_requires_grad = 0
    for param in list(pytorch_module.parameters()):
        t = param.numel()
        n_total += t
        if param.requires_grad:
            n_requires_grad += t
    return n_requires_grad, n_total
