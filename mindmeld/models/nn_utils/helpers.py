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
    """
    A dictionary-like object that exposes its keys as attributes and holds various inputs as well
    as outputs of neural models related to a batch of data, such as tensor encodings, lengths of
    inputs etc.

    Following is the description of the different keys that serve as inputs to neural models:
        - seq_lengths: Number of tokens in each example before adding padding tokens. The number
            includes terminal tokens too if they are added before padding. If using an encoder
            that splits words in sub-words, seq_lengths still implies number of words (instead
            of number of sub-words) along with any added terminal tokens; this number is
            useful in case of token classifiers which require token-level (aka.
            word-level) outputs as well as in sequence classifiers models such as LSTM.
        - split_lengths: The length of each subgroup (i.e. group of sub-words) in each
            example. Due to its definition, it obviously does not include any terminal
            tokens in its counts. This can be seen as fine-grained information to
            seq_lengths values for the encoders with sub-word tokenization. This is again
            useful in cases of token classifiers to flexibly choose between representations
            of first sub-word or mean/max pool of sub-words' representations in order to
            obtain the word-level representations. For lookup table based encoders where
            words are not broken into sub-words, `split_lengths` is simply a sequence of
            ones whose sum indicates the number of words w/o terminal & padding tokens.
        - seq_ids (in case non-pretrained models that require training an embedding layer):
            The encoded ids useful for embedding lookup, including terminal special tokens
            if asked for, and with padding.
        - attention_masks (only in case of huggingface trainable encoders): Boolean flags
            corresponding to each id in seq_ids, set to 0 if padding token else 1.
        - hgf_encodings (only in huggingface pretrained encoders): A dict of outputs from a
            Pretrained Language Model encoder from Huggingface (shortly dubbed as hgf).
        - char_seq_ids (only in dual tokenizers): Similar to seq_ids but from a char
            tokenizer in case of dual tokenization
        - char_seq_lengths (only in dual tokenizers): Similar to seq_lengths but from a char
            tokenizer in case of dual tokenization. Like seq_lengths, this also includes
            terminal special tokens from char vocab in the length count whenever added.

     Following is the description of the different keys that serve as inputs to neural models:

    Following is the description of the different keys that are outputted by neural models:
        - seq_embs: The embeddings produced before final classification (dense) layers by
            sequence-classification classes (generally of shape [batch_size, emd_dim]).
        - token_embs: The embeddings produced before final classification (dense) layers by
            token-classification classes (generally of shape [batch_size, seq_length, emd_dim]).
        - logits: Classification scores (before SoftMax).
        - loss: Classification loss object.
    """
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


class ClassificationType(enum.Enum):
    TEXT = "text"
    TAGGER = "tagger"


class SequenceClassificationType(enum.Enum):
    EMBEDDER = "embedder"
    CNN = "cnn"
    LSTM = "lstm"


class TokenClassificationType(enum.Enum):
    EMBEDDER = "embedder"
    LSTM = "lstm-pytorch"
    CNN_LSTM = "cnn-lstm"
    LSTM_LSTM = "lstm-lstm"


TRAIN_DEV_SPLIT_SEED = 6174
SHUFFLE_TRAINING_SEED = 8128
LABEL_PAD_TOKEN_IDX = -1  # value set based on default label padding idx in pytorch

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

DEFAULT_VANILLA_BERT_MODEL_PARAMS = {
    "batch_size": 16,
    "gradient_accumulation_steps": 2,
    "number_of_epochs": 20,
    "patience": 5,
    "embedder_output_pooling_type": "first",
}

DEFAULT_COMMON_TOKEN_CLASSIFICATION_PARAMS = {
    "output_keep_prob": 0.7,
    "use_crf_layer": True,
    "patience": 10,  # observed in benchmarking that more patience is better when using crf
    "token_spans_pooling_type":
        "first",  # if words split in subgroups, tells which subword representation to consider
    "validation_metric": "f1",
}

DEFAULT_EMB_DIM = 256
# TODO: A whitespace default can be inappropriate for languages like Japanese, Chinese, etc. Other
#  tokenizer types such as character or huggingface-pretrained tokenizers can be used instead.
DEFAULT_TOKENIZER = TokenizerType.WHITESPACE_TOKENIZER.value

DEFAULT_FORWARD_PASS_PARAMS = {
    "EmbedderForSequenceClassification": {
        "embedder_type": EmbedderType.NONE.value,
        "update_embeddings": True,
        "embedder_output_keep_prob": 0.7,
        "embedder_output_pooling_type": "mean",
        "output_keep_prob": 1.0,  # set to 1.0 due to the shallowness of the architecture
    },
    "CnnForSequenceClassification": {
        "embedder_type": EmbedderType.NONE.value,
        "update_embeddings": True,
        "embedder_output_keep_prob": 0.7,
        "output_keep_prob": 0.7,
        "window_sizes": [3, 4, 5],
        "number_of_windows": [100, 100, 100],
    },
    "LstmForSequenceClassification": {
        "embedder_type": EmbedderType.NONE.value,
        "update_embeddings": True,
        "embedder_output_keep_prob": 0.7,
        "output_keep_prob": 0.7,
        "lstm_hidden_dim": 128,
        "lstm_num_layers": 2,
        "lstm_keep_prob": 0.7,
        "lstm_bidirectional": True,
        "lstm_output_pooling_type": "last",
    },
    "BertForSequenceClassification": {
        **DEFAULT_VANILLA_BERT_MODEL_PARAMS,
        "pretrained_model_name_or_path": "bert-base-uncased",
        "update_embeddings": True,
        "embedder_output_keep_prob": 0.7,
        "embedder_output_pooling_type": "first",
        "output_keep_prob": 1.0,  # this dropout unnecessary upon using `embedder_output_keep_prob`
        # other values for following keys will possibly throw errors
        "embedder_type": "bert",
        "tokenizer_type": TokenizerType.HUGGINGFACE_PRETRAINED_TOKENIZER.value,
        # keys that are not mutually exclusive and are valid when some of the above keys are set
        "save_frozen_embedder": False,  # the key is valid only when update_embeddings=False
    },
    "EmbedderForTokenClassification": {
        **DEFAULT_COMMON_TOKEN_CLASSIFICATION_PARAMS,
        "embedder_type": EmbedderType.NONE.value,
        "update_embeddings": True,
        "embedder_output_keep_prob": 0.7,
        "output_keep_prob": 1.0,  # set to 1.0 due to the shallowness of the architecture,
    },
    "LstmForTokenClassification": {
        **DEFAULT_COMMON_TOKEN_CLASSIFICATION_PARAMS,
        "embedder_type": EmbedderType.NONE.value,
        "update_embeddings": True,
        "embedder_output_keep_prob": 0.7,
        "lstm_hidden_dim": 128,
        "lstm_num_layers": 2,
        "lstm_keep_prob": 0.7,
        "lstm_bidirectional": True,
    },
    "CharCnnWithWordLstmForTokenClassification": {
        **DEFAULT_COMMON_TOKEN_CLASSIFICATION_PARAMS,
        "embedder_type": EmbedderType.NONE.value,
        "tokenizer_type": TokenizerType.WHITESPACE_AND_CHAR_DUAL_TOKENIZER.value,
        "update_embeddings": True,
        "embedder_output_keep_prob": 0.7,
        "lstm_hidden_dim": 128,
        "lstm_num_layers": 2,
        "lstm_keep_prob": 0.7,
        "lstm_bidirectional": True,
        "char_emb_dim": 50,
        "char_window_sizes": [3, 4, 5],
        "char_number_of_windows": [100, 100, 100],
        "char_cnn_output_keep_prob": 0.7,
        "char_proj_dim": None,
        "add_terminals": False,
    },
    "CharLstmWithWordLstmForTokenClassification": {
        **DEFAULT_COMMON_TOKEN_CLASSIFICATION_PARAMS,
        "tokenizer_type": TokenizerType.WHITESPACE_AND_CHAR_DUAL_TOKENIZER.value,
        "update_embeddings": True,
        "embedder_output_keep_prob": 0.7,
        "lstm_hidden_dim": 128,
        "lstm_num_layers": 2,
        "lstm_keep_prob": 0.7,
        "lstm_bidirectional": True,
        "char_emb_dim": 50,
        "char_lstm_hidden_dim": 128,
        "char_lstm_num_layers": 2,
        "char_lstm_keep_prob": 0.7,
        "char_lstm_bidirectional": True,
        "char_lstm_output_pooling_type": "last",
        "char_proj_dim": None,
        "add_terminals": False,
    },
    "BertForTokenClassification": {
        **DEFAULT_VANILLA_BERT_MODEL_PARAMS,
        **DEFAULT_COMMON_TOKEN_CLASSIFICATION_PARAMS,
        "pretrained_model_name_or_path": "bert-base-uncased",
        "update_embeddings": True,
        "embedder_output_keep_prob": 0.7,
        "output_keep_prob": 1.0,  # this dropout unnecessary upon using `embedder_output_keep_prob`
        "use_crf_layer": False,  # Following BERT paper's best results,
        # other values for following keys will possibly throw errors
        "embedder_type": "bert",
        "tokenizer_type": TokenizerType.HUGGINGFACE_PRETRAINED_TOKENIZER.value,
        # keys that are not mutually exclusive and are valid when some of the above keys are set
        "save_frozen_embedder": False,  # the key is valid only when update_embeddings=False
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
    Returns the disk space of a pytorch module in MB units. This includes all weights
    (trainable and non-trainable) of the module.

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
