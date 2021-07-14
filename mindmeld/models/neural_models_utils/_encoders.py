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
This module consists of encoders that serve as input to pytorch modules
"""
import json
import logging
import os
from abc import abstractmethod
from typing import Dict, List, Union, Any, Iterable

try:
    import torch
    from torch import nn
except ImportError:
    pass
from ..containers import GloVeEmbeddingsContainer, HuggingfaceTransformersContainer

logger = logging.getLogger(__name__)

LABEL_PAD_TOKEN_IDX = -1  # value set based on default label padding idx in pytorch


# base encoders

class AbstractEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb_dim = None
        self.params_keys = set(["emb_dim"])

    def forward(self, *args, **kwargs):
        self.batch_encode(*args, **kwargs)

    @abstractmethod
    def fit(self, **kwargs):
        msg = f"Subclass {self.__class__.__name__} need to implement this method"
        raise NotImplementedError(msg)

    @abstractmethod
    def dump(self, **kwargs):
        msg = f"Subclass {self.__class__.__name__} need to implement this method"
        raise NotImplementedError(msg)

    @classmethod
    @abstractmethod
    def load(cls, **kwargs):
        msg = f"Subclass {cls.__name__} need to implement this method"
        raise NotImplementedError(msg)

    @abstractmethod
    def batch_encode(self, examples, labels=None, **kwargs) -> Dict[str, Any]:
        msg = f"Subclass {self.__class__.__name__} need to implement this method"
        raise NotImplementedError(msg)

    # common `get` methods

    def get_num_tokens(self):
        if not hasattr(self, "token2idx"):
            return None
        return len(self.token2idx)

    def get_emb_dim(self):
        return self.emb_dim

    def get_pad_token_idx(self):
        if not hasattr(self, "pad_token_idx"):
            return None
        return self.pad_token_idx

    def get_embedding_weights(self):
        if not hasattr(self, "token2emb") or not self.token2emb:
            msg = f"Encoder instance ({self.__class__.__name__}) does not contain " \
                  f"any token-to-embeddings mapping"
            logger.info(msg)
            return None
        embedding_weights = {}
        for token, idx in self.token2idx.items():
            if token in self.token2emb:
                embedding_weights[idx] = self.token2emb[token]
        return embedding_weights

    @staticmethod
    def get_pad_label_idx():
        return LABEL_PAD_TOKEN_IDX


class EncoderWithStaticEmbeddings(AbstractEncoder):
    """
    A base class for joint tokenization-encoding with the vocab derived from the training data and
    using word-level embedders if required.

    TOKENIZATION SUPPORT:
    This class supports three kinds of tokenizations: (1) tokenization at white space,
    (2) character tokenizations

    ENCODING SUPPORT
    This class supports encoding text into 0-based ids for an embedding lookup. When using the
    padding_length, at times, the labels might have to be trimmed as well, which is to be supported
    in the .batch_encode() method of the derived class.
    """

    ALLOWED_TOKENIZERS = ["whitespace-tokenizer", "char-tokenizer"]
    ALLOWED_EMBEDDER_TYPES = ["glove"]

    BASIC_SPECIAL_VOCAB_DICT = {
        "pad_token": "<PAD>",
        "unk_token": "<UNK>",
        "start_token": "<START>",
        "end_token": "<END>",
        "mask_token": "<MASK>",
    }

    def fit(
        self,
        # input data stream
        examples: List[str] = None,
        # params
        embedder_type=None,
        special_vocab_dict=None,
        padding_length=None,
        add_terminals=False,
        emb_dim=None,
        # non-params
        _disable_loading_embedder=False,
        # other params
        **params
    ):
        self.embedder_type = embedder_type
        self.special_vocab_dict = special_vocab_dict
        self.padding_length = padding_length
        self.add_terminals = add_terminals
        self.params_keys.update([
            "embedder_type", "special_vocab_dict", "padding_length", "add_terminals"
        ])

        self.emb_dim = emb_dim
        self.token2idx = {}
        self.token2emb = {}

        # load tokenizer and embedder
        if not self.embedder_type:
            self.tokenizer_type = params.get("tokenizer_type")
            self._tokenizer = self._get_tokenizer(self.tokenizer_type)
            self.params_keys.update(["tokenizer_type"])
        elif self.embedder_type == "glove":
            self.tokenizer_type = params.get("tokenizer_type")
            self._tokenizer = self._get_tokenizer(self.tokenizer_type)
            self.params_keys.update(["tokenizer_type"])
            if not _disable_loading_embedder:
                glove_container = GloVeEmbeddingsContainer()
                self.token2emb = glove_container.get_pretrained_word_to_embeddings_dict()
                self.emb_dim = glove_container.token_dimension
        else:
            msg = f"Unsupported name '{embedder_type}' for 'embedder_type' " \
                  f"found. Supported names are only " \
                  f"{self.__class__.ALLOWED_EMBEDDER_TYPES}."
            raise ValueError(msg)

        # Add special vocab before actual vocab
        special_vocab_dict_ = self.__class__.BASIC_SPECIAL_VOCAB_DICT
        if self.special_vocab_dict:
            special_vocab_dict_.update(self.special_vocab_dict)
        self.special_vocab_dict = dict(special_vocab_dict_)
        self._add_special_vocab(self.special_vocab_dict)

        # add vocab from examples upon tokenizing each text item
        if examples:
            all_tokens = set(sum([self._tokenizer(text) for text in examples], []))
            self._add_vocab(all_tokens)
            msg = f"Vocab size: Number of unique tokens collected from training data: " \
                  f"{len(self.token2idx)}"
            logger.info(msg)

        # some validations
        if not self.emb_dim:
            msg = "Need a valid 'emb_dim' to initialize encoder resource. To specify a " \
                  "particular dimension, either pass-in the 'emb_dim' param or provide a " \
                  "valid 'embedder_type' param."
            raise ValueError(msg)

    def dump(self, dump_folder):
        if os.path.isfile(dump_folder):
            msg = f"Path input to 'dump' method must be a folder, " \
                  f"not a file ({dump_folder})"
            raise ValueError(msg)
        os.makedirs(dump_folder, exist_ok=True)

        with open(os.path.join(dump_folder, "text_encoder_vocab.txt"), "w") as fp:
            for token in self.token2idx:
                fp.write(token + "\n")
            fp.close()
        with open(os.path.join(dump_folder, "text_encoder_config.json"), "w") as fp:
            params = {k: getattr(self, k) for k in self.params_keys}
            json.dump(params, fp, indent=4)
            fp.close()

    @classmethod
    def load(cls, load_folder):
        if os.path.isfile(load_folder):
            msg = f"Path input to 'load' method must be a folder, " \
                  f"not a file ({load_folder})"
            raise ValueError(msg)

        encoder = cls()
        with open(os.path.join(load_folder, "text_encoder_config.json"), "r") as fp:
            params = json.load(fp)
            setattr(encoder, "params_keys", set(params.keys()))
            fp.close()
        encoder.fit(**params, _disable_loading_embedder=True)
        with open(os.path.join(load_folder, "text_encoder_vocab.txt"), "r") as fp:
            for line in fp:
                encoder._add_vocab([line.strip()])
            fp.close()

        return encoder

    @abstractmethod
    def batch_encode(self, examples, labels=None, **kwargs) -> Dict[str, Any]:
        msg = f"Subclass {self.__class__.__name__} need to implement this method"
        raise NotImplementedError(msg)

    def _get_tokenizer(self, tokenizer_type):

        def whitespace_tokenizer(text: str) -> List[str]:
            return text.strip().split()

        def char_tokenizer(text: str) -> List[str]:
            return list(text.strip())

        if not tokenizer_type:
            return whitespace_tokenizer
        elif tokenizer_type == "whitespace-tokenizer":
            return whitespace_tokenizer
        elif tokenizer_type == "char-tokenizer":
            return char_tokenizer
        else:
            msg = f"Unknown tokenizer type specified ('{tokenizer_type}'). Expected to be " \
                  f"among {self.__class__.ALLOWED_TOKENIZERS}. "
            raise ValueError(msg)

    def _add_special_vocab(self, special_tokens: Dict[str, str]):

        # Validate special tokens
        DUPLICATE_RECORD_WARNING = "Ignoring duplicate token {} from special vocab set with name {}"
        for name, token in special_tokens.items():
            if name in self.special_vocab_dict and self.special_vocab_dict[name] != token:
                logger.warning(DUPLICATE_RECORD_WARNING, token, name)
                continue
            self.special_vocab_dict.update({name: token})

        # Add special tokens to vocab as well as their names to self
        for name, token in self.special_vocab_dict.items():
            self._add_vocab([token])
            setattr(self, f"{name}", token)
            setattr(self, f"{name}_idx", self.token2idx[token])

    def _add_vocab(self, vocab: Iterable[str]):
        for token in vocab:
            if token not in self.token2idx:
                self.token2idx.update({token: len(self.token2idx)})


class EncoderWithPlmEmbeddings(AbstractEncoder):
    """
    A base class for joint tokenization-encoding with vocab of a Pretrained Language Model
    (aka. PLM embedders), i.e sentence embedders like BERT, Elmo, etc..
    """

    ALLOWED_EMBEDDER_TYPES = ["bert"]

    def fit(
        self,
        # input data stream
        examples: List[str] = None,
        # params
        embedder_type=None,
        padding_length=None,
        emb_dim=None,
        freeze_embedder=False,
        # non-params
        _disable_loading_embedder=False,
        # other params
        **params
    ):
        del examples

        self.embedder_type = embedder_type
        self.padding_length = padding_length
        self.freeze_embedder = freeze_embedder
        self.params_keys.update(["embedder_type", "padding_length", "freeze_embedder"])

        self.emb_dim = emb_dim

        # load tokenizer and embedder
        if self.embedder_type == "bert":
            self.pretrained_model_name_or_path = params.get("pretrained_model_name_or_path")
            if not self.pretrained_model_name_or_path:
                msg = "Must include a valid 'pretrained_model_name_or_path' param when using " \
                      "embedder_type: 'bert'"
                raise ValueError(msg)
            self.params_keys.update(["pretrained_model_name_or_path"])
            if not _disable_loading_embedder:
                self._load_bert_embedder_from_checkpoint(self.pretrained_model_name_or_path)
                self.emb_dim = self._bert_config.hidden_size
        else:
            msg = f"Unsupported name '{embedder_type}' for 'embedder_type' " \
                  f"found. Supported names are only " \
                  f"{self.__class__.ALLOWED_EMBEDDER_TYPES}."
            raise ValueError(msg)

        # some validations
        if not self.emb_dim:
            msg = "Need a valid 'emb_dim' to initialize encoder resource. To specify a " \
                  "particular dimension, either pass-in the 'emb_dim' param or provide a " \
                  "valid 'embedder_type' param."
            raise ValueError(msg)

    def dump(self, dump_folder):
        if os.path.isfile(dump_folder):
            msg = f"Path input to 'dump' method must be a folder, " \
                  f"not a file ({dump_folder})"
            raise ValueError(msg)
        os.makedirs(dump_folder, exist_ok=True)

        if self.embedder_type == "bert":
            self.pretrained_model_name_or_path = os.path.join(
                os.path.abspath(dump_folder), f"{self.embedder_type}_resources",
                os.path.split(self.pretrained_model_name_or_path)[-1]
            )
            self._dump_bert_embedder_to_checkpoint(self.pretrained_model_name_or_path)
        with open(os.path.join(dump_folder, "text_encoder_config.json"), "w") as fp:
            params = {k: getattr(self, k) for k in self.params_keys}
            json.dump(params, fp, indent=4)
            fp.close()

    @classmethod
    def load(cls, load_folder):
        if os.path.isfile(load_folder):
            msg = f"Path input to 'load' method must be a folder, " \
                  f"not a file ({load_folder})"
            raise ValueError(msg)

        encoder = cls()
        with open(os.path.join(load_folder, "text_encoder_config.json"), "r") as fp:
            params = json.load(fp)
            setattr(encoder, "params_keys", set(params.keys()))
            fp.close()
        encoder.fit(**params, _disable_loading_embedder=True)
        if encoder.embedder_type == "bert":
            encoder._load_bert_embedder_from_checkpoint(encoder.pretrained_model_name_or_path)

        return encoder

    @abstractmethod
    def batch_encode(self, examples, labels=None, **kwargs) -> Dict[str, Any]:
        msg = f"Subclass {self.__class__.__name__} need to implement this method"
        raise NotImplementedError(msg)

    def _load_bert_embedder_from_checkpoint(self, pretrained_model_name_or_path):
        model_bunch = HuggingfaceTransformersContainer(pretrained_model_name_or_path,
                                                       reload=True).get_model_bunch()
        self._bert_config = model_bunch.config
        self._bert_tokenizer = model_bunch.tokenizer
        self._bert_model = model_bunch.model
        if self.freeze_embedder:
            for param in self._bert_model:
                param.requires_grad = False

    def _dump_bert_embedder_to_checkpoint(self, ckpt_folder):
        os.makedirs(ckpt_folder, exist_ok=True)
        self._bert_config.save_pretrained(ckpt_folder)
        self._bert_tokenizer.save_pretrained(ckpt_folder)
        self._bert_model.save_pretrained(ckpt_folder)


# sub-classes that define a custom `batch_encode` on top of base encoders


class SequenceClassificationEncoderWithStaticEmbeddings(EncoderWithStaticEmbeddings):
    """This class produces encoding for the given textual input as a sequence of ids.
    The inputs can be singular or batched.
    """

    def _reformat_and_validate(self, examples, labels=None):

        # reformatting
        if isinstance(examples, str):
            examples = [examples]
            if labels:
                labels = [labels]

        # validation for labels size
        if labels:
            if len(examples) != len(labels):
                msg = f"Number of 'labels' ({len(labels)}) must be same as 'examples' " \
                      f"({len(examples)}) when passing labels to " \
                      f"{self.__class__.__name__}.batch_encode()"
                raise AssertionError(msg)

        return examples, labels

    def batch_encode(
        self,
        examples: Union[str, List[str]],
        labels: Union[int, List[int]] = None,
        padding_length: int = None,
        add_terminals: bool = False
    ):
        """
        Returns batched encodings that can be used as an input to a featurizer
        """

        examples, labels = self._reformat_and_validate(examples, labels)

        # convert to tokens and obtain sequence lengths
        tokenized_sequences = [self._tokenizer(text) for text in examples]
        sequence_lengths = [len(seq) for seq in tokenized_sequences]

        # if padding_length is None, it is computed as max(length of all seqs from inputted text)
        padding_length = padding_length or self.padding_length
        # add 2 for start and end tokens respectively if add_terminals is True
        add_terminals = add_terminals or self.add_terminals
        curr_max = max(sequence_lengths) + 2 if add_terminals else max(sequence_lengths)
        padding_length = min(padding_length, curr_max) if padding_length else curr_max

        # define local constants
        pad_token_idx = getattr(self, "pad_token_idx")
        unk_token_idx = getattr(self, "unk_token_idx")
        start_token_idx = getattr(self, "start_token_idx")
        end_token_idx = getattr(self, "end_token_idx")

        # convert string tokens into ids
        input_ids = [[self.token2idx.get(token, unk_token_idx) for token in seq]
                     for seq in tokenized_sequences]

        # batchify by truncating or padding for each example
        for i in range(len(examples)):
            seq = input_ids[i]
            new_seq = seq[:padding_length - 2] if add_terminals else seq[:padding_length]
            new_seq = ([start_token_idx] + new_seq + [end_token_idx]) \
                if add_terminals else new_seq
            new_seq_length = len(new_seq)
            input_ids[i] = new_seq + [pad_token_idx] * (padding_length - new_seq_length)
            sequence_lengths[i] = new_seq_length

        # create output dict
        return_dict = {
            "seq_ids": torch.as_tensor(input_ids, dtype=torch.long),
            "seq_lengths": torch.as_tensor(sequence_lengths, dtype=torch.long)
        }

        if labels:
            return_dict.update({
                "labels": torch.as_tensor(labels, dtype=torch.long)
            })

        return return_dict


class TokenClassificationEncoderWithStaticEmbeddings(EncoderWithStaticEmbeddings):

    def _reformat_and_validate(self, examples, labels=None):

        # reformatting
        if isinstance(examples, str):
            examples = [examples]
            if labels:
                labels = [labels]

        # validation for labels size
        if labels:
            if len(examples) != len(labels):
                msg = f"Number of 'labels' ({len(labels)}) must be same as 'examples' " \
                      f"({len(examples)}) when passing labels to " \
                      f"{self.__class__.__name__}.batch_encode()"
                raise AssertionError(msg)
            for ex, label_tokens in zip(examples, labels):
                ex_tokens = ex.split()
                if len(ex_tokens) != len(label_tokens):
                    msg = f"Number of tokens in a sentence ({len(ex_tokens)}) must be same as the" \
                          f"number of tokens in the corresponding token labels " \
                          f"({len(label_tokens)}) for sentence '{ex}' with labels '{labels}'"
                    raise AssertionError(msg)

        return examples, labels

    def batch_encode(
        self,
        examples: Union[str, List[str]],
        labels: Union[List[int], List[List[int]]] = None,
        padding_length: int = None,
        add_terminals: bool = False,
    ):
        """
        Returns batched encodings that can be used as an input to a featurizer
        """

        examples, labels = self._reformat_and_validate(examples, labels)

        # convert to tokens and obtain sequence lengths
        tokenized_sequences = [self._tokenizer(text) for text in examples]
        sequence_lengths = [len(seq) for seq in tokenized_sequences]

        # if padding_length is None, it is computed as max(length of all seqs from inputted text)
        padding_length = padding_length or self.padding_length
        # add 2 for start and end tokens respectively if add_terminals is True
        add_terminals = add_terminals or self.add_terminals
        curr_max = max(sequence_lengths) + 2 if add_terminals else max(sequence_lengths)
        padding_length = min(padding_length, curr_max) if padding_length else curr_max

        # define local constants
        pad_token_idx = getattr(self, "pad_token_idx")
        unk_token_idx = getattr(self, "unk_token_idx")
        start_token_idx = getattr(self, "start_token_idx")
        end_token_idx = getattr(self, "end_token_idx")

        # convert string tokens into ids
        input_ids = [[self.token2idx.get(token, unk_token_idx) for token in seq]
                     for seq in tokenized_sequences]

        # batchify by truncating or padding for each example
        if labels:
            new_labels = []
        for i in range(len(examples)):
            # batchify sequences of text
            seq = input_ids[i]
            new_seq = seq[:padding_length - 2] if add_terminals else seq[:padding_length]
            new_seq = ([start_token_idx] + new_seq + [end_token_idx]) \
                if add_terminals else new_seq
            new_seq_length = len(new_seq)
            input_ids[i] = new_seq + [pad_token_idx] * (padding_length - new_seq_length)
            sequence_lengths[i] = new_seq_length
            # batchify labels
            if labels:
                label_seq = labels[i]
                new_label_seq = (
                    label_seq[:padding_length - 2] if add_terminals else
                    label_seq[:padding_length]
                )
                new_label_seq = (
                    [LABEL_PAD_TOKEN_IDX] + new_label_seq + [LABEL_PAD_TOKEN_IDX] if add_terminals
                    else new_label_seq
                )
                assert len(new_label_seq) == new_seq_length
                new_label_seq = (
                    new_label_seq + [LABEL_PAD_TOKEN_IDX] * (padding_length - new_seq_length)
                )
                new_labels.append(new_label_seq)

        return_dict = {
            "seq_ids": torch.as_tensor(input_ids, dtype=torch.long),
            "seq_lengths": torch.as_tensor(sequence_lengths, dtype=torch.long)
        }

        if labels:
            return_dict.update({
                "labels": torch.as_tensor(new_labels, dtype=torch.long)
            })

        return return_dict


class SequenceClassificationEncoderWithBert(EncoderWithPlmEmbeddings):

    def _reformat_and_validate(self, examples, labels=None):

        # reformatting
        if isinstance(examples, str):
            examples = [examples]
            if labels:
                labels = [labels]

        # validation for labels size
        if labels:
            if len(examples) != len(labels):
                msg = f"Number of 'labels' ({len(labels)}) must be same as 'examples' " \
                      f"({len(examples)}) when passing labels to " \
                      f"{self.__class__.__name__}.batch_encode()"
                raise AssertionError(msg)

        return examples, labels

    def batch_encode(
        self,
        examples: Union[str, List[str]],
        labels: Union[int, List[int]] = None,
        padding_length: int = None,
    ):
        """
        Returns batched encodings that can be used as an input to a featurizer
        """

        examples, labels = self._reformat_and_validate(examples, labels)

        return_dict = {}
        padding_length = padding_length or self.padding_length
        if self.embedder_type == "bert":
            # If padding_length is None, padding is done to the max length of the input batch
            _inputs = self._bert_tokenizer(examples, padding=True, truncation=True,
                                           max_length=padding_length, return_tensors="pt")
            _outputs = self._bert_model(**_inputs, return_dict=True)

            # last_hidden_state, pooler_output, seq_lengths are the keys being updated
            return_dict.update({**_outputs, "seq_lengths": _inputs["attention_mask"].sum(dim=-1)})

        if labels:
            return_dict.update({"labels": torch.as_tensor(labels, dtype=torch.long)})

        return return_dict


class TokenClassificationEncoderWithBert(EncoderWithPlmEmbeddings):

    def _reformat_and_validate(self, examples, labels=None):
        raise NotImplementedError

    def batch_encode(
        self,
        examples: Union[str, List[str]],
        labels: Union[int, List[int]] = None,
        padding_length: int = None,
    ):
        raise NotImplementedError
