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

from .._util import _get_module_or_attr
from ..containers import GloVeEmbeddingsContainer, HuggingfaceTransformersContainer

try:
    import torch
    from torch.nn.utils.rnn import pad_sequence

    nn_module = _get_module_or_attr("torch.nn", "Module")
except ImportError:
    nn_module = object
    pass

logger = logging.getLogger(__name__)

LABEL_PAD_TOKEN_IDX = -1  # value set based on default label padding idx in pytorch


# abstract encoder


class AbstractEncoder(nn_module):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.emb_dim = None
        self.params_keys = set(["name", "emb_dim"])

    def forward(self, *args, **kwargs):
        return self.batch_encode(*args, **kwargs)

    @abstractmethod
    def fit(self, **kwargs):
        msg = f"Subclass {self.name} need to implement this method"
        raise NotImplementedError(msg)

    @abstractmethod
    def dump(self, **kwargs):
        msg = f"Subclass {self.name} need to implement this method"
        raise NotImplementedError(msg)

    @classmethod
    @abstractmethod
    def load(cls, **kwargs):
        msg = f"Subclass {cls.__name__} need to implement this method"
        raise NotImplementedError(msg)

    @abstractmethod
    def batch_encode(self, examples, labels=None, **kwargs) -> Dict[str, Any]:
        msg = f"Subclass {self.name} need to implement this method"
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
            msg = f"Encoder instance ({self.name}) does not contain any token-to-embeddings mapping"
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

    def _reformat_and_validate(self, examples, labels=None):

        # reformatting
        if isinstance(examples, str):
            examples = [examples]
            if labels:
                labels = [labels]

        # validation for labels size
        if labels:
            if len(examples) != len(labels):
                msg = f"Number of 'labels' ({len(labels)}) must be same as number of 'examples' " \
                      f"({len(examples)}) when passing labels to {self.name}.batch_encode()"
                raise AssertionError(msg)

        return examples, labels


# base encoders


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

    ALLOWED_TOKENIZERS = [None, "whitespace-tokenizer", "char-tokenizer"]
    ALLOWED_EMBEDDER_TYPES = [None, "glove"]
    BASIC_SPECIAL_VOCAB_DICT = {
        "pad_token": "<PAD>",
        "unk_token": "<UNK>",
        "start_token": "<START>",
        "end_token": "<END>",
    }

    def fit(
        self,
        # input data stream
        examples: List[str] = None,
        tokens: List[str] = None,
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
        if self.embedder_type not in self.__class__.ALLOWED_EMBEDDER_TYPES:
            msg = f"Unsupported name '{self.embedder_type}' for 'embedder_type' " \
                  f"found. Supported names are only " \
                  f"{self.__class__.ALLOWED_EMBEDDER_TYPES}."
            raise ValueError(msg)
        elif self.embedder_type is None:
            self.tokenizer_type = params.get("tokenizer_type")
            self._tokenizer = self._get_tokenizer(self.tokenizer_type)
            self.params_keys.update(["tokenizer_type"])
        elif self.embedder_type == "glove":
            self.tokenizer_type = params.get("tokenizer_type")
            if self.tokenizer_type not in [None, "whitespace-tokenizer"]:
                msg = f"Provided 'tokenizer_type':'{params.get('tokenizer_type')}' cannot be " \
                      f"used with the provided 'embedder_type':'{self.embedder_type}'. " \
                      f"Discarding 'tokenizer_type' information"
                logger.warning(msg)
                self.tokenizer_type = "whitespace-tokenizer"
            self._tokenizer = self._get_tokenizer(self.tokenizer_type)
            self.params_keys.update(["tokenizer_type"])
            if not _disable_loading_embedder:
                glove_container = GloVeEmbeddingsContainer()
                self.token2emb = glove_container.get_pretrained_word_to_embeddings_dict()
                glove_emb_dim = glove_container.token_dimension
                if self.emb_dim and self.emb_dim != glove_emb_dim:
                    msg = f"Provided 'emb_dim':'{self.emb_dim}' cannot be used with the provided " \
                          f"'embedder_type':'{self.embedder_type}'. " \
                          f"Discarding 'emb_dim' information."
                    logger.warning(msg)
                self.emb_dim = glove_emb_dim

        # validate if emb_dim is valid
        if not self.emb_dim:
            msg = "Need a valid 'emb_dim' to initialize encoder resource. To specify a " \
                  "particular dimension, either pass-in the 'emb_dim' param or provide a " \
                  "valid 'embedder_type' param."
            raise ValueError(msg)

        # Add special vocab before actual vocab
        special_vocab_dict_ = self.__class__.BASIC_SPECIAL_VOCAB_DICT
        if self.special_vocab_dict:
            special_vocab_dict_.update(self.special_vocab_dict)
        self.special_vocab_dict = dict(special_vocab_dict_)
        self._add_special_vocab(self.special_vocab_dict)

        # add vocab from tokens inputted directly (useful when loading the encoder from a dump)
        if tokens:
            self._add_vocab(tokens)

        # add vocab from examples upon tokenizing each text item
        if examples:
            all_tokens = set(sum([self._tokenizer(text) for text in examples], []))
            self._add_vocab(all_tokens)

        if tokens is None and examples is None:
            msg = "At least one of 'examples' and 'tokens' must be inputted to create a vocab"
            raise ValueError(msg)
        else:
            msg = f"Vocab size: Number of unique tokens collected from training data: " \
                  f"{len(self.token2idx)}"
            logger.info(msg)

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
        with open(os.path.join(load_folder, "text_encoder_vocab.txt"), "r") as fp:
            all_tokens = [line.strip() for line in fp]
            fp.close()
        encoder.fit(**params, tokens=all_tokens, _disable_loading_embedder=True)

        return encoder

    @abstractmethod
    def batch_encode(self, examples, labels=None, **kwargs) -> Dict[str, Any]:
        msg = f"Subclass {self.name} need to implement this method"
        raise NotImplementedError(msg)

    def encode(self, list_of_tokens, padding_length, add_terminals):
        trimmed_list_of_tokens = (
            list_of_tokens[:padding_length - 2] if add_terminals else
            list_of_tokens[:padding_length]
        )
        bounded_list_of_tokens = (
            [getattr(self, "start_token")] + trimmed_list_of_tokens + [getattr(self, "end_token")]
        ) if add_terminals else trimmed_list_of_tokens
        seq_length = len(bounded_list_of_tokens)
        new_list_of_tokens = (
            bounded_list_of_tokens + [getattr(self, "pad_token")] * (padding_length - seq_length)
        )
        list_of_ids = [
            self.token2idx.get(token, getattr(self, "unk_token_idx"))
            for token in new_list_of_tokens
        ]
        return list_of_ids, seq_length

    def _get_tokenizer(self, tokenizer_type):

        def whitespace_tokenizer(text: str) -> List[str]:
            return text.split(" ")

        def char_tokenizer(text: str) -> List[str]:
            return list(text)

        if tokenizer_type not in self.__class__.ALLOWED_TOKENIZERS:
            msg = f"Unknown tokenizer type specified ('{tokenizer_type}'). Expected to be " \
                  f"among {self.__class__.ALLOWED_TOKENIZERS}. "
            raise ValueError(msg)
        elif tokenizer_type is None:
            return whitespace_tokenizer
        elif tokenizer_type == "whitespace-tokenizer":
            return whitespace_tokenizer
        elif tokenizer_type == "char-tokenizer":
            return char_tokenizer

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

    def _add_vocab(self, vocab: Iterable[str], target_dict=None):
        if target_dict is None:
            target_dict = self.token2idx
        for token in vocab:
            if token not in target_dict:
                target_dict.update({token: len(target_dict)})


class EncoderWithPretrainedLMs(AbstractEncoder):
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
        update_embeddings=True,
        # non-params
        _disable_loading_embedder=False,
        # other params
        **params
    ):
        del examples

        self.embedder_type = embedder_type
        self.padding_length = padding_length
        self.update_embeddings = update_embeddings
        self.params_keys.update(["embedder_type", "padding_length", "update_embeddings"])

        self.device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.params_keys.update(["device"])

        self.emb_dim = emb_dim

        # load tokenizer and embedder
        if self.embedder_type == "bert":
            if params.get("tokenizer_type"):
                msg = f"Provided 'tokenizer_type':'{params.get('tokenizer_type')}' cannot be " \
                      f"used with the provided 'embedder_type':'{self.embedder_type}'. " \
                      f"Discarding 'tokenizer_type' information"
                logger.warning(msg)
            self.pretrained_model_name_or_path = params.get("pretrained_model_name_or_path")
            if not self.pretrained_model_name_or_path:
                msg = "Must include a valid 'pretrained_model_name_or_path' param when using " \
                      "embedder_type: 'bert'"
                raise ValueError(msg)
            self.params_keys.update(["pretrained_model_name_or_path"])
            if not _disable_loading_embedder:
                self._load_bert_embedder_from_checkpoint(self.pretrained_model_name_or_path)
                bert_emb_dim = self._bert_config.hidden_size
                if self.emb_dim and self.emb_dim != bert_emb_dim:
                    msg = f"Provided 'emb_dim':'{self.emb_dim}' cannot be used with the provided " \
                          f"'embedder_type':'{self.embedder_type}'. " \
                          f"Discarding 'emb_dim' information."
                    logger.warning(msg)
                self.emb_dim = bert_emb_dim
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
        msg = f"Subclass {self.name} need to implement this method"
        raise NotImplementedError(msg)

    def _load_bert_embedder_from_checkpoint(self, pretrained_model_name_or_path):
        model_bunch = HuggingfaceTransformersContainer(
            pretrained_model_name_or_path, reload=True).get_model_bunch()
        self._bert_config = model_bunch.config
        self._bert_tokenizer = model_bunch.tokenizer
        self._bert_model = model_bunch.model.to(self.device)
        if not self.update_embeddings:
            for param in self._bert_model:
                param.requires_grad = False

    def _dump_bert_embedder_to_checkpoint(self, ckpt_folder):
        os.makedirs(ckpt_folder, exist_ok=True)
        self._bert_config.save_pretrained(ckpt_folder)
        self._bert_tokenizer.save_pretrained(ckpt_folder)
        self._bert_model.save_pretrained(ckpt_folder)


# sub-classes that define a custom `batch_encode` on top of base encoders


class SeqClsEncoderForEmbLayer(EncoderWithStaticEmbeddings):
    """This class produces encoding for the given textual input as a sequence of ids.
    The inputs can be singular or batched.
    """

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
        examples = [self._tokenizer(text) for text in examples]
        sequence_lengths = [len(seq) for seq in examples]

        # if padding_length is None, it is computed as max(length of all seqs from inputted text)
        padding_length = padding_length or self.padding_length
        # add 2 for start and end tokens respectively if add_terminals is True
        add_terminals = add_terminals or self.add_terminals
        curr_max = max(sequence_lengths) + 2 if add_terminals else max(sequence_lengths)
        padding_length = min(padding_length, curr_max) if padding_length else curr_max

        # batchify by truncating or padding for each example
        seq_ids, sequence_lengths = zip(*[
            self.encode(example, padding_length, add_terminals) for example in examples
        ])

        # create output dict
        return_dict = {
            "seq_ids": torch.as_tensor(seq_ids, dtype=torch.long),
            "seq_lengths": torch.as_tensor(sequence_lengths, dtype=torch.long)
        }

        if labels:
            return_dict.update({
                "labels": torch.as_tensor(labels, dtype=torch.long)
            })

        return return_dict


class TokenClsEncoderForEmbLayer(EncoderWithStaticEmbeddings):

    def _reformat_and_validate(self, examples, labels=None):
        super()._reformat_and_validate(examples, labels)

        # validation for labels' lengths too
        if labels:
            for ex, label_tokens in zip(examples, labels):
                ex_tokens = ex.split(" ")
                if len(ex_tokens) != len(label_tokens):
                    msg = f"Number of tokens in a sentence ({len(ex_tokens)}) must be same as the" \
                          f"number of tokens in the corresponding token labels " \
                          f"({len(label_tokens)}) for sentence '{ex}' with labels '{labels}'"
                    raise AssertionError(msg)

        return examples, labels

    @staticmethod
    def encode_labels(list_of_label_ids, padding_length, add_terminals):
        trimmed_list_of_label_ids = (
            list_of_label_ids[:padding_length - 2] if add_terminals else
            list_of_label_ids[:padding_length]
        )
        bounded_list_of_label_ids = (
            [LABEL_PAD_TOKEN_IDX] + trimmed_list_of_label_ids + [LABEL_PAD_TOKEN_IDX]
        ) if add_terminals else trimmed_list_of_label_ids
        new_list_of_label_ids = (
            bounded_list_of_label_ids +
            [LABEL_PAD_TOKEN_IDX] * (padding_length - len(bounded_list_of_label_ids))
        )
        return new_list_of_label_ids

    def batch_encode(
        self,
        examples: Union[str, List[str]],
        labels: Union[List[int], List[List[int]]] = None,
        padding_length: int = None,
        add_terminals: bool = False,
        _return_tokenized_examples: bool = False
    ):
        """
        Returns batched encodings that can be used as an input to a featurizer
        """

        examples, labels = self._reformat_and_validate(examples, labels)

        # convert to tokens and obtain sequence lengths
        examples = [self._tokenizer(text) for text in examples]
        sequence_lengths = [len(seq) for seq in examples]

        # if padding_length is None, it is computed as max(length of all seqs from inputted text)
        padding_length = padding_length or self.padding_length
        # add 2 for start and end tokens respectively if add_terminals is True
        add_terminals = add_terminals or self.add_terminals
        curr_max = max(sequence_lengths) + 2 if add_terminals else max(sequence_lengths)
        padding_length = min(padding_length, curr_max) if padding_length else curr_max

        # batchify by truncating or padding for each example
        seq_ids, sequence_lengths = zip(*[
            self.encode(example, padding_length, add_terminals) for example in examples
        ])

        return_dict = {
            "seq_ids": torch.as_tensor(seq_ids, dtype=torch.long),
            "seq_lengths": torch.as_tensor(sequence_lengths, dtype=torch.long)
        }

        if _return_tokenized_examples:
            return_dict.update({
                "seq_tokens": examples
            })

        if labels:
            new_labels = [self.encode_labels(label, padding_length, add_terminals) for label in
                          labels]
            return_dict.update({
                "labels": torch.as_tensor(new_labels, dtype=torch.long)
            })

        return return_dict


class SeqClsEncoderWithPlmLayer(EncoderWithPretrainedLMs):

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
            _inputs = self._bert_tokenizer(
                examples, padding=True, truncation=True, max_length=padding_length,
                return_tensors="pt"
            ).to(self.device)
            _outputs = self._bert_model(**_inputs, return_dict=True)

            # last_hidden_state, pooler_output, seq_lengths are the keys being updated
            return_dict.update({
                **_outputs,
                "seq_lengths": _inputs["attention_mask"].sum(dim=-1)
            })

        if labels:
            return_dict.update({"labels": torch.as_tensor(labels, dtype=torch.long)})

        return return_dict


class TokenClsEncoderWithPlmLayer(EncoderWithPretrainedLMs):

    # method similar to TokenClsEncoderForEmbLayer
    def _reformat_and_validate(self, examples, labels=None):
        super()._reformat_and_validate(examples, labels)

        # validation for labels' lengths too
        if labels:
            for ex, label_tokens in zip(examples, labels):
                ex_tokens = ex.split(" ")
                if len(ex_tokens) != len(label_tokens):
                    msg = f"Number of tokens in a sentence ({len(ex_tokens)}) must be same as the" \
                          f"number of tokens in the corresponding token labels " \
                          f"({len(label_tokens)}) for sentence '{ex}' with labels '{labels}'"
                    raise AssertionError(msg)

        return examples, labels

    @staticmethod
    def _trim_combined(x: List[List[Any]], trim_len: int, y: List[Any] = None):
        curr_len = 0
        if y:
            new_x, new_y = [], []
            for _x, _y in zip(x, y):
                if curr_len >= trim_len:
                    return new_x, new_y
                if curr_len + len(_x) > trim_len:
                    new_x.append(_x[trim_len - curr_len])
                    new_y.append(_y)
                elif curr_len + len(_x) <= trim_len:
                    new_x.append(_x)
                    new_y.append(_y)
            return new_x, new_y
        else:
            new_x = []
            for _x in x:
                if curr_len >= trim_len:
                    return new_x
                if curr_len + len(_x) > trim_len:
                    new_x.append(_x[trim_len - curr_len])
                elif curr_len + len(_x) <= trim_len:
                    new_x.append(_x)
            return new_x

    def batch_encode(
        self,
        examples: Union[str, List[str]],
        labels: Union[int, List[int]] = None,
        padding_length: int = None,
    ):
        examples, labels = self._reformat_and_validate(examples, labels)

        return_dict = {}
        padding_length = padding_length or self.padding_length

        if self.embedder_type == "bert":
            # tokenize each word of each input seperately
            tokenized_examples = [
                [self._bert_tokenizer.tokenize(word) for word in example.split(" ")]
                for example in examples
            ]
            # get maximum length of each example
            max_curr_len = max([len(sum(t_ex, [])) for t_ex in tokenized_examples]) + 2  # cls, sep
            # If padding_length is None, padding has to be done to the max length of the input batch
            padding_length = min(max_curr_len, padding_length) if padding_length else max_curr_len

            _trim_length = padding_length - 2  # -2 to sum upto padding_length after adding cls, sep
            if labels:
                trimmed_tokenized_examples, trimmed_labels = zip(*[
                    self._trim_combined(t_ex, _trim_length, _labels)
                    for t_ex, _labels in zip(tokenized_examples, labels)
                ])
            else:
                trimmed_tokenized_examples = [
                    self._trim_combined(t_ex, _trim_length) for t_ex in tokenized_examples
                ]
            split_lengths = [[len(x) for x in ex] for ex in trimmed_tokenized_examples]
            trimmed_tokenized_examples = [" ".join(sum(ex, [])) for ex in
                                          trimmed_tokenized_examples]
            _inputs = self._bert_tokenizer(
                trimmed_tokenized_examples, padding=True, truncation=True, max_length=None,
                return_tensors="pt"
            ).to(self.device)
            # last_hidden_state, pooler_output, seq_lengths are the keys outputted
            _outputs = self._bert_model(**_inputs, return_dict=True)

            # discard [CLS] token, [SEP] token is anyway discarded as the sum of each split length
            # item does not go as far as the SEP token
            last_hidden_state = _outputs["last_hidden_state"][:, 1:]

            return_dict.update({
                "last_hidden_state": last_hidden_state,  # [BS, SEQ_LEN, EMD_DIM]
                "split_lengths": split_lengths,  # List[List[Int]]
                "seq_lengths": torch.as_tensor(
                    [len(_split_lengths) for _split_lengths in split_lengths], dtype=torch.long
                )
            })

            if labels:
                trimmed_labels = pad_sequence(
                    [torch.as_tensor(label) for label in trimmed_labels], batch_first=True,
                    padding_value=LABEL_PAD_TOKEN_IDX
                ).long()
                return_dict.update({"labels": trimmed_labels})

        return return_dict


# sub-classes that define more complex custom `batch_encode` on top of derived encoders


class TokenClsDualEncoderForEmbLayers(TokenClsEncoderForEmbLayer):
    """Dual encoder that encodes both word level as well as character level tokens
    """

    ALLOWED_TOKENIZERS = [None, "whitespace-tokenizer"]
    BASIC_SPECIAL_CHAR_VOCAB_DICT = {
        "char_pad_token": "<CHAR_PAD>",
        "char_unk_token": "<CHAR_UNK>",
        "char_start_token": "<CHAR_START>",
        "char_end_token": "<CHAR_END>",
    }

    def fit(
        self,
        # input data stream
        all_char_tokens: List[str] = None,
        # this class specific params
        char_emb_dim=None,
        char_special_vocab_dict=None,
        char_padding_length=None,
        char_add_terminals=False,
        # all other params
        **params,
    ):
        self.char_emb_dim = char_emb_dim
        self.char_special_vocab_dict = char_special_vocab_dict
        self.char_padding_length = char_padding_length
        self.char_add_terminals = char_add_terminals
        self.params_keys.update([
            "char_emb_dim", "char_special_vocab_dict", "char_padding_length", "char_add_terminals"
        ])

        self.char_token2idx = {}

        # validate if char_emb_dim is valid
        if not self.char_emb_dim:
            msg = "Need a valid 'char_emb_dim' to initialize encoder resource. To specify a " \
                  "particular dimension, either pass-in the 'char_emb_dim' param with a positive " \
                  "integer value"
            raise ValueError(msg)

        # Add special vocab before actual vocab
        special_vocab_dict_ = self.__class__.BASIC_SPECIAL_CHAR_VOCAB_DICT
        if self.char_special_vocab_dict:
            special_vocab_dict_.update(self.char_special_vocab_dict)
        self.char_special_vocab_dict = dict(special_vocab_dict_)
        for name, token in self.char_special_vocab_dict.items():
            self._add_vocab([token], target_dict=self.char_token2idx)
            setattr(self, f"{name}", token)
            setattr(self, f"{name}_idx", self.char_token2idx[token])

        # populate token lookup dictionary
        super().fit(**params)

        # populate char lookup dictionary
        if not all_char_tokens:
            all_char_tokens = set(c for token in self.token2idx for c in list(token))
        self._add_vocab(all_char_tokens, target_dict=self.char_token2idx)

    def dump(self, dump_folder):
        super().dump(dump_folder)
        with open(os.path.join(dump_folder, "char_text_encoder_vocab.txt"), "w") as fp:
            for token in self.char_token2idx:
                fp.write(token + "\n")
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
        with open(os.path.join(load_folder, "text_encoder_vocab.txt"), "r") as fp:
            all_tokens = [line.strip() for line in fp]
            fp.close()
        with open(os.path.join(load_folder, "char_text_encoder_vocab.txt"), "r") as fp:
            all_char_tokens = [line.strip() for line in fp]
            fp.close()
        encoder.fit(
            **params, tokens=all_tokens, all_char_tokens=all_char_tokens,
            _disable_loading_embedder=True
        )

        return encoder

    def batch_encode(
        self,
        examples: Union[str, List[str]],
        labels: Union[List[int], List[List[int]]] = None,
        padding_length: int = None,
        add_terminals: bool = False,
        char_padding_length: int = None,
        char_add_terminals: bool = False
    ):
        """
        Returns batched encodings that can be used as an input to a featurizer
        """

        # validation
        add_terminals = add_terminals or self.add_terminals
        if add_terminals:
            msg = "Setting param 'add_terminals' to True is not supported with dual token encoder"
            raise NotImplementedError(msg)

        return_dict = super().batch_encode(
            examples, labels=labels, padding_length=padding_length, add_terminals=False,
            _return_tokenized_examples=True
        )

        # use tokenize examples to obtain tokens for char tokenization
        examples = return_dict.pop("seq_tokens")

        char_seq_ids, char_sequence_lengths = [], []
        for example in examples:
            # compute padding length for character sequences
            _char_padding_length = char_padding_length or self.char_padding_length
            _char_add_terminals = char_add_terminals or self.char_add_terminals
            _curr_max = max([len(word) for word in example])
            _curr_max = _curr_max + 2 if _char_add_terminals else _curr_max
            _char_padding_length = (
                min(_char_padding_length, _curr_max) if _char_padding_length else _curr_max
            )
            # encode
            _char_seq_ids, _char_sequence_lengths = zip(*[
                self.encode_char(list(word), _char_padding_length, _char_add_terminals)
                for word in example
            ])
            char_seq_ids.append(_char_seq_ids)
            char_sequence_lengths.append(_char_sequence_lengths)

        return_dict.update({
            "char_seq_ids": [torch.as_tensor(_ids, dtype=torch.long) for _ids in char_seq_ids],
            "char_seq_lengths": [
                torch.as_tensor(_lens, dtype=torch.long) for _lens in char_sequence_lengths
            ]
        })

        return return_dict

    def encode_char(self, list_of_tokens, padding_length, add_terminals):
        trimmed_list_of_tokens = (
            list_of_tokens[:padding_length - 2] if add_terminals else
            list_of_tokens[:padding_length]
        )
        bounded_list_of_tokens = (
            [getattr(self, "char_start_token")] + trimmed_list_of_tokens + [
            getattr(self, "char_end_token")]
        ) if add_terminals else trimmed_list_of_tokens
        seq_length = len(bounded_list_of_tokens)
        new_list_of_tokens = (
            bounded_list_of_tokens +
            [getattr(self, "char_pad_token")] * (padding_length - seq_length)
        )
        list_of_ids = [
            self.char_token2idx.get(token, getattr(self, "char_unk_token_idx"))
            for token in new_list_of_tokens
        ]
        return list_of_ids, seq_length

    # some `get` methods for char modeling

    def get_char_num_tokens(self):
        if not hasattr(self, "char_token2idx"):
            return None
        return len(self.char_token2idx)

    def get_char_emb_dim(self):
        return self.char_emb_dim

    def get_char_pad_token_idx(self):
        if not hasattr(self, "char_pad_token_idx"):
            return None
        return self.char_pad_token_idx
