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

import numpy as np

try:
    import torch
except ImportError:
    pass

try:
    import transformers
except ImportError:
    pass

from ..taggers.embeddings import GloVeEmbeddingsContainer

logger = logging.getLogger(__name__)

LABEL_PAD_TOKEN_IDX = -1  # value set based on default label padding idx in pytorch


class ClassificationEncoder:
    """
    A base class for joint tokenization-encoding with the vocab derived from the training data.

    TOKENIZATION SUPPORT:
    This class supports three kinds of tokenizations: (1) tokenization at white space,
    (2) character tokenizations, (3) a mindmeld tokenizer configured in the mindmeld app
    through config.py

    ENCODING SUPPORT
    This class supports encoding text into 0-based ids for an embedding lookup. When using the
    max_seq_length, at times, the labels might have to be trimmed as well, which is to be supported
    in the .batch_encode() method of the derived class.
    """

    ALLOWED_BASIC_TOKENIZERS = ["whitespace-tokenizer", "char-tokenizer"]
    ALLOWED_EMBEDDER_TYPES = [
        "glove",
        # "fastext",  # TODO: implement
        "bert"
    ]

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
        max_seq_length=None,
        # non-params
        _disable_loading_glove_data=False,
        # other params
        **params
    ):
        self.embedder_type = embedder_type
        self.special_vocab_dict = special_vocab_dict
        self.max_seq_length = max_seq_length
        self.params_keys = set(["embedder_type", "special_vocab_dict", "max_seq_length"])

        self.token2idx = {}
        self.token2emb = {}

        # load tokenizer and embedder
        if not self.embedder_type:
            self.tokenizer_type = params.get("tokenizer_type")
            self._tokenizer = self._get_basic_tokenizer(self.tokenizer_type)
            self.emb_dim = params.get("emb_dim")
            self.params_keys.update(["tokenizer_type", "emb_dim"])
        elif self.embedder_type == "glove":
            self.tokenizer_type = params.get("tokenizer_type")
            self._tokenizer = self._get_basic_tokenizer(self.tokenizer_type)
            self.emb_dim = params.get("emb_dim")
            self.params_keys.update(["tokenizer_type", "emb_dim"])
            if not _disable_loading_glove_data:
                glove_container = GloVeEmbeddingsContainer()
                self.token2emb = glove_container.get_pretrained_word_to_embeddings_dict()
                self.emb_dim = glove_container.token_dimension
        elif self.embedder_type == "bert":
            self.pretrained_model_name_or_path = params.get("pretrained_model_name_or_path")
            if not self.pretrained_model_name_or_path:
                msg = "Must include a valid 'pretrained_model_name_or_path' param when using " \
                      "embedder_type: 'bert'"
                raise ValueError(msg)
            self.params_keys.update(["pretrained_model_name_or_path"])
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.pretrained_model_name_or_path).tokenize
            self.emb_dim = transformers.AutoConfig.from_pretrained(
                self.pretrained_model_name_or_path).hidden_size
            # TODO: Add snippet to load bert model
            raise NotImplementedError
        else:
            msg = f"Unsupported name '{embedder_type}' for 'embedder_type' " \
                  f"found. Supported names are only " \
                  f"{ClassificationEncoder.ALLOWED_EMBEDDER_TYPES}."
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
        encoder.fit(**params, _disable_loading_glove_data=True)
        with open(os.path.join(load_folder, "text_encoder_vocab.txt"), "r") as fp:
            for line in fp:
                encoder._add_vocab([line.strip()])
            fp.close()

        return encoder

    @abstractmethod
    def batch_encode(self, **kwargs) -> Dict[str, Any]:
        msg = f"Subclass {self.__class__.__name__} need to implement this"
        raise NotImplementedError(msg)

    def get_num_tokens(self):
        return len(self.token2idx)

    def get_emb_dim(self):
        return self.emb_dim

    def get_embedding_weights(self):
        if not self.token2emb:
            msg = f"The created encoder ({self.__class__.__name__}) does not contain " \
                  f"token-to-embeddings mapping."
            logger.error(msg)
            return None
        embedding_weights = {}
        for token, idx in self.token2idx.items():
            if token in self.token2emb:
                embedding_weights[idx] = self.token2emb[token]
        return embedding_weights

    @staticmethod
    def _get_basic_tokenizer(tokenizer_type):

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
                  f"among {ClassificationEncoder.ALLOWED_BASIC_TOKENIZERS}. "
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


class SequenceClassificationEncoder(ClassificationEncoder):

    def batch_encode(
        self,
        examples: Union[str, List[str]],
        labels: Union[int, List[int]] = None,
        return_embeddings: bool = None,
        max_seq_length: int = None,
        add_terminals: bool = False,
        return_as_tensors: bool = True
    ):
        """
        Returns batched encodings that can be used as an input to a featurizer
        """

        if not examples:
            return {}

        # check if ids are to be returned or embeddings
        return_embeddings = isinstance(return_embeddings, bool) and return_embeddings
        if return_embeddings and add_terminals:
            msg = "Cannot set both 'add_terminals' and 'return_embeddings' at the same " \
                  "time. Discarding 'add_terminals' param"
            logger.error(msg)
            add_terminals = False

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

        # convert to tokens and obtain sequence lengths
        tokenized_sequences = [self._tokenizer(text) for text in examples]
        sequence_lengths = [len(seq) for seq in tokenized_sequences]

        # if max_seq_length is None, it is computed as max(length of all seqs from inputted text)
        #   add 2 for start and end tokens resp. if add_terminals is True
        max_seq_length = max_seq_length or self.max_seq_length
        observed_max_seq_length = (
            max(sequence_lengths) + 2 if add_terminals else max(sequence_lengths)
        )
        if max_seq_length:
            max_seq_length = min(max_seq_length, observed_max_seq_length)
        else:
            max_seq_length = observed_max_seq_length

        if return_embeddings:  # prepare & return a dense 2d embeddings tensor instead of token ids

            # update sequences
            tokenized_sequences = [seq[:max_seq_length] for seq in tokenized_sequences]
            sequence_lengths = [len(seq) for seq in tokenized_sequences]

            # replace unk and pad token's embeddings with zeros
            zeroes_emb = np.zeros(self.emb_dim)
            sequence_embeddings = np.asarray([
                [self.token2emb.get(tok, zeroes_emb) for tok in seq] +
                [zeroes_emb] * (max_seq_length - len(seq))
                for seq in tokenized_sequences
            ])

            if return_as_tensors:
                return_dict = {
                    "seq_lengths": torch.as_tensor(sequence_lengths, dtype=torch.long),
                    "seq_embs": torch.as_tensor(sequence_embeddings, dtype=torch.float32)
                }
            else:
                return_dict = {
                    "seq_lengths": sequence_lengths,
                    "seq_embs": sequence_embeddings
                }

        else:

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
                new_seq = seq[:max_seq_length - 2] if add_terminals else seq[:max_seq_length]
                new_seq = ([start_token_idx] + new_seq + [end_token_idx]) \
                    if add_terminals else new_seq
                new_seq_length = len(new_seq)
                input_ids[i] = new_seq + [pad_token_idx] * (max_seq_length - new_seq_length)
                sequence_lengths[i] = new_seq_length

            # create output dict
            if return_as_tensors:
                return_dict = {
                    "seq_ids": torch.as_tensor(input_ids, dtype=torch.long),
                    "seq_lengths": torch.as_tensor(sequence_lengths, dtype=torch.long)
                }
            else:
                return_dict = {
                    "seq_ids": np.array(input_ids),
                    "seq_lengths": np.array(sequence_lengths)
                }

        if labels:
            if return_as_tensors:
                return_dict.update({"labels": torch.as_tensor(labels, dtype=torch.long)})
            else:
                return_dict.update({"labels": labels})

        return return_dict


class TokenClassificationEncoder(ClassificationEncoder):

    def batch_encode(  # pylint: disable=too-many-locals
        self,
        examples: Union[str, List[str]],
        labels: Union[List[int], List[List[int]]] = None,
        return_embeddings: bool = None,
        max_seq_length: int = None,
        add_terminals: bool = False,
        return_as_tensors: bool = True
    ):
        """
        Returns batched encodings that can be used as an input to a featurizer
        """

        if not examples:
            return {}

        # check if ids are to be returned or embeddings
        return_embeddings = isinstance(return_embeddings, bool) and return_embeddings
        if return_embeddings and add_terminals:
            msg = "Cannot set both 'add_terminals' and 'return_embeddings' at the same " \
                  "time. Discarding 'add_terminals' param"
            logger.error(msg)
            add_terminals = False

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

        # convert to tokens and obtain sequence lengths
        tokenized_sequences = [self._tokenizer(text) for text in examples]
        sequence_lengths = [len(seq) for seq in tokenized_sequences]

        # if max_seq_length is None, it is computed as max(length of all seqs from inputted text)
        #   add 2 for start and end tokens resp. if add_terminals is True
        max_seq_length = max_seq_length or self.max_seq_length
        observed_max_seq_length = (
            max(sequence_lengths) + 2 if add_terminals else max(sequence_lengths)
        )
        if max_seq_length:
            max_seq_length = min(max_seq_length, observed_max_seq_length)
        else:
            max_seq_length = observed_max_seq_length

        if return_embeddings:  # prepare & return a dense 2d embeddings tensor instead of token ids

            # update sequences
            tokenized_sequences = [seq[:max_seq_length] for seq in tokenized_sequences]
            sequence_lengths = [len(seq) for seq in tokenized_sequences]

            # replace unk and pad token's embeddings with zeros
            zeroes_emb = np.zeros(self.emb_dim)
            sequence_embeddings = np.asarray([
                [self.token2emb.get(tok, zeroes_emb) for tok in seq] +
                [zeroes_emb] * (max_seq_length - len(seq))
                for seq in tokenized_sequences
            ])

            if return_as_tensors:
                return_dict = {
                    "seq_lengths": torch.as_tensor(sequence_lengths, dtype=torch.long),
                    "seq_embs": torch.as_tensor(sequence_embeddings, dtype=torch.float32)
                }
            else:
                return_dict = {
                    "seq_lengths": sequence_lengths,
                    "seq_embs": sequence_embeddings
                }

        else:

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
                # batchify sequences of text
                seq = input_ids[i]
                new_seq = seq[:max_seq_length - 2] if add_terminals else seq[:max_seq_length]
                new_seq = ([start_token_idx] + new_seq + [end_token_idx]) \
                    if add_terminals else new_seq
                new_seq_length = len(new_seq)
                input_ids[i] = new_seq + [pad_token_idx] * (max_seq_length - new_seq_length)
                sequence_lengths[i] = new_seq_length
                # batchify labels
                if labels:
                    label_seq = labels[i]
                    new_label_seq = (
                        label_seq[:max_seq_length - 2] if add_terminals else
                        label_seq[:max_seq_length]
                    )
                    new_label_seq = ([LABEL_PAD_TOKEN_IDX] + new_label_seq + [
                        LABEL_PAD_TOKEN_IDX]) if add_terminals else new_label_seq
                    assert len(new_label_seq) == new_seq_length
                    labels[i] = (
                        new_label_seq + [LABEL_PAD_TOKEN_IDX] * (max_seq_length - new_seq_length)
                    )

            # create output dict
            if return_as_tensors:
                return_dict = {
                    "seq_ids": torch.as_tensor(input_ids, dtype=torch.long),
                    "seq_lengths": torch.as_tensor(sequence_lengths, dtype=torch.long)
                }
            else:
                return_dict = {
                    "seq_ids": np.array(input_ids),
                    "seq_lengths": np.array(sequence_lengths)
                }

        if labels:
            if return_as_tensors:
                return_dict.update({"labels": torch.as_tensor(labels, dtype=torch.long)})
            else:
                return_dict.update({"labels": labels})

        return return_dict
