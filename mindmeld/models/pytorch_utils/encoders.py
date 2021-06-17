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
This module consists of encoders that serve as input to featurizers
"""
import json
import logging
import os
from typing import Dict, List, Union

import numpy as np
import torch
from torch import Tensor

from mindmeld.models.embedder_models.embeddings import GloVeEmbeddingsContainer

logger = logging.getLogger(__name__)


class Encoder(ABC):

    @abstractmethod
    def initialize_resources(self, **kwargs) -> self:
        msg = f"Deriving class {self.__class__.__name__} need to implement this"
        raise NotImplementedError(msg)

    @abstractmethod
    def load_resources(self, load_folder) -> None:
        msg = f"Deriving class {self.__class__.__name__} need to implement this"
        raise NotImplementedError(msg)

    @abstractmethod
    def dump_resources(self, dump_folder) -> None:
        msg = f"Deriving class {self.__class__.__name__} need to implement this"
        raise NotImplementedError(msg)

    def encode(self, **kwargs) -> Dict[str, Any]:
        msg = f"Deriving class {self.__class__.__name__} need to implement this"
        raise NotImplementedError(msg)


class TextsEncoder(Encoder):
    """
    A base class for joint tokenization-encoding with the vocab derived from the training data.

    This class supports three kinds of tokenizations: (1) tokenization at white space,
    (2) individual character tokenizations, (3) a mindmeld tokenizer configured in the mindmeld app.

    This class only supports encoding text into 0-based ids for an embedding lookup or directly into
    embeddings based on the featurizer's requirement to finetune or not to finetune the embeddings
    of words found in training data.
    """

    ALLOWED_TOKENIZER_TYPES = ["whitespace-tokenizer", "char-tokenizer", "mindmeld-tokenizer"]
    ALLOWED_EMBEDDER_TYPES = ["glove"]  # only word level embedders are allowed in this encoder

    BASIC_SPECIAL_VOCAB_DICT = {
        "pad_token": "<PAD>",
        "unk_token": "<UNK>",
        "start_token": "<START>",
        "end_token": "<END>",
        "mask_token": "<MASK>",
    }

    def __init__(self, **kwargs):

        self.tokenizer_type = kwargs.get("tokenizer_type", None)
        self.token_embedder_type = kwargs.get("token_embedder_type", None)
        self.special_vocab_dict = kwargs.get("special_vocab_dict", {})
        self.max_seq_length = kwargs.get("max_seq_length", None)
        self.token_emb_dim = kwargs.get("token_emb_dim", None)

        self._token2idx = {}
        self._token2emb = {}

    def _add_special_vocab(self, special_tokens: Dict[str, str]):

        # Validate special tokens
        DUPLICATE_RECORD_WARNING = "Ignoring duplicate token {} from special vocab set with name {}"
        for name, token in special_tokens:
            if name in self.special_vocab_dict and self.special_vocab_dict[name] != token:
                logger.warning(DUPLICATE_RECORD_WARNING, token, name)
                continue
            self.special_vocab_dict.update({name: token})

        # Add special tokens to vocab as well as their names to self
        for name, token in self.special_vocab_dict:
            self._add_vocab([token])
            setattr(self, f"{name}", token)
            setattr(self, f"{name}_idx", self._token2idx[token])

    def _add_vocab(self, vocab: Iterable[str]):
        for token in vocab:
            if token not in self._token2idx:
                self._token2idx.update({token: len(self._token2idx)})

    @property
    def get_vocab_size(self):
        return len(self._token2idx)

    @property
    def get_emb_dim(self):
        return len(self.token_emb_dim)

    @staticmethod
    def _get_tokenizer(tokenizer_type, mm_tokenizer):

        def whitespace_tokenizer(text: str) -> List[str]:
            return text.strip().split()

        def char_tokenizer(text: str) -> List[str]:
            return list(text.strip())

        def mindmeld_tokenizer(text: str) -> List[str]:
            # extracts tokenized texts from self.mm_tokenizer's output format
            bundled_tokens = mm_tokenizer.tokenize_raw(text)
            return [bundle["text"] for bundle in bundled_tokens]

        if not tokenizer_type:
            return whitespace_tokenizer
        elif tokenizer_type == "whitespace-tokenizer":
            return whitespace_tokenizer
        elif tokenizer_type == "char-tokenizer":
            return char_tokenizer
        elif tokenizer_type == "mindmeld-tokenizer":
            if not mm_tokenizer:
                msg = f"Invalid mindmeld-tokenizer: {mm_tokenizer} of type {type(mm_tokenizer)}"
                raise ValueError(msg)
            return mindmeld_tokenizer
        else:
            msg = f"Unknown tokenizer type specified ('{tokenizer_type}'). " \
                  f"Expected to be among {TextsEncoder.ALLOWED_TOKENIZER_TYPES}. "
            raise ValueError(msg)

    def initialize_resources(
        self,
        texts: List[str],
        tokenizer_type=None,
        token_embedder_type=None,
        special_vocab_dict=None,
        max_seq_length=None,
        token_emb_dim=None,
        mm_tokenizer=None
    ):

        self.tokenizer_type = self.tokenizer_type or tokenizer_type
        self._tokenizer = self._get_tokenizer(self.tokenizer_type, mm_tokenizer)

        self.token_embedder_type = self.token_embedder_type or token_embedder_type
        if self.token_embedder_type:
            if self.token_embedder_type == "glove":
                glove_container = GloVeEmbeddingsContainer()
                self._token2emb = glove_container.get_pretrained_word_to_embeddings_dict()
                token_emb_dim = glove_container.token_dimension
                if self.token_emb_dim and self.token_emb_dim != token_emb_dim:
                    msg = f"Overwriting 'token_emb_dim' from {self.token_emb_dim} dim to " \
                          f"{token_emb_dim} dim while using 'token_embedder_type'='glove'"
                    logger.error(msg)
                self.token_emb_dim = token_emb_dim
            else:
                msg = f"Unsupported name '{token_embedder_type}' for 'token_embedder_type' " \
                      f"found. Supported names are only {TextsEncoder.ALLOWED_EMBEDDER_TYPES}."
                raise ValueError(msg)

        _special_vocab_dict = self.__class__.BASIC_SPECIAL_VOCAB_DICT
        if self.special_vocab_dict:
            _special_vocab_dict.update(self.special_vocab_dict)
        if special_vocab_dict:
            _special_vocab_dict.update(special_vocab_dict)
        self.special_vocab_dict = dict(_special_vocab_dict)
        self._add_special_vocab(self.special_vocab_dict)

        self.max_seq_length = self.max_seq_length or max_seq_length

        if not self.token_emb_dim:
            self.token_emb_dim = token_emb_dim
        elif self.token_emb_dim and token_emb_dim and token_emb_dim != self.token_emb_dim:
            msg = f"Found token_emb_dim {self.token_emb_dim} dim not matching the passed-in " \
                  f"value of {token_emb_dim}. Discarding the passed-in value. "
            logger.error(msg)
        elif not (self.token_emb_dim or token_emb_dim):
            msg = "Need a valid 'token_emb_dim' to initialize encoder resource. " \
                  "Either pass-in the argument or load an embedder."
            raise ValueError(msg)

        all_tokens = set(sum([self._tokenizer(text) for text in texts], []))
        self._add_vocab(all_tokens)

        return self

    def to_metadata(self):
        encoder_config = {
            "tokenizer_type": self.tokenizer_type,
            "token_embedder_type": self.token_embedder_type,
            "special_vocab_dict": self.special_vocab_dict,
            "max_seq_length": self.max_seq_length,
            "token_emb_dim": self.token_emb_dim
        }
        return encoder_config

    @classmethod
    def from_metadata(cls, encoder_config):
        return cls(**encoder_config)

    def dump_resources(self, dump_folder):

        if os.path.isfile(dump_folder):
            msg = f"Path input to 'dump' method must be a folder, " \
                  f"not a file ({dump_folder})"
            raise ValueError(msg)
        os.makedirs(dump_folder, exist_ok=True)

        with open(os.path.join(dump_folder, "text_encoder_vocab.txt"), "w") as fp:
            for token in self._token2idx:
                fp.write(token + "\n")
            fp.close()
        with open(os.path.join(dump_folder, "text_encoder_config.json"), "w") as fp:
            json.dump(self.to_metadata(), fp, indent=4)
            fp.close()

    @classmethod
    def load_resources(cls, load_folder) -> TextsEncoder:

        if os.path.isfile(load_folder):
            msg = f"Path input to 'load' method must be a folder, " \
                  f"not a file ({load_folder})"
            raise ValueError(msg)

        with open(os.path.join(load_folder, "text_encoder_config.json"), "r") as fp:
            encoder_config = json.load(fp)
            fp.close()
        encoder = cls.from_metadata(encoder_config)

        with open(os.path.join(load_folder, "text_encoder_vocab.txt"), "r") as fp:
            for line in fp:
                encoder._add_vocab([line.strip()])
            fp.close()

        return encoder

    def encode(
        self,
        texts: Union[str, List[str]],
        from_token_embedder: bool = None,
        max_seq_length: int = None,
        add_terminals: bool = False,
        return_as_tensors: bool = True,
        **kwargs
    ):
        """
        Returns batched encodings that can be used as an input to a featurizer
        """
        if isinstance(texts, str):
            texts = [texts]

        # return a dense 2d embeddings tensor instead of token ids
        if isinstance(from_token_embedder, bool) and from_token_embedder:

            if add_terminals:
                msg = "Cannot set both 'add_terminals' and 'from_token_embedder' at the same " \
                      "time. Discarding 'add_terminals' param."
                logger.error(msg)

            # convert to tokens and obtain sequence lengths
            token_sequences = [self._tokenizer(text) for text in texts]

            max_seq_length = max_seq_length or self.max_seq_length
            observed_max_seq_length = max(token_sequences)
            if max_seq_length:
                max_seq_length = min(max_seq_length, observed_max_seq_length)
            else:
                max_seq_length = observed_max_seq_length

            token_sequences = [seq[:max_seq_length] for seq in token_sequences]
            sequence_lengths = [len(seq) for seq in token_sequences]

            # replace unk and pad token's embeddings with zeros
            zeroes_emb = np.zeros(self.token_emb_dim)
            sequence_embeddings = np.asarray([
                [self._token2emb.get(tok, zeroes_emb) for tok in seq] +
                [zeroes_emb] * (max_seq_length - len(seq))
                for seq in token_sequences
            ])

            if return_as_tensors:
                return {
                    "sequence_lengths": torch.as_tensor(sequence_lengths, dtype=torch.long),
                    "sequence_embeddings": torch.as_tensor(sequence_embeddings, dtype=torch.float32)
                }
            else:
                return {
                    "sequence_lengths": sequence_lengths,
                    "sequence_embeddings": sequence_embeddings
                }

        # define local constants
        pad_token_idx = getattr(self, "pad_token_idx")
        unk_token_idx = getattr(self, "unk_token_idx")
        start_token_idx = getattr(self, "start_token_idx")
        end_token_idx = getattr(self, "end_token_idx")

        # convert string tokens into ids
        input_ids = [self._token2idx.get(token, unk_token_idx) for token in self._tokenizer(text)
                     for text in texts]
        sequence_lengths = [len(seq) for seq in input_ids]

        # if max_seq_length is None, it is computed as max(length of all seqs from inputted text)
        #   add 2 for start and end tokens resp. if add_terminals is True
        max_seq_length = max_seq_length or self.max_seq_length
        observed_max_seq_length = max(sequence_lengths) + 2 if add_terminals else max(
            sequence_lengths)
        if max_seq_length:
            max_seq_length = min(max_seq_length, observed_max_seq_length)
        else:
            max_seq_length = observed_max_seq_length

        # batchify by truncating and/or padding
        for i in range(len(texts)):
            seq, seq_length = input_ids[i], sequence_lengths[i]
            new_seq = seq[:max_seq_length - 2] if add_terminals else seq[:max_seq_length]
            new_seq = [start_token_idx] + new_seq + [end_token_idx]
            new_seq_length = len(new_seq)
            input_ids[i] = [new_seq] + [pad_token_idx] * (max_seq_length - new_seq_length)
            sequence_lengths[i] = new_seq_length

        # create output dict
        if return_as_tensors:
            return {
                "sequence_ids": torch.tensor(input_ids, dtype=torch.long),
                "sequence_lengths": torch.tensor(sequence_lengths, dtype=torch.long)
            }
        else:
            return {
                "sequence_ids": np.array(input_ids),
                "sequence_lengths": np.array(sequence_lengths)
            }


class LabelsEncoderForTextModels(Encoder):
    """
    Encodes labels into ids
    """

    def __init__(self):
        self._token2idx = {}

    def _add_vocab(self, vocab: Iterable[str]):
        for token in vocab:
            if token not in self._token2idx:
                self._token2idx.update({token: len(self._token2idx)})

    def initialize_resources(self, labels):
        self._add_vocab(set(labels))
        self._idx2token = {idx: token for token, idx in self._token2idx.items()}
        return self

    @classmethod
    def load_resources(cls, load_folder) -> TextLabelsEncoder:
        if os.path.isfile(load_folder):
            msg = f"Path input to 'load' method must be a folder, " \
                  f"not a file ({load_folder})"
            raise ValueError(msg)

        encoder = cls()
        with open(os.path.join(load_folder, "text_label_vocab.txt"), "r") as fp:
            labels = [line.strip() for line in fp]
            for line in fp:
                encoder._add_vocab([line.strip()])
            fp.close()

    def dump_resources(self, dump_folder):
        if os.path.isfile(dump_folder):
            msg = f"Path input to 'dump' method must be a folder, " \
                  f"not a file ({dump_folder})"
            raise ValueError(msg)
        os.makedirs(dump_folder, exist_ok=True)
        with open(os.path.join(dump_folder, "text_label_vocab.txt"), "w") as fp:
            for token in self._token2idx:
                fp.write(token + "\n")
            fp.close()

    def encode(self, labels, return_as_tensors: bool = True):
        encoded_labels = [self._token2idx[label] for label in labels]
        if return_as_tensors:
            return torch.as_tensor(encoded_labels, dtype=torch.long)
        return encoded_labels

    def decode(self, encoded_labels: Union[List, Tensor]):
        if isinstance(encoded_labels, Tensor):
            encoded_labels = encoded_labels.numpy().tolist()
        return [self._idx2token[idx] for idx in encoded_labels]


class LabelsEncoderForTaggerModels(Encoder):
    """
    Encodes labels into ids at token level, additionally holds PAD token
    """
    raise NotImplementedError
