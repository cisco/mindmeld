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
import logging
import os
import pickle

import numpy as np
from mindmeld.path import (
    PREVIOUSLY_USED_CHAR_EMBEDDINGS_FILE_PATH,
    PREVIOUSLY_USED_WORD_EMBEDDINGS_FILE_PATH,
)

from ..containers import GloVeEmbeddingsContainer

logger = logging.getLogger(__name__)

EMBEDDING_FILE_PATH_TEMPLATE = "glove.6B.{}d.txt"
ALLOWED_WORD_EMBEDDING_DIMENSIONS = [50, 100, 200, 300]


class WordSequenceEmbedding:
    """WordSequenceEmbedding encodes a sequence of words into a sequence of fixed
    dimension real-numbered vectors by mapping each word as a vector.
    """

    def __init__(
        self,
        sequence_padding_length,
        token_embedding_dimension=None,
        token_pretrained_embedding_filepath=None,
        use_padding=True,
    ):
        """Initializes the WordSequenceEmbedding class

        Args:
            sequence_padding_length (int): padding length of the sequence after which
            the sequence is cut off
            token_embedding_dimension (int): The embedding dimension of the token
            token_pretrained_embedding_filepath (str): The embedding filepath to
            extract the embeddings from.
        """
        self.token_embedding_dimension = token_embedding_dimension
        self.sequence_padding_length = sequence_padding_length

        self.token_to_embedding_mapping = GloVeEmbeddingsContainer(
            token_embedding_dimension, token_pretrained_embedding_filepath
        ).get_pretrained_word_to_embeddings_dict()

        self._add_historic_embeddings()

        self.use_padding = use_padding

    def encode_sequence_of_tokens(self, token_sequence):
        """Encodes a sequence of tokens into real value vectors.

        Args:
            token_sequence (list): A sequence of tokens.

        Returns:
            (list): Encoded sequence of tokens.
        """
        default_encoding = np.zeros(self.token_embedding_dimension)
        if self.use_padding:
            encoded_query = [default_encoding] * self.sequence_padding_length
        else:
            encoded_query = [default_encoding] * len(token_sequence)

        for idx, token in enumerate(token_sequence):
            if idx >= self.sequence_padding_length and self.use_padding:
                break
            encoded_query[idx] = self._encode_token(token)

        return encoded_query

    def _encode_token(self, token):
        """Encodes a token to its corresponding embedding

        Args:
            token (str): Individual token

        Returns:
            corresponding embedding
        """
        if token not in self.token_to_embedding_mapping:
            random_vector = np.random.uniform(
                -1, 1, size=(self.token_embedding_dimension,)
            )
            self.token_to_embedding_mapping[token] = random_vector
        return self.token_to_embedding_mapping[token]

    def _add_historic_embeddings(self):
        historic_word_embeddings = {}

        # load historic word embeddings
        if os.path.exists(PREVIOUSLY_USED_WORD_EMBEDDINGS_FILE_PATH):
            pkl_file = open(PREVIOUSLY_USED_WORD_EMBEDDINGS_FILE_PATH, "rb")
            historic_word_embeddings = pickle.load(pkl_file)
            pkl_file.close()

        for word in historic_word_embeddings:
            if len(historic_word_embeddings[word]) == self.token_embedding_dimension:
                self.token_to_embedding_mapping[word] = historic_word_embeddings.get(
                    word
                )

    def save_embeddings(self):
        """Save extracted embeddings to historic pickle file."""
        output = open(PREVIOUSLY_USED_WORD_EMBEDDINGS_FILE_PATH, "wb")
        pickle.dump(self.token_to_embedding_mapping, output)
        output.close()


class CharacterSequenceEmbedding:
    """CharacterSequenceEmbedding encodes a sequence of words into a sequence of fixed
    dimension real-numbered vectors by mapping each character in the words as vectors.
    """

    def __init__(
        self,
        sequence_padding_length,
        token_embedding_dimension=None,
        max_char_per_word=None,
    ):
        """Initializes the CharacterSequenceEmbedding class

        Args:
            sequence_padding_length (int): padding length of the sequence after which
            the sequence is cut off
            token_embedding_dimension (int): The embedding dimension of the token
            max_char_per_word (int): The maximum number of characters per word
        """
        self.token_embedding_dimension = token_embedding_dimension
        self.sequence_padding_length = sequence_padding_length
        self.max_char_per_word = max_char_per_word
        self.token_to_embedding_mapping = {}
        self._add_historic_embeddings()

    def encode_sequence_of_tokens(self, token_sequence):
        """Encodes a sequence of tokens into real value vectors.

        Args:
            token_sequence (list): A sequence of tokens.

        Returns:
            (list): Encoded sequence of tokens.
        """
        default_encoding = np.zeros(self.token_embedding_dimension)
        default_char_word = [default_encoding] * self.max_char_per_word
        encoded_query = [default_char_word] * self.sequence_padding_length

        for idx, word_token in enumerate(token_sequence):
            if idx >= self.sequence_padding_length:
                break

            encoded_word = [default_encoding] * self.max_char_per_word
            for idx2, char_token in enumerate(word_token):
                if idx2 >= self.max_char_per_word:
                    break

                self._encode_token(char_token)
                encoded_word[idx2] = self.token_to_embedding_mapping[char_token]

            encoded_query[idx] = encoded_word
        return encoded_query

    def _encode_token(self, token):
        """Encodes a token to its corresponding embedding

        Args:
            token (str): Individual token

        Returns:
            corresponding embedding
        """
        if token not in self.token_to_embedding_mapping:
            random_vector = np.random.uniform(
                -1, 1, size=(self.token_embedding_dimension,)
            )
            self.token_to_embedding_mapping[token] = random_vector
        return self.token_to_embedding_mapping[token]

    def _add_historic_embeddings(self):
        historic_char_embeddings = {}

        # load historic word embeddings
        if os.path.exists(PREVIOUSLY_USED_CHAR_EMBEDDINGS_FILE_PATH):
            pkl_file = open(PREVIOUSLY_USED_CHAR_EMBEDDINGS_FILE_PATH, "rb")
            historic_char_embeddings = pickle.load(pkl_file)
            pkl_file.close()

        for char in historic_char_embeddings:
            self.token_to_embedding_mapping[char] = historic_char_embeddings.get(char)

    def save_embeddings(self):
        """Save extracted embeddings to historic pickle file."""
        output = open(PREVIOUSLY_USED_CHAR_EMBEDDINGS_FILE_PATH, "wb")
        pickle.dump(self.token_to_embedding_mapping, output)
        output.close()
