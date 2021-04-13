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

"""This module contains Tokenizers."""

from abc import ABC, abstractmethod
import logging
import unicodedata

from .spacy_model_factory import SpacyModelFactory

logger = logging.getLogger(__name__)


class Tokenizer(ABC):
    """Abstract Tokenizer Base Class."""

    def __init__(self):
        """Creates a tokenizer instance."""
        pass

    @abstractmethod
    def tokenize(self, text):
        """
        Args:
            text (str): Input text.
        Returns:
            tokens (List[str]): List of tokens.
        """
        raise NotImplementedError("Subclasses must implement this method")


class NoOpTokenizer(Tokenizer):
    """A No-Ops tokenizer."""

    def __init__(self):
        pass

    def tokenize(self, text):
        """
        Args:
            text (str): Input text.
        Returns:
            tokens (List[str]): List of tokens.
        """
        return [text]


class CharacterTokenizer(Tokenizer):
    """A Tokenizer that splits text at the character level."""

    def __init__(self):
        """Initializes the CharacterTokenizer."""
        pass

    def tokenize(self, text):
        """
        Identify tokens in text and create normalized tokens that contain the text and start index.
        Args:
            text (str): The text to normalize
        Returns:
            list: A list of normalized tokens
        """
        tokens = []
        token = {}
        token_text = ""
        for i, char in enumerate(text):
            cat = unicodedata.category(char)
            if char.isspace():
                # if we hit a space, close the token and empty the buffer
                if token and token_text:
                    token["text"] = token_text
                    tokens.append(token)
                token = {}
                token_text = ""
                continue
            if (
                token_text and cat[0] != unicodedata.category(token_text[-1])[0]
            ) or cat == "Lo":
                # if we hit a non-Latin character, close the token and restart buffer
                if token and token_text:
                    token["text"] = token_text
                    tokens.append(token)
                token = {"start": i}
                token_text = char
                continue
            if not token_text:
                token = {"start": i}
            token_text += char

        if token and token_text:
            token["text"] = token_text
            tokens.append(token)

        return tokens


class SpaceTokenizer(Tokenizer):
    """A Tokenizer that splits text at spaces."""

    def __init__(self):
        """Initializes the SpaceTokenizer."""
        pass

    def tokenize(self, text):
        """
        Identify tokens in text and create normalized tokens that contain the text and start index.
        Args:
            text (str): The text to normalize
        Returns:
            list: A list of normalized tokens
        """
        tokens = []
        token = {}
        token_text = ""
        for i, char in enumerate(text):
            if char.isspace():
                if token and token_text:
                    token["text"] = token_text
                    tokens.append(token)
                token = {}
                token_text = ""
                continue
            if not token_text:
                token = {"start": i}
            token_text += char

        if token and token_text:
            token["text"] = token_text
            tokens.append(token)

        return tokens


class SpacyTokenizer(Tokenizer):
    """A Tokenizer that splits text at spaces."""

    def __init__(self, language, spacy_model_size):
        """Initializes a SpacyTokenizer."""
        self.spacy_model = SpacyModelFactory.get_spacy_language_model(
            language, spacy_model_size
        )

    def tokenize(self, text):
        """
        Args:
            text (str): Input text.
        Returns:
            tokens (List[str]): List of tokens created using a Spacy Language Model.
        """
        return [token.text for token in self.spacy_model(text)]

    # TODO: Add the start index of each token and return dictionaries instead of token list.
    # Ex: [{'start': 0, 'text': 'I'}, {'start': 2, 'text': 'visited'}, {'start': 10, 'text': 'the'}..]


class TokenizerFactory:
    """Tokenizer Factory Class"""

    @staticmethod
    def get_tokenizer(tokenizer, language=None, spacy_model_size=None):
        """A static method to get a tokenizer

        Args:
            tokenizer (str): Name of the desired tokenizer class
            language (str, optional): Language as specified using a 639-1/2 code.
            spacy_model_size (str, optional): Size of the Spacy model to use. ("sm", "md", or "lg")

        Returns:
            (Tokenizer): Tokenizer Class
        """
        if tokenizer == NoOpTokenizer.__name__:
            return NoOpTokenizer()
        elif tokenizer == CharacterTokenizer.__name__:
            return CharacterTokenizer()
        elif tokenizer == SpaceTokenizer.__name__:
            return SpaceTokenizer()
        elif tokenizer == SpacyTokenizer.__name__:
            return SpacyTokenizer(language, spacy_model_size)
        raise AssertionError(f" {tokenizer} is not a valid Tokenizer.")
