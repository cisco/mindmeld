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
from ..constants import UNICODE_NON_LATIN_CATEGORY, UNICODE_SPACE_CATEGORY

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
        """Initialize the NoOpTokenizer."""
        pass

    def tokenize(self, text):
        """
        Args:
            text (str): Input text.
        Returns:
            tokens (List[str]): List of tokens.
        """
        return [text]


class LetterTokenizer(Tokenizer):
    """A Tokenizer that splits text at the character level."""

    def __init__(self):
        """Initializes the LetterTokenizer."""
        pass

    def tokenize(self, text):
        """
        Identify tokens in text and create normalized tokens that contain the text and start index.
        Args:
            text (str): The text to normalize
        Returns:
            tokens (List[Dict]): List of tokenized tokens which a represented as dictionaries.
                Keys include "start" (token starting index), "end" (token ending index), and
                "text" (token text). For example: [{"start": 0, "text":"hello", "end":4}]
        """
        token_num_by_char = LetterTokenizer.get_token_num_by_char(text)
        return LetterTokenizer.create_tokens(text, token_num_by_char)

    @staticmethod
    def get_token_num_by_char(text):
        """ Determine the token number for each character.
        More details about unicode categories can be found here:
        https://www.compart.com/en/unicode/category.

        Args:
            text (str): The text to process and get actions per character.
        Returns:
            token_num_by_char (List[str]): Token number that each character belongs to.
                Spaces are represented as None. For example: [1,2,2,3,None,4,None,5,5,5]
        """
        category_by_char = [unicodedata.category(x) for x in text]

        token_num_by_char = []
        token_num = 0
        for index, category in enumerate(category_by_char):
            same_category_as_previous = (
                index > 0 and category[0] == category_by_char[index - 1][0]
            )
            if category == UNICODE_SPACE_CATEGORY:
                token_num_by_char.append(None)
                continue
            if (
                category == UNICODE_NON_LATIN_CATEGORY or not same_category_as_previous
            ):
                token_num += 1
            token_num_by_char.append(token_num)
        return token_num_by_char

    @staticmethod
    def create_tokens(text, token_num_by_char):
        """
        Generate token dictionaries from the original text and the token numbers by character.
        Args:
            text (str): The text to normalize
            token_num_by_char (List[str]): Token number that each character belongs to.
                Spaces are represented as None. For example: [1,2,2,3,None,4,None,5,5,5]
        Returns:
            tokens (List[Dict]): List of tokenized tokens which a represented as dictionaries.
                Keys include "start" (token starting index), "end" (token ending index), and
                "text" (token text). For example: [{"start": 0, "text":"hello", "end":4}]
        """
        tokens = []
        token_text = ""
        for index, token_num in enumerate(token_num_by_char):
            if not token_num:
                continue
            if not token_text:
                start = index
            token_text += text[index]
            is_last_char = index == len(token_num_by_char) - 1
            # Close off entity if char is the last or if next char is a different token number
            if is_last_char or (
                not is_last_char and token_num != token_num_by_char[index + 1]
            ):
                tokens.append({"start": start, "text": token_text})
                token_text = ""
        return tokens


class WhiteSpaceTokenizer(Tokenizer):
    """A Tokenizer that splits text at spaces."""

    def __init__(self):
        """Initializes the WhiteSpaceTokenizer."""
        pass

    def tokenize(self, text):
        """
        Identify tokens in text and create normalized tokens that contain the text and start index.
        Args:
            text (str): The text to normalize
        Returns:
            tokens (List[str]): List of tokenized tokens.
        """
        tokens = []
        token = {}
        token_text = ""
        # Space added at the end of text to close off the last token
        for i, char in enumerate(text + " "):
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
        spacy_tokens = [token.text for token in self.spacy_model(text)]

        start_index = 0
        tokens = []
        for token_text in spacy_tokens:
            token = {"start": start_index, "text": token_text}
            tokens.append(token)
            start_index += len(token_text)
        return tokens


class TokenizerFactory:
    """Tokenizer Factory Class"""

    @staticmethod
    def get_tokenizer(tokenizer, language=None, spacy_model_size="sm"):
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
        elif tokenizer == LetterTokenizer.__name__:
            return LetterTokenizer()
        elif tokenizer == WhiteSpaceTokenizer.__name__:
            return WhiteSpaceTokenizer()
        elif tokenizer == SpacyTokenizer.__name__:
            return SpacyTokenizer(language, spacy_model_size)
        raise AssertionError(f" {tokenizer} is not a valid Tokenizer.")
