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
from ..components._config import ENGLISH_LANGUAGE_CODE
from ..constants import (
    UNICODE_NON_LATIN_CATEGORY,
    UNICODE_SPACE_CATEGORY,
)

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

    def tojson(self):
        """
        Method defined to obtain recursive JSON representation of a TextPreparationPipeline.

        Args:
            None.

        Returns:
            JSON representation of TextPreparationPipeline (dict) .
        """
        return {self.__class__.__name__: None}


class NoOpTokenizer(Tokenizer):
    """A No-Ops tokenizer."""

    def __init__(self):
        """Initialize the NoOpTokenizer."""
        pass

    def tokenize(self, text):
        """Returns the original text as a list.
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
        Split characters into separate tokens while skipping spaces.
        Args:
            text (str): the text to tokenize
        Returns:
            tokens (List[Dict]): List of tokenized tokens which a represented as dictionaries.
                Keys include "start" (token starting index), and "text" (token text).
                For example: [{"start": 0, "text":"hello"}]
        """
        if text == "":
            return []
        tokens = []
        for idx, char in enumerate(text):
            if not char.isspace():
                tokens.append({"start": idx, "text": char})
        return tokens


class LetterTokenizer(Tokenizer):
    """A Tokenizer that splits text into a separate token if the character proceeds a space, is a
    non-latin character, or is a different unicode category than the previous character.
    """

    def __init__(self):
        """Initializes the LetterTokenizer."""
        pass

    def tokenize(self, text):
        """
        Identify tokens in text and create normalized tokens that contain the text and start index.
        Args:
            text (str): the text to tokenize
        Returns:
            tokens (List[Dict]): List of tokenized tokens which a represented as dictionaries.
                Keys include "start" (token starting index), and "text" (token text).
                For example: [{"start": 0, "text":"hello"}]
        """
        if text == "":
            return []
        token_num_by_char = LetterTokenizer.get_token_num_by_char(text)
        return LetterTokenizer.create_tokens(text, token_num_by_char)

    @staticmethod
    def get_token_num_by_char(text):
        """Determine the token number for each character.

        More details about unicode categories can be found here:
        http://www.unicode.org/reports/tr44/#General_Category_Values.
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

            if category == UNICODE_SPACE_CATEGORY:
                token_num_by_char.append(None)
                continue

            prev_category = category_by_char[index - 1] if index > 0 else None

            # General Category is represented by the first letter of a Unicode category.
            same_general_category = (
                category[0] == (prev_category[0] if prev_category else None)
            )

            if UNICODE_NON_LATIN_CATEGORY in (category, prev_category) or not same_general_category:
                token_num += 1

            token_num_by_char.append(token_num)
        return token_num_by_char

    @staticmethod
    def create_tokens(text, token_num_by_char):
        """
        Generate token dictionaries from the original text and the token numbers by character.
        Args:
            text (str): the text to tokenize
            token_num_by_char (List[str]): Token number that each character belongs to.
                Spaces are represented as None. For example: [1,2,2,3,None,4,None,5,5,5]
        Returns:
            tokens (List[Dict]): List of tokenized tokens which a represented as dictionaries.
                Keys include "start" (token starting index), and "text" (token text).
                For example: [{"start": 0, "text":"hello"}]
        """
        if text == "":
            return []
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
        Identify tokens in text and token dictionaries that contain the text and start index.
        Args:
            text (str): the text to tokenize
        Returns:
            tokens (List[Dict]): List of tokenized tokens which a represented as dictionaries.
                Keys include "start" (token starting index), and "text" (token text).
                For example: [{"start": 0, "text":"hello"}]
        """
        if text == "":
            return []
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
    """A Tokenizer that uses Spacy to split text into tokens."""

    def __init__(self, language, spacy_model_size="sm"):
        """Initializes a SpacyTokenizer.

        Args:
            language (str, optional): Language as specified using a 639-1/2 code.
            spacy_model_size (str, optional): Size of the Spacy model to use. ("sm", "md", or "lg")
        """
        self.spacy_model = SpacyModelFactory.get_spacy_language_model(
            language, spacy_model_size, disable=["tagger", "parser", "ner", "attribute_ruler", "lemmatizer"]
        )
        assert self.spacy_model.pipeline == []

    def tokenize(self, text):
        """
        Args:
            text (str): Input text.
        Returns:
            tokens (List[Dict]): List of tokenized tokens which a represented as dictionaries.
                Keys include "start" (token starting index), and "text" (token text).
                For example: [{"start": 0, "text":"hello"}]
        """
        if text == "":
            return []
        spacy_tokens = [(token.text, token.idx) for token in self.spacy_model(text)]
        tokens = []
        for token_text, token_idx in spacy_tokens:
            token = {"start": token_idx, "text": token_text}
            tokens.append(token)
        return tokens


class TokenizerFactory:
    """Tokenizer Factory Class"""

    @staticmethod
    def get_tokenizer(
        tokenizer: str, language=ENGLISH_LANGUAGE_CODE, spacy_model_size="sm"
    ):
        """A static method to get a tokenizer

        Args:
            tokenizer (str): Name of the desired tokenizer class
            language (str, optional): Language as specified using a 639-1/2 code.
            spacy_model_size (str, optional): Size of the Spacy model to use. ("sm", "md", or "lg")

        Returns:
            (Tokenizer): Tokenizer Class
        """
        tokenizer_classes = {
            NoOpTokenizer.__name__: NoOpTokenizer,
            CharacterTokenizer.__name__: CharacterTokenizer,
            LetterTokenizer.__name__: LetterTokenizer,
            WhiteSpaceTokenizer.__name__: WhiteSpaceTokenizer,
            SpacyTokenizer.__name__: lambda: SpacyTokenizer(language, spacy_model_size),
        }
        tokenizer_class = tokenizer_classes.get(tokenizer)
        if not tokenizer_class:
            raise TypeError(f"{tokenizer} is not a valid Tokenizer type.")
        return tokenizer_class()

    @staticmethod
    def get_default_tokenizer():
        """Creates the default tokenizer (WhiteSpaceTokenizer) irrespective of the language of the current application.

        Args:
            language (str, optional): Language as specified using a 639-1/2 code.

        Returns:
            (Tokenizer): Tokenizer Class
        """
        return WhiteSpaceTokenizer()
