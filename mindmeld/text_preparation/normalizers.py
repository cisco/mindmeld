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

"""This module contains Normalizers."""
import unicodedata
import codecs
from abc import ABC, abstractmethod
import logging

from ..constants import ASCII_CUTOFF
from ..path import ASCII_FOLDING_DICT_PATH

logger = logging.getLogger(__name__)


class Normalizer(ABC):
    """Abstract Normalizer Base Class."""

    def __init__(self):
        """Creates a Normalizer instance."""

    @abstractmethod
    def normalize(self, text):
        """
        Args:
            text (str): Input text.
        Returns:
            normalized_text (str): Normalized Text.
        """
        raise NotImplementedError("Subclasses must implement this method")


class NoOpNormalizer(Normalizer):
    """A No-Ops Normalizer."""

    def normalize(self, text):
        """
        Args:
            text (str): Input text.
        Returns:
            normalized_text (str): Returns the original text.
        """
        return text


class ASCIIFold(Normalizer):
    """An ASCII Folding Normalizer."""

    def __init__(self):
        super().__init__()
        self.ascii_folding_table = self.load_ascii_folding_table()

    def normalize(self, text):
        """
        Args:
            text (str): Input text.
        Returns:
            normalized_text (str): Normalized Text.
        """
        return self.fold_str_to_ascii(text)

    def fold_char_to_ascii(self, char):
        """
        Return the ASCII character corresponding to the folding token.

        Args:
            char: ASCII folding token

        Returns:
            char: a ASCII character
        """
        char_ord = ord(char)
        if char_ord < ASCII_CUTOFF:
            return char

        try:
            return self.ascii_folding_table[char_ord]
        except KeyError:
            return char

    def fold_str_to_ascii(self, text):
        """
        Return the ASCII character corresponding to the folding token string.

        Args:
            str: ASCII folding token string

        Returns:
            char: a ASCII character
        """
        folded_str = ""
        for char in text:
            folded_str += self.fold_char_to_ascii(char)

        return folded_str

    @staticmethod
    def load_ascii_folding_table():
        """
        Load mapping of ascii code points to ascii characters.
        """
        logger.debug(
            "Loading ascii folding mapping from file: %s.", ASCII_FOLDING_DICT_PATH
        )
        ascii_folding_table = {}
        with codecs.open(
            ASCII_FOLDING_DICT_PATH, "r", encoding="unicode_escape"
        ) as mapping_file:
            for line in mapping_file:
                codepoint, ascii_char = line.split()
                ascii_folding_table[ord(codepoint)] = ascii_char

        return ascii_folding_table


class NFD(Normalizer):
    """Unicode NFD Normalizer Class."""

    def __init__(self):
        """Creates a NFDNormalizer instance."""
        super().__init__()
        self.normalization_type = "NFD"

    def normalize(self, text):
        """
        Args:
            text (str): Input text.
        Returns:
            normalized_text (str): Normalized Text.
        """
        return unicodedata.normalize(self.normalization_type, text)


class NFC(Normalizer):
    """Unicode NFC Normalizer Class."""

    def __init__(self):
        """Creates a NFCNormalizer instance."""
        super().__init__()
        self.normalization_type = "NFC"

    def normalize(self, text):
        """
        Args:
            text (str): Input text.
        Returns:
            normalized_text (str): Normalized Text.
        """
        return unicodedata.normalize(self.normalization_type, text)


class NFKD(Normalizer):
    """Unicode NFKD Normalizer Class."""

    def __init__(self):
        """Creates a NFKDNormalizer instance."""
        super().__init__()
        self.normalization_type = "NFKD"

    def normalize(self, text):
        """
        Args:
            text (str): Input text.
        Returns:
            normalized_text (str): Normalized Text.
        """
        return unicodedata.normalize(self.normalization_type, text)


class NFKC(Normalizer):
    """Unicode NFKC Normalizer Class."""

    def __init__(self):
        """Creates a NFKCNormalizer instance."""
        super().__init__()
        self.normalization_type = "NFKC"

    def normalize(self, text):
        """
        Args:
            text (str): Input text.
        Returns:
            normalized_text (str): Normalized Text.
        """
        return unicodedata.normalize(self.normalization_type, text)


class NormalizerFactory:
    """Normalizer Factory Class"""

    @staticmethod
    def get_normalizer(normalizer):
        """A static method to get a Normalizer

        Args:
            normalizer (str): Name of the desired Normalizer class
        Returns:
            (Normalizer): Normalizer Class
        """
        if normalizer == NoOpNormalizer.__name__:
            return NoOpNormalizer()
        elif normalizer == ASCIIFold.__name__:
            return ASCIIFold()
        elif normalizer == NFC.__name__:
            return NFC()
        elif normalizer == NFD.__name__:
            return NFD()
        elif normalizer == NFKC.__name__:
            return NFKC()
        elif normalizer == NFKD.__name__:
            return NFKD()
        raise AssertionError(f" {normalizer} is not a valid Normalizer.")
