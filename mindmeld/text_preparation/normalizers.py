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
from abc import ABC, abstractmethod
import codecs
import logging
import re
import unicodedata
from ..constants import CURRENCY_SYMBOLS
from ..path import ASCII_FOLDING_DICT_PATH

logger = logging.getLogger(__name__)

ASCII_CUTOFF = ord("\u0080")


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

    def tojson(self):
        """
        Method defined to obtain recursive JSON representation of a TextPreparationPipeline.

        Args:
            None.

        Returns:
            JSON representation of Preprocessor (dict) .
        """
        return {self.__class__.__name__: None}


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
    """Unicode NFD Normalizer Class. (Canonical Decomposition)

    For more details: https://unicode.org/reports/tr15/#Norm_Forms
    """

    def __init__(self):
        """Creates a NFD Normalizer instance."""
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
    """Unicode NFC Normalizer Class.
    (Canonical Decomposition, followed by Canonical Composition)

    For more details: https://unicode.org/reports/tr15/#Norm_Forms
    """

    def __init__(self):
        """Creates a NFC Normalizer instance."""
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
    """Unicode NFKD Normalizer Class. (Compatibility Decomposition)

    For more details: https://unicode.org/reports/tr15/#Norm_Forms
    """

    def __init__(self):
        """Creates a NFKD Normalizer instance."""
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
    """Unicode NFKC Normalizer Class.
    (Compatibility Decomposition, followed by Canonical Composition)

    For more details: https://unicode.org/reports/tr15/#Norm_Forms
    """

    def __init__(self):
        """Creates a NFKC Normalizer instance."""
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


class Lowercase(Normalizer):
    """Lowercase Normalizer Class."""

    def normalize(self, text):
        """
        Args:
            text (str): Input text.
        Returns:
            normalized_text (str): Normalized Text.
        """
        return text.lower()


class RegexNormalizerRule(Normalizer):
    def __init__(self, pattern: str, replacement: str):
        """Creates a RegexNormalizerRule instance."""
        self.pattern = pattern
        self.replacement = replacement
        self._expr = re.compile(self.pattern)

    def normalize(self, s):
        return self._expr.sub(self.replacement, s)

    def tojson(self):
        return {self.__class__.__name__ + "##" + self.pattern + "##" + self.replacement: None}


class RegexNormalizerRuleFactory:
    # exception_chars is a class var so that updates are accessible throughout the application
    EXCEPTION_CHARS = r"\@\[\]'"

    @staticmethod
    def get_default_regex_normalizer_rule(regex_normalizer: str):
        """Creates a RegexNormalizerRule object based on the given rule and the current
        EXCEPTION_CHARS.

        Args:
            regex_normalizer (str): Name of the desired RegexNormalizerRule
        Returns:
            (RegexNormalizerRule): Default Regex Normalizer Rule
        """
        if regex_normalizer in DEFAULT_REGEX_NORM_RULES:
            regex_rule_dict = DEFAULT_REGEX_NORM_RULES[regex_normalizer]
            # Inserts current EXCEPTION_CHARS in pattern string if applicable
            regex_rule_dict["pattern"] = regex_rule_dict["pattern"].format(
                exception_chars=RegexNormalizerRuleFactory.EXCEPTION_CHARS
            )
            return RegexNormalizerRule(**regex_rule_dict)

    @staticmethod
    def get_regex_normalizers(regex_norm_rules):
        """A static method to get a RegexNormalizerRule from regex_norm_rules.

        Args:
            regex_norm_rules (List[Dict], optional): Regex normalization rules represented as
                dictionaries. The example rule below removes any text in parentheses.
                {
                    "pattern": "\(.+?\)",
                    "replacement": ""
                }
        Returns:
            regex_normalizer_rules (List[RegexNormalizerRule]): List of RegexNormalizerRule ojects
                created from the regex_norm_rules_provided.
        """
        return [
            RegexNormalizerRule(pattern=r["pattern"], replacement=r["replacement"])
            for r in regex_norm_rules
        ]


DEFAULT_REGEX_NORM_RULES = {
    "RemoveAposAtEndOfPossesiveForm": {
        "pattern": r"^'(?=\S)|(?<=\S)'$",
        "replacement": "",
    },
    "RemoveAdjacentAposAndSpace": {"pattern": r" '|' ", "replacement": ""},
    "RemoveBeginningSpace": {"pattern": r"^\s+", "replacement": ""},
    "RemoveTrailingSpace": {"pattern": r"\s+$", "replacement": ""},
    "ReplaceSpacesWithSpace": {"pattern": r"\s+", "replacement": " "},
    "ReplaceUnderscoreWithSpace": {"pattern": r"_", "replacement": " "},
    "SeparateAposS": {"pattern": r"(?<=[^\s])'[sS]", "replacement": " 's"},
    "ReplacePunctuationAtWordStartWithSpace": {
        "pattern": r"^[^\w\d&" + CURRENCY_SYMBOLS + "{exception_chars}" + r"]+",
        "replacement": " ",
    },
    "ReplacePunctuationAtWordEndWithSpace": {
        "pattern": r"[^\w\d&" + CURRENCY_SYMBOLS + "{exception_chars}" + r"]+$",
        "replacement": " ",
    },
    "ReplaceSpecialCharsBetweenLettersAndDigitsWithSpace": {
        "pattern": r"(?<=[^\W\d_])[^\w\d\s&" + "{exception_chars}" + r"]+(?=[\d]+)",
        "replacement": " ",
    },
    "ReplaceSpecialCharsBetweenDigitsAndLettersWithSpace": {
        "pattern": r"(?<=[\d])[^\w\d\s&" + "{exception_chars}" + r"]+(?=[^\W\d_]+)",
        "replacement": " ",
    },
    "ReplaceSpecialCharsBetweenLettersWithSpace": {
        "pattern": r"(?<=[^\W\d_])[^\w\d\s&" + "{exception_chars}" + r"]+(?=[^\W\d_]+)",
        "replacement": " ",
    },
}


class NormalizerFactory:
    """Normalizer Factory Class"""

    @staticmethod
    def get_normalizer(normalizer: str):
        """A static method to get a Normalizer

        Args:
            normalizer (str): Name of the desired Normalizer class
        Returns:
            (Normalizer): Normalizer Class
        """
        if normalizer in DEFAULT_REGEX_NORM_RULES:
            return RegexNormalizerRuleFactory.get_default_regex_normalizer_rule(
                normalizer
            )

        normalizer_classes = {
            NoOpNormalizer.__name__: NoOpNormalizer,
            ASCIIFold.__name__: ASCIIFold,
            NFC.__name__: NFC,
            NFD.__name__: NFD,
            NFKC.__name__: NFKC,
            NFKD.__name__: NFKD,
            Lowercase.__name__: Lowercase,
        }
        normalizer_class = normalizer_classes.get(normalizer)
        if not normalizer_class:
            raise TypeError(f"{normalizer} is not a valid Normalizer type.")
        return normalizer_class()
