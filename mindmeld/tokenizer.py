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

"""This module contains the tokenizer."""

import logging
import re
import sre_constants

from .text_preparation.tokenizers import TokenizerFactory
from .text_preparation.normalizers import NormalizerFactory, NoOpNormalizer
from .components._config import get_tokenizer_config, get_language_config
from .constants import CURRENCY_SYMBOLS

logger = logging.getLogger(__name__)


class Tokenizer:
    """The Tokenizer class encapsulates all the functionality for normalizing and tokenizing a
    given piece of text."""

    def __init__(self, app_path=None, exclude_from_norm=None):
        """Initializes the tokenizer.

        Args:
            exclude_from_norm (optional) - list of chars to exclude from normalization
        """
        self.app_path = app_path
        self.exclude_from_norm = exclude_from_norm or []
        self.config = get_tokenizer_config(app_path, self.exclude_from_norm)
        self.language, _ = get_language_config(app_path)
        self._tokenizer = self._get_tokenizer()
        self._normalizer = self._get_normalizer()
        self._custom = False
        self._init_regex()

    def _get_tokenizer(self):
        """Gets the 'tokenizer' specified in the config.

        Returns:
            tokenizer (Tokenizer): Tokenizer specified in the config.
        """
        tokenizer = self.config.get("tokenizer")
        if not tokenizer:
            raise AssertionError("'tokenizer' cannot be None.")
        return TokenizerFactory().get_tokenizer(tokenizer, self.language)

    def _get_normalizer(self):
        """Gets the 'normalizer' specified in the config.

        Returns:
            normalizer (Normalizer): Normalizer specified in the config.
        """
        normalizer = self.config.get("normalizer")
        if not normalizer:
            logger.warning("'normalizer' not found in the config.")
            return NoOpNormalizer()
        return NormalizerFactory().get_normalizer(normalizer)

    def _init_regex(self):
        """
        Initialize the regex for matching and tokenizing text.
        """
        # List of regex's for matching and tokenizing when keep_special_chars=False
        regex_list = []

        letter_pattern_str = "[^\W\d_]+"  # noqa: W605

        to_exclude = CURRENCY_SYMBOLS + "".join(self.exclude_from_norm)

        # Make regex list
        regex_list.append("?P<start>^[^\w\d&" + to_exclude + "]+")  # noqa: W605
        regex_list.append("?P<end>[^\w\d&" + to_exclude + "]+$")  # noqa: W605
        regex_list.append(
            "?P<pattern1>(?P<pattern1_replace>"
            + letter_pattern_str
            + ")"  # noqa: W605
            + "[^\w\d\s&]+(?=[\d]+)"  # noqa: W605
        )
        regex_list.append(
            "?P<pattern2>(?P<pattern2_replace>[\d]+)[^\w\d\s&]+(?="  # noqa: W605
            + letter_pattern_str
            + ")"
        )
        regex_list.append(
            "?P<pattern3>(?P<pattern3_replace>"
            + letter_pattern_str
            + ")"
            + "[^\w\d\s&]+(?="  # noqa: W605
            + letter_pattern_str
            + ")"
        )

        regex_list.append("?P<underscore>_")  # noqa: W605
        regex_list.append("?P<begspace>^\s+")  # noqa: W605
        regex_list.append("?P<trailspace>\s+$")  # noqa: W605
        regex_list.append("?P<spaceplus>\s+")  # noqa: W605
        regex_list.append("?P<apos_space> '|' ")  # noqa: W605
        regex_list.append("?P<apos_s>(?<=[^\\s])'[sS]")  # noqa: W605
        # handle the apostrophes used at the end of a possessive form, e.g. dennis'
        regex_list.append("?P<apos_poss>(^'(?=\S)|(?<=\S)'$)")  # noqa: W605

        # Replace lookup based on regex
        self.replace_lookup = {
            "apos_s": (" 's", None),
            "apos_poss": ("", None),
            "apos_space": (" ", None),
            "begspace": ("", None),
            "end": (" ", None),
            "escape1": ("{0}", "escape1_replace"),
            "escape2": ("{0} ", "escape2_replace"),
            "pattern1": ("{0} ", "pattern1_replace"),
            "pattern2": ("{0} ", "pattern2_replace"),
            "pattern3": ("{0} ", "pattern3_replace"),
            "spaceplus": (" ", None),
            "start": (" ", None),
            "trailspace": ("", None),
            "underscore": (" ", None),
            "apostrophe": (" ", None),
        }

        # Check if custom pattern is being used or MM defined
        if self.config.get("allowed_patterns"):
            self._custom = True

        # Create compiled regex expressions
        combined_re = ")|(".join(
            self.config["allowed_patterns"] or self.config["default_allowed_patterns"]
        )

        try:
            self.keep_special_compiled = re.compile(
                "(%s)" % (combined_re,),
                re.UNICODE,
            )
        except sre_constants.error:
            logger.error(
                "Regex compilation failed for the following patterns: %s",
                combined_re,
            )

        self.compiled = re.compile("(%s)" % ")|(".join(regex_list), re.UNICODE)

    # Needed for train-roles where queries are deep copied (and thus tokenizer).
    # Pre compiled patterns don't deepcopy natively. Bug introduced past python 2.5
    # TODO investigate necessity of deepcopy in train-roles
    def __deepcopy__(self, memo):
        # TODO: optimize this
        return Tokenizer(
            app_path=self.app_path, exclude_from_norm=self.exclude_from_norm
        )

    def _one_xlat(self, match_object):
        """
        Helper function for for multiple replace. Takes match object and looks up replacement.

        Args:
            match_object: A regex match object

        Returns:
            str: A string with punctuation replaced/removed
        """
        replace_str, format_str = self.replace_lookup[match_object.lastgroup]
        if format_str:
            replace_str = replace_str.format(match_object.groupdict()[format_str])
        return replace_str

    def multiple_replace(self, text, compiled):
        """
        Takes text and compiled regex pattern, does lookup for multi rematch.

        Args:
            text (str): The text to perform matching on
            compiled: A compiled regex object that can be used for matching

        Returns:
            str: The text with replacement specified by self.replace_lookup
        """
        # For each match, look-up corresponding value in dictionary
        try:

            # Checks if replacement can be found in pre-defined match object (non-custom).
            # If no key in match object, go to custom tokenizer handling in Exception.
            filtered = compiled.sub(self._one_xlat, text)

            # If no key error and custom tokenizer was involved
            # then the token has unwanted special characters. Remove them and return.
            if self._custom:
                return self.compiled.sub(self._one_xlat, text)

            # Return filtered list if non-custom tokenizer.
            return filtered

        except KeyError:
            # In case of custom/app-specific tokenizer configuration
            logger.info("Using custom tokenizer configuration.")
            re_str = compiled.findall(text)

            # For the custom regex pattern, the following first filters the list of matches to
            # only keep the non-NULL matches. The filtered object is converted to a list and the
            # first matching object is selected.
            return "".join([list(filter(None, e))[0] for e in re_str])

    def normalize(self, text, keep_special_chars=True):
        """
        Normalize a given text string and return the string with each token normalized.

        Args:
            text (str): The text to normalize
            keep_special_chars (bool): If True, the tokenizer excludes a list of special
                characters used in annotations

        Returns:
            str: the original text string with each token in normalized form
        """
        norm_tokens = self.tokenize(text, keep_special_chars)
        normalized_text = " ".join(t["entity"] for t in norm_tokens)

        return normalized_text

    def tokenize(self, text, keep_special_chars=True):
        """Tokenizes the input text, normalizes the token text, and returns normalized tokens.

        Currently it does the following during normalization:
        1. remove leading special characters except dollar sign and ampersand
        2. remove trailing special characters except ampersand
        3. remove special characters except ampersand when the preceding character is a letter and
        the following characters is a number
        4. remove special characters except ampersand when the preceding character is a number and
        the following character is a letter
        5. remove special characters except ampersand when both preceding and following characters
        are letters
        6. remove special character except ampersand when the following character is '|'
        7. remove diacritics and replace it with equivalent ascii character when possible

        Note that the tokenizer also excludes a list of special characters used in annotations when
        the flag keep_special_chars is set to True

        Args:
            text (str): The text to normalize
            keep_special_chars (bool): If True, the tokenizer excludes a list of special
                characters used in annotations

        Returns:
            list: A list of normalized tokens
        """
        raw_tokens = self._tokenizer.tokenize(text)
        norm_tokens = []
        for i, raw_token in enumerate(raw_tokens):
            if not raw_token["text"] or len(raw_token["text"]) == 0:
                continue

            norm_token_start = len(norm_tokens)
            norm_token_text = raw_token["text"]

            if keep_special_chars:
                regex_norm_token_text = self.multiple_replace(
                    norm_token_text, self.keep_special_compiled
                )
            else:
                regex_norm_token_text = self.multiple_replace(norm_token_text, self.compiled)

            # TODO: Move normalization out of the Tokenizer (Phase 2)
            final_norm_token_text = []
            for c in regex_norm_token_text:
                c = c if c in self.exclude_from_norm else self._normalizer.normalize(c)
                final_norm_token_text.append(c)

            # TODO: Make it optional to take the lowercase of text (Phase 2)
            final_norm_token_text = "".join(final_norm_token_text)
            final_norm_token_text = final_norm_token_text.lower()

            norm_token_count = 0
            if len(final_norm_token_text) > 0:
                # remove diacritics and fold the character to equivalent ascii character if possible
                for token in final_norm_token_text.split():
                    norm_token = {}
                    norm_token["entity"] = token
                    norm_token["raw_entity"] = raw_token["text"]
                    norm_token["raw_token_index"] = i
                    norm_token["raw_start"] = raw_token["start"]
                    norm_tokens.append(norm_token)
                    norm_token_count += 1

            raw_token["norm_token_start"] = norm_token_start
            raw_token["norm_token_count"] = norm_token_count
        return norm_tokens

    def get_char_index_map(self, raw_text, normalized_text):  # pylint: disable=R0201
        """
        Generates character index mapping from normalized query to raw query. The entity model
        always operates on normalized query during NLP processing but for entity output we need
        to generate indexes based on raw query.

        The mapping is generated by calculating edit distance and backtracking to get the
        proper alignment.

        Args:
            raw_text (str): Raw query text.
            normalized_text (str): Normalized query text.
        Returns:
            dict: A mapping of character indexes from normalized query to raw query.
        """

        text = raw_text.lower()

        m = len(raw_text)
        n = len(normalized_text)

        # handle case where normalized text is the empty string
        if n == 0:
            raw_to_norm_mapping = {i: 0 for i in range(m)}
            return raw_to_norm_mapping, {0: 0}

        # handle case where normalized text and raw text are identical
        if m == n and raw_text == normalized_text:
            mapping = {i: i for i in range(n)}
            return mapping, mapping

        edit_dis = []
        for i in range(0, n + 1):
            edit_dis.append([0] * (m + 1))
        edit_dis[0] = list(range(0, m + 1))
        for i in range(0, n + 1):
            edit_dis[i][0] = i

        directions = []
        for i in range(0, n + 1):
            directions.append([""] * (m + 1))

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dis = 999
                direction = None

                diag_dis = edit_dis[i - 1][j - 1]
                if normalized_text[i - 1] != text[j - 1]:
                    diag_dis += 1

                # dis from going down
                down_dis = edit_dis[i - 1][j] + 1

                # dis from going right
                right_dis = edit_dis[i][j - 1] + 1

                if down_dis < dis:
                    dis = down_dis
                    direction = "↓"
                if right_dis < dis:
                    dis = right_dis
                    direction = "→"
                if diag_dis < dis:
                    dis = diag_dis
                    direction = "↘"

                edit_dis[i][j] = dis
                directions[i][j] = direction

        mapping = {}

        # backtrack
        m_idx = m
        n_idx = n
        while m_idx > 0 and n_idx > 0:
            if directions[n_idx][m_idx] == "↘":
                mapping[n_idx - 1] = m_idx - 1
                m_idx -= 1
                n_idx -= 1
            elif directions[n_idx][m_idx] == "→":
                m_idx -= 1
            elif directions[n_idx][m_idx] == "↓":
                n_idx -= 1

        # initialize the forward mapping (raw to normalized text)
        raw_to_norm_mapping = {0: 0}

        # naive approach for generating forward mapping. this is naive and probably not robust.
        # all leading special characters will get mapped to index position 0 in normalized text.
        raw_to_norm_mapping.update({v: k for k, v in mapping.items()})
        for i in range(0, m):
            if i not in raw_to_norm_mapping:
                raw_to_norm_mapping[i] = raw_to_norm_mapping[i - 1]

        return raw_to_norm_mapping, mapping

    def __repr__(self):
        return "<Tokenizer exclude_from_norm: {}>".format(
            self.exclude_from_norm.__repr__()
        )

    @staticmethod
    def create_tokenizer(app_path=None):
        """Creates the tokenizer for the app

        Args:
            app_path (str, optional): MindMeld Application Path

        Returns:
            Tokenizer: a tokenizer
        """
        return Tokenizer(app_path=app_path)
