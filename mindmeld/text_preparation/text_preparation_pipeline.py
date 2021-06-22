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

"""This module contains a text Processing Pipeline."""
import logging
from typing import List, Dict
import re

from .preprocessor import Preprocessor, PreprocessorFactory
from .normalizers import Normalizer, NormalizerFactory, RegexNormalizer
from .tokenizers import Tokenizer, TokenizerFactory
from .stemmers import Stemmer, StemmerFactory

from ..components._config import (
    get_text_preparation_config,
    get_language_config,
    ENGLISH_LANGUAGE_CODE,
)

logger = logging.getLogger(__name__)


# Regex Pattern to capture MindMeld entities ("{entity_text|entity_type|optional_role}")
MINDMELD_ANNOTATION_PATTERN = r"\{([^\}\|]*)\|[^\{]*\}"


class TextPreparationPipelineError(Exception):
    pass


class TextPreparationPipeline:
    """Pipeline Class for MindMeld's text processing."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        stemmer: Stemmer,
        preprocessors: List[Preprocessor] = None,
        normalizers: List[Normalizer] = None,
        language: str = ENGLISH_LANGUAGE_CODE,

    ):
        """Creates a Pipeline instance."""
        self.language = language
        self.preprocessors = preprocessors
        self.normalizers = normalizers
        self.tokenizer = tokenizer
        self.stemmer = stemmer

        if self.tokenizer is None:
            raise TextPreparationPipelineError("Tokenizer cannot be None.")

    def preprocess(self, text):
        """
        Args:
            text (str): Input text.
        Returns:
            normalized_text (str): Normalized Text.
            forward_map (Dict): Mapping from raw text to modified text.
            backward_map (Dict): Reverse mapping from modified text to raw text.
        """
        if not self.preprocessors:
            logger.warning(
                "This instance of TextPreparationPipeline does not have preprocessors."
                "Returning the original text."
            )
            return text

        preprocessed_text = text
        for preprocessor in self.preprocessors:
            preprocessed_text = (
                TextPreparationPipeline.modify_around_mindmeld_annotations(
                    text=preprocessed_text, function=preprocessor.process
                )
            )
        return preprocessed_text

    def normalize(self, text):
        """
        Args:
            text (str): Input text.
        Returns:
            normalized_text (str): Normalized Text.
            forward_map (Dict): Mapping from raw text to modified text
            backward_map (Dict): Reverse mapping from modified text to raw text
        """
        if not self.normalizers:
            logger.warning(
                "This instance of TextPreparationPipeline does not have normalizers."
                "Returning the original text."
            )
            return text

        normalized_text = text
        for normalizer in self.normalizers:
            normalized_text = (
                TextPreparationPipeline.modify_around_mindmeld_annotations(
                    text=normalized_text, function=normalizer.normalize
                )
            )
        return normalized_text

    def tokenize(self, text):
        """
        Args:
            text (str): Input text.
        Returns:
            tokens (List[str]): List of tokens.
        """
        return TextPreparationPipeline.tokenize_around_mindmeld_annotations(
            text=text, function=self.tokenizer.tokenize
        )

    def stem_words(self, words):
        """
        Gets the stem of a word. For example, the stem of the word 'fishing' is 'fish'.

        Args:
            words (List[str]): List of words to stem.

        Returns:
            stemmed_words (List[str]): List of stemmed words.
        """
        return [self.stemmer.stem_word(word) for word in words]

    @staticmethod
    def find_mindmeld_annotation_re_matches(text):
        """
        Args:
            text (str): The string to find mindmeld annotation instances
                (" {entity_text|entity_type} ")
        Returns:
            matches (List[sre.SRE_Match object]): Regex match objects.
        """
        COMPILED_MINDMELD_ANNOTATION_PATTERN = re.compile(MINDMELD_ANNOTATION_PATTERN)
        return list(COMPILED_MINDMELD_ANNOTATION_PATTERN.finditer(text))

    @staticmethod
    def modify_around_mindmeld_annotations(text, function):
        """Applied a function around the mindmeld annotation.

        function(pre_entity_text) + { + function(entity_text) + |entity_name}
            + function(post_entity_text)

        Args:
            text (str): Original sentence with markup to modify.
            function (function): Function to apply around the annotation
        Returns:
            modified_text (str): Text modified around annotations.
        """
        matches = TextPreparationPipeline.find_mindmeld_annotation_re_matches(text)

        modified_text = []
        prev_entity_end = 0

        for match in matches:
            entity_start, entity_end = match.span()

            # Adds "function(pre_entity_text)" {..
            text_before_entity = text[prev_entity_end:entity_start]
            modified_text.append(function(text_before_entity))

            # Modify the Inner Entity Text
            entity_text_start, entity_text_end = match.span(1)
            entity_text = match.group(1)

            # Adds " {"
            modified_text.append(text[entity_start:entity_text_start])

            # Adds function(entity_text)
            modified_text.append(function(entity_text))

            # Adds "|"
            modified_text.append(text[entity_text_end:entity_end])

            # Update the previous entity ending index
            prev_entity_end = entity_end

        if prev_entity_end < len(text):
            # Adds the remainder of the text after the last end brace } "function(post_entity_text)"
            modified_text.append(function(text[prev_entity_end : len(text)]))

        return "".join(modified_text)

    @staticmethod
    def tokenize_around_mindmeld_annotations(text, function):
        """Applied a function around the mindmeld annotation.

        tokenize(pre_entity_text) + { + tokenize(entity_text) + |entity_name}
            + tokenize(post_entity_text)

        Args:
            text (str): Original sentence with markup to modify.
            function (function): Function to apply around the annotation
        Returns:
            tokens (List[dict]): List of tokens represented as dictionaries.
        """
        matches = TextPreparationPipeline.find_mindmeld_annotation_re_matches(text)

        tokens = []
        prev_entity_end = 0

        for match in matches:
            entity_start, entity_end = match.span()
            entity_text = match.group(1)

            # Adds tokens from text before the current entity and after the last entity
            tokens_before_entity = function(text[prev_entity_end:entity_start])
            TextPreparationPipeline.offset_token_start_values(
                token_list=tokens_before_entity, offset=prev_entity_end
            )
            tokens.extend(tokens_before_entity)

            # Adds tokens from text within the entity text
            entity_text_start, _ = match.span(1)
            entity_text = match.group(1)
            tokens_within_entity_text = function(entity_text)
            TextPreparationPipeline.offset_token_start_values(
                token_list=tokens_within_entity_text, offset=entity_text_start
            )
            tokens.extend(tokens_within_entity_text)

            # Update the previous entity ending index
            prev_entity_end = entity_end

        if prev_entity_end < len(text):
            # Add tokens from the text after the last MindMeld entity
            tokens_after_last_entity = function(text[prev_entity_end : len(text)])
            TextPreparationPipeline.offset_token_start_values(
                token_list=tokens_after_last_entity, offset=prev_entity_end
            )
            tokens.extend(tokens_after_last_entity)

        return tokens

    @staticmethod
    def offset_token_start_values(token_list, offset):
        """
        Args:
            token_list (List(Dict)): List of tokens represented as dictionaries.
            offset (int): Amount to offset for the start value of each token
        """
        for token in token_list:
            token["start"] = token["start"] + offset

    @staticmethod
    def get_char_index_map(raw_text, normalized_text):
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
        text = raw_text

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


class TextPreparationPipelineFactory:
    """Creates a TextPreparationPipeline object."""

    @staticmethod
    def create_from_app_path(app_path):
        """Static method to create a TextPreparationPipeline instance from an app_path.

        Args:
            app_path (str): The application path.

        Returns:
            TextPreparationPipeline: A TextPreparationPipeline class.
        """
        language, _ = get_language_config(app_path)
        text_preparation_config = get_text_preparation_config(app_path)

        return TextPreparationPipelineFactory.create_text_preparation_pipeline(
            language=language,
            preprocessors=text_preparation_config.get("preprocessors"),
            normalizers=text_preparation_config.get("normalizers"),
            regex_norm_rules=text_preparation_config.get("regex_norm_rules"),
            tokenizer=text_preparation_config.get("tokenizer"),
            stemmer=text_preparation_config.get("stemmer"),
        )

    @staticmethod
    def create_text_preparation_pipeline(
        language: str = ENGLISH_LANGUAGE_CODE,
        preprocessors: List[Preprocessor] = None,
        normalizers: List[Normalizer] = None,
        regex_norm_rules: List[Dict] = None,
        tokenizer: Tokenizer = None,
        stemmer: Stemmer = None,
    ):
        """Static method to create a TextPreparationPipeline instance.

        Args:
        Args:
            language (str, optional): Language as specified using a 639-1/2 code.
            preprocessors (List[Preprocessors]):

        Returns:
            TextPreparationPipeline: A TextPreparationPipeline class.
        """
        preprocessors = (
            [PreprocessorFactory.get_preprocessor(p) for p in preprocessors]
            if preprocessors
            else []
        )

        normalizers = (
            [NormalizerFactory.get_normalizer(n) for n in normalizers]
            if normalizers
            else []
        )

        if regex_norm_rules:
            regex_normalizer = NormalizerFactory.get_normalizer(
                normalizer=RegexNormalizer.__name__, regex_norm_rules=regex_norm_rules
            )
            # Add the Regex Normalizer as the First Normalizer by Default
            normalizers.insert(0, regex_normalizer)

        tokenizer = (
            TokenizerFactory.get_tokenizer(tokenizer, language)
            if tokenizer
            else TokenizerFactory.get_tokenizer_by_language(language)
        )
        stemmer = (
            StemmerFactory.get_stemmer(stemmer)
            if stemmer
            else StemmerFactory.get_stemmer_by_language(language)
        )
        return TextPreparationPipeline(
            language=language,
            preprocessors=preprocessors,
            normalizers=normalizers,
            tokenizer=tokenizer,
            stemmer=stemmer,
        )
