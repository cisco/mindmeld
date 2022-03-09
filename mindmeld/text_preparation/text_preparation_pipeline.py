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
from typing import List, Dict, Tuple, Union
import re
import unicodedata
import json
from hashlib import sha256

from .normalizers import (
    Normalizer,
    NoOpNormalizer,
    NormalizerFactory,
    RegexNormalizerRuleFactory,
)
from .preprocessors import Preprocessor, PreprocessorFactory, NoOpPreprocessor
from .stemmers import Stemmer, StemmerFactory, NoOpStemmer
from .tokenizers import SpacyTokenizer, Tokenizer, TokenizerFactory

from ..components._config import (
    get_text_preparation_config,
    DEFAULT_NORMALIZERS,
    DEFAULT_EN_TEXT_PREPARATION_CONFIG,
    get_language_config,
    ENGLISH_LANGUAGE_CODE,
)
from ..constants import UNICODE_SPACE_CATEGORY, DUCKLING_VERSION
from ..exceptions import MindMeldImportError
from ..path import get_app
from .._version import get_mm_version

logger = logging.getLogger(__name__)


# Regex Pattern to capture MindMeld entities ("{entity_text|entity_type|optional_role}")
MINDMELD_ANNOTATION_PATTERN = re.compile(r"\{([^\}\|]*)\|[^\{]*\}")


class TextPreparationPipelineError(Exception):
    pass


class TextPreparationPipeline:  # pylint: disable=R0904
    """Pipeline Class for MindMeld's text processing."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        stemmer: Stemmer = None,
        preprocessors: List[Preprocessor] = None,
        normalizers: List[Normalizer] = None,
        language: str = ENGLISH_LANGUAGE_CODE,
    ):
        """Creates a Pipeline instance."""
        self._language = language
        self._preprocessors = preprocessors or [NoOpPreprocessor()]
        self._normalizers = normalizers or [NoOpNormalizer()]
        self._tokenizer = tokenizer
        self._stemmer = stemmer or NoOpStemmer()

        if self.tokenizer is None:
            raise TextPreparationPipelineError("Tokenizer cannot be None.")

    # Getters
    @property
    def language(self):
        return self._language

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def preprocessors(self):
        return self._preprocessors

    @property
    def normalizers(self):
        return self._normalizers

    @property
    def stemmer(self):
        return self._stemmer

    # Setters
    @tokenizer.setter
    def tokenizer(self, tokenizer: Tokenizer):
        """Set the tokenizer for the Text Preparation Pipeline
        Args:
            tokenizer (Tokenizer): Tokenizer to use.
        """
        if not isinstance(tokenizer, Tokenizer):
            raise TypeError(f"{tokenizer} must be a Tokenizer object.")
        self._tokenizer = tokenizer

    @preprocessors.setter
    def preprocessors(self, preprocessors: List[Preprocessor]):
        """Set the preprocessors for the Text Preparation Pipeline
        Args:
            preprocessors (List[Preprocessor]): Preprocessors to use.
        """
        for preprocessor in preprocessors:
            if not isinstance(preprocessor, Preprocessor):
                raise TypeError(f"{preprocessor} must be a Preprocessor object.")
        self._preprocessors = preprocessors

    def append_preprocessor(self, preprocessor: Preprocessor):
        """Add a preprocessor to the Text Preparation Pipeline
        Args:
            preprocessor (List[Preprocessor]): Preprocessor to append to current Preprocessors.
        """
        if not isinstance(preprocessor, Preprocessor):
            raise TypeError(f"{preprocessor} must be a Preprocessor object.")
        self._preprocessors.append(preprocessor)

    @normalizers.setter
    def normalizers(self, normalizers: List[Normalizer]):
        """Set the normalizers for the Text Preparation Pipeline
        Args:
            normalizers (List[Normalizer]): Normalizers to use.
        """
        for normalizer in normalizers:
            if not isinstance(normalizer, Normalizer):
                raise TypeError(f"{normalizer} must be a Normalizer object.")
        self._normalizers = normalizers

    def append_normalizer(self, normalizer: Normalizer):
        """Add a normalizer to the Text Preparation Pipeline
        Args:
            normalizer (List[Normalizer]): Normalizer to append to current Normalizers.
        """
        if not isinstance(normalizer, Normalizer):
            raise TypeError(f"{normalizer} must be a Normalizer object.")
        self._normalizers.append(normalizer)

    @stemmer.setter
    def stemmer(self, stemmer: Stemmer):
        """Set the stemmer for the Text Preparation Pipeline
        Args:
            stemmer (Stemmer): Stemmer to use.
        """
        if not isinstance(stemmer, Stemmer):
            raise TypeError(f"{stemmer} must be a Stemmer object.")
        self._stemmer = stemmer

    def preprocess(self, text):
        """
        Args:
            text (str): Input text.
        Returns:
            normalized_text (str): Normalized Text.
            forward_map (Dict): Mapping from raw text to modified text.
            backward_map (Dict): Reverse mapping from modified text to raw text.
        """
        preprocessed_text = text
        for preprocessor in self.preprocessors:
            preprocessed_text = preprocessor.process(preprocessed_text)
        return preprocessed_text

    def custom_preprocessors_exist(self):
        """Checks if the current TextPreparationPipeline has preprocessors that are not
        simply the NoOpPreprocessor or None.

        Returns:
            has_custom_preprocessors (bool): Whether atleast one custom preprocessor exists.
        """
        return (
            self.preprocessors
            and not any(
                [isinstance(elem, NoOpPreprocessor) for elem in self.preprocessors]
            )
        )

    def normalize(self, text, keep_special_chars=None):
        """Normalize Text.
        Args:
            text (str): Text to normalize.
            keep_special_chars (bool): Whether to prevent special characters (such as @, [, ])
                from being removed in the normalization process. No longer supported at the
                function level, can be specified in the config.
        Returns:
            normalized_text (str): Normalized text.
        """
        if keep_special_chars:
            logger.warning(
                "'keep_special_chars' is deprecated as a parameter to normalize(). "
                "You can specify 'keep_special_chars' in the TEXT_PREPARATION_CONFIG."
            )
        normalized_tokens = self.tokenize_and_normalize(text)
        normalized_text = " ".join([t["entity"] for t in normalized_tokens])
        return normalized_text

    def _normalize_text(self, text):
        """Normalize an individual token by processing text with all normalizers.

        Args:
            text (str): Text to normalize.
        Returns:
            normalized_text (str): Normalized text.
        """
        normalized_text = text
        for normalizer in self.normalizers:
            normalized_text = TextPreparationPipeline.modify_around_annotations(
                text=normalized_text,
                function=normalizer.normalize,
            )
        return normalized_text

    def tokenize(self, text, keep_special_chars=None):
        """
        Args:
            text (str): Input text.
            keep_special_chars (bool): Whether to prevent special characters (such as @, [, ])
                from being removed in the normalization process. No longer supported at the
                function level, can be specified in the config.
        Returns:
            tokens (List[dict]): List of tokens represented as dictionaries.
        """
        if keep_special_chars:
            logger.warning(
                "'keep_special_chars' is deprecated as a parameter to normalize(). "
                "You can specify 'keep_special_chars' in the TEXT_PREPARATION_CONFIG."
            )
        # Single-shot tokenization for Spacy-Based Tokenizers (Performance Optimization)
        if isinstance(self.tokenizer, SpacyTokenizer):
            return self.tokenize_using_spacy(text)

        # Non-Spacy Tokenizer Handling
        return self.tokenize_around_mindmeld_annotations(text)

    def tokenize_and_normalize(self, text):
        """
        Args:
            text (str): Text to normalize.
        Returns:
            normalized_tokens (List[Dict]): Normalized tokens represented as dictionaries.
                For Example:
                    norm_token = {
                        "entity": "order",
                        "raw_entity": "order",
                        "raw_token_index": 1,
                        "raw_start": 1
                    }
        """
        raw_tokens = self.tokenizer.tokenize(text)
        normalized_tokens = []
        for i, raw_token in enumerate(raw_tokens):
            if not raw_token["text"]:
                continue
            normalized_text = self._normalize_text(raw_token["text"])
            # We sub-tokenize the post-norm text and split the entity if possible
            # Ex: normalize("o'clock") -> "o clock" -> ["o", "clock"]
            # Skip sub-tokenization call if characters are not added/removed
            if normalized_text.lower() == raw_token["text"].lower():
                normalized_texts = [normalized_text]
            else:
                normalized_texts = [t["text"] for t in self.tokenize(normalized_text)]

            if len(normalized_texts) > 0:
                for token_text in normalized_texts:
                    normalized_tokens.append(
                        {
                            "entity": token_text,
                            "raw_entity": raw_token["text"],
                            "raw_token_index": i,
                            "raw_start": raw_token["start"],
                        }
                    )
        return normalized_tokens

    def get_normalized_tokens_as_tuples(self, text):
        """Gets normalized tokens from input text and returns the result as a tuple.

        Args:
            text (str): Text to normalize.
        Returns:
            normalized_tokens_as_tuples (Tuple(str)): A Tuple of normalized tokens.
        """
        return tuple(t["entity"] for t in self.tokenize_and_normalize(text))

    def stem_word(self, word):
        """
        Gets the stem of a word. For example, the stem of the word 'fishing' is 'fish'.

        Args:
            words (List[str]): List of words to stem.

        Returns:
            stemmed_words (List[str]): List of stemmed words.
        """
        return self.stemmer.stem_word(word)

    def tojson(self):
        """
        Method defined to obtain recursive JSON representation of a TextPreparationPipeline.

        Args:
            None.

        Returns:
            JSON representation of TextPreparationPipeline (dict) .
        """
        return {
            "duckling_version": DUCKLING_VERSION,
            "mm_version": get_mm_version(),
            "language": self.language,
            "preprocessors": self.preprocessors,
            "normalizers": self.normalizers,
            "tokenizer": self.tokenizer,
            "stemmer": self.stemmer,
        }

    def get_hashid(self):
        """
        Method defined to obtain Hash value of TextPreparationPipeline.

        Args:
            None.

        Returns:
            256 character hash representation of current TextPreparationPipeline config (str) .
        """
        string = json.dumps(self, cls=TextPreparationPipelineJSONEncoder, sort_keys=True)
        return sha256(string.encode()).hexdigest()

    @staticmethod
    def find_mindmeld_annotation_re_matches(text):
        """
        Args:
            text (str): The string to find mindmeld annotation instances
                (" {entity_text|entity_type} ")
        Returns:
            matches (List[sre.SRE_Match object]): Regex match objects.
        """
        return list(MINDMELD_ANNOTATION_PATTERN.finditer(text))

    @staticmethod
    def calc_unannotated_spans(text):
        """Calculates the spans of text that exclude mindmeld entity annotations.
        For example, "{Lucien|person_name}" would return [(1,7)] since "Lucien" is
        the only text that is not the annotation.

        Args:
            text (str): Original sentence with markup to modify.
        Returns:
            unannotated_spans (List[Tuple(int, int)]): The list of spans where each span
                is a section of the original text excluding mindmeld entity annotations of
                class type and markup symbols ("{", "|", "}"). The first element of the
                tuple is the start index and the second is the ending index + 1.
        """
        matches = TextPreparationPipeline.find_mindmeld_annotation_re_matches(text)
        unannotated_spans = []
        prev_entity_end = 0

        for match in matches:
            entity_start, entity_end = match.span()
            entity_text = match.group(1)

            unannotated_spans.append((prev_entity_end, entity_start))
            entity_text_start = entity_start + 1
            unannotated_spans.append(
                (entity_text_start, entity_text_start + len(entity_text))
            )
            prev_entity_end = entity_end

        # Append a span from the end of last entity to the end of the text (if it exists) 
        if prev_entity_end < len(text):
            unannotated_spans.append((prev_entity_end, len(text)))

        # Filter out spans that have a length of 0
        unannotated_spans = [
            span for span in unannotated_spans if span[1] - span[0] > 0
        ]
        return unannotated_spans

    @staticmethod
    def unannotated_to_annotated_idx_map(unannotated_spans):
        """Create a vector mapping indexes from the unannotated text to the original
        text.

        Args:
            unannotated_spans (List[Tuple(int, int)]): The list of spans where each span
                is a section of the original text excluding mindmeld entity annotations of
                class type and markup symbols ("{", "|", "}"). The first element of the
                tuple is the start index and the second is the ending index + 1.
        Returns:
            unannotated_to_annotated_idx_map (List[int]): A vector where the value at
                each index represents the mapping of the position of a single character
                in the unannotated text to the position in the original text.
        """
        unannotated_to_annotated_idx_map = []
        for unannotated_span in unannotated_spans:
            start, end = unannotated_span
            for i in range(start, end):
                unannotated_to_annotated_idx_map.append(i)
        return unannotated_to_annotated_idx_map

    @staticmethod
    def convert_token_idx_unannotated_to_annotated(
        tokens, unannotated_to_annotated_idx_map
    ):
        """In-place function that reverts the token start indices to the
        index of the character in the orginal text with annotations.

        Args:
            unannotated_to_annotated_idx_map (List[Tuple(int, int)]): A vector where the value at
                each index represents the mapping of the position of a single character
                in the unannotated text to the position in the original text.
            tokens (List[dict]): List of tokens represented as dictionaries. With "start"
                indices referring to the unannotated text.
        """
        for token in tokens:
            token["start"] = unannotated_to_annotated_idx_map[token["start"]]

    def tokenize_using_spacy(self, text):
        """Wrapper function used before tokenizing with Spacy. Combines all unannoted text spans
        into a single string to pass to spacy for tokenization. Applies the correct offset to
        the resulting tokens to align with the annotated text. This optimization reduces the overall
        time needed for tokenization.

        Args:
            text (str): Input text.
        Returns:
            tokens (List[dict]): List of tokens represented as dictionaries.
        """
        unannotated_spans = TextPreparationPipeline.calc_unannotated_spans(text)
        unannotated_text = "".join([text[i[0] : i[1]] for i in unannotated_spans])
        unannotated_to_annotated_idx_mapping = (
            TextPreparationPipeline.unannotated_to_annotated_idx_map(unannotated_spans)
        )
        tokens = self.tokenizer.tokenize(unannotated_text)
        TextPreparationPipeline.convert_token_idx_unannotated_to_annotated(
            tokens, unannotated_to_annotated_idx_mapping
        )
        tokens = TextPreparationPipeline.filter_out_space_text_tokens(tokens)
        return tokens

    @staticmethod
    def modify_around_annotations(text, function):
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

            # Adds "function(pre_entity_text) "{.. or "function(pre_entity_text)" {..
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

    def tokenize_around_mindmeld_annotations(self, text):
        """Applied a function around the mindmeld annotation.

        tokenize(pre_entity_text) + { + tokenize(entity_text) + |entity_name}
            + tokenize(post_entity_text)

        Args:
            text (str): Original sentence with markup to modify.
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
            tokens_before_entity = self.tokenizer.tokenize(
                text[prev_entity_end:entity_start]
            )
            TextPreparationPipeline.offset_token_start_values(
                tokens=tokens_before_entity, offset=prev_entity_end
            )
            tokens.extend(tokens_before_entity)

            # Adds tokens from text within the entity text
            entity_text_start, _ = match.span(1)
            entity_text = match.group(1)
            tokens_within_entity_text = self.tokenizer.tokenize(entity_text)
            TextPreparationPipeline.offset_token_start_values(
                tokens=tokens_within_entity_text, offset=entity_text_start
            )
            tokens.extend(tokens_within_entity_text)

            # Update the previous entity ending index
            prev_entity_end = entity_end

        if prev_entity_end < len(text):
            # Add tokens from the text after the last MindMeld entity
            tokens_after_last_entity = self.tokenizer.tokenize(
                text[prev_entity_end : len(text)]
            )
            TextPreparationPipeline.offset_token_start_values(
                tokens=tokens_after_last_entity, offset=prev_entity_end
            )
            tokens.extend(tokens_after_last_entity)

        tokens = TextPreparationPipeline.filter_out_space_text_tokens(tokens)
        return tokens

    @staticmethod
    def offset_token_start_values(tokens: List[Dict], offset: int):
        """
        Args:
            tokens (List(Dict)): List of tokens represented as dictionaries.
            offset (int): Amount to offset for the start value of each token
        """
        for token in tokens:
            token["start"] = token["start"] + offset

    @staticmethod
    def filter_out_space_text_tokens(tokens: List[Dict]):
        """Filter out any tokens where the text of the token only consists of space characters.

        Args:
            tokens (List[Dict]): List of tokens represented as dictionaries
        Returns:
            filtered_tokens (List[Dict]): List of filtered tokens.
        """
        filtered_tokens = []
        for token in tokens:
            category_by_char = [unicodedata.category(x) for x in token["text"]]
            all_characters_are_space = all(
                [c == UNICODE_SPACE_CATEGORY for c in category_by_char]
            )
            if not all_characters_are_space:
                filtered_tokens.append(token)
        return filtered_tokens

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


class TextPreparationPipelineFactory:
    """Creates a TextPreparationPipeline object."""

    @staticmethod
    def create_from_app_path(app_path):
        """Static method to create a TextPreparationPipeline instance from an app_path.
        If a custom text_preparation_pipeline is passed into the Application object in the
        app_path/__init__.py file then it will be used. Otherwise, a text_preparation_pipeline
        will be created based on the specifications in the config.

        Args:
            app_path (str): The application path.

        Returns:
            TextPreparationPipeline: A TextPreparationPipeline class.
        """
        if app_path:
            # Check if a custom TextPreparationPipeline has been created in app.py
            try:
                app = get_app(app_path)
                if getattr(app, 'text_preparation_pipeline', None):
                    logger.info(
                        "Using custom text_preparation_pipeline from %s/__init__.py.",
                        app_path,
                    )
                    return app.text_preparation_pipeline
            except MindMeldImportError:
                pass
        return TextPreparationPipelineFactory.create_from_app_config(app_path)

    @staticmethod
    def create_from_app_config(app_path):
        """Static method to create a TextPreparation pipeline based on the specifications in
        the config.

        Args:
            app_path (str): The application path.

        Returns:
            TextPreparationPipeline: A TextPreparationPipeline class.
        """
        language, _ = get_language_config(app_path)
        text_preparation_config = get_text_preparation_config(app_path)

        if (
            text_preparation_config.get("regex_norm_rules")
            and "normalizers" not in text_preparation_config
        ):
            logger.warning(
                "Detected 'regex_norm_rules' in TEXT_PREPARATION_CONFIG, however, 'normalizers' "
                "have not been specified. Will apply specified 'regex_norm_rules' in addition to "
                "default normalizers. To omit default normalizers set 'normalizers' to []."
            )

        stemmer = (
            "NoOpStemmer"
            if "stemmer" in text_preparation_config
               and not text_preparation_config["stemmer"]
            else text_preparation_config.get("stemmer")
        )

        return TextPreparationPipelineFactory.create_text_preparation_pipeline(
            language=language,
            preprocessors=text_preparation_config.get("preprocessors"),
            regex_norm_rules=text_preparation_config.get("regex_norm_rules"),
            keep_special_chars=text_preparation_config.get("keep_special_chars"),
            normalizers=text_preparation_config.get("normalizers", DEFAULT_NORMALIZERS),
            tokenizer=text_preparation_config.get("tokenizer"),
            stemmer=stemmer,
        )

    @staticmethod
    def create_text_preparation_pipeline(
        language: str = ENGLISH_LANGUAGE_CODE,
        preprocessors: Tuple[Union[str, Preprocessor]] = None,
        regex_norm_rules: List[Dict] = None,
        keep_special_chars: str = None,
        normalizers: Tuple[Union[str, Normalizer]] = None,
        tokenizer: Union[str, Tokenizer] = None,
        stemmer: Union[str, Stemmer] = None,
    ):
        """Static method to create a TextPreparationPipeline instance.

        Args:
            language (str, optional): Language as specified using a 639-1/2 code.
            preprocessors (Tuple[Union[str, Preprocessor]]): List of preprocessor class
                names or objects.
            regex_norm_rules (List[Dict]): List of regex normalization rules represented
                as dictionaries. ({"pattern":<pattern>, "replacement":<replacement>})
            normalizers (Tuple[Union[str, Preprocessor]]): List of normalizer class names or
                objects.
            tokenizer (Union[str, Tokenizer]): Class name of Tokenizer to use or Tokenizer object.
            stemmer (Union[str, Stemmer]): Class name of Stemmer to use or Stemmer object.

        Returns:
            TextPreparationPipeline: A TextPreparationPipeline class.
        """
        # Instantiate Preprocessors
        instantiated_preprocessors = (
            TextPreparationPipelineFactory._construct_pipeline_components(
                Preprocessor, preprocessors
            )
            if preprocessors
            else [NoOpPreprocessor()]
        )

        # Update Regex Normalization Exception Characters as Specified in the Config
        if keep_special_chars:
            RegexNormalizerRuleFactory.EXCEPTION_CHARS = keep_special_chars

        # Instantiate Normalizers
        instantiated_normalizers = (
            TextPreparationPipelineFactory._construct_pipeline_components(
                Normalizer, normalizers
            )
            if normalizers
            else [NoOpNormalizer()]
        )

        # Instatiate Regex Norm Rules as Normalizer Classes
        if regex_norm_rules:
            regex_normalizers = RegexNormalizerRuleFactory.get_regex_normalizers(
                regex_norm_rules
            )
            # Adds the regex normalizers as the first normalizers by default
            instantiated_normalizers = regex_normalizers + instantiated_normalizers

        # Instantiate Tokenizer
        instantiated_tokenizer = (
            TextPreparationPipelineFactory._construct_pipeline_component(
                Tokenizer, tokenizer, language
            )
            if tokenizer
            else TokenizerFactory.get_default_tokenizer()
        )

        # Instantiate Stemmer
        instantiated_stemmer = (
            TextPreparationPipelineFactory._construct_pipeline_component(
                Stemmer, stemmer
            )
            if stemmer
            else StemmerFactory.get_stemmer_by_language(language)
        )

        return TextPreparationPipeline(
            language=language,
            preprocessors=instantiated_preprocessors,
            normalizers=instantiated_normalizers,
            tokenizer=instantiated_tokenizer,
            stemmer=instantiated_stemmer,
        )

    @staticmethod
    def create_default_text_preparation_pipeline():
        """ Default text_preparation_pipeline used across MindMeld internally."""
        return TextPreparationPipelineFactory.create_text_preparation_pipeline(
            **DEFAULT_EN_TEXT_PREPARATION_CONFIG
        )

    @staticmethod
    def _construct_pipeline_components(  # pylint: disable=W0640
        expected_component_class, components, language=None
    ):
        """Helper method to instantiate multiple components of a TextPreparationPipeline.

        Args:
            expected_component_class (Class): The expected type of the component.
            components (Tuple[Union[str, Object]]): A List/Tuple of components that are either
                strings representing the object that needs to be instantiated or objects that are
                already instantiated.
            language (str, optional): Language as specified using a 639-1/2 code.

        Returns:
            instantiated_components (List[Object]): A list instantiated components.
        """
        instantiated_components = []
        for component in components:
            instantiated_components.append(
                TextPreparationPipelineFactory._construct_pipeline_component(
                    expected_component_class, component, language
                )
            )
        return instantiated_components

    @staticmethod
    def _construct_pipeline_component(  # pylint: disable=W0640
        expected_component_class, component, language=None
    ):
        """Helper method to instantiate a single component of a TextPreparationPipeline.

        Args:
            expected_component_class (Class): The expected type of the component.
            component (Union[str, Object]): A List/Tuple of components that are either
                strings representing the object that needs to be instantiated or objects that are
                already instantiated.
            language (str, optional): Language as specified using a 639-1/2 code.

        Returns:
            instantiated_component (Object): A single TextPreparationPipeline component.
        """
        if isinstance(component, str):
            component_factory_getter = {
                Preprocessor.__name__: lambda: PreprocessorFactory.get_preprocessor(
                    component
                ),
                Normalizer.__name__: lambda: NormalizerFactory.get_normalizer(
                    component
                ),
                Tokenizer.__name__: lambda: TokenizerFactory.get_tokenizer(
                    component, language
                ),
                Stemmer.__name__: lambda: StemmerFactory.get_stemmer(component),
            }
            return component_factory_getter.get(expected_component_class.__name__)()
        elif isinstance(component, expected_component_class):
            return component
        else:
            raise TypeError(
                f"{component} must be of type String or {expected_component_class.__name__}."
            )


class TextPreparationPipelineJSONEncoder(json.JSONEncoder):
    """
    Custom Encoder class defined to obtain recursive JSON representation of a TextPreparationPipeline.

    Args:
        None.

    Returns:
        Custom JSON Encoder class (json.JSONEncoder) .
    """

    def default(self, o):
        tojson = getattr(o, "tojson", None)
        if callable(tojson):
            return tojson()
        else:
            raise TextPreparationPipelineError(
                f"Missing tojson() for {o.__class__.__name__} to create query cache hash."
            )
