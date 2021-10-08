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

"""This module contains the query factory class."""
from __future__ import absolute_import, unicode_literals

import logging

from .components._config import get_language_config
from .core import TEXT_FORM_NORMALIZED, TEXT_FORM_PROCESSED, TEXT_FORM_RAW, Query
from .system_entity_recognizer import (
    DucklingRecognizer,
    NoOpSystemEntityRecognizer,
    SystemEntityRecognizer,
)
from .text_preparation.text_preparation_pipeline import TextPreparationPipelineFactory

logger = logging.getLogger(__name__)


class QueryFactory:
    """An object which encapsulates the components required to create a Query object.

    Attributes:
        text_preparation_pipeline (TextPreparationPipeline): Pipeline class responsible for
            processing queries.
        language (str): the language of the text
        locale (str): the locale of the text
        system_entity_recognizer (SystemEntityRecognizer): default to NoOpSystemEntityRecognizer
        duckling (bool): if no system entity recognizer is provided,
            initialize a new Duckling recognizer instance.
    """

    def __init__(
        self,
        text_preparation_pipeline=None,
        locale=None,
        language=None,
        system_entity_recognizer=None,
        duckling=False,
    ):
        self.text_preparation_pipeline = text_preparation_pipeline
        self.locale = locale
        self.language = language

        if system_entity_recognizer:
            self.system_entity_recognizer = system_entity_recognizer
        elif duckling:
            self.system_entity_recognizer = DucklingRecognizer.get_instance()
        else:
            logger.warning(
                "No System Entity Recognizer selected, set 'duckling=True' for DucklingRecognizer",
            )
            self.system_entity_recognizer = NoOpSystemEntityRecognizer.get_instance()

    def create_query(
        self, text, time_zone=None, timestamp=None, locale=None, language=None
    ):
        """Creates a query with the given text.

        Args:
            text (str): Text to create a query object for
            time_zone (str, optional): An IANA time zone id to create the query relative to.
            timestamp (int, optional): A reference unix timestamp to create the query relative to,
                in seconds.
            locale (str, optional): The locale representing the ISO 639-1 language code and \
                ISO3166 alpha 2 country code separated by an underscore character.
            language (str, optional): Language as specified using a 639-1/2 code

        Returns:
            Query: A newly constructed query
        """
        if not language and not locale:
            language = self.language
            locale = self.locale

        raw_text = text
        char_maps = {}

        # Step 1: Preprocessing
        if self.text_preparation_pipeline.custom_preprocessors_exist():
            preprocessed_text = self.text_preparation_pipeline.preprocess(raw_text)
            (
                forward_map,
                backward_map,
            ) = self.text_preparation_pipeline.get_char_index_map(
                raw_text, preprocessed_text
            )
            char_maps[(TEXT_FORM_RAW, TEXT_FORM_PROCESSED)] = forward_map
            char_maps[(TEXT_FORM_PROCESSED, TEXT_FORM_RAW)] = backward_map
        else:
            preprocessed_text = raw_text

        # Step 2: Tokenization and Step 3: Normalization
        normalized_tokens = self.text_preparation_pipeline.tokenize_and_normalize(preprocessed_text)

        normalized_text = " ".join([t["entity"] for t in normalized_tokens])

        # Step 4: Stemming
        stemmed_tokens = [
            self.text_preparation_pipeline.stem_word(t["entity"]) for t in normalized_tokens
        ]

        # Create Normalized Maps
        (
            normalization_forward_map,
            normalization_backward_map,
        ) = self.text_preparation_pipeline.get_char_index_map(
            preprocessed_text, normalized_text
        )
        char_maps[
            (TEXT_FORM_PROCESSED, TEXT_FORM_NORMALIZED)
        ] = normalization_forward_map
        char_maps[
            (TEXT_FORM_NORMALIZED, TEXT_FORM_PROCESSED)
        ] = normalization_backward_map

        query = Query(
            raw_text,
            preprocessed_text,
            normalized_tokens,
            char_maps,
            locale=locale,
            language=language,
            time_zone=time_zone,
            timestamp=timestamp,
            stemmed_tokens=stemmed_tokens,
        )
        query.system_entity_candidates = self.system_entity_recognizer.get_candidates(
            query, locale=locale, language=language
        )
        return query

    def normalize(self, text):
        """Normalizes the given text.

        Args:
            text (str): Text to process

        Returns:
            str: Normalized text
        """
        return self.text_preparation_pipeline.normalize(text)

    def __repr__(self):
        return "<{} id: {!r}>".format(self.__class__.__name__, id(self))

    @staticmethod
    def create_query_factory(
        app_path,
        text_preparation_pipeline=None,
        system_entity_recognizer=None,
        duckling=False,
    ):
        """Creates a query factory for the application.

        Args:
            app_path (str, optional): The path to the directory containing the
                app's data. If None is passed, a default query factory will be
                returned.
            text_preparation_pipeline (TextPreparationPipeline, optional): Pipeline class
                responsible for processing queries.
            system_entity_recognizer (SystemEntityRecognizer): If not passed, we use either the one
                from the application's configuration or NoOpSystemEntityRecognizer.
            duckling (bool, optional): if no system entity recognizer is provided,
                 initialize a new Duckling recognizer instance.

        Returns:
            QueryFactory: A QueryFactory object that is used to create Query objects.
        """
        language, locale = get_language_config(app_path)

        if text_preparation_pipeline is None:
            text_preparation_pipeline = TextPreparationPipelineFactory.create_from_app_path(
                app_path
            )
        if system_entity_recognizer:
            sys_entity_recognizer = system_entity_recognizer
        elif app_path:
            sys_entity_recognizer = SystemEntityRecognizer.load_from_app_path(app_path)
        elif duckling:
            sys_entity_recognizer = DucklingRecognizer.get_instance()
        else:
            logger.warning(
                "No System Entity Recognizer selected, set 'duckling=True' for DucklingRecognizer",
            )
            sys_entity_recognizer = NoOpSystemEntityRecognizer.get_instance()
        return QueryFactory(
            text_preparation_pipeline=text_preparation_pipeline,
            language=language,
            locale=locale,
            system_entity_recognizer=sys_entity_recognizer,
        )
