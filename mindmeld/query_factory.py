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

import nltk

from . import ser as sys_ent_rec

from .core import Query, TEXT_FORM_RAW, TEXT_FORM_PROCESSED, TEXT_FORM_NORMALIZED
from .tokenizer import Tokenizer


class QueryFactory:
    """An object which encapsulates the components required to create a Query object.

    Attributes:
        preprocessor (Preprocessor): the object responsible for processing raw text
        tokenizer (Tokenizer): the object responsible for normalizing and tokenizing processed
            text
    """
    def __init__(self, tokenizer, preprocessor=None):
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.stemmer = nltk.stem.PorterStemmer()

    def create_query(self, text, language=None, time_zone=None, timestamp=None):
        """Creates a query with the given text.

        Args:
            text (str): Text to create a query object for
            language (str, optional): Language as specified using a 639-2 code;
                if omitted, English is assumed.
            time_zone (str, optional): An IANA time zone id to create the query relative to.
            timestamp (int, optional): A reference unix timestamp to create the query relative to,
                in seconds.

        Returns:
            Query: A newly constructed query
        """
        raw_text = text

        char_maps = {}

        # create raw, processed maps
        if self.preprocessor:
            processed_text = self.preprocessor.process(raw_text)
            maps = self.preprocessor.get_char_index_map(raw_text, processed_text)
            forward, backward = maps
            char_maps[(TEXT_FORM_RAW, TEXT_FORM_PROCESSED)] = forward
            char_maps[(TEXT_FORM_PROCESSED, TEXT_FORM_RAW)] = backward
        else:
            processed_text = raw_text

        normalized_tokens = self.tokenizer.tokenize(processed_text)
        normalized_text = ' '.join([t['entity'] for t in normalized_tokens])
        stemmed_tokens = [self.stem_word(t['entity']) for t in normalized_tokens]

        # create normalized maps
        maps = self.tokenizer.get_char_index_map(processed_text, normalized_text)
        forward, backward = maps

        char_maps[(TEXT_FORM_PROCESSED, TEXT_FORM_NORMALIZED)] = forward
        char_maps[(TEXT_FORM_NORMALIZED, TEXT_FORM_PROCESSED)] = backward

        query = Query(raw_text, processed_text, normalized_tokens, char_maps,
                      language=language, time_zone=time_zone, timestamp=timestamp,
                      stemmed_tokens=stemmed_tokens)
        query.system_entity_candidates = sys_ent_rec.get_candidates(query)
        return query

    def normalize(self, text):
        """Normalizes the given text.

        Args:
            text (str): Text to process

        Returns:
            str: Normalized text
        """
        return self.tokenizer.normalize(text)

    def stem_word(self, word):
        """
        Gets the stem of a word. For example, the stem of the word 'fishing' is 'fish'.

        Args:
            word (str): The word to stem

        Returns:
            str: Stemmed version of a word.
        """
        stem = word.lower()

        if self.stemmer.mode == self.stemmer.NLTK_EXTENSIONS and word in self.stemmer.pool:
            return self.stemmer.pool[word]

        if self.stemmer.mode != self.stemmer.ORIGINAL_ALGORITHM and len(word) <= 2:
            # With this line, strings of length 1 or 2 don't go through
            # the stemming process, although no mention is made of this
            # in the published algorithm.
            return word

        stem = self.stemmer._step1a(stem)
        stem = self.stemmer._step1b(stem)
        stem = self.stemmer._step1c(stem)
        stem = self.stemmer._step5b(stem)

        # if the stemmed cleaves off the whole token, just return the original one
        if stem == '':
            return word
        else:
            return stem

    def __repr__(self):
        return "<{} id: {!r}>".format(self.__class__.__name__, id(self))

    @staticmethod
    def create_query_factory(app_path=None, tokenizer=None, preprocessor=None):
        """Creates a query factory for the application.

        Args:
            app_path (str, optional): The path to the directory containing the
                app's data. If None is passed, a default query factory will be
                returned.
            tokenizer (Tokenizer, optional): The app's tokenizer. One will be
                created if none is provided
            preprocessor (Processor, optional): The app's preprocessor.

        Returns:
            QueryFactory: A QueryFactory object that is used to create Query objects.
        """
        del app_path
        tokenizer = tokenizer or Tokenizer.create_tokenizer()
        return QueryFactory(tokenizer, preprocessor)
