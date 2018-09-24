# -*- coding: utf-8 -*-
"""This module contains the query factory class."""
from __future__ import absolute_import, unicode_literals
from builtins import object
from porter2stemmer import Porter2Stemmer


from . import ser as sys_ent_rec

from .core import Query, TEXT_FORM_RAW, TEXT_FORM_PROCESSED, TEXT_FORM_NORMALIZED
from .tokenizer import Tokenizer


class QueryFactory(object):
    """An object which encapsulates the components required to create a Query object.

    Attributes:
        preprocessor (Preprocessor): the object responsible for processing raw text
        tokenizer (Tokenizer): the object responsible for normalizing and tokenizing processed
            text
    """
    def __init__(self, tokenizer, preprocessor=None):
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.stemmer = Porter2Stemmer()

    def create_query(self, text, language=None, time_zone=None, timestamp=None):
        """Creates a query with the given text

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
        """Normalizes the given text

        Args:
            text (str): Text to process

        Returns:
            str: Normalized text
        """
        return self.tokenizer.normalize(text)

    def stem_word(self, word):
        if len(word) <= 2:
            return word
        else:
            # Skipped replace_suffixes_3 and replace_suffixes_4
            # from the original stemmer
            word = self.stemmer.remove_initial_apostrophe(word)
            word = self.stemmer.set_ys(word)
            self.stemmer.find_regions(word)
            word = self.stemmer.strip_possessives(word)
            word = self.stemmer.replace_suffixes_1(word)
            word = self.stemmer.replace_suffixes_2(word)
            word = self.stemmer.replace_ys(word)
            word = self.stemmer.delete_suffixes(word)
            word = self.stemmer.process_terminals(word)
            return word

    def __repr__(self):
        return "<{} id: {!r}>".format(self.__class__.__name__, id(self))

    @staticmethod
    def create_query_factory(app_path=None, tokenizer=None, preprocessor=None):
        """Creates a query factory for the app

        Args:
            app_path (str, optional): The path to the directory containing the
                app's data. If None is passed, a default query factory will be
                returned.
            tokenizer (Tokenizer, optional): The app's tokenizer. One will be
                created if none is provided
            preprocessor (Processor, optional): The app's preprocessor.

        Returns:
            QueryFactory:
        """
        tokenizer = tokenizer or Tokenizer.create_tokenizer(app_path)
        return QueryFactory(tokenizer, preprocessor)
