# -*- coding: utf-8 -*-
"""This module contains a collection of the core data structures used in workbench."""

from __future__ import unicode_literals
from builtins import object

TEXT_FORM_RAW = 0
TEXT_FORM_PROCESSED = 1
TEXT_FORM_NORMALIZED = 0
TEXT_FORMS = [TEXT_FORM_RAW, TEXT_FORM_PROCESSED, TEXT_FORM_NORMALIZED]


class Query(object):
    """The query object is responsible for processing and normalizing raw user text input so that
    classifiers can deal with it. A query stores three forms of text: raw text, processed text, and
    normalized text. The query object is also responsible for translating text ranges across these
    forms.

    Attributes:
        normalized_tokens (list): a list of normalized tokens
        preprocessor (Preprocessor): the object responsible for processing raw text
        tokenizer (Tokenizer): the object responsible for normalizing and tokenizing processed text
        raw_text (str): the original input text
        normalized_text (str): the normalized text. TODO: better description here
        processed_text (str): the text after it has been preprocessed. TODO: better description here
    """
    def __init__(self, raw_text, tokenizer, preprocessor=None):
        self._text = {}
        self._text[TEXT_FORM_RAW] = raw_text
        self._char_maps = {}
        if preprocessor:
            processed_text = preprocessor.process(raw_text)
            char_map = preprocessor.generate_character_index_mapping(raw_text, processed_text)
            self._char_maps[(TEXT_FORM_PROCESSED, TEXT_FORM_RAW)] = char_map
            # TODO: create TEXT_FORM_RAW -> TEXT_FORM_PROCESSED mapping
        else:
            processed_text = raw_text
        self._text[TEXT_FORM_PROCESSED] = processed_text

        self.normalized_tokens = tokenizer.tokenize(self.processed_text)
        normalized_text = ' '.join([t['entity'] for t in self.normalized_tokens])
        self._text[TEXT_FORM_NORMALIZED] = normalized_text
        char_map = tokenizer.generate_character_index_mapping(processed_text, normalized_text)
        self._char_maps[(TEXT_FORM_NORMALIZED, TEXT_FORM_PROCESSED)] = char_map
        # TODO: create TEXT_FORM_PROCESSED -> TEXT_FORM_NORMALIZED mapping

    @property
    def raw_text(self):
        return self._text[TEXT_FORM_RAW]

    @property
    def processed_text(self):
        return self._text[TEXT_FORM_PROCESSED]

    @property
    def normalized_text(self):
        return self._text[TEXT_FORM_NORMALIZED]

    def transform_range(self, text_range, form_in, form_out):
        """Transforms a text range from one form to another.

        Args:
            text_range (tuple): the range being transformed
            form_in (int): the input text form. Should be one of TEXT_FORM_RAW, TEXT_FORM_PROCESSED
                or TEXT_FORM_NORMALIZED
            form_out (int): the output text form. Should be one of TEXT_FORM_RAW,
                TEXT_FORM_PROCESSED or TEXT_FORM_NORMALIZED

        Returns:
            tuple: the equivalent range of text in the output form
        """
        return (self._transform_index(text_range[0], form_in, form_out),
                self._transform_index(text_range[1], form_in, form_out))

    def transform_index(self, index, form_in, form_out):
        """Transforms a text index from one form to another.

        Args:
            index (int): the index being transformed
            form_in (int): the input form. should be one of TEXT_FORM_RAW
            form_out (int): the output form

        Returns:
            int: the equivalent index of text in the output form
        """
        if form_in not in TEXT_FORMS or form_out not in TEXT_FORMS:
            raise ValueError('Invalid text form')

        if form_in > form_out:
            while form_in > form_out:
                index = self._unprocess_index(index, form_in)
                form_in -= 1
        else:
            while form_in < form_out:
                index = self._process_index(index, form_in)
                form_in += 1

    def _process_index(self, index, form_in):
        if form_in == TEXT_FORM_NORMALIZED:
            raise ValueError("'{}' form cannot be processed".form(TEXT_FORM_NORMALIZED))
        mapping_key = (form_in, (form_in + 1))
        try:
            mapping = self._char_maps[mapping_key]
        except KeyError:
            # mapping doesn't exist -> use identity
            return index
        return mapping[index]

    def _unprocess_index(self, index, form_in):
        if form_in == TEXT_FORM_RAW:
            raise ValueError("'{}' form cannot be unprocessed".form(TEXT_FORM_RAW))
        mapping_key = (form_in, (form_in - 1))
        try:
            mapping = self._char_maps[mapping_key]
        except KeyError:
            # mapping doesn't exist -> use identity
            return index
        return mapping[index]


class ParsedQuery(object):
    """A parsed query contains a query and the additional metadata that has been labeled or
    predicted.


    Attributes:
        domain (str): The domain of the query
        entities (list): A list of entities present in this query
        intent (str): The intent of the query
        is_gold (bool): Indicates whether the details in this query were predicted or human labeled
        query (Query): The underlying query object.
    """
    def __init__(self, query, domain=None, intent=None, entities=None, is_gold=False):

        self.query = query
        self.domain = domain
        self.intent = intent
        self.entities = entities
        self.is_gold = is_gold


class Entity(object):
    """Summary

    TODO: account for numeric entities

    Attributes:
        display_text (str): A human readable text representation of the entity for use in natural
            language responses.
        query (Query): Description
        role (str): Description
        source_raw_text (str): The raw text that was parsed into this entity
        source_processed_text (str): The processed text that was parsed into this entity
        source_normalized_text (str): The normalized text that was parsed into this entity
        text_start (int): The character index start of the text range that was parsed into this
            entity. This index is based on the normalized text of the query passed in.
        text_end (int): The character index end of the text range that was parsed into this entity.
            This index is based on the normalized text of the query passed in.
        type (str): The type of entity
        value (str): The resolved value of the entity
    """
    def __init__(self, query, text_start, text_end, entity_type, role, display_text=None,
                 value=None):
        """Initializes an entity object

        Args:
            query (Query): The query this entity was parsed from
            text_start (int): The character index start of the text range that was parsed into this
                entity. This index is based on the normalized text of the query passed in.
            text_end (int): The character index end of the text range that was parsed into this
                entity. This index is based on the normalized text of the query passed in.
            entity_type (str): The type of this entity
            role (str): The role of this entity
            display_text (str): A human readable text representation of the entity for use in
                natural language responses.
            value (str): The resolved value of the entity
        """
        self.query = query
        self.text_start = text_start
        self.text_end = text_end
        self.type = entity_type
        self.role = role
        self.display_text = display_text
        self.value = value

    @property
    def source_raw_text(self):
        text_range = self.query.transform_range((self.text_start, self.text_end),
                                                TEXT_FORM_NORMALIZED, TEXT_FORM_RAW)
        return self.query.raw_text[text_range[0]:text_range[1]]

    @property
    def source_processed_text(self):
        text_range = self.query.transform_range((self.text_start, self.text_end),
                                                TEXT_FORM_NORMALIZED, TEXT_FORM_PROCESSED)
        return self.query.processed_text[text_range[0]:text_range[1]]

    @property
    def source_normalized_text(self):
        return self.query.normalized_text[self.text_start:self.text_end]


class Slot(object):
    """A slot represents an entity without the context of the query it came from.

    Attributes:
        display_text (str): Description
        role (str): Description
        type (str): Description
        value (str): Description
    """
    def __init__(self, entity_type, role, value, display_text):
        self.type = entity_type
        self.role = role
        self.value = value
        self.display_text = display_text
