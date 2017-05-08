# -*- coding: utf-8 -*-
"""This module contains a collection of the core data structures used in workbench."""
from __future__ import unicode_literals
from builtins import object, range, super

import logging


TEXT_FORM_RAW = 0
TEXT_FORM_PROCESSED = 1
TEXT_FORM_NORMALIZED = 2
TEXT_FORMS = [TEXT_FORM_RAW, TEXT_FORM_PROCESSED, TEXT_FORM_NORMALIZED]

logger = logging.getLogger(__name__)


class Bunch(dict):
    """Dictionary-like object that exposes its keys as attributes.

    Inspired by scikit learn's Bunches

    >>> b = Bunch(a=1, b=2)
    >>> b['b']
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b['a']
    3
    >>> b.c = 6
    >>> b['c']
    6

    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        pass


class Span(object):
    """Simple object representing a span with start and end indices"""
    __slots__ = ['start', 'end']

    def __init__(self, start, end):
        assert start <= end, "Span 'start' must be less than or equal to 'end'"
        self.start = start
        self.end = end

    def to_dict(self):
        """Converts the span into a dictionary"""
        return {'start': self.start, 'end': self.end}

    def slice(self, obj):
        """Returns the slice of the object for this span

        Args:
            obj: The object to slice

        Returns:
            The slice of the passed in object for this span
        """
        return obj[self.start:self.end + 1]

    def shift(self, offset):
        """Shifts a span by offset

        Args:
            offset (int): The number to change start and end by

        """
        return Span(self.start + offset, self.end + offset)

    def __iter__(self):
        for index in range(self.start, self.end + 1):
            yield index

    def __len__(self):
        return self.end - self.start + 1

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.start == other.start and self.end == other.end
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __repr__(self):
        return "{}(start={}, end={})".format(self.__class__.__name__, self.start, self.end)


class Query(object):
    """The query object is responsible for processing and normalizing raw user text input so that
    classifiers can deal with it. A query stores three forms of text: raw text, processed text, and
    normalized text. The query object is also responsible for translating text ranges across these
    forms.

    Attributes:
        text (str): the original input text
        processed_text (str): the text after it has been preprocessed. TODO: better description here
        normalized_tokens (tuple of str): a list of normalized tokens
        normalized_text (str): the normalized text. TODO: better description here
        system_entity_candidates (tuple): Description
    """

    # TODO: look into using __slots__

    def __init__(self, raw_text, processed_text, normalized_tokens, char_maps):
        """Summary

        Args:
            raw_text (str): the original input text
            processed_text (str): the input text after it has been preprocessed
            normalized_tokens (list of dict): List tokens outputted by
                a tokenizer
            char_maps (dict): Mappings between character indices in raw,
                processed and normalized text
        """
        self._normalized_tokens = normalized_tokens
        norm_text = ' '.join([t['entity'] for t in self._normalized_tokens])
        self._texts = (raw_text, processed_text, norm_text)
        self._char_maps = char_maps
        self.system_entity_candidates = ()

    @property
    def text(self):
        """The original input text"""
        return self._texts[TEXT_FORM_RAW]

    @property
    def processed_text(self):
        """The input text after it has been preprocessed"""
        return self._texts[TEXT_FORM_PROCESSED]

    @property
    def normalized_text(self):
        """The normalized input text"""
        return self._texts[TEXT_FORM_NORMALIZED]

    @property
    def normalized_tokens(self):
        """The tokens of the normalized input text"""
        return tuple((token['entity'] for token in self._normalized_tokens))

    def get_text_form(self, form):
        """Programmatically retrieves text by form

        Args:
            form (int): A valid text form (TEXT_FORM_RAW, TEXT_FORM_PROCESSED, or
                TEXT_FORM_NORMALIZED)

        Returns:
            str: The requested text
        """
        return self._texts[form]

    def get_system_entity_candidates(self, sys_types):
        """
        Args:
            sys_types (set of str): A set of entity types to select

        Returns:
            list: Returns candidate system entities of the types specified
        """
        return [e for e in self.system_entity_candidates if e.entity.type in sys_types]

    def transform_span(self, text_span, form_in, form_out):
        """Transforms a text range from one form to another.

        Args:
            text_span (Span): the text span being transformed
            form_in (int): the input text form. Should be one of TEXT_FORM_RAW, TEXT_FORM_PROCESSED
                or TEXT_FORM_NORMALIZED
            form_out (int): the output text form. Should be one of TEXT_FORM_RAW,
                TEXT_FORM_PROCESSED or TEXT_FORM_NORMALIZED

        Returns:
            tuple: the equivalent range of text in the output form
        """
        return Span(self.transform_index(text_span.start, form_in, form_out),
                    self.transform_index(text_span.end, form_in, form_out))

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
        return index

    def _process_index(self, index, form_in):
        if form_in == TEXT_FORM_NORMALIZED:
            raise ValueError("'{}' form cannot be processed".format(TEXT_FORM_NORMALIZED))
        mapping_key = (form_in, (form_in + 1))
        try:
            mapping = self._char_maps[mapping_key]
        except KeyError:
            # mapping doesn't exist -> use identity
            return index
        # None for mapping means 1-1 mapping
        try:
            return mapping[index] if mapping else index
        except KeyError:
            raise ValueError('Invalid index {}'.format(index))

    def _unprocess_index(self, index, form_in):
        if form_in == TEXT_FORM_RAW:
            raise ValueError("'{}' form cannot be unprocessed".format(TEXT_FORM_RAW))
        mapping_key = (form_in, (form_in - 1))
        try:
            mapping = self._char_maps[mapping_key]
        except KeyError:
            # mapping doesn't exist -> use identity
            return index
        # None for mapping means 1-1 mapping
        try:
            return mapping[index] if mapping else index
        except KeyError:
            raise ValueError('Invalid index {}'.format(index))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.text)


class ProcessedQuery(object):
    """A processed query contains a query and the additional metadata that has been labeled or
    predicted.


    Attributes:
        domain (str): The domain of the query
        entities (list): A list of entities present in this query
        intent (str): The intent of the query
        is_gold (bool): Indicates whether the details in this query were predicted or human labeled
        query (Query): The underlying query object.
    """

    # TODO: look into using __slots__

    def __init__(self, query, domain=None, intent=None, entities=None, entity_groups=None,
                 is_gold=False):
        self.query = query
        self.domain = domain
        self.intent = intent
        self.entities = None if entities is None else tuple(entities)
        self.entity_groups = None if entity_groups is None else tuple(entity_groups)
        self.is_gold = is_gold

    def to_dict(self):
        """Converts the processed query into a dictionary"""
        base = {
            'text': self.query.text,
            'domain': self.domain,
            'intent': self.intent,
            'entities': None if self.entities is None else [e.to_dict() for e in self.entities],
        }
        if self.entity_groups is not None:
            base['entity_groups'] = [g.to_dict() for g in self.entity_groups]
        return base

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __repr__(self):
        msg = "<{} {!r}, domain: {!r}, intent: {!r}, {!r} entities{}>"
        return msg.format(self.__class__.__name__, self.query.text, self.domain, self.intent,
                          len(self.entities), ', gold' if self.is_gold else '')


class NestedEntity(object):
    def __init__(self, texts, spans, token_spans, entity):
        """Initializes an entity node object

        Args:
            texts (tuple): Tuple containing the three forms of text
            spans (tuple): Tuple containing the character index spans of the
                text for this entity for each text form
            token_spans (tuple): Tuple containing the token index spans of the
                text for this entity for each text form
        """
        self._texts = texts
        self._spans = spans
        self._token_spans = token_spans
        self.entity = entity

    @classmethod
    def from_query(cls, query, span=None, normalized_span=None, entity_type=None,
                   entity=None, parent_offset=0):
        """Creates an entity node using a parent entity node

        Args:
            query (Query): Description
            parent_offset (int): The offset of the parent in the query
            span (Span): The span of the entity in the query's raw text
            normalized_span (None, optional): The span of the entity in the
                query's normalized text
            entity_type (str, optional): The entity type. One of this and entity
                must be provided
            entity (Entity, optional): The entity. One of this and entity must
                be provided

        Returns:
            the created entity

        """
        def _get_form_details(query_span, offset, form_in, form_out):
            offset_out = query.transform_index(offset, form_in, form_out)
            span_out = query.transform_span(query_span, form_in, form_out)
            full_text = query.get_text_form(form_out)
            text = span_out.slice(full_text)
            tok_offset = len(full_text[:offset_out].split())
            tok_start = len(full_text[:span_out.start].split()) - tok_offset
            tok_span = Span(tok_start, tok_start - 1 + len(text.split()))

            # convert span from query's indexing to parent's indexing
            span_out = span_out.shift(-offset_out)

            return text, span_out, tok_span

        if span:
            query_span = span.shift(parent_offset)
            form_in = TEXT_FORM_RAW
        elif normalized_span:
            query_span = normalized_span.shift(parent_offset)
            form_in = TEXT_FORM_NORMALIZED

        texts, spans, tok_spans = list(zip(*[_get_form_details(query_span, parent_offset,
                                                               form_in, form_out)
                                             for form_out in TEXT_FORMS]))

        if entity is None:
            if entity_type is None:
                raise ValueError("Either 'entity' or 'entity_type' must be specified")
            entity = Entity(texts[0], entity_type)

        return cls(texts, spans, tok_spans, entity)

    def to_dict(self):
        """Converts the query entity into a dictionary"""
        base = self.entity.to_dict()
        base.update({'span': self.span.to_dict()})
        return base

    @property
    def text(self):
        """The original input text span"""
        return self._texts[TEXT_FORM_RAW]

    @property
    def processed_text(self):
        """The input text after it has been preprocessed"""
        return self._texts[TEXT_FORM_PROCESSED]

    @property
    def normalized_text(self):
        """The normalized input text"""
        return self._texts[TEXT_FORM_NORMALIZED]

    @property
    def span(self):
        """The span of original input text span"""
        return self._spans[TEXT_FORM_RAW]

    @property
    def processed_span(self):
        """The span of the preprocessed text span"""
        return self._spans[TEXT_FORM_PROCESSED]

    @property
    def normalized_span(self):
        """The span of the normalized text span"""
        return self._spans[TEXT_FORM_NORMALIZED]

    @property
    def token_span(self):
        """The token_span of original input text span"""
        return self._token_spans[TEXT_FORM_RAW]

    @property
    def processed_token_span(self):
        """The token_span of the preprocessed text span"""
        return self._token_spans[TEXT_FORM_PROCESSED]

    @property
    def normalized_token_span(self):
        """The token_span of the normalized text span"""
        return self._token_spans[TEXT_FORM_NORMALIZED]

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __str__(self):
        return "{}{}{} '{}' {}-{} ".format(
            self.entity.type, ':' if self.entity.role else '', self.entity.role, self.text,
            self.span.start, self.span.end
        )

    def __repr__(self):
        msg = '<{} {!r} ({!r}) char: [{!r}-{!r}], tok: [{!r}-{!r}]>'
        return msg.format(self.__class__.__name__, self.text, self.entity.type, self.span.start,
                          self.span.end, self.token_span.start, self.token_span.end)


class QueryEntity(NestedEntity):
    """An entity with the context of the query it came from.

    Attributes:
        text (str): The raw text that was processed into this entity
        processed_text (str): The processed text that was processed into
            this entity
        normalized_text (str): The normalized text that was processed into
            this entity
        span (Span): The character index span of the raw text that was
            processed into this entity
        processed_span (Span): The character index span of the raw text that was
            processed into this entity
        span (Span): The character index span of the raw text that was
            processed into this entity
        start (int): The character index start of the text range that was processed into this
            entity. This index is based on the normalized text of the query passed in.
        end (int): The character index end of the text range that was processed into this
            entity. This index is based on the normalized text of the query passed in.
    """


class Entity(object):
    """An Entity is any important piece of text that provides more information about the user
    intent.

    Attributes:
        type (str): The type of the entity
        role (str): The role of the entity
        value (str): The resolved value of the entity
        display_text (str): A human readable text representation of the entity for use in natural
            language responses.
    """

    # TODO: look into using __slots__

    def __init__(self, text, entity_type, role=None, value=None, display_text=None,
                 confidence=None):
        self.text = text
        self.type = entity_type
        self.role = role
        self.value = value
        self.display_text = display_text
        self.confidence = confidence
        self.is_system_entity = self.__class__.is_system_entity(entity_type)

    @staticmethod
    def is_system_entity(entity_type):
        """Checks whether the provided entity type is a Workbench-recognized system entity.

        Args:
            entity_type (str): An entity type

        Returns:
            bool: True if the entity is a system entity type, else False
        """
        return entity_type.startswith('sys:')

    def to_dict(self):
        """Converts the entity into a dictionary"""
        base = {'text': self.text, 'type': self.type, 'role': self.role}
        for field in ['value', 'display_text', 'confidence']:
            value = getattr(self, field)
            if value is not None:
                base[field] = value

        return base

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __repr__(self):
        text = self.display_text or self.text
        return "<{} {!r} ({!r})>".format(self.__class__.__name__, text, self.type)


class EntityGroup(object):
    """An object which represents the relationship between entities.

    Attributes:
        head (QueryEntity): The head of this entity group
        dependents (tuple of QueryEntity or EntityGroup): A list of entities which describe the head
    """

    def __init__(self, head, dependents):
        self.head = head
        self.dependents = tuple(sorted(dependents, key=lambda d: d.span.start))
        start = head.span.start
        end = head.span.end
        for dep in dependents:
            start = min(start, dep.span.start)
            end = max(end, dep.span.end)
        self.span = Span(start, end)

    def __repr__(self):
        text = self.head.entity.display_text or self.head.text
        msg = '<{} {!r} ({!r}) [{!r}-{!r}]>'
        return msg.format(self.__class__.__name__, text, self.head.entity.type,
                          self.span.start, self.span.end)

    def to_dict(self):
        """Converts the entity group into a dictionary"""
        return {
            'head': self.head.to_dict(),
            'dependents': [d.to_dict() for d in self.dependents],
            'span': self.span.to_dict()
        }


def resolve_entity_conflicts(query_entities):
    """This method takes a list containing query entities for a query, and resolves
    any entity conflicts. The resolved list is returned.

    If two entities in a query conflict with each other, use the following logic:
        - If the target entity is a subset of another entity, then delete the
          target entity.
        - If the target entity shares the identical span as another entity,
          then keep the one with the highest confidence.
        - If the target entity overlaps with another entity, then keep the one
          with the highest confidence.

    Args:
        entities (list of QueryEntity): A list of query entities to resolve

    Returns:
        list of QueryEntity: A filtered list of query entities

    """
    filtered = [e for e in query_entities]
    i = 0
    while i < len(filtered):
        include_target = True
        target = filtered[i]
        j = i + 1
        while j < len(filtered):
            other = filtered[j]
            if _is_superset(target, other) and not _is_same_span(target, other):
                logger.debug('Removing {{{1:s}|{2:s}}} entity in query {0:d} since it is a '
                             'subset of another.'.format(i, other.text, other.entity.type))
                del filtered[j]
                continue
            elif _is_subset(target, other) and not _is_same_span(target, other):
                logger.debug('Removing {{{1:s}|{2:s}}} entity in query {0:d} since it is a '
                             'subset of another.'.format(i, target.text, target.entity.type))
                del filtered[i]
                include_target = False
                break
            elif _is_same_span(target, other) or _is_overlapping(target, other):
                if target.entity.confidence >= other.entity.confidence:
                    logger.debug('Removing {{{1:s}|{2:s}}} entity in query {0:d} since it overlaps '
                                 'with another.'.format(i, other.text, other.entity.type))
                    del filtered[j]
                    continue
                elif target.entity.confidence < other.entity.confidence:
                    logger.debug('Removing {{{1:s}|{2:s}}} entity in query {0:d} since it overlaps '
                                 'with another.'.format(i, target.text, target.entity.type))
                    del filtered[i]
                    include_target = False
                    break
            j += 1
        if include_target:
            i += 1

    return filtered


def _is_subset(target, other):
    return ((target.start >= other.start) and
            (target.end <= other.end))


def _is_superset(target, other):
    return ((target.start <= other.start) and
            (target.end >= other.end))


def _is_same_span(target, other):
    return _is_superset(target, other) and _is_subset(target, other)


def _is_overlapping(target, other):
    target_range = range(target.start, target.end + 1)
    predicted_range = range(other.start, other.end + 1)
    overlap = set(target_range).intersection(predicted_range)
    return (overlap and not _is_subset(target, other) and
            not _is_superset(target, other))


def configure_logs(**kwargs):
    """Helper method for easily configuring logs from the python shell.
    Args:
        level (TYPE, optional): A logging level recognized by python's logging module.
    """
    import sys
    level = kwargs.get('level', logging.INFO)
    log_format = kwargs.get('format', '%(message)s')
    logging.basicConfig(stream=sys.stdout, format=log_format)
    package_logger = logging.getLogger(__package__)
    package_logger.setLevel(level)
