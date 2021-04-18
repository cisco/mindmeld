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

"""This module contains a collection of the core data structures used in MindMeld."""
import logging
from typing import Optional, List, Dict
import immutables

TEXT_FORM_RAW = 0
TEXT_FORM_PROCESSED = 1
TEXT_FORM_NORMALIZED = 2
TEXT_FORMS = [TEXT_FORM_RAW, TEXT_FORM_PROCESSED, TEXT_FORM_NORMALIZED]

logger = logging.getLogger(__name__)

# The date keys are extracted from here
# https://github.com/wit-ai/duckling_old/blob/a4bc34e3e945d403a9417df50c1fb2172d56de3e/src/duckling/time/obj.clj#L21 # noqa E722
TIME_GRAIN_TO_ORDER = {
    "year": 8,
    "quarter": 7,
    "month": 6,
    "week": 5,
    "day": 4,
    "hour": 3,
    "minute": 2,
    "second": 1,
    "milliseconds": 0,
}


def _sort_by_lowest_time_grain(system_entities):
    return sorted(
        system_entities,
        key=lambda query_entity: TIME_GRAIN_TO_ORDER[
            query_entity.entity.value["grain"]
        ],
    )


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
        except KeyError as e:
            raise AttributeError(key) from e

    def __setstate__(self, state):
        pass


class Span:
    """Object representing a text span with start and end indices

    Attributes:
        start (int): The index from the original text that represents the start of the span
        end (int): The index from the original text that represents the end of the span
    """

    __slots__ = ["start", "end"]

    def __init__(self, start, end):
        assert start <= end, "Span 'start' must be less than or equal to 'end'"
        self.start = start
        self.end = end

    def to_dict(self):
        """Converts the span into a dictionary"""
        return {"start": self.start, "end": self.end}

    def slice(self, obj):
        """Returns the slice of the object for this span

        Args:
            obj: The object to slice

        Returns:
            The slice of the passed in object for this span
        """
        return obj[self.start : self.end + 1]

    def shift(self, offset):
        """Shifts a span by offset

        Args:
            offset (int): The number to change start and end by

        """
        return Span(self.start + offset, self.end + offset)

    def has_overlap(self, other):
        """Determines whether two spans overlap."""
        return self.end >= other.start and other.end >= self.start

    def __iter__(self):
        for index in range(self.start, self.end + 1):
            yield index

    def __len__(self):
        return self.end - self.start + 1

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.start == other.start and self.end == other.end
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, self.__class__):
            return len(self) > len(other)
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, self.__class__):
            return len(self) >= len(other)
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return len(self) < len(other)
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, self.__class__):
            return len(self) <= len(other)
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __repr__(self):
        return "{}(start={}, end={})".format(
            self.__class__.__name__, self.start, self.end
        )


class Query:
    """The query object is responsible for processing and normalizing raw user text input so that
    classifiers can deal with it. A query stores three forms of text: raw text, processed text, and
    normalized text. The query object is also responsible for translating text ranges across these
    forms.

    Attributes:
        raw_text (str): the original input text
        processed_text (str): the text after it has been preprocessed. The pre-processing happens
            at the application level and is generally used for special characters
        normalized_tokens (tuple of str): a list of normalized tokens
        system_entity_candidates (tuple): A list of system entities extracted from the text
        locale (str, optional): The locale representing the ISO 639-1 language code and \
            ISO3166 alpha 2 country code separated by an underscore character.
        language (str, optional): The language code representing ISO 639-1 language codes.
        time_zone (str): The IANA id for the time zone in which the query originated
            such as 'America/Los_Angeles'
        timestamp (long, optional): A unix timestamp used as the reference time
            If not specified, the current system time is used. If `time_zone`
            is not also specified, this parameter is ignored
        stemmed_tokens (list): A sequence of stemmed tokens for the query text
    """

    # TODO: look into using __slots__

    def __init__(
        self,
        raw_text,
        processed_text,
        normalized_tokens,
        char_maps,
        locale=None,
        language=None,
        time_zone=None,
        timestamp=None,
        stemmed_tokens=None,
    ):
        """Creates a query object

        Args:
            raw_text (str): the original input text
            processed_text (str): the input text after it has been preprocessed
            normalized_tokens (list of dict): List tokens outputted by
                a tokenizer
            char_maps (dict): Mappings between character indices in raw,
                processed and normalized text
        """
        self._normalized_tokens = normalized_tokens
        norm_text = " ".join([t["entity"] for t in self._normalized_tokens])
        self._texts = (raw_text, processed_text, norm_text)
        self._char_maps = char_maps
        self.system_entity_candidates = ()
        self._locale = locale
        self._language = language
        self._time_zone = time_zone
        self._timestamp = timestamp
        self.stemmed_tokens = stemmed_tokens or tuple()

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
    def stemmed_text(self):
        """The stemmed input text"""
        return " ".join(self.stemmed_tokens)

    @property
    def normalized_tokens(self):
        """The tokens of the normalized input text"""
        return tuple((token["entity"] for token in self._normalized_tokens))

    @property
    def language(self):
        """Language of the query specified using a 639-2 code."""
        return self._language

    @property
    def locale(self):
        """The locale representing the ISO 639-1/2 language code and
        ISO3166 alpha 2 country code separated by an underscore character."""
        return self._locale

    @property
    def time_zone(self):
        """The IANA id for the time zone in which the query originated
        such as 'America/Los_Angeles'.
        """
        return self._time_zone

    @property
    def timestamp(self):
        """A unix timestamp for when the time query was created. If `time_zone` is None,
        this parameter is ignored.
        """
        return self._timestamp

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
        return Span(
            self.transform_index(text_span.start, form_in, form_out),
            self.transform_index(text_span.end, form_in, form_out),
        )

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
            raise ValueError("Invalid text form")

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
            raise ValueError(
                "'{}' form cannot be processed".format(TEXT_FORM_NORMALIZED)
            )
        mapping_key = (form_in, (form_in + 1))
        try:
            mapping = self._char_maps[mapping_key]
        except KeyError:
            # mapping doesn't exist -> use identity
            return index
        # None for mapping means 1-1 mapping
        try:
            return mapping[index] if mapping else index
        except KeyError as e:
            raise ValueError("Invalid index {}".format(index)) from e

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
        except KeyError as e:
            raise ValueError("Invalid index {}".format(index)) from e

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


class ProcessedQuery:
    """A processed query contains a query and the additional metadata that has been labeled or
    predicted.


    Attributes:
        query (Query): The underlying query object.
        domain (str): The domain of the query
        entities (list): A list of entities present in this query
        intent (str): The intent of the query
        is_gold (bool): Indicates whether the details in this query were predicted or human labeled
        nbest_transcripts_queries (list): A list of n best transcript queries
        nbest_transcripts_entities (list): A list of lists of entities for each query
        nbest_aligned_entities (list): A list of lists of aligned entities
        confidence (dict): A dictionary of the class probas for the domain and intent classifier
    """

    # TODO: look into using __slots__

    def __init__(
        self,
        query,
        domain=None,
        intent=None,
        entities=None,
        is_gold=False,
        nbest_transcripts_queries=None,
        nbest_transcripts_entities=None,
        nbest_aligned_entities=None,
        confidence=None,
    ):
        self.query = query
        self.domain = domain
        self.intent = intent
        self.entities = None if entities is None else tuple(entities)
        self.is_gold = is_gold
        self.nbest_transcripts_queries = nbest_transcripts_queries
        self.nbest_transcripts_entities = nbest_transcripts_entities
        self.nbest_aligned_entities = nbest_aligned_entities
        self.confidence = confidence

    def to_dict(self):
        """Converts the processed query into a dictionary"""
        base = {
            "text": self.query.text,
            "domain": self.domain,
            "intent": self.intent,
            "entities": None
            if self.entities is None
            else [e.to_dict() for e in self.entities],
        }
        if self.nbest_transcripts_queries:
            base["nbest_transcripts_text"] = [
                q.text for q in self.nbest_transcripts_queries
            ]
        if self.nbest_transcripts_entities:
            base["nbest_transcripts_entities"] = [
                [e.to_dict() for e in n_entities]
                for n_entities in self.nbest_transcripts_entities
            ]
        if self.nbest_aligned_entities:
            base["nbest_aligned_entities"] = [
                [{"text": e.entity.text, "type": e.entity.type} for e in n_entities]
                for n_entities in self.nbest_aligned_entities
            ]

        if self.confidence:
            base["confidences"] = self.confidence
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
        return msg.format(
            self.__class__.__name__,
            self.query.text,
            self.domain,
            self.intent,
            len(self.entities),
            ", gold" if self.is_gold else "",
        )


class NestedEntity:
    """An entity with the context of the query it came from, along with \
        information like the entity's parent and children.

    Attributes:
        texts (tuple): Tuple containing the three forms of text: raw text, \
            processed text, and normalized text
        spans (tuple): Tuple containing the character index spans of the \
            text for this entity for each text form
        token_spans (tuple): Tuple containing the token index spans of the \
            text for this entity for each text form
        entity (Entity): The entity object
        parent (NestedEntity): The parent of the nested entity
        children (tuple of NestedEntity): A tuple of children nested entities
    """

    def __init__(self, texts, spans, token_spans, entity, children=None):
        self._texts = texts
        self._spans = spans
        self._token_spans = token_spans
        self.entity = entity
        self.parent = None

        for child in children or ():
            child.parent = self
        if children:
            self.children = tuple(sorted(children, key=lambda c: c.span.start))
        else:
            self.children = None

    def with_children(self, children):
        """Creates a copy of this entity with the provided children"""
        return self.__class__(
            self._texts, self._spans, self._token_spans, self.entity, children
        )

    @classmethod
    def from_query(
        cls,
        query,
        span=None,
        normalized_span=None,
        entity_type=None,
        role=None,
        entity=None,
        parent_offset=None,
        children=None,
    ):
        """Creates an entity node using a parent entity node

        Args:
            query (Query): Description
            span (Span): The span of the entity in the query's raw text
            normalized_span (None, optional): The span of the entity in the
                query's normalized text
            entity_type (str, optional): The entity type. One of this and entity
                must be provided
            role (str, optional): The entity role. Ignored if entity is provided.
            entity (Entity, optional): The entity. One of this and entity must
                be provided
            parent_offset (int): The offset of the parent in the query
            children (None, optional): Description

        Returns:
            the created entity

        """

        def _get_form_details(query_span, offset, form_in, form_out):
            span_out = query.transform_span(query_span, form_in, form_out)
            full_text = query.get_text_form(form_out)
            text = span_out.slice(full_text)
            # The span range is till the span_out or max to the second last char
            tok_start = 0
            span_range = min(span_out.start, len(full_text) - 1)
            for idx, current_char in enumerate(full_text[:span_range]):
                # Increment the counter only if a whitespace follows a non-whitespace
                next_char = full_text[idx + 1]
                if not current_char.isspace() and next_char.isspace():
                    tok_start += 1
            tok_span = Span(tok_start, tok_start - 1 + len(text.split()))
            # convert span from query's indexing to parent's indexing
            if offset is not None:
                offset_out = query.transform_index(offset, form_in, form_out)
                span_out = span_out.shift(-offset_out)
                tok_offset = len(full_text[:offset_out].split())
                tok_span.shift(-tok_offset)

            return text, span_out, tok_span

        if span:
            query_span = (
                span.shift(parent_offset) if parent_offset is not None else span
            )
            form_in = TEXT_FORM_RAW
        elif normalized_span:
            query_span = (
                normalized_span.shift(parent_offset)
                if parent_offset is not None
                else normalized_span
            )
            form_in = TEXT_FORM_NORMALIZED

        texts, spans, tok_spans = list(
            zip(
                *[
                    _get_form_details(query_span, parent_offset, form_in, form_out)
                    for form_out in TEXT_FORMS
                ]
            )
        )

        if entity is None:
            if entity_type is None:
                raise ValueError("Either 'entity' or 'entity_type' must be specified")
            entity = Entity(texts[0], entity_type, role=role)

        return cls(texts, spans, tok_spans, entity, children)

    def to_dict(self):
        """Converts the query entity into a dictionary"""
        base = self.entity.to_dict()
        base["span"] = self.span.to_dict()
        if self.children:
            base["children"] = [c.to_dict() for c in self.children]
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
        return "{}{} '{}' {}-{}".format(
            self.entity.type,
            ":" + self.entity.role if self.entity.role else "",
            self.text,
            self.span.start,
            self.span.end,
        )

    def __repr__(self):
        msg = "<{} {!r} ({!r}) char: [{!r}-{!r}], tok: [{!r}-{!r}]>"
        return msg.format(
            self.__class__.__name__,
            self.text,
            self.entity.type,
            self.span.start,
            self.span.end,
            self.token_span.start,
            self.token_span.end,
        )


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


class Entity:
    """An Entity is any important piece of text that provides more information about the user
    intent.

    Attributes:
        text (str): The text contents that span the entity
        type (str): The type of the entity
        role (str): The role of the entity
        value (dict): The resolved value of the entity
        display_text (str): A human readable text representation of the entity for use in natural
            language responses.
        confidence (float): A confidence value from 0 to 1 about how confident the entity
            recognizer was for the given class label.
        is_system_entity (bool): True if the entity is a system entity
    """

    # TODO: look into using __slots__

    def __init__(
        self,
        text,
        entity_type,
        role=None,
        value=None,
        display_text=None,
        confidence=None,
    ):
        self.text = text
        self.type = entity_type
        self.role = role
        self.value = value
        self.display_text = display_text
        self.confidence = confidence
        self.is_system_entity = self.__class__.is_system_entity(entity_type)

    @staticmethod
    def is_system_entity(entity_type):  # pylint: disable=method-hidden
        """Checks whether the provided entity type is a MindMeld-recognized system entity.

        Args:
            entity_type (str): An entity type

        Returns:
            bool: True if the entity is a system entity type, else False
        """
        return entity_type.startswith("sys_")

    def to_dict(self):
        """Converts the entity into a dictionary"""
        base = {"text": self.text, "type": self.type, "role": self.role}
        for field in ["value", "display_text", "confidence"]:
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


class FormEntity:
    """A form entity is used for defining custom objects for the entity form used in
    AutoEntityFilling (slot-filling).

    Attributes:
        entity (str): Entity name
        role (str, optional): The role of the entity
        responses(list/str, optional): Message(s) for prompting the user for missing entities
        retry_response (list/str, optional): Message(s) for re-prompting users. If not provided,
        defaults to responses
        value (str, optional): The resolved value of the entity
        default_eval(bool, optional): Use system validation (default: True)
        hints(list, optional): Developer defined list of keywords to verify the
        user input against
        custom_eval(func, optional): custom validation function (should return either bool:
        validated or not) or a custom resolved value for the entity. If custom resolved value
        is returned, the slot response is considered to be valid.
    """

    def __init__(
        self,
        entity: str,
        role: Optional[str] = None,
        responses: Optional[List[str]] = None,
        retry_response: Optional[List[str]] = None,
        value: Optional[Dict] = None,
        default_eval: Optional[bool] = True,
        hints: Optional[List[str]] = None,
        custom_eval: Optional[str] = None,
    ):
        self.entity = entity
        self.role = role

        if isinstance(responses, str):
            responses = [responses]
        self.responses = responses or [
            "Please provide value for: {}".format(self.entity)
        ]

        if isinstance(retry_response, str):
            retry_response = [retry_response]
        self.retry_response = retry_response or self.responses
        self.value = value
        self.default_eval = default_eval
        self.hints = hints
        self.custom_eval = custom_eval

        if not self.entity or not isinstance(self.entity, str):
            raise TypeError("Entity cannot be empty.")
        if self.custom_eval and not callable(custom_eval):
            raise TypeError("Invalid custom validation function type.")

    def to_dict(self):
        """Converts the entity into a dictionary"""
        base = {}
        for field in self.__dict__:
            val = getattr(self, field)
            if val is not None:
                if isinstance(val, immutables.Map):
                    val = dict(val)
                base[field] = val

        return base


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
    filtered = query_entities
    i = 0
    while i < len(filtered):
        include_target = True
        target = filtered[i]
        j = i + 1
        while j < len(filtered):
            other = filtered[j]
            if _is_superset(target, other) and not _is_same_span(target, other):
                logger.debug(
                    "Removing {{%s|%s}} entity in query %d since it is a "
                    "subset of another.",
                    other.text,
                    other.entity.type,
                    i,
                )
                del filtered[j]
                continue

            if _is_subset(target, other) and not _is_same_span(target, other):
                logger.debug(
                    "Removing {{%s|%s}} entity in query %d since it is a "
                    "subset of another.",
                    target.text,
                    target.entity.type,
                    i,
                )
                del filtered[i]
                include_target = False
                break

            if _is_same_span(target, other) or _is_overlapping(target, other):
                if target.entity.confidence >= other.entity.confidence:
                    logger.debug(
                        "Removing {{%s|%s}} entity in query %d since it overlaps "
                        "with another.",
                        other.text,
                        other.entity.type,
                        i,
                    )
                    del filtered[j]
                    continue

                if target.entity.confidence < other.entity.confidence:
                    logger.debug(
                        "Removing {{%s|%s}} entity in query %d since it overlaps "
                        "with another.",
                        target.text,
                        target.entity.type,
                        i,
                    )
                    del filtered[i]
                    include_target = False
                    break
            j += 1
        if include_target:
            i += 1

    return filtered


def _is_subset(target, other):
    return (target.start >= other.start) and (target.end <= other.end)


def _is_superset(target, other):
    return (target.start <= other.start) and (target.end >= other.end)


def _is_same_span(target, other):
    return _is_superset(target, other) and _is_subset(target, other)


def _is_overlapping(target, other):
    overlap = _get_overlap(target, other)
    return overlap and not _is_subset(target, other) and not _is_superset(target, other)


def _get_overlap(target, other):
    target_range = range(target.start, target.end + 1)
    predicted_range = range(other.start, other.end + 1)
    return set(target_range).intersection(predicted_range)
