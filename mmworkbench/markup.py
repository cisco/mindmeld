# -*- coding: utf-8 -*-
"""The markup module contains functions for interacting with the MindMeld Markup language for
representing annotations of query text inline.
"""
from __future__ import unicode_literals
from future.utils import raise_from

from .core import Entity, EntityGroup, NestedEntity, ProcessedQuery, QueryEntity, Span
from .exceptions import MarkupError, SystemEntityMarkupError, SystemEntityResolutionError
from .ser import resolve_system_entity
from .query_factory import QueryFactory

ENTITY_START = '{'
ENTITY_END = '}'
GROUP_START = '['
GROUP_END = ']'
META_SPLIT = '|'

START_CHARACTERS = frozenset({ENTITY_START, GROUP_START})
END_CHARACTERS = frozenset({ENTITY_END, GROUP_END})
SPECIAL_CHARACTERS = frozenset({ENTITY_START, ENTITY_END, GROUP_START, GROUP_END, META_SPLIT})


MINDMELD_FORMAT = 'mindmeld'
BRAT_FORMAT = 'brat'
MARKUP_FORMATS = frozenset({MINDMELD_FORMAT, BRAT_FORMAT})


def load_query(markup, query_factory=None, domain=None, intent=None, is_gold=False):
    """Creates a processed query object from marked up query text.

    Args:
        markup (str): The marked up query text.
        query_factory (QueryFactory, optional): An object which can create
            queries.
        domain (str, optional): The name of the domain annotated for the query.
        intent (str, optional): The name of the intent annotated for the query.
        is_gold (bool, optional): True if the markup passed in is a reference,
            human-labeled example. Defaults to False.

    Returns:
        ProcessedQuery: a processed query
    """
    query_factory = query_factory or QueryFactory.create_query_factory()
    try:
        raw_text, annotations = _parse_tokens(_tokenize_markup(markup))
        query = query_factory.create_query(raw_text)
        entities = _process_annotations(query, annotations)
    except MarkupError as exc:
        msg = 'Invalid markup in query {!r}: {}'
        raise_from(MarkupError(msg.format(markup, exc)), exc)
    except SystemEntityResolutionError as exc:
        msg = "Unable to load query {!r}: {}"
        raise_from(SystemEntityMarkupError(msg.format(markup, exc)), exc)

    return ProcessedQuery(query, domain=domain, intent=intent, entities=entities, is_gold=is_gold)


def load_query_file(file_path, query_factory=None, domain=None, intent=None, is_gold=False):
    """Loads the queries from the specified file

    Args:
        file_path (str): The path of the file to load
        query_factory (QueryFactory, optional): An object which can create
            queries.
        domain (str, optional): The name of the domain annotated for the query.
        intent (str, optional): The name of the intent annotated for the query.
        is_gold (bool, optional): True if the markup passed in is a reference,
            human-labeled example. Defaults to False.

    Returns:
        ProcessedQuery: a processed query
    """
    query_factory = query_factory or QueryFactory.create_query_factory()

    queries = []
    for query_text in _read_query_file(file_path):
        if query_text[0] == '-':
            continue
        query = load_query(query_text, query_factory, domain, intent, is_gold=is_gold)
        queries.append(query)
    return queries


def mark_down_file(file_path):
    """

    Args:
        file_path (str): The path of the file to load
    """
    for markup in _read_query_file(file_path):
        yield mark_down(markup)


def _read_query_file(file_path):
    """Summary

    Args:
        file_path (str): The path of the file to load

    Yields:
        str: query text for each line
    """
    import codecs
    with codecs.open(file_path, encoding='utf-8') as queries_file:
        for line in queries_file:
            line = line.strip()
            # only create query if line is not empty string
            query_text = line.split('\t')[0].strip()
            if query_text:
                yield query_text


def _process_annotations(query, annotations):
    """

    Returns:
        list of ProcessedQuery:
    """
    entities = []
    stack = []

    def _close_ann(ann):
        if ann['ann_type'] == 'group':
            try:
                head = ann['head']
            except KeyError as exc:
                msg = 'Group between {} and {} missing head'.format(ann['start'], ann['end'])
                raise_from(MarkupError(msg), exc)
            try:
                children = ann['children']
            except KeyError as exc:
                msg = 'Group between {} and {} missing children'.format(ann['start'], ann['end'])
                raise_from(MarkupError(msg), exc)
            entity = head.with_children(children)
            entities.remove(head)
            entities.append(entity)
            if ann.get('parent'):
                parent = ann.get('parent')
                children = parent.get('children', [])
                children.append(entity)
                parent['children'] = children

        if ann['ann_type'] == 'entity':
            span = Span(ann['start'], ann['end'])
            if Entity.is_system_entity(ann['type']):
                raw_entity = resolve_system_entity(query, ann['type'], span).entity
            else:
                try:
                    value = {'children': ann['children']}
                except KeyError:
                    value = None
                raw_entity = Entity(ann['text'], ann['type'], role=ann.get('role'), value=value)

            if ann.get('parent'):
                parent = ann.get('parent')
                if parent['ann_type'] == 'entity':
                    children = parent.get('children', [])
                    children.append(NestedEntity.from_query(query, span.shift(-parent['start']),
                                                            entity=raw_entity,
                                                            parent_offset=parent['start']))
                    parent['children'] = children
                if parent['ann_type'] == 'group':
                    entity = QueryEntity.from_query(query, span, entity=raw_entity)
                    entities.append(entity)

                    if parent['type'] == ann['type']:
                        # this is the head
                        parent['head'] = entity
                    else:
                        children = parent.get('children', [])
                        children.append(entity)
                        parent['children'] = children

            else:
                entities.append(QueryEntity.from_query(query, span, entity=raw_entity))

    def _open_ann(ann):
        if stack:
            ann['parent'] = stack[-1]
        stack.append(ann)

    for ann in annotations:
        while stack and stack[-1]['depth'] >= ann['depth']:
            # if there are annotations on the stack of the same or greater depth,
            # they have no more children so close them
            _close_ann(stack.pop())

        _open_ann(ann)

    while stack:
        _close_ann(stack.pop())

    entities = sorted(entities, key=lambda e: e.span.start)

    return entities


def _parse_tokens(tokens):
    text = ''
    annotations = []
    stack = []
    token_is_meta = False
    for token in tokens:
        if token in START_CHARACTERS:
            annotation = {
                'start': len(text),
                'ann_type': 'group' if token == GROUP_START else 'entity',
                'depth': len(stack)
            }
            stack.append(annotation)
        elif token == META_SPLIT:
            token_is_meta = True
        elif token in END_CHARACTERS:
            annotation = stack.pop()
            annotation['end'] = len(text) - 1  # the index of the last character
            annotation['text'] = text[annotation['start']:annotation['end'] + 1]
            token_is_meta = False
            annotations.append(annotation)
        elif token_is_meta:
            annotation = stack[-1]
            if annotation['ann_type'] == 'group':
                key = 'type'
            else:
                key = 'role' if 'type' in annotation else 'type'
            annotation[key] = token
        else:
            text += token

    annotations = sorted(annotations, key=lambda a: a['depth'])
    annotations = sorted(annotations, key=lambda a: a['start'])

    return text, annotations


def _tokenize_markup(markup):
    """Converts markup into a series of 'tokens'.

    A token can fall into one of 5 general categories:
     - raw text
     - a marker indicating the start of an entity or entity group
     - a marker indicating the end of an entity or entity group
     - a marker indicating the start of a label for an entity or entity group
     - a label for an entity or entity group

    Args:
        markup (str): The markup text

    Raises:
        MarkupError: When markup is invalid
    """
    token = ''
    token_is_meta = False
    open_annotations = {
        'group': 0,
        'entity': 0
    }
    for idx, char in enumerate(markup):
        if char in SPECIAL_CHARACTERS:
            if char in START_CHARACTERS:
                if token:
                    yield token
                    token = ''
                if char == GROUP_START:
                    open_annotations['group'] += 1
                else:
                    open_annotations['entity'] += 1
                yield char
            elif char == META_SPLIT:
                # TODO: improve this check
                # if not token:
                #     raise MarkupError('Entity or group text is empty at position {}'.format(idx))
                if token:
                    yield token
                    token = ''
                token_is_meta = True
                yield char
            elif char in END_CHARACTERS:
                if char == GROUP_END:
                    key = 'group'
                else:
                    key = 'entity'
                if open_annotations[key] == 0:
                    raise MarkupError('Mismatched end for {} at position {}'.format(key, idx))
                if not token_is_meta:
                    raise MarkupError('Missing label for {} at position {}'.format(key, idx))
                if not token:
                    raise MarkupError('Empty label for {} at position {}'.format(key, idx))
                open_annotations[key] -= 1

                yield token
                token = ''
                token_is_meta = False
                yield char

            continue

        token += char

    for key in open_annotations:
        if open_annotations[key]:
            raise MarkupError('Mismatched start for {}'.format(key))

    if token:
        yield token


def dump_query(processed_query, markup_format=MINDMELD_FORMAT, **kwargs):
    """Converts a processed query into marked up query text.

    Args:
        processed_query (ProcessedQuery): The query to convert
        markup_format (str, optional): The format to use. Valid formats include
            'mindmeld' and 'brat'. Defaults to 'mindmeld'
        **kwargs: additional format specific parameters may be passed in as
            keyword arguments.

    Returns:
        str: A marked up representation of the query

    Raises:
        ValueError: Description
    """
    if markup_format not in MARKUP_FORMATS:
        raise ValueError('Invalid markup format {!r}'.format(markup_format))
    return {
        MINDMELD_FORMAT: _dump_mindmeld,
        BRAT_FORMAT: _dump_brat
    }[markup_format](processed_query, **kwargs)


def dump_queries(queries, markup_format=MINDMELD_FORMAT, **kwargs):
    """Converts a collection of processed queries to marked up query text

    Args:
        queries (iterable): A collection of processed queries
        markup_format (str, optional): The format to use. Valid formats include
            'mindmeld' and 'brat'. Defaults to 'mindmeld'
        **kwargs: additional format specific parameters may be passed in as
            keyword arguments.

    Yields:
        str or tuple: A marked up representation of the query
    """
    if markup_format == BRAT_FORMAT:
        for result in _dump_brat_queries(queries, **kwargs):
            yield result
        return

    for query in queries:
        yield dump_query(query, markup_format, **kwargs)


def _dump_brat_queries(queries, **kwargs):
    entity_offset = kwargs.get('entity_offset', 0)
    relation_offset = kwargs.get('relation_offset', 0)
    char_offset = kwargs.get('char_offset', 0)

    for query in queries:
        text, annotations = _dump_brat(query, char_offset=char_offset, entity_offset=entity_offset,
                                       relation_offset=relation_offset)
        yield text, annotations

        char_offset += len(text) + 1
        entity_offset += len(query.entities)
        relation_offset += len(annotations.split('\n')) - len(query.entities)


def _dump_brat(processed_query, **kwargs):
    # TODO: support nested entities
    entity_offset = kwargs.get('entity_offset', 0)
    relation_offset = kwargs.get('relation_offset', 0)
    char_offset = kwargs.get('char_offset', 0)
    text = processed_query.query.text
    annotations = []
    entity_dict = {}
    for index, entity in enumerate(processed_query.entities):
        params = {
            'index': entity_offset + index + 1,
            'entity': entity.entity.type.capitalize(),
            'start': char_offset + entity.span.start,
            'end': char_offset + entity.span.end + 1,
            'text': entity.entity.text
        }
        entity_dict[(entity.entity.type, entity.span.start)] = params['index']
        annotations.append('T{index}\t{entity} {start} {end}\t{text}'.format(**params))

    stack = list(reversed(processed_query.entity_groups))
    while stack:
        group = stack.pop()
        for dep in group.dependents:
            if isinstance(dep, EntityGroup):
                stack.append(dep)
                dep = dep.head

            relation_offset += 1  # increment this first so first index is 1
            params = {
                'index': relation_offset,
                'entity': dep.entity.type,
                'head': entity_dict[(group.head.entity.type, group.head.span.start)],
                'dependent': entity_dict[(dep.entity.type, dep.span.start)]
            }
            annotation = 'R{index}\t{entity} Arg1:T{head} Arg2:T{dependent}\t'.format(**params)
            annotations.append(annotation)

    return (text, '\n'.join(annotations))


def _dump_mindmeld(processed_query, **kwargs):
    raw_text = processed_query.query.text
    markup = _mark_up_entities(raw_text, processed_query.entities)
    return markup


def validate_markup(markup, query_factory):
    """Checks whether the markup text is well-formed.

    Args:
        markup (str): The marked up query text
        query_factory (QueryFactory): An object which can create queries

    Returns:
        bool: True if the markup is valid
    """
    return NotImplemented


def _mark_up_entities(query_str, entities):
    annotations = []
    for entity in entities or tuple():
        annotations.extend(_annotations_for_entity(entity))

    # remove duplicates from annotations
    ann_map = {}
    for ann in annotations:
        ann_key = (ann['ann_type'], ann['start'], ann['end'], ann['type'])
        if ann_key in ann_map:
            # a similar annotation has already been found
            if ann['depth'] < ann_map[ann_key]['depth']:
                # keep the annotation already in the map
                ann = ann_map[ann_key]

        ann_map[ann_key] = ann

    annotations = ann_map.values()
    annotations = sorted(annotations, key=lambda a: a['depth'])
    annotations = sorted(annotations, key=lambda a: a['start'])

    stack = []
    cursor = 0
    tokens = []

    def _open_ann(ann, cursor):
        if cursor < ann['start']:
            tokens.append(query_str[cursor:ann['start']])
        tokens.append(GROUP_START if ann['ann_type'] == 'group' else ENTITY_START)
        stack.append(ann)
        return ann['start']

    def _close_ann(ann, cursor):
        if cursor < ann['end'] + 1:
            tokens.append(query_str[cursor:ann['end'] + 1])
        tokens.append(META_SPLIT)
        tokens.append(ann['type'])
        if ann.get('role') is not None:
            tokens.append(META_SPLIT)
            tokens.append(ann['role'])
        tokens.append(GROUP_END if ann['ann_type'] == 'group' else ENTITY_END)
        cursor = ann['end'] + 1
        return cursor

    for ann in annotations:
        while stack and stack[-1]['depth'] >= ann['depth']:
            # if there are annotations on the stack of the same depth, they have no more children
            # so finish them
            cursor = _close_ann(stack.pop(), cursor)

        cursor = _open_ann(ann, cursor)

    while stack:
        cursor = _close_ann(stack.pop(), cursor)

    tokens.append(query_str[cursor:])
    return ''.join(tokens)


def _annotations_for_entity(entity, depth=0, parent_offset=0):
    annotations = []
    start = entity.span.start + parent_offset
    end = entity.span.end + parent_offset
    if entity.children:
        # This entity is the head of a group. Add an annotation for the group.
        g_start = min(start, entity.children[0].span.start)
        g_end = max(end, entity.children[-1].span.end)
        annotations.append({
            'ann_type': 'group',
            'type': entity.entity.type,
            'start': g_start,
            'end': g_end,
            'depth': depth
        })
        depth += 1
        for child in entity.children:
            # Add annotations for each of the dependents
            annotations.extend(_annotations_for_entity(child, depth))
    annotations.append({
        'ann_type': 'entity',
        'type': entity.entity.type,
        'role': entity.entity.role,
        'start': start,
        'end': end,
        'depth': depth
    })

    # Iterate over 'nested' entities
    if entity.entity.value and isinstance(entity.entity.value, dict):
        children = entity.entity.value.get('children', [])
    else:
        children = []

    for child in children:
        annotations.extend(_annotations_for_entity(child, depth+1, start))

    annotations = sorted(annotations, key=lambda a: a['depth'])
    annotations = sorted(annotations, key=lambda a: a['start'])

    return annotations


def mark_down(markup):
    """Removes all entity mark up from a string

    Args:
        markup (str): A marked up string

    Returns:
        str: A clean string with no mark up
    """
    text, _ = _parse_tokens(_tokenize_markup(markup))
    return text
