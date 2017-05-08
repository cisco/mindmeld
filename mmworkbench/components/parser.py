# -*- coding: utf-8 -*-
"""
This module contains the language parser component of the Workbench natural language processor
"""
from __future__ import unicode_literals
from builtins import object

from collections import defaultdict, namedtuple, OrderedDict

from nltk import FeatureChartParser
from nltk.grammar import FeatureGrammar
from nltk.featstruct import Feature

from ._config import get_parser_config

from ..core import EntityGroup, Span

START_SYMBOL = 'S'
HEAD_SYMBOL = 'H'

TYPE_FEATURE = Feature('type', display='prefix')

START_SYMBOLS = frozenset({START_SYMBOL, HEAD_SYMBOL})


class Parser(object):
    """
    A language parser which is used to extract relations between entities in a
    given query and group related entities together.

    The parser uses a context free grammar based on a configuration to generate
    candidate entity groupings. Heuristics are then used to rank and select a
    grouping.

    This rule based parser will be helpful in many situations, but if you have
    a sufficiently sophisticated entity hierarchy, you may benefit from using a
    statistical approach.
    """

    def __init__(self, resource_loader=None, config=None):
        """Summary

        Args:
            resource_loader (ResourceLoader): An object which can load
                resources for the parser.
            config (dict, optional): The configuration for the parser. If none
                is provided the app config will be loaded.
        """
        if not resource_loader and not config:
            raise ValueError('Parser requires either a configuration or a resource loader')
        app_path = resource_loader.app_path if resource_loader else None
        self._resource_loader = resource_loader
        self.config = get_parser_config(app_path, config)
        self._grammar = FeatureGrammar.fromstring(generate_grammar(self.config))
        self._parser = FeatureChartParser(self._grammar)

    def parse_entities(self, query, entities):
        """Finds groupings of entities for the given query.

        Args:
            query (Query): The query being parsed
            entities (list of QueryEntity): The entities to find groupings for

        """
        return self._parse(query, entities)

    def _parse(self, query, entities):
        entity_type_count = defaultdict(int)
        entity_dict = {}
        tokens = []  # tokens to be parsed

        # assumes tokens are sorted
        for entity in entities:
            entity_type = entity.entity.type
            entity_id = '{}{}'.format(entity_type, entity_type_count[entity_type])
            entity_type_count[entity_type] += 1
            entity_dict[entity_id] = entity
            tokens.append(entity_id)

        parses = self._parser.parse(tokens)
        if not parses:
            return []
        resolved = OrderedDict()
        for parse in parses:
            resolved[self._resolve_parse(parse)] = None
        filtered = (p for p in resolved.keys())

        # Prefer parses with fewer groups
        parses = list(sorted(filtered, key=len))
        if not parses:
            return []
        filtered = (p for p in parses if len(p) <= len(parses[0]))

        # Prefer parses with minimal distance from dependents to heads
        parses = list(sorted(filtered, key=lambda p: self._parse_distance(p, query, entity_dict)))
        if not parses:
            return []
        min_parse_dist = self._parse_distance(parses[0], query, entity_dict)
        filtered = (p for p in parses
                    if self._parse_distance(p, query, entity_dict) <= min_parse_dist)

        # TODO: apply precedence

        # if we still have more than one, choose the first
        for parse in parses:
            return sorted((g.to_entity_group(entity_dict) for g in parse if g.dependents),
                          key=lambda g: g.span.start)

    def _parse_distance(self, parse, query, entity_dict):
        total_link_distance = 0
        stack = list(parse)
        while stack:
            node = stack.pop()
            head = entity_dict[node.id]

            for dep in node.dependents:
                if dep.dependents:
                    stack.append(dep)
                    continue
                child = entity_dict[dep.id]
                if child.token_span.start > head.token_span.start:
                    intra_entity_span = Span(head.token_span.end, child.token_span.start)
                else:
                    intra_entity_span = Span(child.token_span.end, head.token_span.start)
                link_distance = 0
                for token in intra_entity_span.slice(query.text.split(' ')):
                    if token in self.config[node.type][dep.type]['linking_words']:
                        link_distance -= 0.5
                    else:
                        link_distance += 1
                total_link_distance += link_distance

        return total_link_distance

    @classmethod
    def _resolve_parse(cls, node):
        groups = set()
        for child in node:
            child_symbol = child.label()[TYPE_FEATURE]
            if child_symbol in START_SYMBOLS:
                groups.update(cls._resolve_parse(child))
            else:
                group = cls._resolve_group(child).freeze()
                groups.add(group)
        return frozenset(groups)

    @classmethod
    def _resolve_group(cls, node):
        symbol = node.label()[TYPE_FEATURE]
        if not symbol[0].isupper():
            # this node is a generic entity of type {symbol}, its child is the terminal
            return _EntityNode(symbol, node[0], None)

        # if first char is capitalized, this is a group!
        group_type = symbol.lower()
        dependents = set()
        for child in node:
            child_symbol = child.label()[TYPE_FEATURE]
            if child_symbol == symbol:
                # this is the ancestor of this group
                group = cls._resolve_group(child)
            elif child_symbol == group_type:
                # this is the root ancestor of this group
                group = cls._resolve_group(child)
                group = _EntityNode(group.type, group.id, set())
            else:
                dependents.add(cls._resolve_group(child).freeze())

        group.dependents.update(dependents)
        return group


class _EntityNode(namedtuple('EntityNode', ('type', 'id', 'dependents'))):
    """A private tree data structure used to parse queries

    EntityNodes use sets and are conditionally hashable. This makes it easy to check the
    equivalence of parse trees represented as entity nodes.
    """

    def freeze(self):
        """Converts to a 'frozen' representation that can be hashed"""
        if self.dependents is None:
            return self

        frozen_dependents = frozenset((d.freeze() for d in self.dependents))
        return _EntityNode(self.type, self.id, frozen_dependents)

    def pretty(self, indent=0):
        """Pretty prints the entity node.

        Primarily useful for debugging."""
        text = ('  ' * indent) + self.id

        if not self.dependents:
            return text

        return text + '\n' + '\n'.join(dep.pretty(indent+1) for dep in self.dependents)

    def to_entity_group(self, entity_dict, is_root=True):
        """Converts a node to an EntityGroup

        Args:
            entity_dict (dict): A mapping from entity ids to the corresponding QueryEntity objects

        """
        if not self.dependents and not is_root:
            return entity_dict[self.id]

        dependents = tuple((c.to_entity_group(entity_dict, is_root=False) for c in self.dependents))
        return EntityGroup(entity_dict[self.id], dependents)


def _build_symbol_template(group, features):
    """Builds a template for a symbol in a feature CFG.

    Args:
        group (str): The group the template is for
        features (iterable): The names of features which should be included in
            the template

    Example:
    >>> _build_symbol_template('Group', {'feat1', 'feat2'})
    "Group[feat1={feat1}, feat2={feat2}]"

    """
    symbol_template = group
    for feature in features:
        if symbol_template is group:
            symbol_template += '['
        else:
            symbol_template += ', '
        symbol_template += '{0}={{{0}}}'.format(feature)
    if symbol_template is not group:
        symbol_template += ']'
    return symbol_template


def _generate_dependent_rules(dep_type, config, symbol_template, features, head_types):
    """Generates the rules for a dependent entity

    Args:
        config (dict): A dictionary containing the configuration for this dependent
        symbol_template (str): A symbol template
        features (iterable): A list of features for this symbol
        head_types (set): All symbols which have dependents

    Yields:
        str: A rule for the dependent
    """
    # If dependent is a group, its symbol should be capitalized
    dep_symbol = dep_type.capitalize() if dep_type in head_types else dep_type

    max_instances = config.get('max_instances')
    if max_instances is None:
        # pass through features unchanged
        lhs = symbol_template.format(**{f: '?' + chr(ord('a') + i)
                                        for i, f in enumerate(features)})
        rhs = lhs
        if config.get('left'):
            yield '{lhs} -> {dep} {rhs}'.format(lhs=lhs, rhs=rhs, dep=dep_symbol)
        if config.get('right'):
            yield '{lhs} -> {rhs} {dep}'.format(lhs=lhs, rhs=rhs, dep=dep_symbol)
    else:
        for dep_count in range(max_instances):
            feature_dict = {f: '?' + chr(ord('a') + i) for i, f in enumerate(features)
                            if f is not dep_type}
            feature_dict[dep_type] = dep_count
            rhs = symbol_template.format(**feature_dict)
            feature_dict[dep_type] = dep_count + 1
            lhs = symbol_template.format(**feature_dict)

            if config.get('left'):
                yield '{lhs} -> {dep} {rhs}'.format(lhs=lhs, rhs=rhs, dep=dep_symbol)
            if config.get('right'):
                yield '{lhs} -> {rhs} {dep}'.format(lhs=lhs, rhs=rhs, dep=dep_symbol)


def generate_grammar(config, unique_entities=20):
    """Generates a feature context free grammar from the provided parser config

    Args:
        config (dict): The parser configuration
        unique_entities (int, optional): The number of entities of the same type that should be
            permitted in the same query

    Returns:
        str: a string containing the grammar with rules separated by line
    """
    # start rules
    rules = ['{} -> {}'.format(START_SYMBOL, HEAD_SYMBOL),  # The start rule
             '{0} -> {0} {0}'.format(HEAD_SYMBOL)]  # Allow multiple heads

    # the set of all heads
    head_types = set(config.keys())

    # the set of all dependents
    dependent_types = set((t for g in config.values() for t in g))

    all_types = head_types.union(dependent_types)

    # create rules for each group
    for entity in head_types:
        # the symbol for a group is the capitalized version of the string
        group = entity.capitalize()
        rules.append('H -> {}'.format(group))

        dep_configs = config[entity]
        # If a dependent has a max number of instances, we will track it as a feature
        features = [t for t, d in dep_configs.items() if d.get('max_instances') is not None]

        symbol_template = _build_symbol_template(group, features)

        # basic rule with features initialized to 0
        rules.append('{} -> {}'.format(symbol_template.format(**{f: 0 for f in features}), entity))

        for dep_type, dep_config in dep_configs.items():
            rules.extend(_generate_dependent_rules(dep_type, dep_config, symbol_template,
                                                   features, head_types))
    for entity in all_types:
        for idx in range(unique_entities):
            rules.append("{0} -> '{0}{1}'".format(entity, idx))

    return '\n'.join(rules)
