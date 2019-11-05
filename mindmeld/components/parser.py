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

"""
This module contains the language parser component of the MindMeld natural language processor
"""
import logging
import time
from collections import OrderedDict, defaultdict, namedtuple

from nltk import FeatureChartParser
from nltk.featstruct import Feature
from nltk.grammar import FeatureGrammar

from .. import path
from ..core import Span
from ..exceptions import ParserTimeout
from ._config import get_parser_config

logger = logging.getLogger(__name__)

START_SYMBOL = "S"
HEAD_SYMBOL = "H"

TYPE_FEATURE = Feature("type", display="prefix")

START_SYMBOLS = frozenset({START_SYMBOL, HEAD_SYMBOL})

MAX_PARSE_TIME = 2.0


class Parser:
    """
    A language parser which is used to extract relations between entities in a
    given query and group related entities together.

    The parser uses a context free grammar based on a configuration to generate
    candidate entity groupings. Heuristics are then used to rank and select a
    grouping.

    This rule based parser will be helpful in many situations, but if you have
    a sufficiently sophisticated entity hierarchy, you may benefit from using a
    statistical approach.

    Attributes:
        config (dict): The parser config.
    """

    def __init__(
        self,
        resource_loader=None,
        config=None,
        allow_relaxed=True,
        domain=None,
        intent=None,
    ):
        """Initializes the parser

        Args:
            resource_loader (ResourceLoader): An object which can load
                resources for the parser.
            config (dict, optional): The configuration for the parser. If none
                is provided the app config will be loaded.
        """
        if not resource_loader and not config:
            raise ValueError(
                "Parser requires either a configuration or a resource loader"
            )
        app_path = resource_loader.app_path if resource_loader else None
        try:
            entity_types = path.get_entity_types(app_path) + ["unk"]
        except TypeError:
            entity_types = {"unk"}
        self._resource_loader = resource_loader
        self.config = get_parser_config(app_path, config, domain, intent) or {}
        configured_entities = set()
        for entity_type, entity_config in self.config.items():
            configured_entities.add(entity_type)
            configured_entities.update(entity_config.keys())

        self._configured_entities = configured_entities
        rules = generate_grammar(self.config, entity_types)
        self._grammar = FeatureGrammar.fromstring(rules)
        self._parser = FeatureChartParser(self._grammar)
        if allow_relaxed:
            relaxed_rules = generate_grammar(self.config, entity_types, relaxed=True)
            self._relaxed_grammar = FeatureGrammar.fromstring(relaxed_rules)
            self._relaxed_parser = FeatureChartParser(self._relaxed_grammar)
        else:
            self._relaxed_grammar = None
            self._relaxed_parser = None

    def parse_entities(
        self,
        query,
        entities,
        all_candidates=False,
        handle_timeout=True,
        timeout=MAX_PARSE_TIME,
    ):
        """Determines groupings of entities for the given query.

        Args:
            query (Query): The query being parsed.
            entities (list[QueryEntity]): The entities to find groupings for.
            all_candidates (bool, optional): Whether to return all the entity candidates.
            handle_timeout (bool, optional): False if an exception should be raised in the event of
                a parsing times out. Defaults to True.
            timeout (float, optional): The amount of time to wait for the parsing to complete.
                By default this is set to MAX_PARSE_TIME. If None is passed, the passing will never
                time out

        Returns:
            (tuple[QueryEntity]): An updated version of the entities collection passed in with \
                their parent and children attributes set appropriately.
        """
        if not self._configured_entities:
            return entities

        if not handle_timeout:
            return self._parse(
                query, entities, all_candidates=all_candidates, timeout=timeout
            )

        try:
            return self._parse(
                query, entities, all_candidates=all_candidates, timeout=timeout
            )
        except ParserTimeout:
            logger.warning("Parser timed out parsing query %r", query.text)
            return entities

    def _parse(self, query, entities, all_candidates, timeout):
        entity_type_count = defaultdict(int)
        entity_dict = {}
        tokens = []  # tokens to be parsed

        # generate sentential form (assumes entities are sorted)
        for entity in entities:
            entity_type = entity.entity.type
            role_type = entity.entity.role
            if role_type:
                # Append role type to entity type with - separator
                entity_with_role_type = entity_type + "--" + role_type
                if entity_with_role_type in self._configured_entities:
                    entity_type = entity_with_role_type
            if entity_type not in self._configured_entities:
                entity_type = "unk"
            entity_id = "{}{}".format(entity_type, entity_type_count[entity_type])
            entity_type_count[entity_type] += 1
            entity_dict[entity_id] = entity
            tokens.append(entity_id)

        logger.debug("Parsing sentential form: %r", " ".join(tokens))
        start_time = time.time()
        parses = []
        for parse in self._parser.parse(tokens):
            parses.append(parse)
            if timeout is not None and (time.time() - start_time) > timeout:
                raise ParserTimeout("Parsing took too long")

        if not parses and self._relaxed_parser:
            for parse in self._relaxed_parser.parse(tokens):
                parses.append(parse)
                if timeout is not None and (time.time() - start_time) > MAX_PARSE_TIME:
                    raise ParserTimeout("Parsing took too long")

        if not parses:
            if all_candidates:
                return []
            return entities

        ranked_parses = self._rank_parses(
            query, entity_dict, parses, timeout, start_time
        )
        if all_candidates:
            return ranked_parses

        # if we still have more than one, choose the first
        entities = self._get_flat_entities(ranked_parses[0], entities, entity_dict)
        return tuple(sorted(entities, key=lambda e: e.span.start))

    def _rank_parses(self, query, entity_dict, parses, timeout, start_time=None):
        start_time = start_time or time.time()
        resolved = OrderedDict()

        for parse in parses:
            if timeout is not None and time.time() - start_time > timeout:
                raise ParserTimeout("Parsing took too long")
            resolved[self._resolve_parse(parse)] = None
        filtered = (p for p in resolved.keys())

        # Prefer parses with fewer groups
        parses = list(sorted(filtered, key=len))
        filtered = (p for p in parses if len(p) <= len(parses[0]))

        # Prefer parses with minimal distance from dependents to heads
        parses = list(
            sorted(filtered, key=lambda p: self._parse_distance(p, query, entity_dict))
        )
        min_parse_dist = self._parse_distance(parses[0], query, entity_dict)
        filtered = (
            p
            for p in parses
            if self._parse_distance(p, query, entity_dict) <= min_parse_dist
        )

        # TODO: apply precedence

        return list(filtered)

    def _parse_distance(self, parse, query, entity_dict):
        total_link_distance = 0
        stack = list(parse)
        while stack:
            node = stack.pop()
            head = entity_dict[node.id]
            for dep in node.dependents or set():
                if dep.dependents:
                    stack.append(dep)
                    continue
                child = entity_dict[dep.id]
                if child.token_span.start > head.token_span.start:
                    intra_entity_span = Span(
                        head.token_span.end, child.token_span.start
                    )
                else:
                    intra_entity_span = Span(
                        child.token_span.end, head.token_span.start
                    )
                link_distance = 0
                for token in intra_entity_span.slice(query.text.split(" ")):
                    if token in self.config[node.type][dep.type]["linking_words"]:
                        link_distance -= 0.5
                    else:
                        link_distance += 1
                total_link_distance += link_distance

        return total_link_distance

    @staticmethod
    def _get_flat_entities(parse, entities, entity_dict):
        stack = [g.to_query_entity(entity_dict) for g in parse]
        new_dict = {}
        while stack:
            entity = stack.pop()
            new_dict[(entity.entity.type, entity.span.start)] = entity

            for child in entity.children or ():
                stack.append(child)

        return [new_dict.get((e.entity.type, e.span.start), e) for e in entities]

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


class _EntityNode(namedtuple("EntityNode", ("type", "id", "dependents"))):
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
        text = ("  " * indent) + self.id

        if not self.dependents:
            return text

        return (
            text + "\n" + "\n".join(dep.pretty(indent + 1) for dep in self.dependents)
        )

    def to_query_entity(self, entity_dict, is_root=True):
        """Converts a node to an QueryEntity

        Args:
            entity_dict (dict): A mapping from entity ids to the corresponding
                original QueryEntity objects
        """
        if not self.dependents and not is_root:
            return entity_dict[self.id]

        head = entity_dict[self.id]
        if self.dependents is None:
            return head
        dependents = tuple(
            (c.to_query_entity(entity_dict, is_root=False) for c in self.dependents)
        )
        return head.with_children(dependents)


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
            symbol_template += "["
        else:
            symbol_template += ", "
        symbol_template += "{0}={{{0}}}".format(feature)
    if symbol_template is not group:
        symbol_template += "]"
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

    max_instances = config.get("max_instances")
    if max_instances is None:
        # pass through features unchanged
        lhs = symbol_template.format(
            **{f: "?" + chr(ord("a") + i) for i, f in enumerate(features)}
        )
        rhs = lhs
        if config.get("left"):
            yield "{lhs} -> {dep} {rhs}".format(lhs=lhs, rhs=rhs, dep=dep_symbol)
        if config.get("right"):
            yield "{lhs} -> {rhs} {dep}".format(lhs=lhs, rhs=rhs, dep=dep_symbol)
    else:
        for dep_count in range(max_instances):
            feature_dict = {
                f: "?" + chr(ord("a") + i)
                for i, f in enumerate(features)
                if f is not dep_type
            }
            feature_dict[dep_type] = dep_count
            rhs = symbol_template.format(**feature_dict)
            feature_dict[dep_type] = dep_count + 1
            lhs = symbol_template.format(**feature_dict)

            if config.get("left"):
                yield "{lhs} -> {dep} {rhs}".format(lhs=lhs, rhs=rhs, dep=dep_symbol)
            if config.get("right"):
                yield "{lhs} -> {rhs} {dep}".format(lhs=lhs, rhs=rhs, dep=dep_symbol)


def generate_grammar(config, entity_types=None, relaxed=False, unique_entities=20):
    """Generates a feature context free grammar from the provided parser config.

    Args:
        config (dict): The parser configuration
        unique_entities (int, optional): The number of entities of the same type that should be
            permitted in the same query

    Returns:
        str: a string containing the grammar with rules separated by line
    """
    entity_types = set(entity_types or ())
    # start rules
    rules = [
        "{} -> {}".format(START_SYMBOL, HEAD_SYMBOL),  # The start rule
        "{0} -> {0} {0}".format(HEAD_SYMBOL),
    ]  # Allow multiple heads

    # the set of all heads
    head_types = set(config.keys())

    # the set of all dependents
    dependent_types = set((t for g in config.values() for t in g))

    all_types = head_types.union(dependent_types).union(entity_types)

    for entity in all_types:
        if entity not in head_types and entity not in dependent_types:
            # Add entities which are not mentioned in config as standalones
            rules.append("H -> {}".format(entity))
        elif relaxed and entity not in head_types and entity in dependent_types:
            # Add dependent entities as standalones in relaxed mode
            rules.append("H -> {}".format(entity))

    # create rules for each group
    for entity in head_types:
        # the symbol for a group is the capitalized version of the string
        group = entity.capitalize()
        rules.append("H -> {}".format(group))

        dep_configs = config[entity]
        # If a dependent has a max number of instances, we will track it as a feature
        features = [
            t for t, d in dep_configs.items() if d.get("max_instances") is not None
        ]

        symbol_template = _build_symbol_template(group, features)

        # basic rule with features initialized to 0
        rules.append(
            "{} -> {}".format(
                symbol_template.format(**{f: 0 for f in features}), entity
            )
        )

        for dep_type, dep_config in dep_configs.items():
            rules.extend(
                _generate_dependent_rules(
                    dep_type, dep_config, symbol_template, features, head_types
                )
            )
    for entity in all_types:
        for idx in range(unique_entities):
            rules.append("{0} -> '{0}{1}'".format(entity, idx))

    return "\n".join(rules)
