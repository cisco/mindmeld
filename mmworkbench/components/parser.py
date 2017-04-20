# -*- coding: utf-8 -*-
"""
This module contains the language parser component of the Workbench natural language processor
"""

from __future__ import unicode_literals
from builtins import object

from ._config import get_parser_config

START_SYMBOL = 'S'
HEAD_SYMBOL = 'H'

START_SYMBOLS = set(START_SYMBOL, HEAD_SYMBOL)


class Parser(object):
    """A language parser which is used to extract relations between entities in a given query and
    group related entities together."""

    def __init__(self, resource_loader, domain, intent, config=None):
        self._resource_loader = resource_loader
        self.domain = domain
        self.intent = intent
        self.config = get_parser_config(resource_loader.app_path, config)


def _build_symbol_template(group, features):
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


def _generate_dependent_rules(config, symbol_template, features, head_types):
    dep_type = config['type']
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
            feature_dict = {f: '%' + str(i) for i, f in enumerate(features)
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
    """Generates a context free grammar from the provided parser config

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
    dependent_types = set((d['type'] for g in config.values() for d in g['dependents']))

    all_types = head_types.union(dependent_types)

    # create rules for each group
    for entity in head_types:
        # the symbol for a group is the capitalized version of the string
        group = entity.capitalize()
        rules.append('H -> {}'.format(group))

        dep_configs = config[entity]['dependents']
        # If a dependent has a max number of instances, we will track it as a feature
        features = [d['type'] for d in dep_configs if d.get('max_instances') is not None]

        symbol_template = _build_symbol_template(group, features)

        # basic rule with features initialized to 0
        rules.append('{} -> {}'.format(symbol_template.format(**{f: 0 for f in features}), entity))

        for dep_config in dep_configs:
            rules.extend(_generate_dependent_rules(dep_config, symbol_template,
                                                   features, head_types))
    for entity in all_types:
        for idx in range(unique_entities):
            rules.append("{0} -> '{0}{1}'".format(entity, idx))

    return '\n'.join(rules)
