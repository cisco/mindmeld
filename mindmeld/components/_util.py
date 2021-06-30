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

"""A module containing various utility functions for MindMeld NLP Components.
"""
import importlib
from typing import Union, Optional


def _is_module_available(module_name: str):
    """
    checks if a module is available or not (eg. _is_module_available("sentence_transformers"))
    Args:
        module_name (str): name of the model to check
    Returns:
        bool, if or not the given module exists
    """
    return bool(importlib.util.find_spec(module_name) is not None)


def _get_module_or_attr(module_name: str, func_name: str = None):
    """
    Loads an attribute from a module or a module itself
    (check if the module exists before calling this function)
    """
    m = importlib.import_module(module_name)
    if not func_name:
        return m
    if func_name not in dir(m):
        raise ImportError(f"Cannot import {func_name} from {module_name}")
    return getattr(m, func_name)


class TreeNode:
    def __init__(self, value: str, parent=None, children=None, allow=None):
        self.value = value
        self.allow = allow
        self.parent = parent
        self.children = children


class TreeNlp:
    def __init__(self, nlp, allow=None):
        # root
        self.root = TreeNode('root', allow=allow, children=[])
        # construct NLP tree
        for domain in nlp.domains:
            domain_node = TreeNode(domain, parent=self.root, allow=allow, children=[])
            self.root.children.append(domain_node)
            for intent in nlp.domains[domain].intents:
                intent_node = TreeNode(intent, parent=domain_node, allow=allow, children=[])
                domain_node.children.append(intent_node)
                entities = nlp.domains[domain].intents[intent].entities
                for entity in entities:
                    entity_node = TreeNode(entity, parent=intent_node, allow=allow, children=[])
                    intent_node.children.append(entity_node)
                    for role in entities[entity].role_classifier.roles:
                        role_node = TreeNode(role, parent=intent_node, allow=allow, children=None)
                        entity_node.children.append(role_node)

    def get_domains(self):
        return self.root.children or []

    def get_intents(self, domain: Union[str, TreeNode]):
        if isinstance(domain, TreeNode):
            domain = domain.value

        for domain_node in self.root.children:
            if domain_node.value == domain:
                return domain_node.children
        return []

    def get_entities(self, domain: Union[str, TreeNode],
                     intent: Union[str, TreeNode]):

        if isinstance(domain, TreeNode):
            domain = domain.value

        if isinstance(intent, TreeNode):
            intent = intent.value

        for intent_node in self.get_intents(domain):
            if intent_node.value == intent:
                return intent_node.children
        return []

    def get_roles(self, domain: Union[str, TreeNode],
                  intent: Union[str, TreeNode],
                  entity: Union[str, TreeNode]):
        if isinstance(domain, TreeNode):
            domain = domain.value

        if isinstance(intent, TreeNode):
            intent = intent.value

        if isinstance(entity, TreeNode):
            entity = entity.value

        for entity_node in self.get_entities(domain, intent):
            if entity_node.value == entity:
                return entity_node.children
        return []

    def update(self, allow: bool, domain: Union[str, TreeNode],
               intent: Optional[Union[str, TreeNode]] = None,
               entity: Optional[Union[str, TreeNode]] = None,
               role: Optional[Union[str, TreeNode]] = None):
        """
        This function updates the NLP tree with mask values
        Args:
            allow: True is mask off, False is mask on
            domain: domain of NLP
            intent: intent of NLP
            entity: entity of NLP
            role: role of NLP
        """

        if isinstance(domain, TreeNode):
            domain = domain.value

        if isinstance(intent, TreeNode):
            intent = intent.value

        if isinstance(entity, TreeNode):
            entity = entity.value

        if isinstance(role, TreeNode):
            role = role.value

        for domain_node in self.get_domains():
            if domain_node.value != domain:
                continue

            if not intent:
                domain_node.allow = allow
                return

            for intent_node in self.get_intents(domain):
                if intent_node.value != intent:
                    continue

                if not entity:
                    intent_node.allow = allow
                    return

                for entity_node in self.get_entities(domain, intent):
                    if entity_node.value != entity:
                        continue

                    if not role:
                        entity_node.allow = allow
                        return

                    for role_node in self.get_roles(domain, intent, entity):
                        if role_node.value != role:
                            continue

                        role_node.allow = allow
                        return

    def _sync_nodes(self):
        """
        This function does two actions sequentially:
            1. down-flow: flow mask decisions down the tree
            2. up-flow: flow mask decisions up the tree

        Each node has three allow states: True, False and None. True and False
        are explicitly set by the user while None is the default state.

        For 1., if a parent is allowed, then all it's "eligible" descendant components
        are allowed as well. An "eligible" component is a node set to None (ie non-user defined),
        since a user might have explicitly set a child.

        For 2., if all children of a NLP component are not allowed, then the parent
        will not be allowed as well. When we do an up-flow, we update nodes regardless of being
        explicitly set or not. This is because of the rule that if all the descendants are masked,
        the parent should be masked as well, even if it's explicitly set to the contrary.
        """
        for domain in self.get_domains():
            intents = self.get_intents(domain)

            # sync down
            if domain.allow is not None:
                for intent in intents:
                    if intent.allow is None:
                        intent.allow = domain.allow

            for intent in intents:
                entities = self.get_entities(domain, intent)
                # sync down
                if intent.allow is not None:
                    for entity in entities:
                        if entity.allow is None:
                            entity.allow = intent.allow

                for entity in entities:
                    roles = self.get_roles(domain, intent, entity)
                    # sync down
                    if entity.allow is not None:
                        for role in roles:
                            if role.allow is None:
                                role.allow = entity.allow

                    # sync up
                    if roles and all(role.allow is False for role in roles):
                        entity.allow = False

                # sync up
                if entities and all(entity.allow is False for entity in entities):
                    intent.allow = False

            # sync up
            if intents and all(intent.allow is False for intent in intents):
                domain.allow = False

    def to_dict(self):
        self._sync_nodes()
        result = {}
        for domain in self.get_domains():
            if not domain.allow:
                continue
            result[domain.value] = {}
            for intent in self.get_intents(domain.value):
                if not intent.allow:
                    continue
                result[domain.value][intent.value] = {}
                for entity in self.get_entities(domain.value, intent.value):
                    if not entity.allow:
                        continue
                    result[domain.value][intent.value][entity.value] = {}
                    for role in self.get_roles(domain.value, intent.value, entity.value):
                        if not role.allow:
                            continue
                        result[domain.value][intent.value][entity.value][role.value] = {}
        return result
