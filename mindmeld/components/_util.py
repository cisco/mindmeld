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
    def __init__(self, value: str, parent=None, children=None, allow=False):
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

    def get_intents(self, domain:str):
        for domain_node in self.root.children:
            if domain_node.value == domain:
                return domain_node.children
        return []

    def get_entities(self, domain:str, intent:str):
        for intent_node in self.get_intents(domain):
            if intent_node.value == intent:
                return intent_node.children
        return []

    def get_roles(self, domain:str, intent:str, entity:str):
        for entity_node in self.get_entities(domain, intent):
            if entity_node.value == entity:
                return entity_node.children
        return []

    def update(self, domain, intent=None, entity=None, role=None, allow=True):
        for domain_node in self.get_domains():
            if domain_node.value != domain:
                continue
            if intent:
                for intent_node in self.get_intents(domain):
                    if intent_node.value != intent:
                        continue
                    if entity:
                        for entity_node in self.get_entities(domain, intent):
                            if entity_node.value != entity:
                                continue
                            if role:
                                for role_node in self.get_roles(domain, intent, entity):
                                    if role_node.value != role:
                                        continue
                                    role_node.allow = allow
                            else:
                                entity_node.allow = allow
                    else:
                        intent_node.allow = allow
            else:
                domain_node.allow = allow

    def _sync_nodes(self):
        for domain in self.get_domains():
            intents = self.get_intents(domain.value)
            # sync down
            if domain.allow is not None:
                for intent in intents:
                    if intent.allow is None:
                        intent.allow = domain.allow
            for intent in intents:
                entities = self.get_entities(domain.value, intent.value)
                # sync down
                if intent.allow is not None:
                    for entity in entities:
                        if entity.allow is None:
                            entity.allow = intent.allow
                for entity in entities:
                    roles = self.get_roles(domain.value, intent.value, entity.value)
                    # sync down
                    if entity.allow is not None:
                        for role in roles:
                            if role.allow is None:
                                role.allow = entity.allow
                    # sync up
                    if roles and all(not role.allow for role in roles):
                        entity.allow = False
                # sync up
                if entities and all(not entity.allow for entity in entities):
                    intent.allow = False
            # sync up
            if intents and all(not intent.allow for intent in intents):
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
