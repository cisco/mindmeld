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
import logging
import enum
from typing import Union, Optional, List
from collections import defaultdict
from ..exceptions import InvalidMaskError

logger = logging.getLogger(__name__)


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


class MaskState(enum.Enum):
    """
    This class encoded three NLP states:
        unset: state when the user has neither allowed/denied the NLP component.
            This state is needed to propagate state up/down the tree since we only
            propagate state to unset nodes, never to user-defined nodes
        allow: state when the user has explicitly allowed a node.
        deny: state when the user has explicitly denied a node.
    """
    unset = enum.auto()
    allow = enum.auto()
    deny = enum.auto()

    def __bool__(self):
        return self == self.allow


class TreeNode:
    def __init__(self, nlp_name: str,
                 parent: Optional['TreeNode'] = None,
                 children: Optional[List['TreeNode']] = None,
                 mask_state: Optional[MaskState] = None):
        """
        Constructor for the tree node
        Args:
            nlp_name: The name of the NLP component. eg. "weather"
                is a name for a domain
            parent: The parent of the NLP component. eg. parent of
                an intent is a domain
            children: The children of the NLP component. eg.
                children of an intent are entities
            mask_state: The mask state of the NLP component
        """
        self.nlp_name = nlp_name
        self.mask_state = mask_state
        self.parent = parent
        self.children = children or []


class TreeNlp:
    """
    This data structure encodes a NLP tree hierarchy where each node
    encodes a mask state, based on which certain NLP components are allowed
    or denied based on user input
    """
    def __init__(self, nlp, mask_state=MaskState.unset):
        # root
        self.root = TreeNode('root', mask_state=mask_state)
        # construct NLP tree
        for domain in nlp.domains:
            domain_node = TreeNode(domain, parent=self.root, mask_state=mask_state)
            self.root.children.append(domain_node)
            for intent in nlp.domains[domain].intents:
                intent_node = TreeNode(intent, parent=domain_node, mask_state=mask_state)
                domain_node.children.append(intent_node)
                entities = nlp.domains[domain].intents[intent].entities
                for entity in entities:
                    entity_node = TreeNode(entity, parent=intent_node, mask_state=mask_state)
                    intent_node.children.append(entity_node)
                    for role in entities[entity].role_classifier.roles:
                        role_node = TreeNode(role, parent=intent_node, mask_state=mask_state)
                        entity_node.children.append(role_node)

    @staticmethod
    def _convert_tree_node_to_values(*nlp_components):
        result = [None for _ in ['domain', 'intent', 'entity', 'role']]
        for idx, component in enumerate(nlp_components):
            component_name = component.nlp_name if isinstance(
                component, TreeNode) else component
            result[idx] = component_name
        return result

    def get_domain_nodes(self):
        return self.root.children or []

    def get_intent_nodes(self, domain: Union[str, TreeNode]):
        domain, _, _, _ = self._convert_tree_node_to_values(domain)
        for domain_node in self.root.children:
            if domain_node.nlp_name == domain:
                return domain_node.children
        return []

    def get_entity_nodes(self, domain: Union[str, TreeNode],
                         intent: Union[str, TreeNode]):
        domain, intent, _, _ = self._convert_tree_node_to_values(domain, intent)
        for intent_node in self.get_intent_nodes(domain):
            if intent_node.nlp_name == intent:
                return intent_node.children
        return []

    def get_role_nodes(self, domain: Union[str, TreeNode],
                       intent: Union[str, TreeNode],
                       entity: Union[str, TreeNode]):
        domain, intent, entity, _ = self._convert_tree_node_to_values(
            domain, intent, entity)
        for entity_node in self.get_entity_nodes(domain, intent):
            if entity_node.nlp_name == entity:
                return entity_node.children
        return []

    def update(self, mask_state: bool,
               domain: Union[str, TreeNode],
               intent: Optional[Union[str, TreeNode]] = None,
               entity: Optional[Union[str, TreeNode]] = None,
               role: Optional[Union[str, TreeNode]] = None):
        """
        This function updates the NLP tree with mask values. Note:
        Args:
            mask_state: True is mask off, False is mask on
            domain: domain of NLP
            intent: intent of NLP
            entity: entity of NLP
            role: role of NLP
        """
        domain_name, intent_name, entity_name, role_name = self._convert_tree_node_to_values(
            domain, intent, entity, role)

        # validation check
        nlp_components = [domain_name, intent_name, entity_name, role_name]
        for i in range(1, len(nlp_components)):
            if any(not component for component in nlp_components[:i]) and nlp_components[i]:
                raise InvalidMaskError(
                    f"Unable to resolve NLP hierarchy since "
                    f"{str(nlp_components[i])} does not have an valid ancestor")

        for domain_node in self.get_domain_nodes():
            if domain_node.nlp_name != domain_name:
                continue

            if not intent_name:
                domain_node.mask_state = mask_state
                return

            for intent_node in self.get_intent_nodes(domain_node.nlp_name):
                if intent_name not in ('*', intent_node.nlp_name):
                    continue

                if not entity_name:
                    intent_node.mask_state = mask_state
                    # If the intent is * and it's terminal, eg. "domain.*", then
                    # we mask the intent AND continue to iterate through the other
                    # intents of the domain
                    if intent_name == '*':
                        continue
                    # If the intent is not *, then it's terminal, eg. "domain.intent",
                    # then we mask the intent and end the function's operations
                    return

                for entity_node in self.get_entity_nodes(domain_node.nlp_name,
                                                         intent_node.nlp_name):
                    if entity_name not in ('*', entity_node.nlp_name):
                        continue

                    if not role_name:
                        entity_node.mask_state = mask_state

                        # If the entity is * and it's terminal, eg. "domain.intent.*", then
                        # we mask the entity AND continue to iterate through the other
                        # entities of the intent
                        if entity_name == '*':
                            continue
                        # If the entity is not *, then it's terminal, eg. "domain.intent.entity",
                        # then we mask the entity and end the function's operations
                        return

                    for role_node in self.get_role_nodes(domain_node.nlp_name,
                                                         intent_node.nlp_name,
                                                         entity_node.nlp_name):
                        if role_name not in ('*', role_node.nlp_name):
                            continue

                        role_node.mask_state = mask_state
                        if role_name == '*':
                            continue
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
        for domain in self.get_domain_nodes():
            intents = self.get_intent_nodes(domain)
            for intent in intents:
                # sync down
                if domain.mask_state != MaskState.unset and \
                        intent.mask_state == MaskState.unset:
                    intent.mask_state = domain.mask_state

                entities = self.get_entity_nodes(domain, intent)
                for entity in entities:
                    # sync down
                    if intent.mask_state != MaskState.unset and \
                            entity.mask_state == MaskState.unset:
                        entity.mask_state = intent.mask_state

                    roles = self.get_role_nodes(domain, intent, entity)
                    for role in roles:
                        # sync down
                        if entity.mask_state != MaskState.unset and \
                                role.mask_state == MaskState.unset:
                            role.mask_state = entity.mask_state

                    # sync up entity-role
                    if roles and all(role.mask_state == MaskState.deny for role in roles):
                        entity.mask_state = MaskState.deny

                # We do not perform sync ups for entities since tagger models cannot
                # deny their parent text classification models. For example,
                # just because the developer wants to deny all the entities in a particular
                # intent, doesn't mean the intent should be denied as well.

            # sync up domain-intent
            if intents and all(intent.mask_state == MaskState.deny for intent in intents):
                domain.mask_state = MaskState.deny

    def _default_to_regular(self, d):
        if isinstance(d, defaultdict):
            d = {k: self._default_to_regular(v) for k, v in d.items()}
        return d

    def to_dict(self) -> dict:
        """
        This function serializes TreeNlp into a dict structure by only adding keys representing
        allow MaskState nodes and not adding keys for deny and unset MaskState nodes.
        """
        self._sync_nodes()
        # The results has three nested dicts: {domain: {intent: {entity: role: {}}}}
        result = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for domain in self.get_domain_nodes():
            if domain.mask_state:
                result[domain.nlp_name] = defaultdict(lambda: defaultdict(dict))

            for intent in self.get_intent_nodes(domain.nlp_name):
                if intent.mask_state:
                    result[domain.nlp_name][intent.nlp_name] = defaultdict(dict)

                for entity in self.get_entity_nodes(domain.nlp_name,
                                                    intent.nlp_name):
                    if entity.mask_state:
                        result[domain.nlp_name][intent.nlp_name][entity.nlp_name] = {}

                    for role in self.get_role_nodes(domain.nlp_name,
                                                    intent.nlp_name,
                                                    entity.nlp_name):
                        if role.mask_state:
                            result[domain.nlp_name][intent.nlp_name][
                                entity.nlp_name][role.nlp_name] = {}

        serialize_results = self._default_to_regular(result)
        return serialize_results
