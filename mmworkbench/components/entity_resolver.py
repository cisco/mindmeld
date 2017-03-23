# -*- coding: utf-8 -*-
"""
This module contains the entity resolver component of the Workbench natural language processor.
"""
from __future__ import unicode_literals
from builtins import object

import copy
import logging

from ..core import Entity

logger = logging.getLogger(__name__)


class EntityResolver(object):
    """An entity resolver is used to resolve entities in a given query to their canonical values
    (usually linked to specific entries in a knowledge base).
    """

    def __init__(self, resource_loader, entity_type):
        """Initializes an entity resolver

        Args:
            resource_loader (ResourceLoader): An object which can load resources for the resolver
            entity_type: The entity type associated with this entity resolver
        """
        self._resource_loader = resource_loader
        self._normalizer = resource_loader.query_factory.normalize
        self.type = entity_type

        self._mapping = None
        self._is_system_entity = Entity.is_system_entity(self.type)

    @staticmethod
    def process_mapping(entity_type, mapping, normalizer):
        """
        Description

        Args:
            entity_type: The entity type associated with this entity resolver
            mapping: Description
            normalizer: Description
        """
        item_map = {}
        syn_map = {}
        for item in mapping:
            cname = item['cname']
            if cname in item_map:
                msg = 'Canonical name {!r} specified in {!r} entity map multiple times'
                # TODO: is there a better exception type for this?
                raise ValueError(msg.format(cname, entity_type))

            aliases = [cname] + item.pop('whitelist', [])
            item_map[cname] = item
            for alias in aliases:
                norm_alias = normalizer(alias)
                if norm_alias in syn_map:
                    msg = 'Synonym {!r} specified in {!r} entity map multiple times'
                    # TODO: is there a better exception type for this?
                    raise ValueError(msg.format(cname, entity_type))
                syn_map[norm_alias] = cname

        return {'items': item_map, 'synonyms': syn_map}

    def fit(self):
        """Loads an entity mapping file (if one exists) or trains a machine-learned entity
        resolution model using the provided training examples"""
        if not self._is_system_entity:
            mapping = self._resource_loader.get_entity_map(self.type)
            self._mapping = self.process_mapping(self.type, mapping, self._normalizer)

    def predict(self, entity):
        """Predicts the resolved value for the given entity using the loaded entity map or the
        trained entity resolution model

        Args:
            entity (Entity): An entity found in an input query

        Returns:
            The resolved value for the provided entity
        """
        if self._is_system_entity:
            # system entities are already resolved
            return entity.value

        normed = self._normalizer(entity.text)
        try:
            cname = self._mapping['synonyms'][normed]
        except KeyError:
            logger.warning('Failed to resolve entity %r for type %r', entity.text, entity.type)
            return entity.text

        value = copy.copy(self._mapping['items'][cname])
        value.pop('whitelist', None)
        return value

    def predict_proba(self, entity):
        """Runs prediction on a given entity and generates multiple hypotheses with their
        associated probabilities using the trained entity resolution model

        Args:
            entity (Entity): An entity found in an input query

        Returns:
            list: a list of tuples of the form (str, float) grouping resolved values and their
                probabilities
        """
        pass

    def evaluate(self, use_blind=False):
        """Evaluates the trained entity resolution model on the given test data

        Returns:
            TYPE: Description
        """
        pass

    def dump(self, model_path):
        """Persists the trained entity resolution model to disk.

        Args:
            model_path (str): The location on disk where the model should be stored
        """
        # joblib.dump(self._model, model_path)
        pass

    def load(self):
        """Loads the trained entity resolution model from disk

        Args:
            model_path (str): The location on disk where the model is stored
        """
        self.fit()
