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
        seen_ids = []
        for item in mapping:
            cname = item['cname']
            item_id = item.get('id')
            if cname in item_map:
                msg = 'Canonical name {!r} specified in {!r} entity map multiple times'
                logger.debug(msg.format(cname, entity_type))
            if item_id:
                if item_id in seen_ids:
                    msg = 'Item id {!r} specified in {!r} entity map multiple times'
                    raise ValueError(msg.format(item_id, entity_type))
                seen_ids.append(item_id)

            aliases = [cname] + item.pop('whitelist', [])
            items_for_cname = item_map.get(cname, [])
            items_for_cname.append(item)
            item_map[cname] = items_for_cname
            for alias in aliases:
                norm_alias = normalizer(alias)
                if norm_alias in syn_map:
                    msg = 'Synonym {!r} specified in {!r} entity map multiple times'
                    logger.debug(msg.format(cname, entity_type))
                cnames_for_syn = syn_map.get(norm_alias, [])
                cnames_for_syn.append(cname)
                syn_map[norm_alias] = list(set(cnames_for_syn))

        return {'items': item_map, 'synonyms': syn_map}

    def fit(self):
        """Loads an entity mapping file (if one exists) or trains a machine-learned entity
        resolution model using the provided training examples"""
        if not self._is_system_entity:
            mapping = self._resource_loader.get_entity_map(self.type)
            self._mapping = self.process_mapping(self.type, mapping, self._normalizer)

    def predict(self, entity):
        """Predicts the resolved value(s) for the given entity using the loaded entity map or the
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
            cnames = self._mapping['synonyms'][normed]
        except KeyError:
            logger.warning('Failed to resolve entity %r for type %r', entity.text, entity.type)
            return entity.text

        if len(cnames) > 1:
            logger.info('Multiple possible canonical names for %r entity for type %r',
                        entity.text, entity.type)

        values = []
        for cname in cnames:
            for item in self._mapping['items'][cname]:
                item_value = copy.copy(item)
                item_value.pop('whitelist', None)
                values.append(item_value)

        return values

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
