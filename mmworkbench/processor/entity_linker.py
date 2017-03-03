# -*- coding: utf-8 -*-
"""
This module contains the named entity linker component.
"""
from __future__ import unicode_literals
from builtins import object

import copy
import logging

from ..core import Entity

logger = logging.getLogger(__name__)


class EntityLinker(object):
    """A named entity linker which is used to link entities to their specific
    values in a given query.
    """

    def __init__(self, resource_loader, entity_type, normalizer):
        self._resource_loader = resource_loader
        self._normalizer = normalizer
        self.type = entity_type

        self._mapping = None
        self._is_system_entity = Entity.is_system_entity(self.type)
        if not self._is_system_entity:
            mapping = self._resource_loader.get_entity_map(self.type)
            self._mapping = self.process_mapping(self.type, mapping, normalizer)

    @staticmethod
    def process_mapping(entity_type, mapping, normalizer):
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
        """Trains the model"""
        # self._model = something
        pass

    def predict(self, entity):
        """Predicts linked values for the entities provided

        Args:
            entity (Entity): An entity found in an input query

        Returns:
            The value for the entity passed in
        """
        if self._is_system_entity:
            # system entities are already linked
            return entity.value

        normed = self._normalizer(entity.text)
        try:
            cname = self._mapping['synonyms'][normed]
        except KeyError:
            logger.warning('Failed to link entity %r for type %r', entity.text, entity.type)
            return entity.text

        value = copy.copy(self._mapping['items'][cname])
        value.pop('whitelist', None)
        return value

    def predict_proba(self, entity):
        """Generates multiple hypotheses and returns their associated probabilities

        Args:
            entity (Entity): An entity found in an input query

        Returns:
            list: a list of tuples of the form (str, float) grouping roles and their probabilities
        """
        pass

    def evaluate(self, use_blind=False):
        """Evaluates the model on the specified data

        Returns:
            TYPE: Description
        """
        pass

    def dump(self, model_path):
        """Persists the model to disk.

        Args:
            model_path (str): The location on disk where the model should be stored

        """
        # joblib.dump(self._model, model_path)
        pass

    def load(self, model_path):
        """Loads the model from disk

        Args:
            model_path (str): The location on disk where the model is stored

        """
        # self._model = joblib.load(model_path)
        pass
