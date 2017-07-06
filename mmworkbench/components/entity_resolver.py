# -*- coding: utf-8 -*-
"""
This module contains the entity resolver component of the Workbench natural language processor.
"""
from __future__ import absolute_import, unicode_literals
from builtins import object

import copy
import logging

from ..core import Entity
from ._config import get_app_name, get_classifier_config, DOC_TYPE, DEFAULT_ES_SYNONYM_MAPPING

from ._elasticsearch_helpers import (create_es_client, load_index, get_scoped_index_name,
                                     delete_index)

logger = logging.getLogger(__name__)


class EntityResolver(object):
    """An entity resolver is used to resolve entities in a given query to their canonical values
    (usually linked to specific entries in a knowledge base).
    """
    # prefix for Elasticsearch indices used to store synonyms for entity resolution
    ES_SYNONYM_INDEX_PREFIX = "synonym"

    def __init__(self, app_path, resource_loader, entity_type, es_host=None, es_client=None):
        """Initializes an entity resolver

        Args:
            app_path (str): The application path
            resource_loader (ResourceLoader): An object which can load resources for the resolver
            entity_type: The entity type associated with this entity resolver
            es_host (str): The Elasticsearch host server
        """
        self._app_name = get_app_name(app_path)
        self._resource_loader = resource_loader
        self._normalizer = resource_loader.query_factory.normalize
        self.type = entity_type
        self._is_system_entity = Entity.is_system_entity(self.type)

        self._exact_match_mapping = None

        er_config = get_classifier_config('entity_resolution', app_path=app_path)
        self._use_text_rel = er_config['model_type'] == 'text_relevance'
        self._es_host = es_host
        self.__es_client = es_client
        self._es_index_name = EntityResolver.ES_SYNONYM_INDEX_PREFIX + "_" + entity_type

    @property
    def _es_client(self):
        # Lazily connect to Elasticsearch
        if self.__es_client is None:
            self.__es_client = create_es_client()
        return self.__es_client

    @classmethod
    def ingest_synonym(cls, app_name, index_name, data, es_host=None, es_client=None):
        """Loads synonym documents from the mapping.json data into the specified index. If an index
        with the specified name doesn't exist, a new index with that name will be created.

        Args:
            app_name (str): The name of the app
            index_name (str): The name of the new index to be created
            data (list): A list of documents to be loaded into the index
            es_host (str): The Elasticsearch host server
            es_client (Elasticsearch): The Elasticsearch client
        """
        def _doc_generator(docs):
            for doc in docs:
                base = {}
                if doc.get('id'):
                    base['_id'] = doc['id']
                whitelist = doc['whitelist']
                new_list = []
                new_list.append({"name": doc['cname']})
                for syn in whitelist:
                    new_list.append({"name": syn})
                doc['whitelist'] = new_list
                base.update(doc)

                yield base

        load_index(app_name, index_name, data, _doc_generator, DEFAULT_ES_SYNONYM_MAPPING, DOC_TYPE,
                   es_host, es_client)

    def fit(self, clean=False):
        """Loads an entity mapping file to Elasticsearch for text relevance based entity resolution

        Args:
            clean (bool): If True, deletes and recreates the index from scratch instead of
                          updating the existing index with synonyms in the mapping.json
        """
        if self._is_system_entity:
            return

        if not self._use_text_rel:
            self._fit_exact_match()
            return

        if clean:
            delete_index(self._app_name, self._es_index_name, self._es_host,
                         self._es_client)
        entity_map = self._resource_loader.get_entity_map(self.type)
        logger.info("Importing synonym data to ES index '{}'".format(self._es_index_name))
        EntityResolver.ingest_synonym(self._app_name, self._es_index_name, entity_map,
                                      self._es_host, self._es_client)

    @staticmethod
    def _process_entity_map(entity_type, entity_map, normalizer):
        """Loads in the mapping.json file and stores the synonym mappings in a item_map and a
        synonym_map for exact match entity resolution when Elasticsearch is unavailable

        Args:
            entity_type: The entity type associated with this entity resolver
            entity_map: The loaded mapping.json file for the given entity type
            normalizer: The normalizer to use
        """
        item_map = {}
        syn_map = {}
        seen_ids = []
        for item in entity_map:
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

    def _fit_exact_match(self):
        """Fits a simple exact match entity resolution model when Elasticsearch is not available.
        """
        entity_map = self._resource_loader.get_entity_map(self.type)
        self._exact_match_mapping = self._process_entity_map(self.type, entity_map,
                                                             self._normalizer)

    def predict(self, entity):
        """Predicts the resolved value(s) for the given entity using the loaded entity map or the
        trained entity resolution model

        Args:
            entity (Entity): An entity found in an input query

        Returns:
            The top 20 resolved values for the provided entity
        """
        if self._is_system_entity:
            # system entities are already resolved
            return [entity.value]

        if not self._use_text_rel:
            return self._predict_exact_match(entity)

        text_relevance_query = {
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "should": [
                                {
                                    "bool": {
                                        "should": [
                                            {
                                                "match": {
                                                    "cname.normalized_keyword": {
                                                        "query": entity.text,
                                                        "boost": 10
                                                    }
                                                }
                                            },
                                            {
                                                "match": {
                                                    "cname.raw": {
                                                        "query": entity.text,
                                                        "boost": 10
                                                    }
                                                }
                                            }
                                        ]
                                    }
                                },
                                {
                                    "nested": {
                                        "path": "whitelist",
                                        "score_mode": "max",
                                        "query": {
                                            "bool": {
                                                "should": [
                                                    {
                                                        "match": {
                                                            "whitelist.name.normalized_keyword": {
                                                                "query": entity.text,
                                                                "boost": 10
                                                            }
                                                        }
                                                    },
                                                    {
                                                        "match": {
                                                            "whitelist.name": {
                                                                "query": entity.text
                                                            }
                                                        }
                                                    },
                                                    {
                                                        "match": {
                                                            "whitelist.name.char_ngram": {
                                                                "query": entity.text
                                                            }
                                                        }
                                                    }
                                                ]
                                            }
                                        },
                                        "inner_hits": {}
                                    }
                                }
                            ]
                        }
                    },
                    "field_value_factor": {
                        "field": "sort_factor",
                        "modifier": "log1p",
                        "factor": 10
                    },
                    "boost_mode": "sum"
                }
            }
        }

        index = get_scoped_index_name(self._app_name, self._es_index_name)
        response = self._es_client.search(index=index, body=text_relevance_query)
        hits = response['hits']['hits']

        results = []
        for hit in hits:
            result = {
                'cname': hit['_source']['cname'],
                'score': hit['_score'],
                'top_synonym': hit['inner_hits']['whitelist']['hits']['hits'][0]['_source']['name']}

            if hit['_source'].get('id'):
                result['id'] = hit['_source'].get('id')

            if hit['_source'].get('sort_factor'):
                result['sort_factor'] = hit['_source'].get('sort_factor')

            results.append(result)

        return results[0:20]

    def _predict_exact_match(self, entity):
        """Predicts the resolved value(s) for the given entity using the loaded entity map.

        Args:
            entity (Entity): An entity found in an input query
        """
        normed = self._normalizer(entity.text)
        try:
            cnames = self._exact_match_mapping['synonyms'][normed]
        except KeyError:
            logger.warning('Failed to resolve entity %r for type %r', entity.text, entity.type)
            return None

        if len(cnames) > 1:
            logger.info('Multiple possible canonical names for %r entity for type %r',
                        entity.text, entity.type)

        values = []
        for cname in cnames:
            for item in self._exact_match_mapping['items'][cname]:
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
        pass

    def load(self):
        """Loads the trained entity resolution model from disk

        Args:
            model_path (str): The location on disk where the model is stored
        """
        if self._use_text_rel:
            scoped_index_name = get_scoped_index_name(self._app_name, self._es_index_name)
            if not self._es_client.indices.exists(index=scoped_index_name):
                self.fit()
        else:
            self.fit()
