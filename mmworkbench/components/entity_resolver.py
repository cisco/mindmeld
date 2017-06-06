# -*- coding: utf-8 -*-
"""
This module contains the entity resolver component of the Workbench natural language processor.
"""
from __future__ import unicode_literals
from builtins import object

import logging
import copy

from ..core import Entity

from .elastic_search_helpers import create_es_client, load_index

logger = logging.getLogger(__name__)

DOC_TYPE = "document"


class EntityResolver(object):
    """An entity resolver is used to resolve entities in a given query to their canonical values
    (usually linked to specific entries in a knowledge base).
    """

    # default ElasticSearch mapping to define text analysis settings for text fields
    ES_SYNONYM_INDEX_PREFIX = "synonym_test"

    DEFAULT_SYN_ES_MAPPING = {
        "mappings": {
            "document": {
                "properties": {
                    "cname": {
                        "type": "text",
                        "fields": {
                            "raw": {
                                "type": "keyword",
                                "ignore_above": 256
                            },
                            "normalized_keyword": {
                                "type": "text",
                                "analyzer": "keyword_match_analyzer"
                            },
                            "char_ngram": {
                                "type": "text",
                                "analyzer": "char_ngram_analyzer"
                            }
                        },
                        "analyzer": "default_analyzer"
                    },
                    "id": {
                        "type": "keyword"
                    },
                    "whitelist": {
                        "type": "nested",
                        "properties": {
                            "name": {
                                "type": "text",
                                "fields": {
                                    "raw": {
                                        "type": "keyword",
                                        "ignore_above": 256
                                    },
                                    "normalized_keyword": {
                                        "type": "text",
                                        "analyzer": "keyword_match_analyzer"
                                    },
                                    "char_ngram": {
                                        "type": "text",
                                        "analyzer": "char_ngram_analyzer"
                                    }
                                },
                                "analyzer": "default_analyzer"
                            }
                        }
                    }
                }
            }
        },
        "settings": {
            "analysis": {
                "filter": {
                    "token_shingle": {
                        "max_shingle_size": "4",
                        "min_shingle_size": "2",
                        "output_unigrams": "true",
                        "type": "shingle"
                    },
                    "autocomplete_filter": {
                        "type": "edge_ngram",
                        "min_gram": "4",
                        "max_gram": "20"
                    }
                },
                "analyzer": {
                    "default_analyzer": {
                        "filter": [
                            "lowercase",
                            "asciifolding",
                            "token_shingle"
                        ],
                        "char_filter": [
                            "remove_comma",
                            "remove_tm_and_r",
                            "remove_loose_apostrophes",
                            "space_possessive_apostrophes",
                            "remove_special_beginning",
                            "remove_special_end",
                            "remove_special1",
                            "remove_special2",
                            "remove_special3"
                        ],
                        "type": "custom",
                        "tokenizer": "whitespace"
                    },
                    "keyword_match_analyzer": {
                        "filter": [
                            "lowercase",
                            "asciifolding"
                        ],
                        "char_filter": [
                            "remove_comma",
                            "remove_tm_and_r",
                            "remove_loose_apostrophes",
                            "space_possessive_apostrophes",
                            "remove_special_beginning",
                            "remove_special_end",
                            "remove_special1",
                            "remove_special2",
                            "remove_special3"
                        ],
                        "type": "custom",
                        "tokenizer": "keyword"
                    },
                    "char_ngram_analyzer": {
                        "filter": [
                            "lowercase",
                            "asciifolding",
                            "autocomplete_filter"
                        ],
                        "char_filter": [
                            "remove_comma",
                            "remove_tm_and_r",
                            "remove_loose_apostrophes",
                            "space_possessive_apostrophes",
                            "remove_special_beginning",
                            "remove_special_end",
                            "remove_special1",
                            "remove_special2",
                            "remove_special3"
                        ],
                        "type": "custom",
                        "tokenizer": "whitespace"
                    }
                },
                "char_filter": {
                    "remove_comma": {
                        "pattern": ",",
                        "type": "pattern_replace",
                        "replacement": ""
                    },
                    "remove_loose_apostrophes": {
                        "pattern": " '|' ",
                        "type": "pattern_replace",
                        "replacement": ""
                    },
                    "remove_special2": {
                        "pattern": "([\\p{N}]+)[^\\p{L}\\p{N}&']+(?=[\\p{L}\\s]+)",
                        "type": "pattern_replace",
                        "replacement": "$1 "
                    },
                    "remove_tm_and_r": {
                        "pattern": "™|®",
                        "type": "pattern_replace",
                        "replacement": ""
                    },
                    "remove_special3": {
                        "pattern": "([\\p{L}]+)[^\\p{L}\\p{N}&']+(?=[\\p{L}]+)",
                        "type": "pattern_replace",
                        "replacement": "$1 "
                    },
                    "remove_special1": {
                        "pattern": "([\\p{L}]+)[^\\p{L}\\p{N}&']+(?=[\\p{N}\\s]+)",
                        "type": "pattern_replace",
                        "replacement": "$1 "
                    },
                    "remove_special_end": {
                        "pattern": "[^\\p{L}\\p{N}&']+$",
                        "type": "pattern_replace",
                        "replacement": ""
                    },
                    "space_possessive_apostrophes": {
                        "pattern": "([^\\p{N}\\s]+)'s ",
                        "type": "pattern_replace",
                        "replacement": "$1 's "
                    },
                    "remove_special_beginning": {
                        "pattern": "^[^\\p{L}\\p{N}\\p{Sc}&']+",
                        "type": "pattern_replace",
                        "replacement": ""
                    }
                }
            }
        }
    }

    def __init__(self, resource_loader, entity_type, es_host=None):
        """Initializes an entity resolver

        Args:
            resource_loader (ResourceLoader): An object which can load resources for the resolver
            entity_type: The entity type associated with this entity resolver
        """
        self._resource_loader = resource_loader
        self._normalizer = resource_loader.query_factory.normalize
        self.type = entity_type

        self._is_system_entity = Entity.is_system_entity(self.type)
        self._es_host = es_host
        self.__es_client = None
        self._es_index_name = EntityResolver.ES_SYNONYM_INDEX_PREFIX + "_" + entity_type
        self._mapping = EntityResolver.DEFAULT_SYN_ES_MAPPING

    @property
    def _es_client(self):
        # Lazily connect to Elasticsearch
        if self.__es_client is None:
            self.__es_client = create_es_client()
        return self.__es_client

    @classmethod
    def ingest_synonym(cls, index_name, data, es_host=None, es_client=None):
        """Loads documents from disk into the specified index in the knowledge base. If an index
        with the specified name doesn't exist, a new index with that name will be created in the
        knowledge base.

        Args:
            index_name (str): The name of the new index to be created
            data_file (str): The path to the data file containing the documents to be imported
                into the knowledge base index
            es_host (str): The Elasticsearch host server
            es_client: Description
        """
        def _doc_generator(docs):
            for doc in docs:
                base = {'_id': doc['id']}
                whitelist = doc['whitelist']
                new_list = []
                new_list.append({"name": doc['cname']})
                for syn in whitelist:
                    new_list.append({"name": syn})
                doc['whitelist'] = new_list
                base.update(doc)
                yield base

        load_index(index_name, data, _doc_generator, cls.DEFAULT_SYN_ES_MAPPING, DOC_TYPE,
                   es_host, es_client)

    def fit(self, clean=False):
        """Loads an entity mapping file (if one exists) or trains a machine-learned entity
        resolution model using the provided training examples

        Args:
            clean (bool): If True, deletes and recreates the index from scratch instead of
                          updating the existing index with synonyms in the mapping.json
        """
        if not self._is_system_entity:
            # create index if specified index does not exist
            # TODO: refactor things around ES calls.
            # if not self._es_client.indices.exists(index=self._es_index_name):
            #     create_index(self._es_index_name, self._mapping, es_client=self._es_client)

            data = self._resource_loader.get_entity_map(self.type)
            logger.info("Importing synonym data to ES index '{}'".format(self._es_index_name))
            EntityResolver.ingest_synonym(self._es_index_name, data, self._es_host, self._es_client)

    def predict(self, entity, exact_match_only=False):
        """Predicts the resolved value(s) for the given entity using the loaded entity map or the
        trained entity resolution model

        Args:
            entity (Entity): An entity found in an input query

        Returns:
            The resolved value for the provided entity
        """
        if self._is_system_entity:
            # system entities are already resolved
            return [entity.value]

        if exact_match_only:
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

        text_relevance_query = {
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
            }
        }

        response = self._es_client.search(index=self._es_index_name, body=text_relevance_query)
        results = response['hits']['hits']
        results = [{'id': result['_source']['id'],
                    'cname': result['_source']['cname'],
                    'score': result['_score'],
                    'top_synonym': result['inner_hits']['whitelist']['hits']['hits'][0]['_source']
                                         ['name'],
                    'all_synonyms': result['_source']['whitelist']}
                   for result in results]

        return results[0:20]

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
