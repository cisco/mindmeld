# -*- coding: utf-8 -*-
"""
This module contains the question answerer component.
"""
from __future__ import unicode_literals
from builtins import object

import json
import logging
import os

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

DOC_TYPE = 'document'

logger = logging.getLogger(__name__)


class QuestionAnswerer(object):

    # default ElasticSearch mapping to define text analysis settings for text fields
    DEFAULT_ES_MAPPING = {
        "mappings": {
            DOC_TYPE: {
                "dynamic_templates": [
                    {
                        "default_text": {
                            "match": "*",
                            "match_mapping_type": "string",
                            "mapping": {
                                "type": "text",
                                "analyzer": "default_analyzer",
                                "fields": {
                                    "raw": {
                                        "type": "keyword",
                                        "ignore_above": 256
                                    }
                                }
                            }
                        }
                    }
                ],
                "properties": {
                    "location": {
                        "type": "geo_point"
                    },
                    "id": {
                        "type": "keyword"
                    }
                }
            }
        },
        "settings": {
            "analysis": {
                "char_filter": {
                    "remove_loose_apostrophes": {
                        "pattern": " '|' ",
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
                    },
                    "remove_special_end": {
                        "pattern": "[^\\p{L}\\p{N}&']+$",
                        "type": "pattern_replace",
                        "replacement": ""
                    },
                    "remove_special1": {
                        "pattern": "([\\p{L}]+)[^\\p{L}\\p{N}&']+(?=[\\p{N}\\s]+)",
                        "type": "pattern_replace",
                        "replacement": "$1 "
                    },
                    "remove_special2": {
                        "pattern": "([\\p{N}]+)[^\\p{L}\\p{N}&']+(?=[\\p{L}\\s]+)",
                        "type": "pattern_replace",
                        "replacement": "$1 "
                    },
                    "remove_special3": {
                        "pattern": "([\\p{L}]+)[^\\p{L}\\p{N}&']+(?=[\\p{L}]+)",
                        "type": "pattern_replace",
                        "replacement": "$1 "
                    }
                },
                "analyzer": {
                    "default_analyzer": {
                        "type": "custom",
                        "tokenizer": "whitespace",
                        "char_filter": [
                            "remove_loose_apostrophes",
                            "space_possessive_apostrophes",
                            "remove_special_beginning",
                            "remove_special_end",
                            "remove_special1",
                            "remove_special2",
                            "remove_special3"
                        ],
                        "filter": [
                            "lowercase",
                            "asciifolding",
                            "shingle"
                        ]
                    }
                },
                "filter": {
                    "token_shingle": {
                        "type": "shingle",
                        "max_shingle_size": 4,
                        "min_shingle_size": 2,
                        "output_unigrams": "true"
                    }
                }
            }
        }
    }

    def __init__(self, resource_loader, es_host=None):
        self._resource_loader = resource_loader
        self._es_host = self._get_es_host(es_host)
        self.__es_client = None

    @property
    def _es_client(self):
        # Lazily connect to Elasticsearch
        if self.__es_client is None:
            self.__es_client = Elasticsearch(self._es_host)
        return self.__es_client

    def get(self, query_string=None, **kwargs):
        """Gets a collection of documents from the knowledge base.

        Args:
            search_query (str, optional): Description
            index (str): The name of an index
            sort (TYPE): Description
            location (TYPE): Description

        Returns:
            list: list of matching documents
        """
        es_query = {}
        try:
            index = kwargs['index']
        except KeyError:
            raise TypeError("get() missing required keyword argument 'index'")

        doc_id = kwargs.get('id')
        if doc_id:
            # If an id was passed in, simply retrieve the specified document
            response = self._es_client.get(index=index, doc_type=DOC_TYPE, id=doc_id)
            return [response['_source']]

        sort = kwargs.get('sort')
        location = kwargs.get('location')
        if location and 'latitude' in location and 'longitude' in location:
            location = {'lat': location['latitude'], 'lon': location['longitude']}

        if sort == 'location':
            es_query = {
                "sort": [{
                    "_geo_distance": {
                        "location": location,
                        "order": "asc",
                        "unit": "km",
                        "distance_type": "plane"
                    }
                }]
            }
        # TODO: handle other sorts
        response = self._es_client.search(index=index, body=es_query, q=query_string)

        results = [hit['_source'] for hit in response['hits']['hits']]
        return results

    def config(self, config):
        raise NotImplementedError

    @staticmethod
    def _get_es_host(es_host=None):
        es_host = es_host or os.environ.get('MM_ES_HOST')
        return es_host

    @classmethod
    def create_index(cls, index_name, es_host=None, es_client=None):
        es_host = cls._get_es_host(es_host)
        es_client = es_client or Elasticsearch(es_host)

        mapping = QuestionAnswerer.DEFAULT_ES_MAPPING

        if not es_client.indices.exists(index=index_name):
            logger.info("Creating index '{}'".format(index_name))
            es_client.indices.create(index_name, body=mapping)
        else:
            logger.error("Index '{}' already exists.".format(index_name))

    @classmethod
    def load_index(cls, index_name, data_file, es_host=None, es_client=None):
        es_host = cls._get_es_host(es_host)
        es_client = es_client or Elasticsearch(es_host)

        with open(data_file) as data_fp:
            data = json.load(data_fp)

        def _doc_generator(docs):
            for doc in docs:
                base = {'_id': doc['id']}
                base.update(doc)
                yield base

        # create index if specified index does not exist
        if not es_client.indices.exists(index=index_name):
            QuestionAnswerer.create_index(index_name, es_host=es_host, es_client=es_client)

        count = 0
        for okay, result in streaming_bulk(es_client, _doc_generator(data), index=index_name,
                                           doc_type=DOC_TYPE, chunk_size=50):

            action, result = result.popitem()
            doc_id = '/%s/%s/%s' % (index_name, DOC_TYPE, result['_id'])
            # process the information from ES whether the document has been
            # successfully indexed
            if not okay:
                logger.error('Failed to %s document %s: %r', action, doc_id, result)
            else:
                count += 1
                logger.debug('Loaded document: %s', doc_id)
        logger.info('Loaded %s document%s', count, '' if count == 1 else 's')
