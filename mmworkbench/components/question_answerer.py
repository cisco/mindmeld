# -*- coding: utf-8 -*-
"""
This module contains the question answerer component of Workbench.
"""
from __future__ import unicode_literals
from builtins import object

import json
import logging
import os

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

from ._config import get_app_name

from ..resource_loader import ResourceLoader


DOC_TYPE = 'document'

logger = logging.getLogger(__name__)


class QuestionAnswerer(object):
    """The question answerer is primarily an information retrieval system that provides all the
    necessary functionality for interacting with the application's knowledge base.
    """

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

    def __init__(self, app_path, resource_loader=None, es_host=None):
        """Initializes a question answerer

        Args:
            app_path (str): The path to the directory containing the app's data
            resource_loader (ResourceLoader): An object which can load resources for the answerer
            es_host (str): The Elasticsearch host server
        """
        self._resource_loader = resource_loader or ResourceLoader.create_resource_loader(app_path)
        self._es_host = es_host
        self.__es_client = None
        self._app_name = get_app_name(app_path)

    @property
    def _es_client(self):
        # Lazily connect to Elasticsearch
        if self.__es_client is None:
            self.__es_client = self._create_es_client(self._es_host)
        return self.__es_client

    def get(self, query_string=None, **kwargs):
        """Gets a collection of documents from the knowledge base matching the provided search
        criteria.

        Args:
            query_string (str, optional): A lucene style query string
            index (str): The name of an index
            sort (str): The sort method that should be used
            location (dict): A location to use in the query
            id (str): The id of a particular document to retrieve

        Returns:
            list: list of matching documents
        """
        es_query = {}
        try:
            index = kwargs['index']
            index = '{}${}'.format(self._app_name, index)
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
        """Summary

        Args:
            config: Description
        """
        raise NotImplementedError

    @staticmethod
    def _create_es_client(es_host=None, es_user=None, es_pass=None):
        es_host = es_host or os.environ.get('MM_ES_HOST')
        es_user = es_user or os.environ.get('MM_ES_USERNAME')
        es_pass = es_pass or os.environ.get('MM_ES_PASSWORD')

        http_auth = (es_user, es_pass) if es_user and es_pass else None
        es_client = Elasticsearch(es_host, http_auth=http_auth, request_timeout=60, timeout=60)
        return es_client

    @classmethod
    def create_index(cls, app_name, index_name, es_host=None, es_client=None):
        """Creates a new index in the knowledge base.

        Args:
            index_name (str): The name of the new index to be created
            es_host (str): The Elasticsearch host server
            es_client: Description
        """
        es_client = es_client or cls._create_es_client(es_host)

        mapping = QuestionAnswerer.DEFAULT_ES_MAPPING
        scoped_index_name = '{}${}'.format(app_name, index_name)

        if not es_client.indices.exists(index=scoped_index_name):
            logger.info("Creating index '{}'".format(index_name))
            es_client.indices.create(scoped_index_name, body=mapping)
        else:
            logger.error("Index '{}' already exists.".format(index_name))

    @classmethod
    def load_index(cls, app_name, index_name, data_file, es_host=None, es_client=None):
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
        es_client = es_client or cls._create_es_client(es_host)

        with open(data_file) as data_fp:
            data = json.load(data_fp)

        def _doc_generator(docs):
            for doc in docs:
                base = {'_id': doc['id']}
                base.update(doc)
                yield base

        scoped_index_name = '{}${}'.format(app_name, index_name)

        # create index if specified index does not exist
        if not es_client.indices.exists(index=scoped_index_name):
            QuestionAnswerer.create_index(app_name, index_name, es_host=es_host,
                                          es_client=es_client)

        count = 0
        for okay, result in streaming_bulk(es_client, _doc_generator(data), index=scoped_index_name,
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
