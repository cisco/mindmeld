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

    def __init__(self, resource_loader, es_host=None):
        self._resource_loader = resource_loader
        self._es_host = es_host or os.environ.get('MM_ES_HOST')
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


def create_index(es_host, index_name):
    es_client = Elasticsearch(es_host)
    # TODO: add analysis methods for other fields
    mapping = {
        'mappings': {
            DOC_TYPE: {
                'properties': {
                    'location': {'type': 'geo_point'}
                }
            }
        }
    }
    es_client.indices.create(index_name, body=mapping)


def load_index(es_host, index_name, data_file):
    es_client = Elasticsearch(es_host)

    with open(data_file) as data_fp:
        data = json.load(data_fp)

    def _doc_generator(docs):
        for doc in docs:
            base = {'_id': doc['id']}
            base.update(doc)
            yield base

    for result, okay in streaming_bulk(es_client, _doc_generator(data), index=index_name,
                                       doc_type=DOC_TYPE, chunk_size=50):

        action, result = result.popitem()
        doc_id = '/%s/%s/%s' % (index_name, DOC_TYPE, result['_id'])
        # process the information from ES whether the document has been
        # successfully indexed
        if not okay:
            logger.error('Failed to %s document %s: %r', action, doc_id, result)
        else:
            logger.info('Loaded document: %s', doc_id)
