# -*- coding: utf-8 -*-
"""
This module contains the question answerer component of Workbench.
"""
from __future__ import absolute_import, unicode_literals
from builtins import object

import json
import logging

from ._config import get_app_name, DOC_TYPE, DEFAULT_ES_QA_MAPPING
from ._elasticsearch_helpers import create_es_client, load_index, get_scoped_index_name

from ..resource_loader import ResourceLoader

logger = logging.getLogger(__name__)


class QuestionAnswerer(object):
    """The question answerer is primarily an information retrieval system that provides all the
    necessary functionality for interacting with the application's knowledge base.
    """
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
            self.__es_client = create_es_client(self._es_host)
        return self.__es_client

    def get(self, query_string=None, **kwargs):
        """Gets a collection of documents from the knowledge base matching the provided
        search criteria.

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
            index = get_scoped_index_name(self._app_name, index)
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

    @classmethod
    def load_kb(cls, app_name, index_name, data_file, es_host=None, es_client=None,
                connect_timeout=2):
        """Loads documents from disk into the specified index in the knowledge base. If an index
        with the specified name doesn't exist, a new index with that name will be created in the
        knowledge base.

        Args:
            app_name (str): The name of the app
            index_name (str): The name of the new index to be created
            data_file (str): The path to the data file containing the documents to be imported
                into the knowledge base index
            es_host (str): The Elasticsearch host server
            es_client (Elasticsearch): The Elasticsearch client
            connect_timeout (int, optional): The amount of time for a connection to the
            Elasticsearch host
        """
        with open(data_file) as data_fp:
            data = json.load(data_fp)

        def _doc_generator(docs):
            for doc in docs:
                base = {'_id': doc['id']}
                base.update(doc)
                yield base

        load_index(app_name, index_name, data, _doc_generator, DEFAULT_ES_QA_MAPPING, DOC_TYPE,
                   es_host, es_client, connect_timeout=connect_timeout)
