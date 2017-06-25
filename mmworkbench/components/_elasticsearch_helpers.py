# -*- coding: utf-8 -*-
"""This module contains helper methods for consuming Elasticsearch."""
from __future__ import absolute_import, unicode_literals

import os
import logging

from elasticsearch import Elasticsearch, ConnectionError as ESConnectionError
from elasticsearch.helpers import streaming_bulk

from ..exceptions import KnowledgeBaseConnectionError

logger = logging.getLogger(__name__)


def get_scoped_index_name(app_name, index_name):
    return '{}${}'.format(app_name, index_name)


def create_es_client(es_host=None, es_user=None, es_pass=None):
    """Creates a new Elasticsearch client

    Args:
        es_host (str): The Elasticsearch host server
        es_user (str): The Elasticsearch username for http auth
        es_pass (str): The Elasticsearch password for http auth
    """
    es_host = es_host or os.environ.get('MM_ES_HOST')
    es_user = es_user or os.environ.get('MM_ES_USERNAME')
    es_pass = es_pass or os.environ.get('MM_ES_PASSWORD')

    http_auth = (es_user, es_pass) if es_user and es_pass else None
    es_client = Elasticsearch(es_host, http_auth=http_auth)
    return es_client


def create_index(app_name, index_name, mapping, es_host=None, es_client=None, connect_timeout=2):
    """Creates a new index.

    Args:
        app_name (str): The name of the app
        index_name (str): The name of the new index to be created
        mapping (str): The Elasticsearch index mapping to use
        es_host (str): The Elasticsearch host server
        es_client: The Elasticsearch client
        connect_timeout (int, optional): The amount of time for a connection to the
            Elasticsearch host
    """
    es_client = es_client or create_es_client(es_host)
    scoped_index_name = get_scoped_index_name(app_name, index_name)

    try:
        # Confirm ES connection with a shorter timeout
        es_client.cluster.health(request_timeout=connect_timeout)

        if not es_client.indices.exists(index=scoped_index_name):
            logger.info('Creating index %r', index_name)
            es_client.indices.create(scoped_index_name, body=mapping)
        else:
            logger.error('Index %r already exists.', index_name)
    except ESConnectionError:
        logger.error('Unable to connect to Elasticsearch cluster at %r', es_host)
        raise KnowledgeBaseConnectionError()


def delete_index(app_name, index_name, es_host=None, es_client=None, connect_timeout=2):
    """Deletes an index.

    Args:
        app_name (str): The name of the app
        index_name (str): The name of the index to be deleted
        es_host (str): The Elasticsearch host server
        es_client: The Elasticsearch client
        connect_timeout (int, optional): The amount of time for a connection to the
            Elasticsearch host
    """
    es_client = es_client or create_es_client(es_host)
    scoped_index_name = get_scoped_index_name(app_name, index_name)

    try:
        # Confirm ES connection with a shorter timeout
        es_client.cluster.health(request_timeout=connect_timeout)

        if es_client.indices.exists(index=scoped_index_name):
            logger.info('Deleting index %r', index_name)
            es_client.indices.delete(scoped_index_name)
    except ESConnectionError:
        logger.error('Unable to connect to Elasticsearch cluster at {!r}'.format(es_host))
        raise KnowledgeBaseConnectionError()


def load_index(app_name, index_name, data, doc_generator, mapping, doc_type, es_host=None,
               es_client=None, connect_timeout=2):
    """Loads documents from data into the specified index. If an index with the specified name
    doesn't exist, a new index with that name will be created.

    Args:
        app_name (str): The name of the app
        index_name (str): The name of the new index to be created
        data (list): A list of the documents loaded from disk to be imported into the index
        doc_generator (func): A generator which processes the loaded documents and yeilds them in
                              the correct format to insert into Elasticsearch
        mapping (str): The Elasticsearch index mapping to use
        doc_type (str): The document type
        es_host (str): The Elasticsearch host server
        es_client (Elasticsearch): The Elasticsearch client
        connect_timeout (int, optional): The amount of time for a connection to the
            Elasticsearch host
    """
    scoped_index_name = get_scoped_index_name(app_name, index_name)
    es_client = es_client or create_es_client(es_host)

    try:
        # Confirm ES connection with a shorter timeout
        es_client.cluster.health(request_timeout=connect_timeout)

        # create index if specified index does not exist
        if es_client.indices.exists(index=scoped_index_name):
            logger.info('Loading index %r', index_name)
        else:
            create_index(app_name, index_name, mapping, es_host=es_host, es_client=es_client)

        count = 0
        for okay, result in streaming_bulk(es_client, doc_generator(data),
                                           index=scoped_index_name, doc_type=doc_type,
                                           chunk_size=50):

            action, result = result.popitem()
            doc_id = '/%s/%s/%s' % (index_name, doc_type, result['_id'])
            # process the information from ES whether the document has been
            # successfully indexed
            if not okay:
                logger.error('Failed to %s document %s: %r', action, doc_id, result)
            else:
                count += 1
                logger.debug('Loaded document: %s', doc_id)
        logger.info('Loaded %s document%s', count, '' if count == 1 else 's')
    except ESConnectionError:
        logger.error('Unable to connect to Elasticsearch cluster at {!r}'.format(es_host))
        raise KnowledgeBaseConnectionError()
