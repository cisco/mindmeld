# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Cisco Systems, Inc. and others.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains helper methods for consuming Elasticsearch."""
import os
import logging

from elasticsearch5 import (Elasticsearch, ImproperlyConfigured, ElasticsearchException,
                            ConnectionError as EsConnectionError, TransportError)
from elasticsearch5.helpers import streaming_bulk
from tqdm import tqdm

from ._config import DEFAULT_ES_INDEX_TEMPLATE, DEFAULT_ES_INDEX_TEMPLATE_NAME

from ..exceptions import KnowledgeBaseConnectionError, KnowledgeBaseError

logger = logging.getLogger(__name__)

INDEX_TYPE_SYNONYM = 'syn'
INDEX_TYPE_KB = 'kb'


def get_scoped_index_name(app_namespace, index_name):
    return '{}${}'.format(app_namespace, index_name)


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

    try:
        http_auth = (es_user, es_pass) if es_user and es_pass else None
        es_client = Elasticsearch(es_host, http_auth=http_auth)
        return es_client
    except ElasticsearchException:
        raise KnowledgeBaseError
    except ImproperlyConfigured:
        raise KnowledgeBaseError


def does_index_exist(app_namespace, index_name, es_host=None, es_client=None, connect_timeout=2):
    """Return boolean flag to indicate whether the specified index exists."""

    es_client = es_client or create_es_client(es_host)
    scoped_index_name = get_scoped_index_name(app_namespace, index_name)

    try:
        # Confirm ES connection with a shorter timeout
        es_client.cluster.health(request_timeout=connect_timeout)
        return es_client.indices.exists(index=scoped_index_name)
    except EsConnectionError as e:
        logger.debug('Unable to connect to Elasticsearch: %s details: %s', e.error, e.info)
        raise KnowledgeBaseConnectionError(es_host=es_client.transport.hosts)
    except TransportError as e:
        logger.error('Unexpected error occurred when sending requests to Elasticsearch: %s '
                     'Status code: %s details: %s', e.error, e.status_code, e.info)
        raise KnowledgeBaseError
    except ElasticsearchException:
        raise KnowledgeBaseError


def get_field_names(app_namespace, index_name, es_host=None, es_client=None, connect_timeout=2):
    """Return a list of field names available in the specified index."""

    es_client = es_client or create_es_client(es_host)
    scoped_index_name = get_scoped_index_name(app_namespace, index_name)

    try:
        if not does_index_exist(app_namespace, index_name, es_host, es_client, connect_timeout):
            raise ValueError('Elasticsearch index \'{}\' does not exist.'.format(index_name))

        res = es_client.indices.get(index=scoped_index_name)
        all_field_info = res[scoped_index_name]['mappings']['document']['properties']
        return all_field_info.keys()
    except EsConnectionError as e:
        logger.debug('Unable to connect to Elasticsearch: %s details: %s', e.error, e.info)
        raise KnowledgeBaseConnectionError(es_host=es_client.transport.hosts)
    except TransportError as e:
        logger.error('Unexpected error occurred when sending requests to Elasticsearch: %s '
                     'Status code: %s details: %s', e.error, e.status_code, e.info)
        raise KnowledgeBaseError
    except ElasticsearchException:
        raise KnowledgeBaseError


def create_index(app_namespace, index_name, mapping, es_host=None, es_client=None,
                 connect_timeout=2):
    """Creates a new index.

    Args:
        app_namespace (str): The namespace of the app
        index_name (str): The name of the new index to be created
        mapping (str): The Elasticsearch index mapping to use
        es_host (str): The Elasticsearch host server
        es_client: The Elasticsearch client
        connect_timeout (int, optional): The amount of time for a connection to the
            Elasticsearch host
    """
    es_client = es_client or create_es_client(es_host)
    scoped_index_name = get_scoped_index_name(app_namespace, index_name)

    try:
        if not does_index_exist(app_namespace, index_name, es_host, es_client, connect_timeout):
            # checks the existence of default index template, if not then creates it.
            if not es_client.indices.exists_template(name=DEFAULT_ES_INDEX_TEMPLATE_NAME):
                es_client.indices.put_template(name=DEFAULT_ES_INDEX_TEMPLATE_NAME,
                                               body=DEFAULT_ES_INDEX_TEMPLATE)
            logger.info('Creating index %r', index_name)
            es_client.indices.create(scoped_index_name, body=mapping)
        else:
            logger.error('Index %r already exists.', index_name)
    except EsConnectionError as e:
        logger.debug('Unable to connect to Elasticsearch: %s details: %s', e.error, e.info)
        raise KnowledgeBaseConnectionError(es_host=es_client.transport.hosts)
    except TransportError as e:
        logger.error('Unexpected error occurred when sending requests to Elasticsearch: %s '
                     'Status code: %s details: %s', e.error, e.status_code, e.info)
        raise KnowledgeBaseError('Unexpected error occurred when sending requests to '
                                 'Elasticsearch: {} Status code: {} details: '
                                 '{}'.format(e.error, e.status_code, e.info))
    except ElasticsearchException:
        raise KnowledgeBaseError


def delete_index(app_namespace, index_name, es_host=None, es_client=None, connect_timeout=2):
    """Deletes an index.

    Args:
        app_namespace (str): The namespace of the app
        index_name (str): The name of the index to be deleted
        es_host (str): The Elasticsearch host server
        es_client: The Elasticsearch client
        connect_timeout (int, optional): The amount of time for a connection to the
            Elasticsearch host
    """
    es_client = es_client or create_es_client(es_host)
    scoped_index_name = get_scoped_index_name(app_namespace, index_name)

    try:
        if does_index_exist(app_namespace, index_name, es_host, es_client, connect_timeout):
            logger.info('Deleting index %r', index_name)
            es_client.indices.delete(scoped_index_name)
        else:
            raise ValueError('Elasticsearch index \'{}\' for application \'{}\' does not exist.'
                             .format(index_name, app_namespace))
    except EsConnectionError as e:
        logger.debug('Unable to connect to Elasticsearch: %s details: %s', e.error, e.info)
        raise KnowledgeBaseConnectionError(es_host=es_client.transport.hosts)
    except TransportError as e:
        logger.error('Unexpected error occurred when sending requests to Elasticsearch: %s '
                     'Status code: %s details: %s', e.error, e.status_code, e.info)
        raise KnowledgeBaseError
    except ElasticsearchException:
        raise KnowledgeBaseError


def load_index(app_namespace, index_name, docs, docs_count, mapping, doc_type, es_host=None,
               es_client=None, connect_timeout=2):
    """Loads documents from data into the specified index. If an index with the specified name
    doesn't exist, a new index with that name will be created.

    Args:
        app_namespace (str): The namespace of the app
        index_name (str): The name of the new index to be created
        docs (iterable): An iterable which contains a collection of documents in the correct format
                         which should be imported into the index
        docs_count (int): The number of documents in doc
        mapping (str): The Elasticsearch index mapping to use
        doc_type (str): The document type
        es_host (str): The Elasticsearch host server
        es_client (Elasticsearch): The Elasticsearch client
        connect_timeout (int, optional): The amount of time for a connection to the
            Elasticsearch host
    """
    scoped_index_name = get_scoped_index_name(app_namespace, index_name)
    es_client = es_client or create_es_client(es_host)
    try:
        # create index if specified index does not exist
        if does_index_exist(app_namespace, index_name, es_host, es_client, connect_timeout):
            logger.warning('Elasticsearch index \'%s\' for application \'%s\' already exists!',
                           index_name, app_namespace)
            logger.info('Loading index %r', index_name)
        else:
            create_index(app_namespace, index_name, mapping, es_host=es_host, es_client=es_client)

        count = 0
        # create the progess bar with docs count
        pbar = tqdm(total=docs_count)

        for okay, result in streaming_bulk(es_client, docs,
                                           index=scoped_index_name, doc_type=doc_type,
                                           chunk_size=50, raise_on_error=False):

            action, result = result.popitem()
            doc_id = '/%s/%s/%s' % (index_name, doc_type, result['_id'])
            # process the information from ES whether the document has been
            # successfully indexed
            if not okay:
                logger.error('Failed to %s document %s: %r', action, doc_id, result)
            else:
                count += 1
            pbar.update(1)
        # close the progress bar and flush all output
        pbar.close()
        logger.info('Loaded %s document%s', count, '' if count == 1 else 's')
    except EsConnectionError as e:
        logger.debug('Unable to connect to Elasticsearch: %s details: %s', e.error, e.info)
        raise KnowledgeBaseConnectionError(es_host=es_client.transport.hosts)
    except TransportError as e:
        logger.error('Unexpected error occurred when sending requests to Elasticsearch: %s '
                     'Status code: %s details: %s', e.error, e.status_code, e.info)
        raise KnowledgeBaseError
    except ElasticsearchException:
        raise KnowledgeBaseError
