import os
import logging

from elasticsearch import Elasticsearch, ConnectionError as ESConnectionError
from elasticsearch.helpers import streaming_bulk

logger = logging.getLogger(__name__)


def create_es_client(es_host=None, es_user=None, es_pass=None):
    es_host = es_host or os.environ.get('MM_ES_HOST')
    es_user = es_user or os.environ.get('MM_ES_USERNAME')
    es_pass = es_pass or os.environ.get('MM_ES_PASSWORD')

    http_auth = (es_user, es_pass) if es_user and es_pass else None
    es_client = Elasticsearch(es_host, http_auth=http_auth, request_timeout=60, timeout=60)
    return es_client


def create_index(index_name, scoped_index_name, mapping, es_host=None, es_client=None):
    """Creates a new index in the knowledge base.

    Args:
        index_name (str): The name of the new index to be created
        es_host (str): The Elasticsearch host server
        es_client: Description
    """
    es_client = es_client or create_es_client(es_host)

    try:
        if not es_client.indices.exists(index=scoped_index_name):
            logger.info('Creating index %r', index_name)
            es_client.indices.create(scoped_index_name, body=mapping)
        else:
            logger.error('Index %r already exists.', index_name)
    except ESConnectionError as ex:
        logger.error('Unable to connect to Elasticsearch cluster at {!r}'.format(es_host))
        raise ex


def load_index(app_name, index_name, data, doc_generator, mapping, doc_type, es_host=None,
               es_client=None):
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
    scoped_index_name = '{}${}'.format(app_name, index_name)
    es_client = es_client or create_es_client(es_host)

    # create index if specified index does not exist
    try:
        if es_client.indices.exists(index=scoped_index_name):
            logger.info('Loading index %r', index_name)
        else:
            create_index(index_name, scoped_index_name, mapping, es_host=es_host,
                         es_client=es_client)

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
    except ESConnectionError as ex:
        logger.error('Unable to connect to Elasticsearch cluster at {!r}'.format(es_host))
        raise ex
