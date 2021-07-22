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
import logging
import os

from tqdm.auto import tqdm

from ._util import _get_module_or_attr as _getattr
from ..exceptions import ElasticsearchKnowledgeBaseConnectionError, KnowledgeBaseError

logger = logging.getLogger(__name__)

INDEX_TYPE_SYNONYM = "syn"
INDEX_TYPE_KB = "kb"
DOC_TYPE = "document"

# ElasticSearch mapping to define text analysis settings for text fields.
# It defines specific index configuration for synonym indices. The common index configuration
# is in default index template.
DEFAULT_ES_SYNONYM_MAPPING = {
    "mappings": {
        "properties": {
            "sort_factor": {"type": "double"},
            "whitelist": {
                "type": "nested",
                "properties": {
                    "name": {
                        "type": "text",
                        "fields": {
                            "raw": {"type": "keyword", "ignore_above": 256},
                            "normalized_keyword": {
                                "type": "text",
                                "analyzer": "keyword_match_analyzer",
                            },
                            "char_ngram": {
                                "type": "text",
                                "analyzer": "char_ngram_analyzer",
                            },
                        },
                        "analyzer": "default_analyzer",
                    }
                },
            },
        }
    }
}

PHONETIC_ES_SYNONYM_MAPPING = {
    "mappings": {
        "properties": {
            "sort_factor": {"type": "double"},
            "whitelist": {
                "type": "nested",
                "properties": {
                    "name": {
                        "type": "text",
                        "fields": {
                            "raw": {"type": "keyword", "ignore_above": 256},
                            "normalized_keyword": {
                                "type": "text",
                                "analyzer": "keyword_match_analyzer",
                            },
                            "char_ngram": {
                                "type": "text",
                                "analyzer": "char_ngram_analyzer",
                            },
                            "double_metaphone": {
                                "type": "text",
                                "analyzer": "phonetic_analyzer",
                            },
                        },
                        "analyzer": "default_analyzer",
                    }
                },
            },
            "cname": {
                "type": "text",
                "analyzer": "default_analyzer",
                "fields": {
                    "raw": {"type": "keyword", "ignore_above": 256},
                    "normalized_keyword": {
                        "type": "text",
                        "analyzer": "keyword_match_analyzer",
                    },
                    "char_ngram": {
                        "type": "text",
                        "analyzer": "char_ngram_analyzer",
                    },
                    "double_metaphone": {
                        "type": "text",
                        "analyzer": "phonetic_analyzer",
                    },
                },
            },
        }
    },
    "settings": {
        "analysis": {
            "filter": {
                "phonetic_filter": {
                    "type": "phonetic",
                    "encoder": "doublemetaphone",
                    "replace": True,
                    "max_code_len": 7,
                }
            },
            "analyzer": {
                "phonetic_analyzer": {
                    "filter": [
                        "lowercase",
                        "asciifolding",
                        "token_shingle",
                        "phonetic_filter",
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
                        "remove_special3",
                        "remove_dot",
                    ],
                    "type": "custom",
                    "tokenizer": "whitespace",
                }
            },
        }
    },
}

DEFAULT_ES_INDEX_TEMPLATE_NAME = "mindmeld_default"

# Default ES index template that contains the base index configuration shared across different
# types of indices. Currently all ES indices will be created using this template.
# - custom text analysis settings such as custom analyzers, token filters and character filters.
# - dynamic field mapping template for text fields
# - common fields, e.g. id.
DEFAULT_ES_INDEX_TEMPLATE = {
    "template": "*",
    "mappings": {
        "dynamic_templates": [
            {
                "default_text": {
                    "match": "*",
                    "match_mapping_type": "string",
                    "mapping": {
                        "type": "text",
                        "analyzer": "default_analyzer",
                        "fields": {
                            "raw": {"type": "keyword", "ignore_above": 256},
                            "normalized_keyword": {
                                "type": "text",
                                "analyzer": "keyword_match_analyzer",
                            },
                            "processed_text": {
                                "type": "text",
                                "analyzer": "english",
                            },
                            "char_ngram": {
                                "type": "text",
                                "analyzer": "char_ngram_analyzer",
                            },
                        },
                    },
                }
            }
        ],
        "properties": {"id": {"type": "keyword"}},
    },
    "settings": {
        "analysis": {
            "char_filter": {
                "remove_loose_apostrophes": {
                    "pattern": " '|' ",
                    "type": "pattern_replace",
                    "replacement": "",
                },
                "space_possessive_apostrophes": {
                    "pattern": "([^\\p{N}\\s]+)'s ",
                    "type": "pattern_replace",
                    "replacement": "$1 's ",
                },
                "remove_special_beginning": {
                    "pattern": "^[^\\p{L}\\p{N}\\p{Sc}&']+",
                    "type": "pattern_replace",
                    "replacement": "",
                },
                "remove_special_end": {
                    "pattern": "[^\\p{L}\\p{N}&']+$",
                    "type": "pattern_replace",
                    "replacement": "",
                },
                "remove_special1": {
                    "pattern": "([\\p{L}]+)[^\\p{L}\\p{N}&']+(?=[\\p{N}\\s]+)",
                    "type": "pattern_replace",
                    "replacement": "$1 ",
                },
                "remove_special2": {
                    "pattern": "([\\p{N}]+)[^\\p{L}\\p{N}&']+(?=[\\p{L}\\s]+)",
                    "type": "pattern_replace",
                    "replacement": "$1 ",
                },
                "remove_special3": {
                    "pattern": "([\\p{L}]+)[^\\p{L}\\p{N}&']+(?=[\\p{L}]+)",
                    "type": "pattern_replace",
                    "replacement": "$1 ",
                },
                "remove_comma": {
                    "pattern": ",",
                    "type": "pattern_replace",
                    "replacement": "",
                },
                "remove_tm_and_r": {
                    "pattern": "™|®",
                    "type": "pattern_replace",
                    "replacement": "",
                },
                "remove_dot": {
                    "pattern": "([\\p{L}]+)[.]+(?=[\\p{L}\\s]+)",
                    "type": "pattern_replace",
                    "replacement": "$1",
                },
            },
            "filter": {
                "token_shingle": {
                    "max_shingle_size": "4",
                    "min_shingle_size": "2",
                    "output_unigrams": "true",
                    "type": "shingle",
                },
                "ngram_filter": {"type": "ngram", "min_gram": "3", "max_gram": "3"},
            },
            "analyzer": {
                "default_analyzer": {
                    "filter": ["lowercase", "asciifolding", "token_shingle"],
                    "char_filter": [
                        "remove_comma",
                        "remove_tm_and_r",
                        "remove_loose_apostrophes",
                        "space_possessive_apostrophes",
                        "remove_special_beginning",
                        "remove_special_end",
                        "remove_special1",
                        "remove_special2",
                        "remove_special3",
                    ],
                    "type": "custom",
                    "tokenizer": "whitespace",
                },
                "keyword_match_analyzer": {
                    "filter": ["lowercase", "asciifolding"],
                    "char_filter": [
                        "remove_comma",
                        "remove_tm_and_r",
                        "remove_loose_apostrophes",
                        "space_possessive_apostrophes",
                        "remove_special_beginning",
                        "remove_special_end",
                        "remove_special1",
                        "remove_special2",
                        "remove_special3",
                    ],
                    "type": "custom",
                    "tokenizer": "keyword",
                },
                "char_ngram_analyzer": {
                    "filter": ["lowercase", "asciifolding", "ngram_filter"],
                    "char_filter": [
                        "remove_comma",
                        "remove_tm_and_r",
                        "remove_loose_apostrophes",
                        "space_possessive_apostrophes",
                        "remove_special_beginning",
                        "remove_special_end",
                        "remove_special1",
                        "remove_special2",
                        "remove_special3",
                    ],
                    "type": "custom",
                    "tokenizer": "whitespace",
                },
            },
        }
    },
}

# Elasticsearch mapping to define knowledge base index specific configuration:
# - dynamic field mapping to index all synonym whitelist in fields with "$whitelist" suffix.
# - location field
#
# The common configuration is defined in default index template
DEFAULT_ES_QA_MAPPING = {
    "mappings": {
        "dynamic_templates": [
            {
                "synonym_whitelist_text": {
                    "match": "*$whitelist",
                    "match_mapping_type": "object",
                    "mapping": {
                        "type": "nested",
                        "properties": {
                            "name": {
                                "type": "text",
                                "fields": {
                                    "raw": {"type": "keyword", "ignore_above": 256},
                                    "normalized_keyword": {
                                        "type": "text",
                                        "analyzer": "keyword_match_analyzer",
                                    },
                                    "char_ngram": {
                                        "type": "text",
                                        "analyzer": "char_ngram_analyzer",
                                    },
                                },
                                "analyzer": "default_analyzer",
                            }
                        },
                    },
                }
            }
        ],
        "properties": {"location": {"type": "geo_point"}},
    }
}

DEFAULT_ES_RANKING_CONFIG = {"query_clauses_operator": "or"}


def get_scoped_index_name(app_namespace, index_name):
    return "{}${}".format(app_namespace, index_name)


def create_es_client(es_host=None, es_user=None, es_pass=None):
    """Creates a new Elasticsearch client

    Args:
        es_host (str): The Elasticsearch host server
        es_user (str): The Elasticsearch username for http auth
        es_pass (str): The Elasticsearch password for http auth
    """
    es_host = es_host or os.environ.get("MM_ES_HOST")
    es_user = es_user or os.environ.get("MM_ES_USERNAME")
    es_pass = es_pass or os.environ.get("MM_ES_PASSWORD")

    try:
        http_auth = (es_user, es_pass) if es_user and es_pass else None
        es_client = _getattr("elasticsearch", "Elasticsearch")(es_host, http_auth=http_auth)
        return es_client
    except _getattr("elasticsearch", "ElasticsearchException") as e:
        raise KnowledgeBaseError from e
    except _getattr("elasticsearch", "ImproperlyConfigured") as e:
        raise KnowledgeBaseError from e


def is_es_version_7(es_client):
    major_version = int(es_client.info()["version"]["number"].split(".")[0])
    if major_version < 5:
        logger.warning(
            "Major version of ElasticSearch %d is not officially supported.",
            major_version,
        )
    if major_version >= 7:
        return True
    return False


def resolve_es_config_for_version(config, es_client):
    """ElasticSearch 7 no longer supports mapping types: https://www.elastic.co/guide/en/
    elasticsearch/reference/current/removal-of-types.html#removal-of-types"""
    if not is_es_version_7(es_client):
        if DOC_TYPE not in config.get("mappings", {}):
            mappings = config.pop("mappings")
            config["mappings"] = {DOC_TYPE: mappings}
    return config


def does_index_exist(
    app_namespace, index_name, es_host=None, es_client=None, connect_timeout=2
):
    """Return boolean flag to indicate whether the specified index exists."""

    es_client = es_client or create_es_client(es_host)
    scoped_index_name = get_scoped_index_name(app_namespace, index_name)

    try:
        # Confirm ES connection with a shorter timeout
        es_client.cluster.health(request_timeout=connect_timeout)
        return es_client.indices.exists(index=scoped_index_name)
    except _getattr("elasticsearch", "ConnectionError") as e:
        logger.debug(
            "Unable to connect to Elasticsearch: %s details: %s", e.error, e.info
        )
        raise ElasticsearchKnowledgeBaseConnectionError(es_host=es_client.transport.hosts) from e
    except _getattr("elasticsearch", "TransportError") as e:
        logger.error(
            "Unexpected error occurred when sending requests to Elasticsearch: %s "
            "Status code: %s details: %s",
            e.error,
            e.status_code,
            e.info,
        )
        raise KnowledgeBaseError from e
    except _getattr("elasticsearch", "ElasticsearchException") as e:
        raise KnowledgeBaseError from e


def get_field_names(
    app_namespace, index_name, es_host=None, es_client=None, connect_timeout=2
):
    """Return a list of field names available in the specified index."""

    es_client = es_client or create_es_client(es_host)
    scoped_index_name = get_scoped_index_name(app_namespace, index_name)

    try:
        if not does_index_exist(
            app_namespace, index_name, es_host, es_client, connect_timeout
        ):
            raise ValueError(
                "Elasticsearch index '{}' does not exist.".format(index_name)
            )

        res = es_client.indices.get(index=scoped_index_name)

        if is_es_version_7(es_client):
            all_field_info = res[scoped_index_name]["mappings"]["properties"]
        else:
            all_field_info = res[scoped_index_name]["mappings"][DOC_TYPE]["properties"]
        return all_field_info.keys()
    except _getattr("elasticsearch", "ConnectionError") as e:
        logger.debug(
            "Unable to connect to Elasticsearch: %s details: %s", e.error, e.info
        )
        raise ElasticsearchKnowledgeBaseConnectionError(es_host=es_client.transport.hosts) from e
    except _getattr("elasticsearch", "TransportError") as e:
        logger.error(
            "Unexpected error occurred when sending requests to Elasticsearch: %s "
            "Status code: %s details: %s",
            e.error,
            e.status_code,
            e.info,
        )
        raise KnowledgeBaseError from e
    except _getattr("elasticsearch", "ElasticsearchException") as e:
        raise KnowledgeBaseError from e


def create_index(
    app_namespace, index_name, mapping, es_host=None, es_client=None, connect_timeout=2
):
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
        if not does_index_exist(
            app_namespace, index_name, es_host, es_client, connect_timeout
        ):
            # TODO: add support for non-english texts by allowing configurable `langauge` as input
            template = resolve_es_config_for_version(
                DEFAULT_ES_INDEX_TEMPLATE, es_client
            )
            es_client.indices.put_template(
                name=DEFAULT_ES_INDEX_TEMPLATE_NAME, body=template
            )
            logger.info("Creating index %r", index_name)
            es_client.indices.create(scoped_index_name, body=mapping)
        else:
            logger.error("Index %r already exists.", index_name)
    except _getattr("elasticsearch", "ConnectionError") as e:
        logger.debug(
            "Unable to connect to Elasticsearch: %202s details: %s", e.error, e.info
        )
        raise ElasticsearchKnowledgeBaseConnectionError(es_host=es_client.transport.hosts) from e
    except _getattr("elasticsearch", "TransportError") as e:
        logger.error(
            "Unexpected error occurred when sending requests to Elasticsearch: %s "
            "Status code: %s details: %s",
            e.error,
            e.status_code,
            e.info,
        )
        raise KnowledgeBaseError(
            "Unexpected error occurred when sending requests to "
            "Elasticsearch: {} Status code: {} details: "
            "{}".format(e.error, e.status_code, e.info)
        ) from e
    except _getattr("elasticsearch", "ElasticsearchException") as e:
        raise KnowledgeBaseError from e


def delete_index(
    app_namespace, index_name, es_host=None, es_client=None, connect_timeout=2
):
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
        if does_index_exist(
            app_namespace, index_name, es_host, es_client, connect_timeout
        ):
            logger.info("Deleting index %r", index_name)
            es_client.indices.delete(scoped_index_name)
        else:
            raise ValueError(
                "Elasticsearch index '{}' for application '{}' does not exist.".format(
                    index_name, app_namespace
                )
            )
    except _getattr("elasticsearch", "ConnectionError") as e:
        logger.debug(
            "Unable to connect to Elasticsearch: %s details: %s", e.error, e.info
        )
        raise ElasticsearchKnowledgeBaseConnectionError(es_host=es_client.transport.hosts) from e
    except _getattr("elasticsearch", "TransportError") as e:
        logger.error(
            "Unexpected error occurred when sending requests to Elasticsearch: %s "
            "Status code: %s details: %s",
            e.error,
            e.status_code,
            e.info,
        )
        raise KnowledgeBaseError from e
    except _getattr("elasticsearch", "ElasticsearchException") as e:
        raise KnowledgeBaseError from e


def create_index_mapping(base_mapping, mapping_data):
    """Creates an index mapping given provided base mapping template and additional data.

    Args:
        base_mapping (dict): The base mapping template
        mapping_data (dict): The dictionary with metadata needed to create the mapping.
    """
    properties = base_mapping.get("mappings", {}).get("properties", {})
    embedding_properties = mapping_data.get("embedding_properties", [])
    for emb in embedding_properties:
        properties[emb["field"]] = {"type": "dense_vector", "dims": emb["dims"]}
    if "mappings" not in base_mapping:
        base_mapping["mappings"] = base_mapping
    base_mapping["mappings"]["properties"] = properties
    return base_mapping


def version_compatible_streaming_bulk(
    es_client, docs, index, chunk_size, raise_on_error, doc_type
):
    if is_es_version_7(es_client):
        return _getattr("elasticsearch.helpers", "streaming_bulk")(
            es_client,
            docs,
            index=index,
            chunk_size=chunk_size,
            raise_on_error=raise_on_error,
        )
    else:
        return _getattr("elasticsearch.helpers", "streaming_bulk")(
            es_client,
            docs,
            index=index,
            doc_type=doc_type,
            chunk_size=chunk_size,
            raise_on_error=raise_on_error,
        )


def load_index(
    app_namespace,
    index_name,
    docs,
    docs_count,
    mapping,
    doc_type=None,
    es_host=None,
    es_client=None,
    connect_timeout=2,
):
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
        if does_index_exist(
            app_namespace, index_name, es_host, es_client, connect_timeout
        ):
            logger.warning(
                "Elasticsearch index '%s' for application '%s' already exists!",
                index_name,
                app_namespace,
            )
            logger.info("Loading index %r", index_name)
        else:
            create_index(
                app_namespace, index_name, mapping, es_host=es_host, es_client=es_client
            )

        count = 0
        # create the progess bar with docs count
        pbar = tqdm(
            total=docs_count, desc="Loading Elasticsearch index {}".format(index_name)
        )

        es_version_7 = is_es_version_7(es_client)
        for okay, result in version_compatible_streaming_bulk(
            es_client, docs, scoped_index_name, 50, False, DOC_TYPE
        ):
            action, result = result.popitem()
            if es_version_7:
                doc_id = "/%s/%s" % (index_name, result["_id"])
            else:
                doc_id = "/%s/%s/%s" % (index_name, doc_type, result["_id"])

            # process the information from ES whether the document has been
            # successfully indexed
            if not okay:
                logger.error("Failed to %s document %s: %r", action, doc_id, result)
            else:
                count += 1
            pbar.update(1)

        # close the progress bar and flush all output
        pbar.close()
        # Refresh to make sure all data stored is available for search.
        es_client.indices.refresh(index=scoped_index_name)
        logger.info("Loaded %s document%s", count, "" if count == 1 else "s")
    except _getattr("elasticsearch", "ConnectionError") as e:
        logger.debug(
            "Unable to connect to Elasticsearch: %s details: %s", e.error, e.info
        )
        raise ElasticsearchKnowledgeBaseConnectionError(es_host=es_client.transport.hosts) from e
    except _getattr("elasticsearch", "TransportError") as e:
        logger.error(
            "Unexpected error occurred when sending requests to Elasticsearch: %s "
            "Status code: %s details: %s",
            e.error,
            e.status_code,
            e.info,
        )
        raise KnowledgeBaseError from e
    except _getattr("elasticsearch", "ElasticsearchException") as e:
        raise KnowledgeBaseError from e
