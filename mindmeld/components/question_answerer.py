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

"""
This module contains the question answerer component of MindMeld.
"""
import copy
import json
import logging
import re
from abc import ABC, abstractmethod

from elasticsearch import ConnectionError as EsConnectionError
from elasticsearch import ElasticsearchException, TransportError

from ..exceptions import (
    KnowledgeBaseConnectionError,
    KnowledgeBaseError,
    ElasticsearchVersionError,
)
from ..resource_loader import ResourceLoader
from ._config import (
    DEFAULT_ES_QA_MAPPING,
    DEFAULT_RANKING_CONFIG,
    get_app_namespace,
    get_classifier_config,
)
from ._elasticsearch_helpers import (
    DOC_TYPE,
    create_es_client,
    delete_index,
    does_index_exist,
    get_scoped_index_name,
    load_index,
    create_index_mapping,
    is_es_version_7,
    resolve_es_config_for_version,
)
from ..models import create_embedder_model

logger = logging.getLogger(__name__)


DEFAULT_QUERY_TYPE = "keyword"
ALL_QUERY_TYPES = ["keyword", "text", "embedder", "embedder_keyword", "embedder_text"]
EMBEDDING_FIELD_STRING = "_embedding"


class QuestionAnswerer:
    """The question answerer is primarily an information retrieval system that provides all the
    necessary functionality for interacting with the application's knowledge base.
    """

    def __init__(self, app_path, resource_loader=None, es_host=None, config=None):
        """Initializes a question answerer

        Args:
            app_path (str): The path to the directory containing the app's data
            resource_loader (ResourceLoader): An object which can load resources for the answerer
            es_host (str): The Elasticsearch host server
        """
        self._resource_loader = (
            resource_loader or ResourceLoader.create_resource_loader(app_path)
        )
        self._es_host = es_host
        self.__es_client = None
        self._app_namespace = get_app_namespace(app_path)
        self._es_field_info = {}
        if config:
            self._qa_config = config
        else:
            self._qa_config = get_classifier_config(
                "question_answering", app_path=app_path
            )

        self._embedder_model = None
        if self._qa_config.get("model_type") == "embedder":
            self._embedder_model = create_embedder_model(app_path, self._qa_config)

    @property
    def _es_client(self):
        # Lazily connect to Elasticsearch
        if self.__es_client is None:
            self.__es_client = create_es_client(self._es_host)
        return self.__es_client

    @property
    def _query_type(self):
        if self._qa_config.get("model_type") in ALL_QUERY_TYPES:
            return self._qa_config.get("model_type")
        else:
            return DEFAULT_QUERY_TYPE

    def get(self, index, size=10, query_type=None, **kwargs):
        """Gets a collection of documents from the knowledge base matching the provided
        search criteria. This API provides a simple interface for developers to specify a list of
        knowledge base field and query string pairs to find best matches in a similar way as in
        common Web search interfaces. The knowledge base fields to be used depend on the mapping
        between NLU entity types and corresponding knowledge base objects. For example, a “cuisine”
        entity type can be mapped to either a knowledge base object or an attribute of a knowledge
        base object. The mapping is often application specific and is dependent on the data model
        developers choose to use when building the knowledge base.

        Examples:

            >>> question_answerer.get(index='menu_items',
                                      name='pork and shrimp',
                                      restaurant_id='B01CGKGQ40',
                                      _sort='price',
                                      _sort_type='asc')

        Args:
            index (str): The name of an index.
            size (int): The maximum number of records, default to 10.
            query_type (str): Whether the search is over structured, unstructured and whether to use
                              text signals for ranking, embedder signals, or both.
            id (str): The id of a particular document to retrieve.
            _sort (str): Specify the knowledge base field for custom sort.
            _sort_type (str): Specify custom sort type. Valid values are 'asc', 'desc' and
                              'distance'.
            _sort_location (dict): The origin location to be used when sorting by distance.

        Returns:
            list: A list of matching documents.
        """
        doc_id = kwargs.get("id")

        query_type = query_type or self._query_type

        # If an id was passed in, simply retrieve the specified document
        if doc_id:
            logger.info(
                "Retrieve object from KB: index= '%s', id= '%s'.", index, doc_id
            )
            s = self.build_search(index)
            s = s.filter(query_type=query_type, id=doc_id)
            results = s.execute(size=size)
            return results

        sort_clause = {}
        query_clauses = []

        # iterate through keyword arguments to get KB field and value pairs for search and custom
        # sort criteria
        for key, value in kwargs.items():
            logger.debug("Processing argument: key= %s value= %s.", key, value)
            if key == "_sort":
                sort_clause["field"] = value
            elif key == "_sort_type":
                sort_clause["type"] = value
            elif key == "_sort_location":
                sort_clause["location"] = value
            elif "embedder" in query_type and self._embedder_model:
                if "text" in query_type or "keyword" in query_type:
                    query_clauses.append({key: value})
                embedded_value = self._embedder_model.get_encodings([value])[0]
                embedded_key = key + EMBEDDING_FIELD_STRING
                query_clauses.append({embedded_key: embedded_value})
            else:
                query_clauses.append({key: value})
                logger.debug("Added query clause: field= %s value= %s.", key, value)

        logger.debug("Custom sort criteria %s.", sort_clause)

        # build Search object with overriding ranking setting to require all query clauses are
        # matched.
        s = self.build_search(index, {"query_clauses_operator": "and"})

        # add query clauses to Search object.
        for clause in query_clauses:
            s = s.query(query_type=query_type, **clause)

        # add custom sort clause if specified.
        if sort_clause:
            s = s.sort(
                field=sort_clause.get("field"),
                sort_type=sort_clause.get("type"),
                location=sort_clause.get("location"),
            )

        results = s.execute(size=size)
        return results

    def build_search(self, index, ranking_config=None):
        """Build a search object for advanced filtered search.

        Args:
            index (str): index name of knowledge base object.
            ranking_config (dict): overriding ranking configuration parameters.
        Returns:
            Search: a Search object for filtered search.
        """

        if not does_index_exist(app_namespace=self._app_namespace, index_name=index):
            raise ValueError("Knowledge base index '{}' does not exist.".format(index))

        # get index name with app scope
        index = get_scoped_index_name(self._app_namespace, index)

        # load knowledge base field information for the specified index.
        self._load_field_info(index)

        return Search(
            client=self._es_client,
            index=index,
            ranking_config=ranking_config,
            field_info=self._es_field_info[index],
        )

    def _load_field_info(self, index):
        """load knowledge base field metadata information for the specified index.

        Args:
            index (str): index name.
        """

        # load field info from local cache
        index_info = self._es_field_info.get(index, {})

        if not index_info:
            try:
                # TODO: move the ES API call logic to ES helper
                self._es_field_info[index] = {}
                res = self._es_client.indices.get(index=index)
                if is_es_version_7(self._es_client):
                    all_field_info = res[index]["mappings"]["properties"]
                else:
                    all_field_info = res[index]["mappings"][DOC_TYPE]["properties"]
                for field_name in all_field_info:
                    field_type = all_field_info[field_name].get("type")
                    self._es_field_info[index][field_name] = FieldInfo(
                        field_name, field_type
                    )
            except EsConnectionError as e:
                logger.error(
                    "Unable to connect to Elasticsearch: %s details: %s",
                    e.error,
                    e.info,
                )
                raise KnowledgeBaseConnectionError(
                    es_host=self._es_client.transport.hosts
                ) from e
            except TransportError as e:
                logger.error(
                    "Unexpected error occurred when sending requests to Elasticsearch: %s "
                    "Status code: %s details: %s",
                    e.error,
                    e.status_code,
                    e.info,
                )
                raise KnowledgeBaseError from e
            except ElasticsearchException as e:
                raise KnowledgeBaseError from e

    def config(self, config):
        """Summary

        Args:
            config: Description
        """
        raise NotImplementedError

    def save_embedder_model(self):
        self._embedder_model.dump()

    @classmethod
    def load_kb(
        cls,
        app_namespace,
        index_name,
        data_file,
        es_host=None,
        es_client=None,
        connect_timeout=2,
        clean=False,
        app_path=None,
        config=None,
    ):
        """Loads documents from disk into the specified index in the knowledge
        base. If an index with the specified name doesn't exist, a new index
        with that name will be created in the knowledge base.

        Args:
            app_namespace (str): The namespace of the app. Used to prevent
                collisions between the indices of this app and those of other
                apps.
            index_name (str): The name of the new index to be created.
            data_file (str): The path to the data file containing the documents
                to be imported into the knowledge base index. It could be
                either json or jsonl file.
            es_host (str): The Elasticsearch host server.
            es_client (Elasticsearch): The Elasticsearch client.
            connect_timeout (int, optional): The amount of time for a
                connection to the Elasticsearch host.
            clean (bool): Set to true if you want to delete an existing index
                and reindex it
            app_path (str): The path to the directory containing the app's data
            config (dict): The QA config if passed directly rather than loaded from the app config
        """
        embedder_model = None
        embedding_fields = []
        if not app_path and not config:
            logger.warning(
                "You must provide either the application path to upload embeddings as specified"
                " in the app config or directly provide the QA config."
            )
        else:
            if config:
                qa_config = config
            else:
                qa_config = get_classifier_config(
                    "question_answering", app_path=app_path
                )
            embedder_model = create_embedder_model(app_path, qa_config)
            embedding_fields = (
                qa_config.get("model_settings", {})
                .get("embedding_fields", {})
                .get(index_name, [])
            )

        def _doc_data(data_file):
            with open(data_file) as data_fp:
                line = data_fp.readline()
                data_fp.seek(0)
                if line.strip() == "[":
                    docs = json.load(data_fp)
                    count = len(docs)
                else:
                    count = 0
                    for line in data_fp:
                        count += 1
                return count

        def _doc_generator(data_file, embedder_model=None, embedding_fields=None):
            def match_regex(string, pattern_list):
                for pattern in pattern_list:
                    if re.match(pattern, string):
                        return True

            def transform(doc, embedder_model, embedding_fields):
                if embedder_model:
                    embed_fields = [
                        (key, str(val))
                        for key, val in doc.items()
                        if match_regex(key, embedding_fields)
                    ]
                    embed_keys = list(zip(*embed_fields))[0]
                    embed_vals = embedder_model.get_encodings(
                        list(zip(*embed_fields))[1]
                    )
                    embedded_doc = {
                        key + EMBEDDING_FIELD_STRING: emb.tolist()
                        for key, emb in zip(embed_keys, embed_vals)
                    }
                    doc.update(embedded_doc)
                if not doc.get("id"):
                    return doc
                base = {"_id": doc["id"]}
                base.update(doc)
                return base

            with open(data_file) as data_fp:
                line = data_fp.readline()
                data_fp.seek(0)
                if line.strip() == "[":
                    logging.debug("Loading data from a json file.")
                    docs = json.load(data_fp)
                    for doc in docs:
                        yield transform(doc, embedder_model, embedding_fields)
                else:
                    logging.debug("Loading data from a jsonl file.")
                    for line in data_fp:
                        doc = json.loads(line)
                        yield transform(doc, embedder_model, embedding_fields)

        docs_count = _doc_data(data_file)
        if embedder_model and len(embedding_fields) == 0:
            logger.warning(
                "No embedding fields specified in the app config, "
                "continuing without generating embeddings..."
            )
            embedder_model = None
        docs = _doc_generator(data_file, embedder_model, embedding_fields)

        if clean:
            try:
                delete_index(app_namespace, index_name, es_host, es_client)
            except ValueError:
                logger.warning(
                    "Index %s does not exist for app %s, creating a new index",
                    index_name,
                    app_namespace,
                )

        def _generate_mapping_data(embedder_model, embedding_fields):
            # generates a dictionary with any metadata needed to create the mapping"
            if not embedder_model:
                return {}
            MAX_ES_VECTOR_LEN = 2048
            embedding_properties = []
            mapping_data = {"embedding_properties": embedding_properties}

            dims = len(embedder_model.get_encodings(["encoding"])[0])
            if dims > MAX_ES_VECTOR_LEN:
                logger.error(
                    "Vectors in ElasticSearch must be less than size: %d",
                    MAX_ES_VECTOR_LEN,
                )
            for field in embedding_fields:
                embedding_properties.append(
                    {"field": field + EMBEDDING_FIELD_STRING, "dims": dims}
                )

            return mapping_data

        es_client = es_client or create_es_client(es_host)
        if is_es_version_7(es_client):
            mapping_data = _generate_mapping_data(embedder_model, embedding_fields)
            qa_mapping = create_index_mapping(DEFAULT_ES_QA_MAPPING, mapping_data)
        else:
            if embedder_model:
                logger.error(
                    "You must upgrade to ElasticSearch 7 to use the embedding features."
                )
                raise ElasticsearchVersionError
            qa_mapping = resolve_es_config_for_version(DEFAULT_ES_QA_MAPPING, es_client)

        load_index(
            app_namespace,
            index_name,
            docs,
            docs_count,
            qa_mapping,
            DOC_TYPE,
            es_host,
            es_client,
            connect_timeout=connect_timeout,
        )

        # Saves the embedder model cache to disk
        if embedder_model:
            embedder_model.dump()


class FieldInfo:
    """This class models an information source of a knowledge base field metadata"""

    NUMBER_TYPES = {
        "long",
        "integer",
        "short",
        "byte",
        "double",
        "float",
        "half_float",
        "scaled_float",
    }
    TEXT_TYPES = {"text", "keyword"}
    DATE_TYPES = {"date"}
    GEO_TYPES = {"geo_point"}
    VECTOR_TYPES = {"dense_vector"}

    def __init__(self, name, field_type):
        self.name = name
        self.type = field_type

    def get_name(self):
        """Returns knowledge base field name"""

        return self.name

    def get_type(self):
        """Returns knowledge base field type"""

        return self.type

    def is_number_field(self):
        """Returns True if the knowledge base field is a number field, otherwise returns False"""

        return self.type in self.NUMBER_TYPES

    def is_date_field(self):
        """Returns True if the knowledge base field is a date field, otherwise returns False"""

        return self.type in self.DATE_TYPES

    def is_location_field(self):
        """Returns True if the knowledge base field is a location field, otherwise returns False"""

        return self.type in self.GEO_TYPES

    def is_text_field(self):
        """Returns True if the knowledge base field is a text field, otherwise returns False"""

        return self.type in self.TEXT_TYPES

    def is_vector_field(self):
        """Returns True if the knowledge base field is a vector field, otherwise returns False"""

        return self.type in self.VECTOR_TYPES


class Search:
    """This class models a generic filtered search in knowledge base. It allows developers to
    construct more complex knowledge base search criteria based on the application requirements.

    """

    SYN_FIELD_SUFFIX = "$whitelist"

    def __init__(self, client, index, ranking_config=None, field_info=None):
        """Initialize a Search object.

        Args:
            client (Elasticsearch): Elasticsearch client.
            index (str): index name of knowledge base object.
            ranking_config (dict): overriding ranking configuration parameters for current search.
            field_info (dict): dictionary contains knowledge base matadata objects.
        """
        self.index = index
        self.client = client

        self._clauses = {"query": [], "filter": [], "sort": []}

        self._ranking_config = ranking_config
        if not ranking_config:
            self._ranking_config = copy.deepcopy(DEFAULT_RANKING_CONFIG)

        self._kb_field_info = field_info

    def _clone(self):
        """Clone a Search object.

        Returns:
            Search: cloned copy of the Search object.
        """
        s = Search(client=self.client, index=self.index)
        s._clauses = copy.deepcopy(self._clauses)
        s._ranking_config = copy.deepcopy(self._ranking_config)
        s._kb_field_info = copy.deepcopy(self._kb_field_info)

        return s

    def _build_query_clause(self, query_type=DEFAULT_QUERY_TYPE, **kwargs):
        field, value = next(iter(kwargs.items()))
        field_info = self._kb_field_info.get(field)
        if not field_info:
            raise ValueError("Invalid knowledge base field '{}'".format(field))

        # check whether the synonym field is available. By default the synonyms are
        # imported to "<field_name>$whitelist" field.
        synonym_field = (
            field + self.SYN_FIELD_SUFFIX
            if self._kb_field_info.get(field + self.SYN_FIELD_SUFFIX)
            else None
        )
        clause = Search.QueryClause(field, field_info, value, query_type, synonym_field)
        clause.validate()
        self._clauses[clause.get_type()].append(clause)

    def _build_filter_clause(self, query_type=DEFAULT_QUERY_TYPE, **kwargs):
        # set the filter type to be 'range' if any range operator is specified.
        if (
            kwargs.get("gt")
            or kwargs.get("gte")
            or kwargs.get("lt")
            or kwargs.get("lte")
        ):
            field = kwargs.get("field")
            gt = kwargs.get("gt")
            gte = kwargs.get("gte")
            lt = kwargs.get("lt")
            lte = kwargs.get("lte")

            if field not in self._kb_field_info:
                raise ValueError("Invalid knowledge base field '{}'".format(field))

            clause = Search.FilterClause(
                field=field,
                field_info=self._kb_field_info.get(field),
                range_gt=gt,
                range_gte=gte,
                range_lt=lt,
                range_lte=lte,
            )
        else:
            key, value = next(iter(kwargs.items()))
            if key not in self._kb_field_info:
                raise ValueError("Invalid knowledge base field '{}'".format(key))
            clause = Search.FilterClause(field=key, value=value, query_type=query_type)
        clause.validate()
        self._clauses[clause.get_type()].append(clause)

    def _build_sort_clause(self, **kwargs):
        sort_field = kwargs.get("field")
        sort_type = kwargs.get("sort_type")
        sort_location = kwargs.get("location")

        field_info = self._kb_field_info.get(sort_field)
        if not field_info:
            raise ValueError("Invalid knowledge base field '{}'".format(sort_field))

        # only compute field stats if sort field is number or date type.
        field_stats = None
        if field_info.is_number_field() or field_info.is_date_field():
            field_stats = self._get_field_stats(sort_field)

        clause = Search.SortClause(
            sort_field, field_info, sort_type, field_stats, sort_location
        )
        clause.validate()
        self._clauses[clause.get_type()].append(clause)

    def _build_clause(self, clause_type, query_type=DEFAULT_QUERY_TYPE, **kwargs):
        """Helper method to build query, filter and sort clauses.

        Args:
            clause_type (str): type of clause
        """
        if clause_type == "query":
            self._build_query_clause(query_type, **kwargs)
        elif clause_type == "filter":
            self._build_filter_clause(query_type, **kwargs)
        elif clause_type == "sort":
            self._build_sort_clause(**kwargs)
        else:
            raise Exception("Unknown clause type.")

    def query(self, query_type=DEFAULT_QUERY_TYPE, **kwargs):
        """Specify the query text to match on a knowledge base text field. The query text is
        normalized and processed (based on query_type) to find matches in knowledge base using
        several text relevance scoring factors including exact matches, phrase matches and partial
        matches.

        Examples:

            >>> s = question_answerer.build_search(index='dish')
            >>> s.query(name='pad thai')

        In the example above the query text "pad thai" will be used to match against document field
        "name" in knowledge base index "dish".

        Args:
            a keyword argument to specify the query text and the knowledge base document field along
            with the query type (keyword/text/embedder/embedder_keyword/embedder_text).
        Returns:
            Search: a new Search object with added search criteria.
        """
        new_search = self._clone()
        new_search._build_clause("query", query_type, **kwargs)

        return new_search

    def filter(self, query_type=DEFAULT_QUERY_TYPE, **kwargs):
        """Specify filter condition to be applied to specified knowledge base field. In MindMeld
        two types of filters are supported: text filter and range filters.

        Text filters are used to apply hard filters on specified knowledge base text fields.
        The filter text value is normalized and matched using entire text span against the
        knowledge base field.

        It's common to have filter conditions based on other resolved canonical entities.
        For example, in food ordering domain the resolved restaurant entity can be used as a filter
        to resolve dish entities. The exact knowledge base field to apply these filters depends on
        the knowledge base data model of the application.
        If the entity is not in the canonical form, a fuzzy filter can be applied by setting the
        query_type to 'text'.

        Range filters are used to filter with a value range on specified knowledge base number or
        date fields. Example use cases include price range filters and date range filters.

        Examples:

        add text filter:
            >>> s = question_answerer.build_search(index='menu_items')
            >>> s.filter(restaurant_id='B01CGKGQ40')

        add range filter:
                >>> s = question_answerer.build_search(index='menu_items')
                >>> s.filter(field='price', gte=1, lt=10)

        Args:
            query_type (str): Whether the filter is over structured or unstructured text.
            kwargs: A keyword argument to specify the filter text and the knowledge base text field.
            field (str): knowledge base field name for range filter.
            gt (number or str): range filter operator for greater than.
            gte (number or str): range filter operator for greater than or equal to.
            lt (number or str): range filter operator for less than.
            lte (number or str): range filter operator for less or equal to.

        Returns:
            Search: A new Search object with added search criteria.
        """
        new_search = self._clone()
        new_search._build_clause("filter", query_type, **kwargs)

        return new_search

    def sort(self, field, sort_type=None, location=None):
        """Specify custom sort criteria.

        Args:
            field (str): knowledge base field for sort.
            sort_type (str): sorting type. valid values are 'asc', 'desc' and 'distance'. 'asc' and
                             'desc' can be used to sort numeric or date fields and 'distance' can
                             be used to sort by distance on geo_point fields. Default sort type
                             is 'desc' if not specified.
            location (str): location (lat, lon) in geo_point format to be used as origin when
                            sorting by 'distance'
        """
        new_search = self._clone()
        new_search._build_clause(
            "sort", field=field, sort_type=sort_type, location=location
        )
        return new_search

    def _get_field_stats(self, field):
        """Get knowledge field statistics for custom sort functions. The field statistics is
        only available for number and date typed fields.

        Args:
            field(str): knowledge base field name

        Returns:
            dict: dictionary that contains knowledge base field statistics.
        """

        stats_query = {"aggs": {}, "size": 0}
        stats_query["aggs"][field + "_min"] = {"min": {"field": field}}
        stats_query["aggs"][field + "_max"] = {"max": {"field": field}}

        res = self.client.search(
            index=self.index, body=stats_query, search_type="query_then_fetch"
        )

        return {
            "min_value": res["aggregations"][field + "_min"]["value"],
            "max_value": res["aggregations"][field + "_max"]["value"],
        }

    def _build_es_query(self, size=10):
        """Build knowledge base search syntax based on provided search criteria.

        Args:
            size (int): The maximum number of records to fetch, default to 10.

        Returns:
            str: knowledge base search syntax for the current search object.
        """
        es_query = {
            "query": {
                "function_score": {
                    "query": {},
                    "functions": [],
                    "score_mode": "sum",
                    "boost_mode": "sum",
                }
            },
            "_source": {"excludes": ["*" + self.SYN_FIELD_SUFFIX]},
            "size": size,
        }

        if not self._clauses["query"] and not self._clauses["filter"]:
            # no query/filter clauses - use match_all
            es_query["query"]["function_score"]["query"] = {"match_all": {}}
        else:
            es_query["query"]["function_score"]["query"]["bool"] = {}

            if self._clauses["query"]:
                es_query_clauses = []
                es_boost_functions = []
                for clause in self._clauses["query"]:
                    query_clause, boost_functions = clause.build_query()
                    if query_clause:
                        es_query_clauses.append(query_clause)
                    es_boost_functions.extend(boost_functions)

                if self._ranking_config["query_clauses_operator"] == "and":
                    es_query["query"]["function_score"]["query"]["bool"][
                        "must"
                    ] = es_query_clauses
                else:
                    es_query["query"]["function_score"]["query"]["bool"][
                        "should"
                    ] = es_query_clauses

                # add all boost functions for the query clause
                # right now the only boost functions supported are exact match boosting for
                # CNAME and synonym whitelists.
                es_query["query"]["function_score"]["functions"].extend(
                    es_boost_functions
                )

            if self._clauses["filter"]:
                es_filter_clauses = {"bool": {"must": []}}
                for clause in self._clauses["filter"]:
                    es_filter_clauses["bool"]["must"].append(clause.build_query())

                es_query["query"]["function_score"]["query"]["bool"][
                    "filter"
                ] = es_filter_clauses

        # add scoring function for custom sort criteria
        for clause in self._clauses["sort"]:
            sort_function = clause.build_query()
            es_query["query"]["function_score"]["functions"].append(sort_function)

        logger.debug("ES query syntax: %s.", es_query)

        return es_query

    def execute(self, size=10):
        """Executes the knowledge base search with provided criteria and returns matching documents.

        Args:
            size (int): The maximum number of records to fetch, default to 10.

        Returns:
            a list of matching documents.
        """
        try:
            # TODO: move the ES API call logic to ES helper
            es_query = self._build_es_query(size=size)

            response = self.client.search(index=self.index, body=es_query)

            # construct results, removing embedding metadata and exposing score
            results = []
            for hit in response["hits"]["hits"]:
                item = {key: val for (key, val) in hit["_source"].items()
                        if not key.endswith(EMBEDDING_FIELD_STRING)}
                item['_score'] = hit['_score']
                results.append(item)
            return results
        except EsConnectionError as e:
            logger.error(
                "Unable to connect to Elasticsearch: %s details: %s", e.error, e.info
            )
            raise KnowledgeBaseConnectionError(es_host=self.client.transport.hosts) from e
        except TransportError as e:
            logger.error(
                "Unexpected error occurred when sending requests to Elasticsearch: %s "
                "Status code: %s details: %s",
                e.error,
                e.status_code,
                e.info,
            )
            raise KnowledgeBaseError from e
        except ElasticsearchException as e:
            raise KnowledgeBaseError from e

    class Clause(ABC):
        """This class models an abstract knowledge base clause."""

        def __init__(self):
            """Initialize a knowledge base clause"""
            self.clause_type = None

        @abstractmethod
        def validate(self):
            """Validate the clause."""
            raise NotImplementedError("Must override validate()")

        @abstractmethod
        def build_query(self):
            """Build knowledge base query."""
            raise NotImplementedError("Must override build_query()")

        def get_type(self):
            """Returns clause type"""
            return self.clause_type

    class QueryClause(Clause):
        """This class models a knowledge base query clause."""

        DEFAULT_EXACT_MATCH_BOOSTING_WEIGHT = 100

        def __init__(
            self,
            field,
            field_info,
            value,
            query_type=DEFAULT_QUERY_TYPE,
            synonym_field=None,
        ):
            """Initialize a knowledge base query clause."""
            self.field = field
            self.field_info = field_info
            self.value = value
            self.query_type = query_type
            self.syn_field = synonym_field

            self.clause_type = "query"

        def build_query(self):
            """build knowledge base query for query clause"""

            # ES syntax is generated based on specified knowledge base field
            # the following ranking factors are considered:
            # 1. exact matches (with boosted weight)
            # 2. word N-gram matches
            # 3. character N-gram matches
            # 4. matches on synonym if available (exact, word N-gram and character N-gram):
            # for a knowledge base text field the synonym are indexed in a separate field
            # "<field name>$whitelist" if available.
            functions = []

            if "embedder" in self.query_type and self.field_info.is_vector_field():
                clause = None
                functions = [
                    {
                        "script_score": {
                            "script": {
                                "source": "cosineSimilarity(params.field_embedding,"
                                " doc[params.matching_field]) + 1.0",
                                "params": {
                                    "field_embedding": self.value.tolist(),
                                    "matching_field": self.field,
                                },
                            }
                        },
                        "weight": 10,
                    }
                ]
            elif "text" in self.query_type:
                clause = {
                    "bool": {
                        "should": [
                            {"match": {self.field: {"query": self.value}}},
                            {
                                "match": {
                                    self.field
                                    + ".processed_text": {"query": self.value}
                                }
                            },
                        ]
                    }
                }
            elif "keyword" in self.query_type:
                clause = {
                    "bool": {
                        "should": [
                            {"match": {self.field: {"query": self.value}}},
                            {
                                "match": {
                                    self.field
                                    + ".normalized_keyword": {"query": self.value}
                                }
                            },
                            {
                                "match": {
                                    self.field + ".char_ngram": {"query": self.value}
                                }
                            },
                        ]
                    }
                }
            else:
                raise Exception("Unknown query type.")

            if self.field_info.is_text_field():
                # Boost function for boosting conditions, e.g. exact match boosting
                functions = [
                    {
                        "filter": {
                            "match": {self.field + ".normalized_keyword": self.value}
                        },
                        "weight": self.DEFAULT_EXACT_MATCH_BOOSTING_WEIGHT,
                    }
                ]

            # generate ES syntax for matching on synonym whitelist if available.
            if self.syn_field:
                clause["bool"]["should"].append(
                    {
                        "nested": {
                            "path": self.syn_field,
                            "score_mode": "max",
                            "query": {
                                "bool": {
                                    "should": [
                                        {
                                            "match": {
                                                self.syn_field
                                                + ".name.normalized_keyword": {
                                                    "query": self.value
                                                }
                                            }
                                        },
                                        {
                                            "match": {
                                                self.syn_field
                                                + ".name": {"query": self.value}
                                            }
                                        },
                                        {
                                            "match": {
                                                self.syn_field
                                                + ".name.char_ngram": {
                                                    "query": self.value
                                                }
                                            }
                                        },
                                    ]
                                }
                            },
                            "inner_hits": {},
                        }
                    }
                )

                functions.append(
                    {
                        "filter": {
                            "nested": {
                                "path": self.syn_field,
                                "query": {
                                    "match": {
                                        self.syn_field
                                        + ".name.normalized_keyword": self.value
                                    }
                                },
                            }
                        },
                        "weight": self.DEFAULT_EXACT_MATCH_BOOSTING_WEIGHT,
                    }
                )

            return clause, functions

        def validate(self):
            if (
                not self.field_info.is_text_field()
                and not self.field_info.is_vector_field()
            ):
                raise ValueError(
                    "Query can only be defined on text and vector fields. If it is,"
                    " try running load_kb with clean=True and reinitializing your"
                    " QuestionAnswerer object."
                )

    class FilterClause(Clause):
        """This class models a knowledge base filter clause."""

        def __init__(
            self,
            field,
            field_info=None,
            value=None,
            query_type=DEFAULT_QUERY_TYPE,
            range_gt=None,
            range_gte=None,
            range_lt=None,
            range_lte=None,
        ):
            """Initialize a knowledge base filter clause. The filter type is determined by whether
            the range operators or value is passed in.
            """

            self.field = field
            self.field_info = field_info
            self.value = value
            self.query_type = query_type
            self.range_gt = range_gt
            self.range_gte = range_gte
            self.range_lt = range_lt
            self.range_lte = range_lte

            if self.value:
                self.filter_type = "text"
            else:
                self.filter_type = "range"

            self.clause_type = "filter"

        def build_query(self):
            """build knowledge base query for filter clause"""
            clause = {}
            if self.filter_type == "text":
                if self.field == "id":
                    clause = {"term": {"id": self.value}}
                else:
                    if self.query_type == "text":
                        clause = {
                            "match": {self.field + ".char_ngram": {"query": self.value}}
                        }
                    else:
                        clause = {
                            "match": {
                                self.field
                                + ".normalized_keyword": {"query": self.value}
                            }
                        }
            elif self.filter_type == "range":
                lower_bound = None
                upper_bound = None
                if self.range_gt:
                    lower_bound = ("gt", self.range_gt)
                elif self.range_gte:
                    lower_bound = ("gte", self.range_gte)

                if self.range_lt:
                    upper_bound = ("lt", self.range_lt)
                elif self.range_lte:
                    upper_bound = ("lte", self.range_lte)

                clause = {"range": {self.field: {}}}

                if lower_bound:
                    clause["range"][self.field][lower_bound[0]] = lower_bound[1]

                if upper_bound:
                    clause["range"][self.field][upper_bound[0]] = upper_bound[1]
            else:
                raise Exception("Unknown filter type.")

            return clause

        def validate(self):
            if self.filter_type == "range":
                if (
                    not self.range_gt
                    and not self.range_gte
                    and not self.range_lt
                    and not self.range_lte
                ):
                    raise ValueError("No range parameter is specified")
                elif self.range_gte and self.range_gt:
                    raise ValueError(
                        "Invalid range parameters. Cannot specify both 'gte' and 'gt'."
                    )
                elif self.range_lte and self.range_lt:
                    raise ValueError(
                        "Invalid range parameters. Cannot specify both 'lte' and 'lt'."
                    )
                elif (
                    not self.field_info.is_number_field()
                    and not self.field_info.is_date_field()
                ):
                    raise ValueError(
                        "Range filter can only be defined for number or date field."
                    )

    class SortClause(Clause):
        """This class models a knowledge base sort clause."""

        SORT_ORDER_ASC = "asc"
        SORT_ORDER_DESC = "desc"
        SORT_DISTANCE = "distance"
        SORT_TYPES = {SORT_ORDER_ASC, SORT_ORDER_DESC, SORT_DISTANCE}

        # default weight for adjusting sort scores so that they will be on the same scale when
        # combined with text relevance scores.
        DEFAULT_SORT_WEIGHT = 30

        def __init__(
            self,
            field,
            field_info=None,
            sort_type=None,
            field_stats=None,
            location=None,
        ):
            """Initialize a knowledge base sort clause"""
            self.field = field
            self.location = location
            self.sort_type = sort_type if sort_type else self.SORT_ORDER_DESC
            self.field_stats = field_stats
            self.field_info = field_info

            self.clause_type = "sort"

        def build_query(self):
            """build knowledge base query for sort clause"""

            # sort by distance based on passed in origin
            if self.sort_type == "distance":
                origin = self.location
                scale = "5km"
            else:
                max_value = self.field_stats["max_value"]
                min_value = self.field_stats["min_value"]

                if self.field_info.is_date_field():
                    # ensure the timestamps for date fields are integer values
                    max_value = int(max_value)
                    min_value = int(min_value)

                    # add time unit for date field
                    scale = (
                        "{}ms".format(int(0.5 * (max_value - min_value)))
                        if max_value != min_value
                        else 1
                    )
                else:
                    scale = (
                        0.5 * (max_value - min_value) if max_value != min_value else 1
                    )

                if self.sort_type == "asc":
                    origin = min_value
                else:
                    origin = max_value

            sort_clause = {
                "linear": {self.field: {"origin": origin, "scale": scale}},
                "weight": self.DEFAULT_SORT_WEIGHT,
            }

            return sort_clause

        def validate(self):
            # validate the sort type to be valid.
            if self.sort_type not in self.SORT_TYPES:
                raise ValueError(
                    "Invalid value for sort type '{}'".format(self.sort_type)
                )

            if self.field == "location" and self.sort_type != self.SORT_DISTANCE:
                raise ValueError(
                    "Invalid value for sort type '{}'".format(self.sort_type)
                )

            if self.field == "location" and not self.location:
                raise ValueError(
                    "No origin location specified for sorting by distance."
                )

            if self.sort_type == self.SORT_DISTANCE and self.field != "location":
                raise ValueError(
                    "Sort by distance is only supported using 'location' field."
                )

            # validate the sort field is number, date or location field
            if not (
                self.field_info.is_number_field()
                or self.field_info.is_date_field()
                or self.field_info.is_location_field()
            ):
                raise ValueError(
                    "Custom sort criteria can only be defined for"
                    + " 'number', 'date' or 'location' fields."
                )
