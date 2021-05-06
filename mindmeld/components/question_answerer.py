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
import numbers
import os
import pickle
import re
from abc import ABC, abstractmethod

from elasticsearch import ConnectionError as EsConnectionError
from elasticsearch import ElasticsearchException, TransportError

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
from .entity_resolver import EmbedderCosSimEntityResolver
from .entity_resolver import TfIdfSparseCosSimEntityResolver
from ..core import Bunch
from ..exceptions import (
    KnowledgeBaseConnectionError,
    KnowledgeBaseError,
    ElasticsearchVersionError,
)
from ..models import create_embedder_model
from ..path import get_question_answerer_index_cache_file_path
from ..resource_loader import Hasher
from ..resource_loader import ResourceLoader

logger = logging.getLogger(__name__)

DEFAULT_QUERY_TYPE = "keyword"
ALL_QUERY_TYPES = ["keyword", "text", "embedder", "embedder_keyword", "embedder_text"]
EMBEDDING_FIELD_STRING = "_embedding"
DEFAULT_INDICES_STORAGE_PATH = os.path.join(os.path.expanduser("~"), ".cache/mindmeld")


class BaseQuestionAnswerer(ABC):

    def __init__(self, app_path, **kwargs):
        """
        Args:
            app_path (str, optional): The path to the directory containing the app's data. If
                provided, used to obtain default `app_namespace` and QA configurations
            app_namespace (str, optional): The namespace of the app. Used to prevent
                collisions between the indices of this app and those of other apps.

            config (dict, optional): The QA config if passed directly rather than loaded from the
                app config
            resource_loader (ResourceLoader, optional): An object which can load resources for the
                question answerer.
        """
        self.app_path = app_path
        self.app_namespace = kwargs.get("app_namespace", None)
        self._resource_loader = kwargs.get("resource_loader", None)
        self.__qa_config = kwargs.get("config", None)

        if not self.app_namespace:
            self.app_namespace = get_app_namespace(self.app_path) if self.app_path else None
        if not self._resource_loader:
            self._resource_loader = ResourceLoader.create_resource_loader(self.app_path)
        if not self.__qa_config:
            self.__qa_config = get_classifier_config("question_answering", app_path=app_path)

    def __repr__(self):
        return f"<{self.__class__.__name__} model_type: {self._query_type}>"

    @property
    def _query_type(self) -> str:
        if self.__qa_config.get("model_type") in ALL_QUERY_TYPES:
            return self.__qa_config.get("model_type")
        else:
            return DEFAULT_QUERY_TYPE

    @_query_type.setter
    def _query_type(self, query_type):
        if query_type not in ALL_QUERY_TYPES:
            msg = f"Cannot set query_type to a vlaue outside {ALL_QUERY_TYPES}. Found {query_type}"
            logger.error(msg)
            return
        self.__qa_config.update({"model_type": query_type})

    @property
    def _query_settings(self) -> dict:
        return {"model_settings": self.__qa_config.get("model_settings", {})}

    @abstractmethod
    def get(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def build_search(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def load_kb(self, index_name, data_file, **kwargs):
        raise NotImplementedError


class ElasticsearchQuestionAnswerer(BaseQuestionAnswerer):
    """The question answerer is primarily an information retrieval system that provides all the
    necessary functionality for interacting with the application's knowledge base.
    """

    def __init__(self, app_path, **kwargs):
        """Initializes a question answerer

        Args:
            app_path (str): The path to the directory containing the app's data
            resource_loader (ResourceLoader): An object which can load resources for the answerer
            es_host (str): The Elasticsearch host server
            config (dict): The QA config if passed directly rather than loaded from the app config
        """
        super().__init__(app_path, **kwargs)
        self._es_host = kwargs.get("es_host", None)
        self.__es_client = None
        self._es_field_info = {}

        # bug-fix: previously, `_embedder_model` is created only when `model_type` is `embedder`
        self._embedder_model = None
        if "embedder" in self._query_type:
            self._embedder_model = create_embedder_model(self.app_path, self._query_settings)

    @property
    def _es_client(self):
        # Lazily connect to Elasticsearch
        if self.__es_client is None:
            self.__es_client = create_es_client(self._es_host)
        return self.__es_client

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
                    self._es_field_info[index][field_name] = (
                        ElasticsearchQuestionAnswerer.FieldInfo(field_name, field_type)
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

    def save_embedder_model(self):
        self._embedder_model.dump()

    def get(self, index, size=10, query_type=None, app_namespace=None, **kwargs):
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

        # fix related to Issue 219: https://github.com/cisco/mindmeld/issues/219
        app_namespace = app_namespace or self.app_namespace

        # If an id was passed in, simply retrieve the specified document
        if doc_id:
            logger.info(
                "Retrieve object from KB: index= '%s', id= '%s'.", index, doc_id
            )
            s = self.build_search(index, app_namespace=app_namespace)
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

    def build_search(self, index, ranking_config=None, app_namespace=None):
        """Build a search object for advanced filtered search.

        Args:
            index (str): index name of knowledge base object.
            ranking_config (dict): overriding ranking configuration parameters.
        Returns:
            Search: a Search object for filtered search.
        """

        # fix related to Issue 219: https://github.com/cisco/mindmeld/issues/219
        app_namespace = app_namespace or self.app_namespace

        if not does_index_exist(app_namespace=app_namespace, index_name=index):
            raise ValueError("Knowledge base index '{}' does not exist.".format(index))

        # get index name with app scope
        index = get_scoped_index_name(app_namespace, index)

        # load knowledge base field information for the specified index.
        self._load_field_info(index)

        return ElasticsearchQuestionAnswerer.Search(
            client=self._es_client,
            index=index,
            ranking_config=ranking_config,
            field_info=self._es_field_info[index],
        )

    def load_kb(
        self,
        index_name,
        data_file,
        app_namespace=None,
        es_host=None,
        es_client=None,
        connect_timeout=2,
        clean=False,
        app_path=None,
        config=None,
        **kwargs
    ):
        """Loads documents from disk into the specified index in the knowledge
        base. If an index with the specified name doesn't exist, a new index
        with that name will be created in the knowledge base.

        Args:
            index_name (str): The name of the new index to be created.
            data_file (str): The path to the data file containing the documents
                to be imported into the knowledge base index. It could be
                either json or jsonl file.
            app_namespace (str, optional): The namespace of the app. Used to prevent
                collisions between the indices of this app and those of other apps.
            es_host (str, optional): The Elasticsearch host server.
            es_client (Elasticsearch, optional): The Elasticsearch client.
            connect_timeout (int, optional): The amount of time for a
                connection to the Elasticsearch host.
            clean (bool, optional): Set to true if you want to delete an existing index
                and reindex it
            app_path (str, optional): The path to the directory containing the app's data
            config (dict, optional): The QA config if passed directly rather than loaded from the
            app config
        """

        # fix related to Issue 219: https://github.com/cisco/mindmeld/issues/219
        app_namespace = app_namespace or self.app_namespace

        es_host = es_host or self._es_host
        es_client = es_client or self._es_client

        # clean by deleting
        if clean:
            try:
                delete_index(app_namespace, index_name, es_host, es_client)
            except ValueError:
                logger.warning(
                    "Index %s does not exist for app %s, creating a new index",
                    index_name,
                    app_namespace,
                )

        # determine config: precedence is first given to argument `config`,
        #   then argument `app_path`, and then fallback option is self._query_settings
        if not app_path and not config:
            logger.warning(
                "You must provide either the application path to upload embeddings as specified"
                " in the app config or directly provide the QA config."
            )
            config = self._query_settings
        elif not config:
            config = get_classifier_config("question_answering", app_path=app_path)

        # determine embedding fields and load embedder model
        embedding_fields = (
            kwargs.get("embedding_fields", []) or
            config.get("model_settings", {}).get("embedding_fields", {}).get(index_name, [])
        )
        embedder_model = None
        if embedding_fields:
            if "embedder" not in self._query_type:
                msg = f"Found KB fields to upload embedding for ({embedding_fields}) fields in " \
                      f"index `{index_name}` but specified model_type `{self._query_type}` has no" \
                      f" `embedder` phrase in it. Ignoring provided `embedding_fields`."
                logger.error(msg)
                embedding_fields = []
            else:
                embedder_model = create_embedder_model(app_path, config)
        else:
            if "embedder" in self._query_type:
                logger.warning(
                    "No embedding fields specified in the app config, "
                    "continuing without generating embeddings..."
                )

        def _doc_data_count(data_file):
            with open(data_file) as data_fp:
                line = data_fp.readline()
                data_fp.seek(0)
                # fix related to Issue 220: https://github.com/cisco/mindmeld/issues/220
                if line.strip().startswith("["):
                    docs = json.load(data_fp)
                    count = len(docs)
                else:
                    count = 0
                    for line in data_fp:
                        count += 1
                return count

        def _doc_generator(data_file, embedder_model=None, embedding_fields=None):
            def match_regex(string, pattern_list):
                return any([re.match(pattern, string) for pattern in pattern_list])

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
                # fix related to Issue 220: https://github.com/cisco/mindmeld/issues/220
                if line.strip().startswith("["):
                    logging.debug("Loading data from a json file.")
                    docs = json.load(data_fp)
                    for doc in docs:
                        yield transform(doc, embedder_model, embedding_fields)
                else:
                    logging.debug("Loading data from a jsonl file.")
                    for line in data_fp:
                        doc = json.loads(line)
                        yield transform(doc, embedder_model, embedding_fields)

        docs_count = _doc_data_count(data_file)
        docs = _doc_generator(data_file, embedder_model, embedding_fields)

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
                ranking_config (dict): overriding ranking configuration parameters for current
                                        search.
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
            s = ElasticsearchQuestionAnswerer.Search(client=self.client, index=self.index)
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
            clause = ElasticsearchQuestionAnswerer.Search.QueryClause(
                field, field_info, value, query_type, synonym_field)
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

                clause = ElasticsearchQuestionAnswerer.Search.FilterClause(
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
                clause = ElasticsearchQuestionAnswerer.Search.FilterClause(
                    field=key, value=value, query_type=query_type)
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

            clause = ElasticsearchQuestionAnswerer.Search.SortClause(
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
            several text relevance scoring factors including exact matches, phrase matches and
            partial matches.

            Examples:

                >>> s = question_answerer.build_search(index='dish')
                >>> s.query(name='pad thai')

            In the example above the query text "pad thai" will be used to match against document
            field "name" in knowledge base index "dish".

            Args:
                a keyword argument to specify the query text and the knowledge base document field
                along with the query type (keyword/text/embedder/embedder_keyword/embedder_text).
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
            For example, in food ordering domain the resolved restaurant entity can be used as a
            filter to resolve dish entities. The exact knowledge base field to apply these filters
            depends on the knowledge base data model of the application.
            If the entity is not in the canonical form, a fuzzy filter can be applied by setting the
            query_type to 'text'.

            Range filters are used to filter with a value range on specified knowledge base number
            or date fields. Example use cases include price range filters and date range filters.

            Examples:

            add text filter:
                >>> s = question_answerer.build_search(index='menu_items')
                >>> s.filter(restaurant_id='B01CGKGQ40')

            add range filter:
                    >>> s = question_answerer.build_search(index='menu_items')
                    >>> s.filter(field='price', gte=1, lt=10)

            Args:
                query_type (str): Whether the filter is over structured or unstructured text.
                kwargs: A keyword argument to specify the filter text and the knowledge base text
                        field.
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
                sort_type (str): sorting type. valid values are 'asc', 'desc' and 'distance'.
                                 'asc' and 'desc' can be used to sort numeric or date fields and
                                 'distance' can be used to sort by distance on geo_point fields.
                                 Default sort type is 'desc' if not specified.
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
            """Executes the knowledge base search with provided criteria and returns matching
            documents.

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
                """Initialize a knowledge base filter clause. The filter type is determined by
                whether the range operators or value is passed in.
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
            """Returns True if the knowledge base field is a number field, otherwise returns False
            """

            return self.type in self.NUMBER_TYPES

        def is_date_field(self):
            """Returns True if the knowledge base field is a date field, otherwise returns False"""

            return self.type in self.DATE_TYPES

        def is_location_field(self):
            """Returns True if the knowledge base field is a location field, otherwise returns False
            """

            return self.type in self.GEO_TYPES

        def is_text_field(self):
            """Returns True if the knowledge base field is a text field, otherwise returns False"""

            return self.type in self.TEXT_TYPES

        def is_vector_field(self):
            """Returns True if the knowledge base field is a vector field, otherwise returns False
            """

            return self.type in self.VECTOR_TYPES


class NonElasticsearchQuestionAnswerer(BaseQuestionAnswerer):
    """
    A question answerer class not using Elasticsearch
    """

    def __init__(self, app_path, **kwargs):
        """Initializes a question answerer

        Args:
            app_path (str): The path to the directory containing the app's data
            resource_loader (ResourceLoader): An object which can load resources for the answerer
            config (dict): The QA config if passed directly rather than loaded from the app config
        """
        super().__init__(app_path, **kwargs)

        self._query_type = "embedder_keyword" if "embedder" in self._query_type else "keyword"

        self._embedder_model = None
        if "embedder" in self._query_type:
            self._embedder_model = create_embedder_model(self.app_path, self._query_settings)

    def save_embedder_model(self):
        self._embedder_model.dump()

    def get(self, index, size=10, query_type=None, app_namespace=None, **kwargs):

        doc_id = kwargs.get("id")

        query_type = query_type or self._query_type

        # fix related to Issue 219: https://github.com/cisco/mindmeld/issues/219
        app_namespace = app_namespace or self.app_namespace

        # If an id was passed in, simply retrieve the specified document
        if doc_id:
            logger.info(
                "Retrieve object from KB: index= '%s', id= '%s'.", index, doc_id
            )
            s = self.build_search(index, app_namespace=app_namespace)
            s = s.filter(id=doc_id)
            results = s.execute(size=size)
            return results

        s = self.build_search(index, app_namespace=app_namespace).query(**kwargs)
        results = s.execute(size=size)

        return results

    def build_search(self, index, ranking_config=None, app_namespace=None):

        # fix related to Issue 219: https://github.com/cisco/mindmeld/issues/219
        app_namespace = app_namespace or self.app_namespace

        # get index name with app scope
        scoped_index_name = get_scoped_index_name(app_namespace, index)

        if not NonElasticsearchQuestionAnswerer.ALL_INDICES.is_available(scoped_index_name):
            raise ValueError("Knowledge base index '{}' does not exist.".format(index))

        return NonElasticsearchQuestionAnswerer.Search(
            index=scoped_index_name,
            ranking_config=ranking_config)

    def load_kb(
        self,
        index_name,
        data_file,
        app_namespace=None,
        clean=False,
        app_path=None,
        config=None,
        **kwargs
    ):
        """Loads documents from disk into the specified index in the knowledge
        base. If an index with the specified name doesn't exist, a new index
        with that name will be created in the knowledge base.

        Args:
            index_name (str): The name of the new index to be created.
            data_file (str): The path to the data file containing the documents
                to be imported into the knowledge base index. It could be
                either json or jsonl file.
            app_namespace (str, optional): The namespace of the app. Used to prevent collisions
                between the indices of this app and those of other apps.
            clean (bool, optional): Set to true if you want to delete an existing index
                and reindex it
            app_path (str, optional): The path to the directory containing the app's data
            config (dict, optional): The QA config if passed directly rather than loaded from the
                app config
        """

        # fix related to Issue 219: https://github.com/cisco/mindmeld/issues/219
        app_namespace = app_namespace or self.app_namespace

        scoped_index_name = get_scoped_index_name(app_namespace, index_name)

        if clean:
            if NonElasticsearchQuestionAnswerer.ALL_INDICES.is_available(scoped_index_name):
                msg = f"Index `{index_name}` exists for app `{app_namespace}`, deleting index."
                logger.info(msg)
                NonElasticsearchQuestionAnswerer.ALL_INDICES.delete_index(scoped_index_name)
            else:
                msg = f"Index `{index_name}` does not exist for app `{app_namespace}`, " \
                      f"creating a new index."
                logger.warning(msg)

        # determine config: precedence is first given to argument `config`,
        #   then argument `app_path`, and then fallback option is self._query_settings
        if not app_path and not config:
            logger.warning(
                "You must provide either the application path to upload embeddings as specified"
                " in the app config or directly provide the QA config."
            )
            config = self._query_settings
        elif not config:
            config = get_classifier_config("question_answering", app_path=app_path)

        # determine embedding fields and load embedder model
        embedding_fields = (
            kwargs.get("embedding_fields", []) or
            config.get("model_settings", {}).get("embedding_fields", {}).get(index_name, [])
        )
        if embedding_fields and "embedder" not in self._query_type:
            msg = f"Found KB fields to upload embedding for ({embedding_fields}) fields in " \
                  f"index `{index_name}` but specified model_type `{self._query_type}` has no " \
                  f"`embedder` phrase in it. Ignoring provided `embedding_fields`."
            logger.error(msg)
            embedding_fields = []
        if not embedding_fields and "embedder" in self._query_type:
            logger.warning(
                "No embedding fields specified in the app config, "
                "continuing without generating embeddings..."
            )

        def _doc_generator(_data_file):
            with open(_data_file) as data_fp:
                line = data_fp.readline()
                data_fp.seek(0)
                # fix related to Issue 220: https://github.com/cisco/mindmeld/issues/220
                if line.strip().startswith("["):
                    logging.debug("Loading data from a json file.")
                    docs = json.load(data_fp)
                    for doc in docs:
                        yield doc
                else:
                    logging.debug("Loading data from a jsonl file.")
                    for line in data_fp:
                        doc = json.loads(line)
                        yield doc

        def match_regex(string, pattern_list):
            return any([re.match(pattern, string) for pattern in pattern_list])

        all_id2value = {}  # a mapping from id to value(s) for each kb field
        id_counter = 0
        for doc in _doc_generator(data_file):
            # determine _id, remains same for all keys of the doc
            _id = doc.get("id", None)
            if not _id:
                _id = id_counter
                id_counter += 1
            _id = str(_id)
            # update data
            for key, value in doc.items():
                if key not in all_id2value:
                    all_id2value[key] = {}
                all_id2value[key].update({_id: value})

        # for each key/field in doc, reuse an already existing FieldResource or create one
        index_resources = NonElasticsearchQuestionAnswerer.ALL_INDICES.get_index_metadata(
            scoped_index_name)
        for key, id2value in all_id2value.items():
            field_resource = index_resources.get(key, None)
            if not field_resource:
                field_resource = NonElasticsearchQuestionAnswerer.FieldResource(
                    index_name=scoped_index_name, field_name=key)
            field_resource.update_resource(
                id2value,
                DEFAULT_INDICES_STORAGE_PATH,
                has_embedding_resolver=match_regex(key, embedding_fields),
                resolver_settings=config.get("model_settings", {}),
                clean_fit=clean
            )
            index_resources.update({key: field_resource})

        # update and dump
        NonElasticsearchQuestionAnswerer.ALL_INDICES.update_index(scoped_index_name,
                                                                  index_resources)
        NonElasticsearchQuestionAnswerer.ALL_INDICES.dump_index(scoped_index_name)

    class Indices:
        """
        An object that hold all the indices for an app_path

        `self._indices` has the following dictionary format
        ```
            {
                index_name1: {key1: FieldResource1, key2: FieldResource2, ...),
                index_name2: {...},
                ...
            }
        ```
        """

        def __init__(self):
            self._indices = {}
            self._app_path = DEFAULT_INDICES_STORAGE_PATH
            self.is_loaded = False

        def _get_cache_path_for_index(self, index_name):
            return get_question_answerer_index_cache_file_path(self._app_path, index_name)

        def _load_index_metadata(self, index_name):
            index_resources = {}
            cache_path = self._get_cache_path_for_index(index_name)

            try:
                opfile = open(cache_path, "rb")
                dump = pickle.load(opfile)
                opfile.close()

                for field, cache_object in dump.items():
                    field_resource = (
                        NonElasticsearchQuestionAnswerer.FieldResource.from_metadata(
                            cache_object)
                    )
                    index_resources.update({field: field_resource})
            except Exception as e:
                msg = f"Couldn't find {index_name} related file in {self._app_path}"
                raise IndexError(msg) from e

            return index_resources

        def get_index_metadata(self, index_name):
            """
            - This method return metadata for each of the fields available in the index.
            - It does not fit any resolvers in the process, in case metadata is obtained through
                `_load_index_metadata()`. Be advised to fit resolvers after obtaining field
                resources for any index name in such case.
            """

            # get directly if already loaded
            if not self.is_available(index_name):
                msg = f"Index {index_name} does not exist in scope of {self._app_path}. " \
                      f"Consider creating one."
                logger.error(msg)
                return {}

            # check if metadata exists and unloaded
            if index_name not in self._indices:
                try:
                    return self._load_index_metadata(index_name)
                except IndexError:
                    return {}

            # in case if metadata is already loaded, return that
            return self._indices[index_name]

        def delete_index(self, index_name):

            # clear field resources
            if index_name in self._indices:
                del self._indices[index_name]

            # clear index dump cache path, if required
            cache_path = self._get_cache_path_for_index(index_name)
            if cache_path and os.path.exists(cache_path):
                os.remove(cache_path)

        def dump_index(self, index_name):
            """
            dumps data to cache path
            """
            index_resources = self.get_index_metadata(index_name)
            dump = {
                field: resource.to_metadata() for field, resource in index_resources.items()
            }
            cache_path = self._get_cache_path_for_index(index_name)

            dir_name = os.path.dirname(cache_path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            opfile = open(cache_path, "wb")
            pickle.dump(dump, opfile)
            opfile.close()

        def update_index(self, index_name, index_resources):
            self._indices.update({index_name: index_resources})

        def is_available(self, index_name):
            if (
                index_name in self._indices or
                os.path.exists(self._get_cache_path_for_index(index_name))
            ):
                return True

            return False

        def get(self, index_name):
            try:
                return self._indices[index_name]
            except KeyError as e:
                msg = f"Index {index_name} does not exist in scope of {self._app_path}. " \
                      f"Consider creating one."
                raise KeyError(msg) from e

    class FieldResource:

        def __init__(self, index_name, field_name):

            self.index_name = index_name
            self.field_name = field_name

            # warning: duplicated contents of `id2value` also exists in a resolver object if created
            self.id2value = {}
            self.data_type = None
            self.hash = None  # to identify any data changes before build resolver
            self.has_text_resolver = None
            self.has_embedding_resolver = None

            # any QA-style or ER-style configs dict with "model_settings" key field in it
            self.resolver_settings = {}

            self._text_resolver = None  # an entity resolver if string type data
            self._embedding_resolver = None  # an embedding based entity resolver

            # Currently we do not support `date` filed type sperately and consider it as a `string`
            if field_name.startswith("location"):
                self.data_type = "location"

        def __repr__(self):
            return f"{self.__class__.__name__} field_name: {self.field_name} " \
                   f"data_type: {self.data_type}"

        @classmethod
        def from_metadata(cls, cache_object: Bunch):
            field_resource = cls(index_name=cache_object.index_name,
                                 field_name=cache_object.field_name)
            field_resource.id2value = cache_object.id2value
            field_resource.data_type = cache_object.data_type
            field_resource.hash = cache_object.hash
            field_resource.has_text_resolver = cache_object.has_text_resolver
            field_resource.has_embedding_resolver = cache_object.has_embedding_resolver
            # field_resource.update_resource({}, DEFAULT_INDICES_STORAGE_PATH)
            return field_resource

        def to_metadata(self):
            cache_object = Bunch(
                index_name=self.index_name,
                field_name=self.field_name,
                id2value=self.id2value,
                data_type=self.data_type,
                hash=self.hash,
                has_text_resolver=self.has_text_resolver,
                has_embedding_resolver=self.has_embedding_resolver
            )
            return cache_object

        def _validate_and_reformat_value(self, _id, value):

            errmsg = "Formatting error in field {} in doc id {}. Found an unexpected type {}"

            if self.data_type == "location":
                # convert it into standard format "37.77,122.41"
                if isinstance(value, list):  # eg. [37.77, 122.41]
                    value = ",".join([str(_value) for _value in value])
                elif isinstance(value, dict):  # eg. {"lat": 37.77, "lon": 122.41}
                    value = ",".join([str(value["lat"]), str(value["lon"])])
                elif isinstance(value, str):  # eg. "37.77,122.41"
                    if "," not in value or len(value.split(",")) != 2:
                        raise TypeError("incorrect `location` field value format")
                else:
                    errmsg.format(self.field_name, _id, type(value))
                    raise TypeError(errmsg)
            elif self.data_type == "bool":
                if not isinstance(value, bool):
                    errmsg.format(self.field_name, _id, type(value))
                    raise TypeError(errmsg)
            elif self.data_type == "number":
                if not isinstance(value, numbers.Number):
                    errmsg.format(self.field_name, _id, type(value))
                    raise TypeError(errmsg)
            elif self.data_type == "string":
                if not (isinstance(value, str) or all([isinstance(val, str) for val in value])):
                    errmsg.format(self.field_name, _id, type(value))
                    raise TypeError(errmsg)

            return value

        def update_resource(self,
                            id2value,
                            app_path,
                            has_text_resolver=None,
                            has_embedding_resolver=None,
                            resolver_settings=None,
                            clean_fit=False):

            if not id2value and not self.data_type:
                return  # else, if required, update resolvers

            # update id2cname and compute hash
            for _id, value in id2value.items():
                # TODO: must consider those cases where the value is `null`
                if value is None:
                    continue
                # determine data_type if not previously determined (only once for all input data),
                #   note: it is already determined for `location` field type
                if not self.data_type:
                    if isinstance(value, bool):
                        self.data_type = "bool"
                    elif isinstance(value, numbers.Number):
                        self.data_type = "number"
                    elif isinstance(value, str) or all([isinstance(val, str) for val in value]):
                        self.data_type = "string"
                    else:
                        msg = f"Unknown field type: {isinstance(value, str)}"
                        raise TypeError(msg)
                # validation and re-formatting and update database
                value = self._validate_and_reformat_value(_id, value)
                self.id2value.update({_id: value})
            new_hash = Hasher(algorithm="sha1").hash(
                string=json.dumps(self.id2value, sort_keys=True)
            )

            # ascertain resolver requirement
            if self.data_type in ["location", "bool", "number"]:
                # discard input arguments
                self.has_text_resolver = False
                self.has_embedding_resolver = False
                self.resolver_settings = {}
            else:
                self.has_text_resolver = (
                    has_text_resolver if has_text_resolver is not None else True
                )
                self.has_embedding_resolver = (
                    has_embedding_resolver if has_embedding_resolver is not None else False
                )
                self.resolver_settings.update(resolver_settings or {})
                if not self.has_text_resolver and not self.has_embedding_resolver:
                    msg = "Atleast one of text or embedder resolver needs to be applied " \
                          "for string type field. "
                    raise Exception(msg)

            # update resolvers

            def get_entity_map():
                """
                converts id2value into an entity map format and returns it
                """

                # new: https://github.com/cisco/mindmeld/issues/291
                #   making `value` into a list for cname and whitelist conversion.
                #   If more than one items in `value`, all items after first one go into whitelist
                entity_map = {
                    "entities":
                        [
                            (
                                {"id": key, "cname": list(value)[0], "whitelist": list(value)[1:]}
                                if isinstance(value, (set, list)) else
                                {"id": key, "cname": value, "whitelist": []}
                            )
                            for key, value in self.id2value.items()
                        ]
                }
                return entity_map

            # tfidf based text resolver
            if (
                self.has_text_resolver and
                (self._text_resolver is None or new_hash != self.hash)
            ):
                # log info
                msg = f"Creating a text resolver for field `{self.field_name}` in " \
                      f"index `{self.index_name}`."
                logger.info(msg)
                # create a new resolver and fit
                er_config = {
                    "model_type": "resolver",
                    "model_settings": {
                        # "resolver_type": "" # not required as using a resolver class directly
                        **self.resolver_settings,
                        # "scores_normalizer": "min_max_scaler"
                    }
                }
                self._text_resolver = (
                    TfIdfSparseCosSimEntityResolver(
                        app_path=app_path,
                        entity_type=get_scoped_index_name(self.index_name, self.field_name),
                        er_config=er_config)
                )
                # format id2value data into an `entity_map` format for resolvers
                entity_map = get_entity_map()
                self._text_resolver.fit(entity_map=entity_map, clean=clean_fit)

            # embedder based resolver
            if (
                self.has_embedding_resolver and
                (self._embedding_resolver is None or new_hash != self.hash)
            ):
                # log info
                msg = f"Creating an embedder resolver for field `{self.field_name}` in " \
                      f"index `{self.index_name}`."
                logger.info(msg)
                # create a new resolver and fit
                er_config = {
                    "model_type": "resolver",
                    "model_settings": {
                        # "resolver_type": "" # not required as using a resolver class directly
                        **self.resolver_settings,
                        # "scores_normalizer": "min_max_scaler"
                    }
                }
                self._embedding_resolver = (
                    EmbedderCosSimEntityResolver(
                        app_path=app_path,
                        entity_type=get_scoped_index_name(self.index_name, self.field_name),
                        er_config=er_config)
                )
                # use same data as text resolver!
                entity_map = get_entity_map()
                # fit() also includes dumping cache in case of embedding resolver
                self._embedding_resolver.fit(entity_map=entity_map, clean=clean_fit)

            # update hash
            self.hash = new_hash

        def do_search(self, query):
            """
            Retrieves doc ids with corresponsing similarity scores for the given query

            Args:
                query (str): A string to do similarity search
            """

            # maps doc ids with their similarity scores
            this_field_scores = {}
            n_scores = 0
            for resolver in [self._text_resolver, self._embedding_resolver]:
                if resolver:
                    # obtain all synonyms' scores in predictions
                    predictions = resolver.predict(query, top_n=None)
                    # retain only top scored entries for each id
                    _best_scores = {}
                    for prediction in predictions:
                        _id, _score = prediction["id"], prediction["score"]
                        if _id not in _best_scores:
                            _best_scores[_id] = _score
                        else:
                            _best_scores[_id] = max(_best_scores[_id], _score)
                    # for missing doc ids, populate the minimum score
                    min_best_scores = min([*_best_scores.values()])
                    for _id in self.id2value.keys():
                        if _id not in _best_scores:
                            _best_scores[_id] = min_best_scores
                    # retain only top scored entries for each id
                    this_field_scores = {
                        _id: (this_field_scores.get(_id, 0.0) + _score) / (n_scores + 1)
                        for _id, _score in _best_scores.items()
                    }
                    n_scores += 1

            # Case where-in no resolver exists (eg. other data types)
            if not this_field_scores:
                this_field_scores = {_id: 0.0 for _id in self.id2value.keys()}

            return this_field_scores

        def do_filter(self, curated_docs, query_text=None,
                      gt=None, gte=None, lt=None, lte=None, boolean=None):

            if not curated_docs:
                return curated_docs

            # sanity check
            if self.data_type not in ["number", "string", "bool"]:
                msg = f"Filtering is not allowed for data type `{self.data_type}`. "
                logger.error(msg)
                return curated_docs

            if self.data_type in ["number"]:

                if gt is None and gte is None and lt is None and lte is None:
                    return curated_docs

                def is_valid(value):
                    if gt and not (value > gt):
                        return False
                    if gte and not (value >= gte):
                        return False
                    if lt and not (value < lt):
                        return False
                    if lte and not (value <= lte):
                        return False
                    return True

            elif self.data_type in ["string"]:

                if query_text is None:
                    return curated_docs

                query_text_aliases = set([query_text, query_text.lower()])

                def is_valid(value):
                    if value in query_text_aliases:
                        return True
                    if value.lower() in query_text_aliases:
                        return True
                    return False

            elif self.data_type == "bool":

                if not isinstance(boolean, bool):
                    return curated_docs

                def is_valid(value):
                    return value == boolean

            valid_indices = []
            for i, doc in enumerate(curated_docs):
                _id = doc["_id"]
                if _id in self.id2value and is_valid(self.id2value[_id]):
                    valid_indices.append(i)

            return [curated_docs[i] for i in valid_indices]

        def do_sort(self, curated_docs, sort_type=None, distance=None, location=None):

            if not curated_docs:
                return curated_docs

            # sanity check
            if self.data_type not in ["string", "number", "location"]:
                msg = f"Sorting is not allowed for data type `{self.data_type}`. "
                logger.error(msg)
                return curated_docs

            if self.data_type in ["string", "number"]:
                sort_type = sort_type or "desc"  # "asc" and "desc" are standardized values
                if sort_type not in ["asc", "desc"]:
                    return curated_docs

                return sorted(curated_docs, key=lambda x: x[self.field_name],
                              reverse=sort_type != "asc")

            elif self.data_type == "location":

                if distance is None and location is None:
                    return curated_docs

                if distance is not None or location is not None:
                    msg = "NotImplemented: Using location based sorting isn't implemented yet"
                    logger.error(msg)
                    return curated_docs

        @property
        def docs_ids(self):
            return [*self.id2value.keys()]

        @staticmethod
        def curate_docs_to_return(index_resources, _ids=None, _scores=None):
            """
            Collates all field names into docs

            Args:
                index_resources: a dict of filed names and corresponding FieldResource instances
                _ids (List[str]): if provided as a list of strings, only docs with those ids are
                    obtained in the same order of the ids, else all ids are used
            Returns:
                list[dict]: compiled docs
            """
            docs = {}

            all_ids = _ids or []
            if not all_ids:
                for field_name, field_resource in index_resources.items():
                    all_ids.extend(field_resource.docs_ids)

            if _scores:
                if len(_scores) != len(all_ids):
                    msg = f"Number of ids ({len(all_ids)}) did not match number of " \
                          f"scores ({len(_scores)}). Discarding inputted scores. "
                    logger.warning(msg)
                    _scores = None

            # initialize docs
            for i, _id in enumerate(all_ids):
                docs[_id] = {"_id": _id, "_score": _scores[i]} if _scores else {"_id": _id}

            # populate docs
            for field_name, field_resource in index_resources.items():
                for _id in all_ids:
                    # 1103024456
                    try:
                        docs[_id][field_name] = field_resource.id2value[_id]
                    except KeyError:
                        print(_id)

            return [*docs.values()]

    class Search:

        def __init__(self, index, ranking_config=None):
            """Initialize a Search object.
            """

            if ranking_config:
                msg = f"`ranking_config` is currently discarded in {self.__class__.__name__}."
                logger.warning(msg)

            self.index_name = index

            # TODO: Populate these as lists to entertain more flexible order of building search
            self._search_queries = {}
            self._filter_queries = {}
            self._sort_queries = {}

        def query(self, **kwargs):
            for key, value in kwargs.items():
                if key in self._search_queries:
                    msg = f"Found a duplicate search clause against `{key}` field name. " \
                          "Utilizing only latest input."
                    logger.warning(msg)
                self._search_queries.update({key: value})
            return self

        def filter(self, **kwargs):

            gt = kwargs.pop("gt", None)
            gte = kwargs.pop("gte", None)
            lt = kwargs.pop("lt", None)
            lte = kwargs.pop("lte", None)
            field = kwargs.pop("field", None)

            # NEW! to support boolean filtering as well
            boolean = kwargs.pop("boolean", None)  # if inputted, must be a bool- True or False

            # filter that operates on numeric values
            if field:
                if field in self._filter_queries:
                    msg = f"Found a duplicate filter clause against `{field}` field name. " \
                          "Utilizing only latest input."
                    logger.warning(msg)
                self._filter_queries.update(
                    {field: {"gt": gt, "gte": gte, "lt": lt, "lte": lte, "boolean": boolean}}
                )

            # filter that operates on strings; extract field name and query strings then
            else:
                for key, value in kwargs.items():
                    if key in self._filter_queries:
                        msg = f"Found a duplicate filter clause against `{key}` field name. " \
                              "Utilizing only latest input."
                        logger.warning(msg)
                    self._filter_queries.update({key: {"query_text": value}})

            return self

        def sort(self, field, **kwargs):

            sort_type = kwargs.pop("sort_type", None)
            distance = kwargs.pop("distance", None)
            location = kwargs.pop("location", None)

            if field in self._sort_queries:
                msg = f"Found a duplicate sort clause against `{field}` field name. " \
                      "Utilizing only latest input."
                logger.warning(msg)
            self._sort_queries.update(
                {field: {"sort_type": sort_type, "distance": distance, "location": location}})
            return self

        def execute(self, size=10):

            try:
                index_resources = (
                    NonElasticsearchQuestionAnswerer.ALL_INDICES.get(self.index_name)
                )
            except KeyError:
                msg = f"The index `{self.index_name}` looks unavailable. " \
                      f"Consider running `.load_kb(...)` to create indices for " \
                      f"running search queries. "
                logger.error(msg)
                return []

            # get results (aka. curated_docs) for `query` clauses
            n_scores = 0
            scores = {}
            for field_name, query_text in self._search_queries.items():
                field_resource = index_resources.get(field_name, None)
                if not field_resource:
                    msg = f"The field `{field_name}` is not available in index `{self.index_name}`."
                    logger.error(msg)
                    continue
                this_field_scores = field_resource.do_search(query_text)
                scores = {_id: (scores.get(_id, 0.0) + _score) / (n_scores + 1)
                          for _id, _score in this_field_scores.items()}
                n_scores += 1
            _ids, _scores = (
                zip(*sorted(scores.items(), key=lambda x: x[1], reverse=True))
                if scores else (None, None)
            )
            curated_docs = (
                NonElasticsearchQuestionAnswerer.FieldResource.curate_docs_to_return(
                    index_resources, _ids=_ids, _scores=_scores)
            )

            # get narrowed results for `filter` clause
            for field_name, kwargs in self._filter_queries.items():
                field_resource = index_resources.get(field_name, None)
                if not field_resource:
                    msg = f"The field `{field_name}` is not available in index `{self.index_name}`."
                    logger.error(msg)
                    continue
                curated_docs = field_resource.do_filter(curated_docs, **kwargs)

            # get sorted results for `sort` clause
            for field_name, kwargs in self._sort_queries.items():
                field_resource = index_resources.get(field_name, None)
                if not field_resource:
                    msg = f"The field `{field_name}` is not available in index `{self.index_name}`."
                    logger.error(msg)
                    continue
                curated_docs = field_resource.do_sort(curated_docs, **kwargs)

            # pop `_id` key fields
            for doc in curated_docs:
                doc.pop("_id")

            # reduce to the required size
            curated_docs = curated_docs[:size]
            if len(curated_docs) < size:
                msg = f"Retrieved only {len(curated_docs)} matches instead of asked number " \
                      f"{size} for index `{self.index_name}`."
                logger.info(msg)

            return curated_docs

    ALL_INDICES = Indices()


class QuestionAnswerer:

    def __new__(cls, app_path, **kwargs):
        """
        This method is used to initialize a QuestionAnswerer based on model_type
        To keep the code base backwards compatible, we use a `__new__()` way of creating instances
        alongside using a factory approach.

        The input arguments are kept as-is wrt to the `__init__()` of
        `ElasticsearchQuestionAnswerer` class which was the `QuestionAnswerer` class in previous
        version of `question_answerer.py`
        """

        config = cls._get_config(kwargs.get("config", None), app_path)
        kwargs.update({"config": config})
        return cls._get_question_answerer(config)(app_path, **kwargs)

    @classmethod
    def create_question_answerer(cls, app_path, **kwargs):
        return cls(app_path, **kwargs)

    @staticmethod
    def _get_config(config=None, app_path=None):
        if not config:
            return get_classifier_config("question_answering", app_path=app_path)
        return config

    @staticmethod
    def _get_question_answerer(config):

        use_elastic_search = config.get("use_elastic_search", True)

        if not use_elastic_search:
            return NonElasticsearchQuestionAnswerer
        else:
            return ElasticsearchQuestionAnswerer

    @classmethod
    def load_kb(cls,
                app_namespace,
                index_name,
                data_file,
                es_host=None,
                es_client=None,
                connect_timeout=2,
                clean=False,
                app_path=None,
                config=None,
                **kwargs):
        """
        Implemented to maintain backward compatability. Should be removed in future versions.

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
        msg = "DeprecationWarning: Refer the `load_kb(...)` method from object of a " \
              "QuestionAnswerer. Deprecated Usage: `QuestionAnswerer.load_kb(...)`. New usage: " \
              "`qa = QuestionAnswerer(...)`, then `qa.load_kb(...)`. " \
              "See https://www.mindmeld.com/docs/userguide/kb.html for more details. "
        logger.warning(msg)

        try:
            kwargs.update({
                "app_namespace": app_namespace,
                "es_host": es_host,
                "es_client": es_client,
                "connect_timeout": connect_timeout,
                "clean": clean,
                "config": config,
            })
            question_answerer = cls.create_question_answerer(app_path, **kwargs)
            question_answerer.load_kb(index_name, data_file, **kwargs)

        except TypeError as e:
            exp_msg = f"`model_type` in the provided QuestionAnswerer config must be among " \
                      f"{ALL_QUERY_TYPES} to use `.load_kb(...)` as a classmethod through " \
                      f"Elasticsearch. "
            raise Exception(exp_msg + msg) from e
