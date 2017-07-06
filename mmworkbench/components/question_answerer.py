# -*- coding: utf-8 -*-
"""
This module contains the question answerer component of Workbench.
"""
from __future__ import absolute_import, unicode_literals
from builtins import object

import json
import logging
import copy
import six

from ._config import get_app_name, DOC_TYPE, DEFAULT_ES_QA_MAPPING, DEFAULT_RANKING_CONFIG
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

    def get(self, index, **kwargs):
        """Gets a collection of documents from the knowledge base matching the provided
        search criteria. This API provides a simple interface for developers to specify a list of
        knowledge base field and query string pairs to find best matches in a similar way as in
        common Web search interfaces. The knowledge base fields to be used depend on the mapping
        between NLU entity types and corresponding knowledge base objects, e.g. “cuisine” entity
        type can be mapped to a knowledge base object or an attribute of a knowledge base object.
        The mapping is often application specific and is dependent on the data model developers
        choose to use when building knowledge base.

        Examples:

        question_answerer.get(index='menu_items',
                              name='pork and shrimp',
                              restaurant_id='B01CGKGQ40',
                              _sort='price',
                              _sort_type='asc')

        Args:
            index (str): The name of an index
            id (str): The id of a particular document to retrieve.
            _sort (str): Specify the knowledge base field for custom sort.
            _sort_type (str): Specify custom sort type. Valid values are 'asc', 'desc' and
                              'distance'.
            _sort_location (dict): The origin location to be used when sorting by distance.

        Returns:
            a list of matching documents
        """

        # get index name with app scope
        index = get_scoped_index_name(self._app_name, index)

        doc_id = kwargs.get('id')

        # If an id was passed in, simply retrieve the specified document
        if doc_id:
            logger.info("Retrieve object from KB: index= '{}', id= '{}'.".format(index, doc_id))
            s = self.build_search(index)
            s = s.filter(id=doc_id)
            results = s.execute()
            return results

        sort_clause = {}
        query_clauses = []

        # iterate through keyword arguments to get KB field and value pairs for search and custom
        # sort criteria
        for key, value in kwargs.items():
            logger.debug("Processing argument: key= {} value= {}.".format(key, value))
            if key == '_sort':
                sort_clause['field'] = value
            elif key == '_sort_type':
                sort_clause['type'] = value
            elif key == '_sort_location':
                sort_clause['location'] = value
            else:
                query_clauses.append({key: value})
                logger.debug("Added query clause: field= {} value= {}.".format(key, value))

        logger.debug("Custom sort criteria {}.".format(sort_clause))

        # build Search object with overriding ranking setting to require all query clauses are
        # matched.
        s = self.build_search(index, {'query_clauses_operator': 'and'})

        # add query clauses to Search object.
        for clause in query_clauses:
            s = s.query(**clause)

        # add custom sort clause if specified.
        if sort_clause:
            s = s.sort(field=sort_clause['field'],
                       sort_type=sort_clause['type'],
                       location=sort_clause.get('location'))

        results = s.execute()
        return results

    def build_search(self, index, ranking_config=None):
        """build a search object for advanced filtered search.

        Args:
            index (str): index name of knowledge base object.
            ranking_config (dict): overriding ranking configuration parameters.
        Returns:
            Search: a Search object for filtered search.
        """
        return Search(client=self._es_client, index=index, ranking_config=ranking_config)

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


class Search:
    """This class models a generic filtered search in knowledge base. It allows developers to
    construct more complex knowledge base search criteria based on the application requirements.

    """
    def __init__(self, client, index, ranking_config=None):
        """Initialize a Search object.

        Args:
            client (Elasticsearch): Elasticsearch client.
            index (str): index name of knowledge base object.
            ranking_config (dict): overriding ranking configuration parameters for current search.
        """
        self.index = index
        self.client = client

        self._clauses = {
            "query": [],
            "filter": [],
            "sort": []
        }

        self._ranking_config = ranking_config
        if not ranking_config:
            self._ranking_config = copy.deepcopy(DEFAULT_RANKING_CONFIG)

    def _clone(self):
        """Clone a Search object.

        Returns:
            Search: cloned copy of the Search object.
        """
        s = Search(client=self.client, index=self.index)
        s._clauses = copy.deepcopy(self._clauses)
        s._ranking_config = copy.deepcopy(self._ranking_config)

        return s

    def _build_clause(self, type, **kwargs):
        """Helper method to build query, filter and sort clauses.

        Args:
            type (str): type of clause
        """
        if type == "query":
            key, value = six.next(six.iteritems(kwargs))
            clause = Search.QueryClause(key, value)
        elif type == "filter":
            # set the filter type to be 'range' if any range operator is specified.
            if kwargs.get('gt') or kwargs.get('gte') or kwargs.get('lt') or kwargs.get('lte'):
                field = kwargs.get('field')
                gt = kwargs.get('gt')
                gte = kwargs.get('gte')
                lt = kwargs.get('lt')
                lte = kwargs.get('lte')

                clause = Search.FilterClause(field=field,
                                             range_gt=gt,
                                             range_gte=gte,
                                             range_lt=lt,
                                             range_lte=lte)
            else:
                key, value = six.next(six.iteritems(kwargs))
                clause = Search.FilterClause(field=key, value=value)

        elif type == "sort":
            sort_field = kwargs.get('field')
            sort_type = kwargs.get('sort_type')
            sort_location = kwargs.get('location')

            clause = Search.SortClause(sort_field,
                                       sort_type,
                                       self._get_field_stats(sort_field),
                                       sort_location)

        clause.validate()
        self._clauses[clause.get_type()].append(clause)

    def query(self, **kwargs):
        """Specify the query text to match on a knowledge base text field. The query text is
        normalized and processed to find matches in knowledge base using several text relevance
        scoring factors including exact matches, phrase matches and partial matches.

        Examples:

        s = question_answerer.build_search(index='dish')
        s.query(name='pad thai')

        In the example above the query text "pad thai" will be used to match against document field
        "name" in knowledge base index "dish".

        Args:
            a keyword argument to specify the query text and the knowledge base document field.
        Returns:
            Search: a new Search object with added search criteria.
        """
        new_search = self._clone()
        new_search._build_clause("query", **kwargs)

        return new_search

    def filter(self, **kwargs):
        """Specify filter condition to be applied to specified knowledge base field. In Workbench
        two types of filters are supported: text filter and range filters.

        Text filters are used to apply hard filters on specified knowledge base text fields.
        The filter text value is normalized and matched using entire text span against the
        knowledge base field.

        It's common to have filter conditions based on other resolved canonical entities.
        For example, in food ordering domain the resolved restaurant entity can be used as a filter
        to resolve dish entities. The exact knowledge base field to apply these filters depends on
        the knowledge base data model of the application.

        Range filters are used to filter with a value range on specified knowledge base number or
        date fields. Example use cases include price range filters and date range filters.


        Examples:

        add text filter:
        s = question_answerer.build_search(index='menu_items')
        s.filter(restaurant_id='B01CGKGQ40')

        add range filter:
        s = question_answerer.build_search(index='menu_items')
        s.filter(filter_type='range', field='price', gte=1, lt=10)

        Args:
            filter_type(str): type of filter. Valid values are 'text' and 'range'.
            a keyword argument to specify the filter text and the knowledge base document field.
        Returns:
            Search: a new Search object with added search criteria.
        """
        new_search = self._clone()
        new_search._build_clause("filter", **kwargs)

        return new_search

    def sort(self, field, sort_type, location=None):
        """Specify custom sort criteria.

        Args:
            field (str): knowledge base field for sort.
            sort_type (str): sorting type. valid values are 'asc', 'desc' and 'distance'. 'asc' and
                             'desc' can be used to sort numeric or date fields and 'distance' can
                             be used to sort by distance on geo_point fields
            location (str): location (lat, lon) in geo_point format to be used as origin when
                            sorting by 'distance'
        """
        new_search = self._clone()
        new_search._build_clause("sort", field=field, sort_type=sort_type, location=location)
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
        stats_query['aggs'][field + '_min'] = {"min": {"field": field}}
        stats_query['aggs'][field + '_max'] = {"max": {"field": field}}

        res = self.client.search(index=self.index, body=stats_query, search_type="query_then_fetch")

        return {'min_value': res['aggregations'][field + '_min']['value'],
                'max_value': res['aggregations'][field + '_max']['value']}

    def _build_es_query(self):
        """Build knowledge base search syntax based on provided search criteria.

        Returns:
            str: knowledge base search syntax for the current search object.
        """
        es_query = {
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "filter": {
                                "bool": {
                                    "must": []
                                }
                            }
                        }
                    },
                    "functions": []
                }
            }
        }

        if self._clauses['query']:
            query_clauses = []

            for clause in self._clauses['query']:
                query_clauses.append(clause.build_query())

            if self._ranking_config['query_clauses_operator'] == 'and':
                es_query['query']['function_score']['query']['bool']['must'] = query_clauses
            else:
                es_query['query']['function_score']['query']['bool']['should'] = query_clauses

        if self._clauses['filter']:
            for clause in self._clauses['filter']:
                es_query['query']['function_score']['query']['bool']['filter']['bool']['must']\
                    .append(clause.build_query())

        for clause in self._clauses['sort']:
            sort_function = clause.build_query()
            es_query['query']['function_score']['functions'].append(sort_function)

        logger.debug("ES query syntax: {}.".format(es_query))
        return es_query

    def execute(self):
        """Executes the knowledge base search with provided criteria and returns matching documents.

        Returns:
            a list of matching documents.
        """
        es_query = self._build_es_query()
        response = self.client.search(index=self.index, body=es_query)
        results = [hit['_source'] for hit in response['hits']['hits']]
        return results

    class Clause:
        """This class models an abstract knowledge base clause."""

        def __init__(self):
            self.clause_type = None

        def validate(self):
            self._validate()

        def _validate(self):
            pass

        def get_type(self):
            return self.clause_type

    class QueryClause(Clause):
        """This class models a knowledge base query clause."""

        def __init__(self, field, value):
            """Initialize a knowledge base query clause."""
            self.field = field
            self.value = value

            self.clause_type = 'query'

        def build_query(self):

            clause = {
                "bool": {
                    "should": [
                        {
                            "match": {
                                self.field: {
                                    "query": self.value
                                }
                            }
                        },
                        {
                            "match": {
                                self.field + ".normalized_keyword": {
                                    "query": self.value,
                                    "boost": 10
                                }
                            }
                        },
                        {
                            "match": {
                                self.field + ".name.char_ngram": {
                                    "query": self.value
                                }
                            }
                        }
                    ]
                }
            }

            return clause

    class FilterClause(Clause):
        """This class models a knowledge base filter clause."""

        def __init__(self, field, value=None, range_gt=None, range_gte=None, range_lt=None,
                     range_lte=None):
            """Initialize a knowledge base filter clause. The filter type is determined by whether
            the range operators or value is passed in.
            """

            self.field = field
            self.value = value
            self.range_gt = range_gt
            self.range_gte = range_gte
            self.range_lt = range_lt
            self.range_lte = range_lte

            if self.value:
                self.filter_type = 'text'
            else:
                self.filter_type = 'range'

            self.clause_type = 'filter'

        def build_query(self):
            clause = {}
            if self.filter_type == 'text':
                clause = {
                    "match": {
                        self.field + ".normalized_keyword": {
                            "query": self.value
                        }
                    }
                }
            elif self.filter_type == 'range':
                lower_bound = None
                upper_bound = None
                if self.range_gt:
                    lower_bound = ('gt', self.range_gt)
                elif self.range_gte:
                    lower_bound = ('gte', self.range_gte)

                if self.range_lt:
                    upper_bound = ('lt', self.range_lt)
                elif self.range_lte:
                    upper_bound = ('lte', self.range_lte)

                clause = {
                    "range": {
                        self.field: {}
                    }
                }

                if lower_bound:
                    clause['range'][self.field][lower_bound[0]] = lower_bound[1]

                if upper_bound:
                    clause['range'][self.field][upper_bound[0]] = upper_bound[1]

            return clause

        def _validate(self):
            if self.filter_type == 'range':
                if not self.range_gt and not self.range_gte and not self.range_lt and \
                   not self.range_lte:
                    raise ValueError('No range parameter is specified')
                elif self.range_gte and self.range_gt:
                    raise ValueError(
                        'Invalid range parameters. Cannot specify both \'gte\' and \'gt\'.')
                elif self.range_lte and self.range_lt:
                    raise ValueError(
                        'Invalid range parameters. Cannot specify both \'lte\' and \'lt\'.')

    class SortClause(Clause):
        """This class models a knowledge base sort clause."""

        def __init__(self, field, sort_type='desc', field_stats=None, location=None):
            """Initialize a knowledge base sort clause"""
            self.field = field
            self.type = type
            self.location = location
            self.sort_type = sort_type
            self.field_stats = field_stats

            self.clause_type = 'sort'

        def build_query(self):

            # sort by distance based on passed in origin
            if self.sort_type == 'distance':
                origin = self.location
                scale = "1km"
            else:
                max_value = self.field_stats['max_value']
                min_value = self.field_stats['min_value']

                if self.sort_type == 'asc':
                    origin = self.field_stats['min_value']
                elif self.sort_type == 'desc':
                    origin = self.field_stats['max_value']
                else:
                    raise ValueError('Invalid value for sort type {}'.format(self.sort_type))

                scale = 0.5 * (max_value - min_value) if max_value != min_value else 1

            sort_clause = {
                "linear": {
                    self.field: {
                        "origin": origin,
                        "scale": scale
                    }
                }
            }

            return sort_clause

