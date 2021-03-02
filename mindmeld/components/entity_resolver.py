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
This module contains the entity resolver component of the MindMeld natural language processor.
"""
import copy
import hashlib
import importlib
import logging
import os
import pickle
from abc import ABC, abstractmethod

import numpy as np
from elasticsearch.exceptions import ConnectionError as EsConnectionError
from elasticsearch.exceptions import ElasticsearchException, TransportError
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.autonotebook import trange

from ._config import (
    DEFAULT_ES_SYNONYM_MAPPING,
    PHONETIC_ES_SYNONYM_MAPPING,
    get_app_namespace,
    get_classifier_config,
)
from ._elasticsearch_helpers import (
    INDEX_TYPE_KB,
    INDEX_TYPE_SYNONYM,
    DOC_TYPE,
    create_es_client,
    delete_index,
    does_index_exist,
    get_field_names,
    get_scoped_index_name,
    load_index,
    resolve_es_config_for_version,
)
from .. import path
from ..core import Entity
from ..exceptions import EntityResolverConnectionError, EntityResolverError

logger = logging.getLogger(__name__)


def _is_module_available(module_name: str):
    """
    checks if a module is available or not (eg. _is_module_available("sentence_transformers"))

    Args:
        module_name (str): name of the model to check

    Returns:
        bool, if or not the given module exists
    """
    return bool(importlib.util.find_spec(module_name) is not None)


def _load_module_or_attr(module_name: str, func_name: str = None):
    """
    Loads an attribute from a module or a module itself
    (check if the module exists before calling this function)
    """
    m = importlib.import_module(module_name)
    if not func_name:
        return m
    if func_name not in dir(m):
        raise ImportError(f"Cannot import {func_name} from {module_name}")
    return getattr(m, func_name)


SBERT_AVAILABLE = _is_module_available("sentence_transformers")
if SBERT_AVAILABLE:
    sentence_transformers = _load_module_or_attr("sentence_transformers")
    torch = _load_module_or_attr("torch")


class EntityResolver:
    """An entity resolver is used to resolve entities in a given query to their canonical values
    (usually linked to specific entries in a knowledge base).
    """

    @classmethod
    def validate_resolver_name(cls, name):
        if name not in ENTITY_RESOLVER_MODEL_TYPES:
            msg = "Expected 'model_type' in ENTITY_RESOLVER_CONFIG among {!r}"
            raise Exception(msg.format(ENTITY_RESOLVER_MODEL_TYPES))
        if name == "sbert_cosine_similarity" and not SBERT_AVAILABLE:
            raise ImportError(
                "Must install the extra [bert] to use the built in embbedder for entity "
                "resolution. See https://www.mindmeld.com/docs/userguide/getting_started.html")

    def __new__(cls, app_path, resource_loader, entity_type, **kwargs):
        """Identifies appropriate entity resolver based on input config and
        initializes it.

        Args:
            app_path (str): The application path.
            resource_loader (ResourceLoader): An object which can load resources for the resolver.
            entity_type (str): The entity type associated with this entity resolver.
            er_config (dict): A classifier config
            es_host (str): The Elasticsearch host server.
            es_client (Elasticsearch): The Elasticsearch client.
        """
        er_config = (
            kwargs.pop("er_config", None) or
            get_classifier_config("entity_resolution", app_path=app_path)
        )
        name = er_config.get("model_type", None)
        cls.validate_resolver_name(name)
        return ENTITY_RESOLVER_MODEL_MAPPINGS.get(name)(
            app_path, resource_loader, entity_type, er_config, **kwargs
        )

    def fit(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def predict(self, entity):
        raise NotImplementedError


class EntityResolverBase(ABC):
    """
    Base class for Entity Resolvers
    """

    def __init__(self, app_path, resource_loader, entity_type, er_config, **kwargs):
        """Initializes an entity resolver"""
        self.app_path = app_path
        self.type = entity_type
        self.er_config = er_config
        self.kwargs = kwargs

        self._app_namespace = get_app_namespace(self.app_path)
        self._is_system_entity = Entity.is_system_entity(self.type)
        self.name = self.er_config.get("model_type")
        self.entity_map = resource_loader.get_entity_map(self.type)
        self.normalizer = resource_loader.query_factory.normalize

        if self._use_double_metaphone:
            self._enable_double_metaphone()

        self.dirty = False  # bool, True if exists any unsaved generated data that can be saved
        self.ready = False  # bool, True if the model is fit by calling .fit()

    @property
    def _use_double_metaphone(self):
        return "double_metaphone" in self.er_config.get("phonetic_match_types", [])

    def _enable_double_metaphone(self):
        """
        By default, resolvers are assumed to not support double metaphone usage
        If supported, override this method definition in the derived class
        (eg. see ElasticsearchEntityResolver)
        """
        logger.warning(
            "%r not configured to use double_metaphone",
            self.name
        )
        raise NotImplementedError

    def cache_path(self, tail_name=""):
        name = f"{self.name}_{tail_name}" if tail_name else self.name
        return path.get_entity_resolver_cache_file_path(
            self.app_path, self.type, name
        )

    @abstractmethod
    def _fit(self, clean):
        raise NotImplementedError

    def fit(self, clean=False):
        """Fits the resolver model, if required

        Args:
            clean (bool, optional): If ``True``, deletes and recreates the index from scratch
                                    with synonyms in the mapping.json.
        """

        if self.ready:
            return

        if self._is_system_entity or not self.entity_map.get("entities", []):
            return

        self._fit(clean)
        self.ready = True

    @abstractmethod
    def _predict(self, nbest_entities, top_n):
        raise NotImplementedError

    def predict(self, entity, top_n: int = 20):
        """Predicts the resolved value(s) for the given entity using the loaded entity map or the
        trained entity resolution model.

        Args:
            entity (Entity, tuple): An entity found in an input query, or a list of n-best entity \
                objects.
            top_n (int): maximum number of results to populate

        Returns:
            (list): The top 20 resolved values for the provided entity.
        """
        if isinstance(entity, (list, tuple)):
            top_entity = entity[0]
            nbest_entities = tuple(entity)
        else:
            top_entity = entity
            nbest_entities = tuple([entity])

        if self._is_system_entity:
            # system entities are already resolved
            return [top_entity.value]

        if not self.entity_map.get("entities", []):
            return []

        return self._predict(nbest_entities, top_n)

    @abstractmethod
    def _load(self):
        raise NotImplementedError

    def load(self):
        """If available, loads embeddings of synonyms that are previously dumped
        """
        self._load()

    def dump(self):
        raise NotImplementedError

    @staticmethod
    def process_entity_map(entity_type, entity_map, normalizer=None, augment_lower_case=False):
        """Loads in the mapping.json file and stores the synonym mappings in a item_map and a
        synonym_map for exact match entity resolution when Elasticsearch is unavailable

        Args:
            entity_type: The entity type associated with this entity resolver
            entity_map: The loaded mapping.json file for the given entity type
            normalizer: The normalizer to use
            augment_lower_case: If to extend the synonyms list with their lower-cased values
        """
        item_map = {}
        syn_map = {}
        seen_ids = []
        for item in entity_map.get("entities", []):
            cname = item["cname"]
            item_id = item.get("id")
            if cname in item_map:
                msg = "Canonical name %s specified in %s entity map multiple times"
                logger.debug(msg, cname, entity_type)
            if item_id:
                if item_id in seen_ids:
                    msg = "Item id {!r} specified in {!r} entity map multiple times"
                    raise ValueError(msg.format(item_id, entity_type))
                seen_ids.append(item_id)

            aliases = [cname] + item.pop("whitelist", [])
            items_for_cname = item_map.get(cname, [])
            items_for_cname.append(item)
            item_map[cname] = items_for_cname
            for alias in aliases:
                norm_alias = normalizer(alias) if normalizer else alias
                if norm_alias in syn_map:
                    msg = "Synonym %s specified in %s entity map multiple times"
                    logger.debug(msg, cname, entity_type)
                cnames_for_syn = syn_map.get(norm_alias, [])
                cnames_for_syn.append(cname)
                syn_map[norm_alias] = list(set(cnames_for_syn))

        # extend synonyms map by adding keys which are lowercases of the existing keys
        if augment_lower_case:
            msg = "Adding lowercased whitelist and cnames to list of possible synonyms"
            logger.info(msg)
            initial_num_syns = len(syn_map)
            aug_syn_map = {}
            for alias, alias_map in syn_map.items():
                alias_lower = alias.lower()
                if alias_lower not in syn_map:
                    aug_syn_map.update({alias_lower: alias_map})
            syn_map.update(aug_syn_map)
            final_num_syns = len(syn_map)
            msg = "Added %d additional synonyms by lower-casing. Upped from %d to %d"
            logger.info(msg, final_num_syns - initial_num_syns, initial_num_syns, final_num_syns)

        return {"items": item_map, "synonyms": syn_map}

    def __repr__(self):
        msg = "<{} {!r} ready: {!r}, dirty: {!r}>"
        return msg.format(self.__class__.__name__, self.name, self.ready, self.dirty)


class ElasticsearchEntityResolver(EntityResolverBase):
    """
    Resolver class based on Elastic Search
    """

    # prefix for Elasticsearch indices used to store synonyms for entity resolution
    ES_SYNONYM_INDEX_PREFIX = "synonym"
    """The prefix of the ES index."""

    def __init__(self, app_path, resource_loader, entity_type, er_config, **kwargs):
        super().__init__(app_path, resource_loader, entity_type, er_config, **kwargs)
        self._es_host = self.kwargs.get("es_host", None)
        self._es_config = {"client": self.kwargs.get("es_client", None), "pid": os.getpid()}

    def _enable_double_metaphone(self):
        pass

    @property
    def _es_index_name(self):
        return f"{ElasticsearchEntityResolver.ES_SYNONYM_INDEX_PREFIX}_{self.type}"

    @property
    def _es_client(self):
        # Lazily connect to Elasticsearch.  Make sure each subprocess gets it's own connection
        if self._es_config["client"] is None or self._es_config["pid"] != os.getpid():
            self._es_config = {"pid": os.getpid(), "client": create_es_client()}
        return self._es_config["client"]

    @classmethod
    def ingest_synonym(
        cls,
        app_namespace,
        index_name,
        index_type=INDEX_TYPE_SYNONYM,
        field_name=None,
        data=None,
        es_host=None,
        es_client=None,
        use_double_metaphone=False,
    ):
        """Loads synonym documents from the mapping.json data into the
        specified index. If an index with the specified name doesn't exist, a
        new index with that name will be created.

        Args:
            app_namespace (str): The namespace of the app. Used to prevent
                collisions between the indices of this app and those of other
                apps.
            index_name (str): The name of the new index to be created.
            index_type (str): specify whether to import to synonym index or
                knowledge base object index. INDEX_TYPE_SYNONYM is the default
                which indicates the synonyms to be imported to synonym index,
                while INDEX_TYPE_KB indicates that the synonyms should be
                imported into existing knowledge base index.
            field_name (str): specify name of the knowledge base field that the
                synonym list corresponds to when index_type is
                INDEX_TYPE_SYNONYM.
            data (list): A list of documents to be loaded into the index.
            es_host (str): The Elasticsearch host server.
            es_client (Elasticsearch): The Elasticsearch client.
            use_double_metaphone (bool): Whether to use the phonetic mapping or not.
        """
        data = data or []

        def _action_generator(docs):

            for doc in docs:
                action = {}

                # id
                if doc.get("id"):
                    action["_id"] = doc["id"]
                else:
                    # generate hash from canonical name as ID
                    action["_id"] = hashlib.sha256(
                        doc.get("cname").encode("utf-8")
                    ).hexdigest()

                # synonym whitelist
                whitelist = doc["whitelist"]
                syn_list = []
                syn_list.append({"name": doc["cname"]})
                for syn in whitelist:
                    syn_list.append({"name": syn})

                # If index type is INDEX_TYPE_KB  we import the synonym into knowledge base object
                # index by updating the knowledge base object with additional synonym whitelist
                # field. Otherwise, by default we import to synonym index in ES.
                if index_type == INDEX_TYPE_KB and field_name:
                    syn_field = field_name + "$whitelist"
                    action["_op_type"] = "update"
                    action["doc"] = {syn_field: syn_list}
                else:
                    action.update(doc)
                    action["whitelist"] = syn_list

                yield action

        mapping = (
            PHONETIC_ES_SYNONYM_MAPPING
            if use_double_metaphone
            else DEFAULT_ES_SYNONYM_MAPPING
        )
        es_client = es_client or create_es_client(es_host)
        mapping = resolve_es_config_for_version(mapping, es_client)
        load_index(
            app_namespace,
            index_name,
            _action_generator(data),
            len(data),
            mapping,
            DOC_TYPE,
            es_host,
            es_client,
        )

    def _fit(self, clean):
        """Loads an entity mapping file to Elasticsearch for text relevance based entity resolution.

        In addition, the synonyms in entity mapping are imported to knowledge base indexes if the
        corresponding knowledge base object index and field name are specified for the entity type.
        The synonym info is then used by Question Answerer for text relevance matches.

        Args:
            clean (bool): If ``True``, deletes and recreates the index from scratch instead of
                          updating the existing index with synonyms in the mapping.json.
        """
        try:
            if clean:
                delete_index(
                    self._app_namespace, self._es_index_name, self._es_host, self._es_client
                )
        except ValueError as e:  # when `clean = True` but no index to delete
            logger.info(e)

        entity_map = self.entity_map

        # list of canonical entities and their synonyms
        entities = entity_map.get("entities", [])

        # create synonym index and import synonyms
        logger.info("Importing synonym data to synonym index '%s'", self._es_index_name)
        ElasticsearchEntityResolver.ingest_synonym(
            app_namespace=self._app_namespace,
            index_name=self._es_index_name,
            data=entities,
            es_host=self._es_host,
            es_client=self._es_client,
            use_double_metaphone=self._use_double_metaphone,
        )

        # It's supported to specify the KB object type and field name that the NLP entity type
        # corresponds to in the mapping.json file. In this case the synonym whitelist is also
        # imported to KB object index and the synonym info will be used when using Question Answerer
        # for text relevance matches.
        kb_index = entity_map.get("kb_index_name")
        kb_field = entity_map.get("kb_field_name")

        # if KB index and field name is specified then also import synonyms into KB object index.
        if kb_index and kb_field:
            # validate the KB index and field are valid.
            # TODO: this validation can probably be in some other places like resource loader.
            if not does_index_exist(
                self._app_namespace, kb_index, self._es_host, self._es_client
            ):
                raise ValueError(
                    "Cannot import synonym data to knowledge base. The knowledge base "
                    "index name '{}' is not valid.".format(kb_index)
                )
            if kb_field not in get_field_names(
                self._app_namespace, kb_index, self._es_host, self._es_client
            ):
                raise ValueError(
                    "Cannot import synonym data to knowledge base. The knowledge base "
                    "field name '{}' is not valid.".format(kb_field)
                )
            if entities and not entities[0].get("id"):
                raise ValueError(
                    "Knowledge base index and field cannot be specified for entities "
                    "without ID."
                )
            logger.info("Importing synonym data to knowledge base index '%s'", kb_index)
            ElasticsearchEntityResolver.ingest_synonym(
                app_namespace=self._app_namespace,
                index_name=kb_index,
                index_type="kb",
                field_name=kb_field,
                data=entities,
                es_host=self._es_host,
                es_client=self._es_client,
                use_double_metaphone=self._use_double_metaphone,
            )

    def _predict(self, nbest_entities, top_n):
        """Predicts the resolved value(s) for the given entity using the loaded entity map or the
        trained entity resolution model.

        Args:
            nbest_entities (tuple): List of one entity object found in an input query, or a list  \
                of n-best entity objects.
            top_n (int): maximum number of results to populate

        Returns:
            (list): The top 20 resolved values for the provided entity.
        """

        top_entity = nbest_entities[0]

        weight_factors = [1 - float(i) / len(nbest_entities) for i in range(len(nbest_entities))]

        def _construct_match_query(entity, weight=1):
            return [
                {
                    "match": {
                        "cname.normalized_keyword": {
                            "query": entity.text,
                            "boost": 10 * weight,
                        }
                    }
                },
                {"match": {"cname.raw": {"query": entity.text, "boost": 10 * weight}}},
                {
                    "match": {
                        "cname.char_ngram": {"query": entity.text, "boost": weight}
                    }
                },
            ]

        def _construct_nbest_match_query(entity, weight=1):
            return [
                {
                    "match": {
                        "cname.normalized_keyword": {
                            "query": entity.text,
                            "boost": weight,
                        }
                    }
                }
            ]

        def _construct_phonetic_match_query(entity, weight=1):
            return [
                {
                    "match": {
                        "cname.double_metaphone": {
                            "query": entity.text,
                            "boost": 2 * weight,
                        }
                    }
                }
            ]

        def _construct_whitelist_query(entity, weight=1, use_phons=False):
            query = {
                "nested": {
                    "path": "whitelist",
                    "score_mode": "max",
                    "query": {
                        "bool": {
                            "should": [
                                {
                                    "match": {
                                        "whitelist.name.normalized_keyword": {
                                            "query": entity.text,
                                            "boost": 10 * weight,
                                        }
                                    }
                                },
                                {
                                    "match": {
                                        "whitelist.name": {
                                            "query": entity.text,
                                            "boost": weight,
                                        }
                                    }
                                },
                                {
                                    "match": {
                                        "whitelist.name.char_ngram": {
                                            "query": entity.text,
                                            "boost": weight,
                                        }
                                    }
                                },
                            ]
                        }
                    },
                    "inner_hits": {},
                }
            }

            if use_phons:
                query["nested"]["query"]["bool"]["should"].append(
                    {
                        "match": {
                            "whitelist.double_metaphone": {
                                "query": entity.text,
                                "boost": 3 * weight,
                            }
                        }
                    }
                )

            return query

        text_relevance_query = {
            "query": {
                "function_score": {
                    "query": {"bool": {"should": []}},
                    "field_value_factor": {
                        "field": "sort_factor",
                        "modifier": "log1p",
                        "factor": 10,
                        "missing": 0,
                    },
                    "boost_mode": "sum",
                    "score_mode": "sum",
                }
            }
        }

        match_query = []
        top_transcript = True
        for e, weight in zip(nbest_entities, weight_factors):
            if top_transcript:
                match_query.extend(_construct_match_query(e, weight))
                top_transcript = False
            else:
                match_query.extend(_construct_nbest_match_query(e, weight))
            if self._use_double_metaphone:
                match_query.extend(_construct_phonetic_match_query(e, weight))
        text_relevance_query["query"]["function_score"]["query"]["bool"][
            "should"
        ].append({"bool": {"should": match_query}})

        whitelist_query = _construct_whitelist_query(
            top_entity, use_phons=self._use_double_metaphone
        )
        text_relevance_query["query"]["function_score"]["query"]["bool"][
            "should"
        ].append(whitelist_query)

        try:
            index = get_scoped_index_name(self._app_namespace, self._es_index_name)
            response = self._es_client.search(index=index, body=text_relevance_query)
        except EsConnectionError as ex:
            logger.error(
                "Unable to connect to Elasticsearch: %s details: %s", ex.error, ex.info
            )
            raise EntityResolverConnectionError(es_host=self._es_client.transport.hosts) from ex
        except TransportError as ex:
            logger.error(
                "Unexpected error occurred when sending requests to Elasticsearch: %s "
                "Status code: %s details: %s",
                ex.error,
                ex.status_code,
                ex.info,
            )
            raise EntityResolverError(
                "Unexpected error occurred when sending requests to "
                "Elasticsearch: {} Status code: {} details: "
                "{}".format(ex.error, ex.status_code, ex.info)
            ) from ex
        except ElasticsearchException as ex:
            raise EntityResolverError from ex
        else:
            hits = response["hits"]["hits"]

            results = []
            for hit in hits:
                if self._use_double_metaphone and len(nbest_entities) > 1:
                    if hit["_score"] < 0.5 * len(nbest_entities):
                        continue

                top_synonym = None
                synonym_hits = hit["inner_hits"]["whitelist"]["hits"]["hits"]
                if synonym_hits:
                    top_synonym = synonym_hits[0]["_source"]["name"]
                result = {
                    "cname": hit["_source"]["cname"],
                    "score": hit["_score"],
                    "top_synonym": top_synonym,
                }

                if hit["_source"].get("id"):
                    result["id"] = hit["_source"].get("id")

                if hit["_source"].get("sort_factor"):
                    result["sort_factor"] = hit["_source"].get("sort_factor")

                results.append(result)

            if len(results) < top_n:
                logger.info(
                    "Retrieved only %d entity resolutions instead of asked number %d for "
                    "entity %r for type %r",
                    len(results), top_n, top_entity.text, top_entity.type,
                )

            return results[:top_n]

    def _load(self):
        """Loads the trained entity resolution model from disk."""
        try:
            scoped_index_name = get_scoped_index_name(
                self._app_namespace, self._es_index_name
            )
            if not self._es_client.indices.exists(index=scoped_index_name):
                self.fit()
        except EsConnectionError as e:
            logger.error(
                "Unable to connect to Elasticsearch: %s details: %s", e.error, e.info
            )
            raise EntityResolverConnectionError(es_host=self._es_client.transport.hosts) from e
        except TransportError as e:
            logger.error(
                "Unexpected error occurred when sending requests to Elasticsearch: %s "
                "Status code: %s details: %s",
                e.error,
                e.status_code,
                e.info,
            )
            raise EntityResolverError from e
        except ElasticsearchException as e:
            raise EntityResolverError from e

    def dump(self):
        pass


class ExactmatchEntityResolver(EntityResolverBase):
    """
    Resolver class based on exact matching
    """

    def __init__(self, app_path, resource_loader, entity_type, er_config, **kwargs):
        super().__init__(app_path, resource_loader, entity_type, er_config, **kwargs)
        self._exact_match_mapping = None

    def _fit(self, clean):
        """Loads an entity mapping file to resolve entities using exact match.
        """
        if clean:
            logger.info(
                "clean=True ignored while fitting exact_match algo for entity resolution"
            )

        entity_map = self.entity_map
        augment_lower_case = (
            self.er_config.get("model_settings", {})
                .get("augment_lower_case", False)
        )
        self._exact_match_mapping = self.process_entity_map(
            self.type, entity_map, normalizer=self.normalizer,
            augment_lower_case=augment_lower_case
        )

    def _predict(self, nbest_entities, top_n):
        """Looks for exact name in the synonyms data
        """

        entity = nbest_entities[0]  # top_entity

        normed = self.normalizer(entity.text)
        try:
            cnames = self._exact_match_mapping["synonyms"][normed]
        except (KeyError, TypeError):
            logger.warning(
                "Failed to resolve entity %r for type %r", entity.text, entity.type
            )
            return None

        if len(cnames) > 1:
            logger.info(
                "Multiple possible canonical names for %r entity for type %r",
                entity.text,
                entity.type,
            )

        values = []
        for cname in cnames:
            for item in self._exact_match_mapping["items"][cname]:
                item_value = copy.copy(item)
                item_value.pop("whitelist", None)
                values.append(item_value)

        if len(values) < top_n:
            logger.info(
                "Retrieved only %d entity resolutions instead of asked number %d for "
                "entity %r for type %r",
                len(values), top_n, entity.text, entity.type,
            )

        return values[:top_n]

    def _load(self):
        self.fit()


class SentencebertCossimEntityResolver(EntityResolverBase):
    """
    Resolver class for bert models as described here:
    https://github.com/UKPLab/sentence-transformers
    """

    def __init__(self, app_path, resource_loader, entity_type, er_config, **kwargs):
        super().__init__(app_path, resource_loader, entity_type, er_config, **kwargs)
        self._exact_match_mapping = None
        self._preloaded_mappings_embs = {}
        self._sbert_model_pretrained_name_or_abspath = (
            self.er_config.get("model_settings", {})
                .get("pretrained_name_or_abspath", "distilbert-base-nli-stsb-mean-tokens")
        )
        self._sbert_model = None

        # TODO: _lazy_resolution is set to a default value, can be modified to be an input
        self._lazy_resolution = False
        if not self._lazy_resolution:
            msg = "sentence-bert embeddings are cached for entity_type: `%s` " \
                  "for fast entity resolution; can possibly consume more disk space"
            logger.info(msg, self.type)

    @property
    def pretrained_name(self):
        return os.path.split(self._sbert_model_pretrained_name_or_abspath)[-1]

    def _encode(self, phrases):
        """Encodes input text(s) into embeddings, one vector for each phrase

        Args:
            phrases (str, list[str]): textual inputs that are to be encoded using sentence \
                                        transformers' model

        Returns:
            list[np.array]: one numpy array of embeddings for each phrase,
                            if ``phrases`` is ``str``, a list of one numpy aray is returned
        """

        if not phrases:
            return []

        if isinstance(phrases, str):
            phrases = [phrases]

        if not isinstance(phrases, (str, list)):
            raise TypeError(f"argument phrases must be of type str or list, not {type(phrases)}")

        batch_size = (
            self.er_config.get("model_settings", {})
                .get("batch_size", 16)
        )
        concat_last_n_layers = (
            self.er_config.get("model_settings", {})
                .get("concat_last_n_layers", 1)
        )
        normalize_token_embs = (
            self.er_config.get("model_settings", {})
                .get("normalize_token_embs", False)
        )
        # show_progress = False  # len(phrases) > 1
        show_progress = len(phrases) > 1
        convert_to_numpy = True

        if concat_last_n_layers != 1 or normalize_token_embs:
            results = self._encode_custom(phrases, batch_size=batch_size,
                                          convert_to_numpy=convert_to_numpy,
                                          show_progress_bar=show_progress,
                                          concat_last_n_layers=concat_last_n_layers,
                                          normalize_token_embs=normalize_token_embs)
        else:
            results = self._sbert_model.encode(phrases, batch_size=batch_size,
                                               convert_to_numpy=convert_to_numpy,
                                               show_progress_bar=show_progress)

        return results

    def _encode_custom(self, sentences,
                       batch_size: int = 32,
                       show_progress_bar: bool = None,
                       output_value: str = 'sentence_embedding',
                       convert_to_numpy: bool = True,
                       convert_to_tensor: bool = False,
                       device: str = None,
                       concat_last_n_layers: int = 1,
                       normalize_token_embs: bool = False):
        """
        Computes sentence embeddings (Note: Method largely derived from Sentence Transformers
            library to improve flexibility in encoding and pooling. Notably, `is_pretokenized` and
            `num_workers` are ignored due to deprecation in their library, retrieved 23-Feb-2021)

        Args:
            sentences (list[str]): the sentences to embed
            batch_size (int): the batch size used for the computation
            show_progress_bar (bool): Output a progress bar when encode sentences
            output_value (str): Default sentence_embedding, to get sentence embeddings.
                Can be set to token_embeddings to get wordpiece token embeddings.
            convert_to_numpy (bool): If true, the output is a list of numpy vectors. Else, it is a
                list of pytorch tensors.
            convert_to_tensor (bool): If true, you get one large tensor as return. Overwrites any
                setting from convert_to_numpy
            device: Which torch.device to use for the computation
            concat_last_n_layers (int): number of hidden outputs to concat starting from last layer
            normalize_token_embs (bool): if the (sub-)token embs are to be individually normalized

        Returns:
            (Union[List[Tensor], ndarray, Tensor]): By default, a list of tensors is returned.
                If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy
                matrix is returned.
        """

        if concat_last_n_layers != 1:
            assert 1 <= concat_last_n_layers <= len(
                self.transformer_model.auto_model.transformer.layer)

        self.transformer_model.eval()
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or
                logger.getEffectiveLevel() == logging.DEBUG
            )

        if convert_to_tensor:
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.transformer_model.to(device)
        self.pooling_model.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches",
                                  disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]
            features = self.transformer_model.tokenize(sentences_batch)
            features = sentence_transformers.util.batch_to_device(features, device)

            with torch.no_grad():
                out_features_transformer = self.transformer_model.forward(features)
                token_embeddings = out_features_transformer["token_embeddings"]
                if concat_last_n_layers > 1:
                    _all_layer_embeddings = out_features_transformer["all_layer_embeddings"]
                    token_embeddings = torch.cat(_all_layer_embeddings[-concat_last_n_layers:],
                                                 dim=-1)
                if normalize_token_embs:
                    _norm_token_embeddings = torch.linalg.norm(token_embeddings, dim=2,
                                                               keepdim=True)
                    token_embeddings = token_embeddings.div(_norm_token_embeddings)
                out_features_transformer.update({"token_embeddings": token_embeddings})
                out_features = self.pooling_model.forward(out_features_transformer)

                embeddings = out_features[output_value]

                if output_value == 'token_embeddings':
                    # Set token embeddings to 0 for padding tokens
                    input_mask = out_features['attention_mask']
                    input_mask_expanded = input_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    embeddings = embeddings * input_mask_expanded

                embeddings = embeddings.detach()

                # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                if convert_to_numpy:
                    embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    @staticmethod
    def _text_length(text):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).

        Union[List[int], List[List[int]]]
        """
        if isinstance(text, dict):
            return len(next(iter(text.values())))
        elif len(text) == 0 or isinstance(text[0], int):
            return len(text)
        else:
            return sum([len(t) for t in text])

    @staticmethod
    def _compute_cosine_similarity(syn_embs, entity_emb, return_as_dict=False):
        """Uses cosine similarity metric on synonym embeddings to sort most relevant ones
            for entity resolution

        Args:
            syn_embs (dict): a dict of synonym and its corresponding embedding from bert
            entity_emb (np.array): embedding of the input entity text, an array of size 1
        Returns:
            Union[dict, list[tuple]]: if return_as_dict, returns a dictionary of synonyms and their
                                        scores, else a list of sorted synonym names, paired with
                                        their similarity scores (descending)
        """

        entity_emb = entity_emb.reshape(1, -1)
        synonyms, synonyms_encodings = zip(*syn_embs.items())
        similarity_scores = cosine_similarity(np.array(synonyms_encodings), entity_emb).reshape(-1)
        similarity_scores = np.around(similarity_scores, decimals=2)

        if return_as_dict:
            return dict(zip(synonyms, similarity_scores))

        # results in descending scores
        return sorted(list(zip(synonyms, similarity_scores)), key=lambda x: x[1], reverse=True)

    def _fit(self, clean):
        """
        Fits the resolver model

        Args:
            clean (bool): If ``True``, deletes existing dump of synonym embeddings file
        """

        _output_type = (
            self.er_config.get("model_settings", {})
                .get("bert_output_type", "mean")
        )

        # load model
        try:
            _sbert_model_pretrained_name_or_abspath = \
                "sentence-transformers/" + self._sbert_model_pretrained_name_or_abspath
            self.transformer_model = sentence_transformers.models.Transformer(
                _sbert_model_pretrained_name_or_abspath,
                model_args={"output_hidden_states": True})
            self.pooling_model = sentence_transformers.models.Pooling(
                self.transformer_model.get_word_embedding_dimension(),
                pooling_mode_cls_token=_output_type == "cls",
                pooling_mode_max_tokens=False,
                pooling_mode_mean_tokens=_output_type == "mean",
                pooling_mode_mean_sqrt_len_tokens=False)
            modules = [self.transformer_model, self.pooling_model]
            self._sbert_model = sentence_transformers.SentenceTransformer(modules=modules)
        except OSError:
            logger.error(
                "Could not initialize the model name through sentence-transformers models in "
                "huggingface; Checking - %s - model directly in huggingface models",
                self._sbert_model_pretrained_name_or_abspath)
            try:
                self.transformer_model = sentence_transformers.models.Transformer(
                    self._sbert_model_pretrained_name_or_abspath,
                    model_args={"output_hidden_states": True})
                self.pooling_model = sentence_transformers.models.Pooling(
                    self.transformer_model.get_word_embedding_dimension(),
                    pooling_mode_cls_token=_output_type == "cls",
                    pooling_mode_max_tokens=False,
                    pooling_mode_mean_tokens=_output_type == "mean",
                    pooling_mode_mean_sqrt_len_tokens=False)
                modules = [self.transformer_model, self.pooling_model]
                self._sbert_model = sentence_transformers.SentenceTransformer(modules=modules)
            except OSError:
                logger.error("Could not initialize the model name through huggingface models; Not r"
                             "esorting to model names in sbert.net due to limited exposed features")

        # load mappings.json data
        entity_map = self.entity_map
        augment_lower_case = (
            self.er_config.get("model_settings", {})
                .get("augment_lower_case", False)
        )
        self._exact_match_mapping = self.process_entity_map(
            self.type, entity_map, augment_lower_case=augment_lower_case
        )

        # load embeddings for this data
        cache_path = self.cache_path(self.pretrained_name)
        if clean and os.path.exists(cache_path):
            os.remove(cache_path)
        if not self._lazy_resolution and os.path.exists(cache_path):
            self._load()
            self.dirty = False
        else:
            synonyms = [*self._exact_match_mapping["synonyms"]]
            synonyms_encodings = self._encode(synonyms)
            self._preloaded_mappings_embs = dict(zip(synonyms, synonyms_encodings))
            self.dirty = True

        if self.dirty and not self._lazy_resolution:
            self.dump()

    def _predict(self, nbest_entities, top_n):
        """Predicts the resolved value(s) for the given entity using cosine similarity.

        Args:
            nbest_entities (tuple): List of one entity object found in an input query, or a list  \
                of n-best entity objects.
            top_n (int): maximum number of results to populate

        Returns:
            (list): The top 20 resolved values for the provided entity.
        """

        syn_embs = self._preloaded_mappings_embs
        # TODO: Use all provided entities like elastic search
        top_entity = nbest_entities[0]  # top_entity
        top_entity_emb = self._encode(top_entity.text)[0]

        try:
            sorted_items = self._compute_cosine_similarity(syn_embs, top_entity_emb)
            values = []
            for synonym, score in sorted_items:
                cnames = self._exact_match_mapping["synonyms"][synonym]
                for cname in cnames:
                    for item in self._exact_match_mapping["items"][cname]:
                        item_value = copy.copy(item)
                        item_value.pop("whitelist", None)
                        item_value.update({"score": score})
                        item_value.update({"top_synonym": synonym})
                        values.append(item_value)
        except KeyError:
            logger.warning(
                "Failed to resolve entity %r for type %r; "
                "set 'clean=True' for computing embeddings of newly added items in mappings.json",
                top_entity.text, top_entity.type
            )
            return None
        except TypeError:
            logger.warning(
                "Failed to resolve entity %r for type %r", top_entity.text, top_entity.type
            )
            return None

        if len(values) < top_n:
            logger.info(
                "Retrieved only %d entity resolutions instead of asked number %d for "
                "entity %r for type %r",
                len(values), top_n, top_entity.text, top_entity.type,
            )

        return values[:top_n]

    def _load(self):
        """Loads embeddings for all synonyms, previously dumped into a .pkl file
        """
        cache_path = self.cache_path(self.pretrained_name)
        with open(cache_path, "rb") as fp:
            self._preloaded_mappings_embs = pickle.load(fp)

    def dump(self):
        """Dumps embeddings of synonyms into a .pkl file when the .fit() method is called
        """
        cache_path = self.cache_path(self.pretrained_name)
        if self.dirty:
            folder = os.path.split(cache_path)[0]
            if folder and not os.path.exists(folder):
                os.makedirs(folder)
            with open(cache_path, "wb") as fp:
                pickle.dump(self._preloaded_mappings_embs, fp)


ENTITY_RESOLVER_MODEL_MAPPINGS = {
    "exact_match": ExactmatchEntityResolver,
    "text_relevance": ElasticsearchEntityResolver,
    "sbert_cosine_similarity": SentencebertCossimEntityResolver
}
ENTITY_RESOLVER_MODEL_TYPES = [*ENTITY_RESOLVER_MODEL_MAPPINGS]
