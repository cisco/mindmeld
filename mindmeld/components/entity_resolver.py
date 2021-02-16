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
import logging
import os
import pickle
from abc import ABC, abstractmethod

from elasticsearch.exceptions import ConnectionError as EsConnectionError
from elasticsearch.exceptions import ElasticsearchException, TransportError

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .. import path
from ..core import Entity
from ..exceptions import EntityResolverConnectionError, EntityResolverError
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

try:
    from sentence_transformers import SentenceTransformer

    sbert_available = True
except ImportError:
    sbert_available = False

logger = logging.getLogger(__name__)


class EntityResolver:
    """An entity resolver is used to resolve entities in a given query to their canonical values
    (usually linked to specific entries in a knowledge base).
    """

    @classmethod
    def validate_resolver_name(cls, name):
        if name not in ENTITY_RESOLVER_MODEL_TYPES:
            msg = "Expected 'model_type' in ENTITY_RESOLVER_CONFIG among {!r}"
            raise Exception(msg.format(ENTITY_RESOLVER_MODEL_TYPES))
        if not sbert_available and name == "sbert_cosine_similarity":
            raise ImportError(
                "Must install the extra [bert] to use the built in embbedder for entity resolution.")

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
        er_config = kwargs.pop("er_config", None) or \
                    get_classifier_config("entity_resolution", app_path=app_path)
        name = er_config.get("model_type", None)
        cls.validate_resolver_name(name)
        return ENTITY_RESOLVER_MODEL_MAPPINGS.get(name)(
            app_path, resource_loader, entity_type, er_config, **kwargs
        )


class EntityResolverBase(ABC):
    """
    Base class for Entity Resolvers
    """

    def __init__(self, app_path, resource_loader, entity_type, er_config, **kwargs):
        """Initializes an entity resolver"""
        self._app_path = app_path
        self._resource_loader = resource_loader
        self.type = entity_type
        self._er_config = er_config

        self._app_namespace = get_app_namespace(self._app_path)
        self._is_system_entity = Entity.is_system_entity(self.type)
        self.name = self._er_config.get("model_type")
        self.cache_path = path.get_entity_resolver_cache_file_path(
            self._app_path, self.type, self.name
        )
        self.dirty = False  # bool, True if there exists any unsaved generated data that can be saved
        self.ready = False  # bool, True if the model is fit by calling .fit()

        if self._is_system_entity:
            canonical_entities = []
        else:
            canonical_entities = self._resource_loader.get_entity_map(self.type).get(
                "entities", []
            )
        self._no_canonical_entity_map = len(canonical_entities) == 0

        if self._use_double_metaphone:
            self._invoke_double_metaphone_usage()

    @property
    def _use_double_metaphone(self):
        return "double_metaphone" in self._er_config.get("phonetic_match_types", [])

    def _invoke_double_metaphone_usage(self):
        """
        By default, resolvers are assumed to not support double metaphone usage
        If supported, override this method definition in the derived class (eg. see EntityResolverUsingElasticSearch)
        """
        logger.warning(
            "%r not configured to use double_metaphone",
            self.name
        )
        raise NotImplementedError

    @abstractmethod
    def _fit(self):
        raise NotImplementedError

    def fit(self, clean=False):
        """Fits the resolver model, if required

        Args:
            clean (bool, optional): If ``True``, deletes and recreates the index from scratch
                                    with synonyms in the mapping.json.
        """

        if self.ready:
            return

        if self._no_canonical_entity_map:
            return

        self._fit(clean)
        self.ready = True

    @abstractmethod
    def _predict(self):
        raise NotImplementedError

    def predict(self, entity):
        """Predicts the resolved value(s) for the given entity using the loaded entity map or the
        trained entity resolution model.

        Args:
            entity (Entity, tuple): An entity found in an input query, or a list of n-best entity \
                objects.

        Returns:
            (list): The top 20 resolved values for the provided entity.
        """
        if isinstance(entity, (list, tuple)):
            top_entity = entity[0]
            entity = tuple(entity)
        else:
            top_entity = entity
            entity = tuple([entity])

        if self._is_system_entity:
            # system entities are already resolved
            return [top_entity.value]

        if self._no_canonical_entity_map:
            return []

        return self._predict(entity)

    @abstractmethod
    def _load(self):
        raise NotImplementedError

    def load(self):
        """If available, loads embeddings of synonyms that are previously dumped
        """
        self._load()

    def _dump(self):
        raise NotImplementedError

    def __repr__(self):
        msg = "<{} {!r} ready: {!r}, dirty: {!r}>"
        return msg.format(self.__class__.__name__, self.name, self.ready, self.dirty)


class EntityResolverUsingElasticSearch(EntityResolverBase):
    """
    Resolver class based on Elastic Search
    """

    # prefix for Elasticsearch indices used to store synonyms for entity resolution
    ES_SYNONYM_INDEX_PREFIX = "synonym"
    """The prefix of the ES index."""

    def __init__(self, app_path, resource_loader, entity_type, er_config, **kwargs):
        super(EntityResolverUsingElasticSearch, self).__init__(app_path, resource_loader,
                                                               entity_type, er_config,
                                                               **kwargs)
        self._es_host = kwargs.get("es_host", None)
        self._es_config = {"client": kwargs.get("es_client", None), "pid": os.getpid()}

    def _invoke_double_metaphone_usage(self):
        pass

    @property
    def _es_index_name(self):
        return EntityResolverUsingElasticSearch.ES_SYNONYM_INDEX_PREFIX + "_" + self.type

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
        if clean:
            delete_index(
                self._app_namespace, self._es_index_name, self._es_host, self._es_client
            )

        entity_map = self._resource_loader.get_entity_map(self.type)

        # list of canonical entities and their synonyms
        entities = entity_map.get("entities", [])

        # create synonym index and import synonyms
        logger.info("Importing synonym data to synonym index '%s'", self._es_index_name)
        EntityResolverUsingElasticSearch.ingest_synonym(
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
            EntityResolverUsingElasticSearch.ingest_synonym(
                app_namespace=self._app_namespace,
                index_name=kb_index,
                index_type="kb",
                field_name=kb_field,
                data=entities,
                es_host=self._es_host,
                es_client=self._es_client,
                use_double_metaphone=self._use_double_metaphone,
            )

    def _predict(self, entity):
        """Predicts the resolved value(s) for the given entity using the loaded entity map or the
        trained entity resolution model.

        Args:
            entity (Entity, tuple): An entity found in an input query, or a list of n-best entity \
                objects.

        Returns:
            (list): The top 20 resolved values for the provided entity.
        """

        top_entity = entity[0]

        weight_factors = [1 - float(i) / len(entity) for i in range(len(entity))]

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
        for e, weight in zip(entity, weight_factors):
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
                if self._use_double_metaphone and len(entity) > 1:
                    if hit["_score"] < 0.5 * len(entity):
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

            return results[0:20]

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


class EntityResolverUsingExactMatch(EntityResolverBase):
    """
    Resolver class based on exact matching
    """

    def __init__(self, app_path, resource_loader, entity_type, er_config, **kwargs):
        super(EntityResolverUsingExactMatch, self).__init__(app_path, resource_loader, entity_type,
                                                            er_config, **kwargs)
        self._normalizer = self._resource_loader.query_factory.normalize
        self._exact_match_mapping = None

    @staticmethod
    def _process_entity_map(entity_type, entity_map, normalizer):
        """Loads in the mapping.json file and stores the synonym mappings in a item_map and a
        synonym_map for exact match entity resolution when Elasticsearch is unavailable

        Args:
            entity_type: The entity type associated with this entity resolver
            entity_map: The loaded mapping.json file for the given entity type
            normalizer: The normalizer to use
        """
        item_map = {}
        syn_map = {}
        seen_ids = []
        for item in entity_map.get("entities"):
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
                norm_alias = normalizer(alias)
                if norm_alias in syn_map:
                    msg = "Synonym %s specified in %s entity map multiple times"
                    logger.debug(msg, cname, entity_type)
                cnames_for_syn = syn_map.get(norm_alias, [])
                cnames_for_syn.append(cname)
                syn_map[norm_alias] = list(set(cnames_for_syn))

        return {"items": item_map, "synonyms": syn_map}

    def _fit(self, clean):
        """Loads an entity mapping file to resolve entities using exact match.
        """
        if clean:
            logger.info(
                "clean=True ignored while fitting exact_match algo for entity resolution"
            )

        entity_map = self._resource_loader.get_entity_map(self.type)
        self._exact_match_mapping = self._process_entity_map(
            self.type, entity_map, self._normalizer
        )

    def _predict(self, entity):
        """Looks for exact name in the synonyms data
        """

        entity = entity[0]  # top_entity

        normed = self._normalizer(entity.text)
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

        return values

    def _load(self):
        self.fit()


class EntityResolverUsingSentenceBertEmbedder(EntityResolverBase):
    """
    Resolver class for bert models as described here: https://github.com/UKPLab/sentence-transformers
    """

    def __init__(self, app_path, resource_loader, entity_type, er_config, **kwargs):
        super(EntityResolverUsingSentenceBertEmbedder, self).__init__(app_path, resource_loader,
                                                                      entity_type, er_config,
                                                                      **kwargs)
        self._exact_match_mapping = None
        self._sbert_model = None
        self._preloaded_mappings_embs = {}

        # TODO: _lazy_resolution is set to a default value, can be modified to be an input
        self._lazy_resolution = False
        if not self._lazy_resolution:
            logger.info(
                f"sentence-bert embeddings are cached for entity_type: {self.type} for fast entity resolution; "
                "can possibly consume more disk-memory footprint"
            )

    def _encode(self, phrases):
        """Encodes input text(s) into embeddings, one vector for each phrase

        Args:
            phrases (str, list[str]): textual inputs that are to be encoded using sentence transformers' model

        Returns:
            list[np.array]: one numpy array of embeddings for each phrase,
                            if ``phrases`` is ``str``, a list of one numpy aray is returned
        """

        if not phrases:
            return []

        if isinstance(phrases, str):
            phrases = [phrases]
        elif isinstance(phrases, list):
            phrases = phrases
        else:
            raise TypeError(f"argument phrases must be of type str or list, not {type(phrases)}")

        # batch_size (int): The maximum size of each batch while encoding using on a deep embedder like BERT
        _batch_size = (
            self._er_config.get("model_settings", {})
                .get("batch_size", 16)
        )
        show_progress = len(phrases) > 1
        return self._sbert_model.encode(phrases, batch_size=_batch_size,
                                        is_pretokenized=False, convert_to_numpy=True,
                                        convert_to_tensor=False, show_progress_bar=show_progress)

    def _sort_using_cosine_dist(self, syn_embs, entity_emb):
        """Uses cosine similarity metric on synonym embeddings to sort most relevant ones
            for entity resolution

        Args:
            syn_embs (list[np.array]): a list of synonym embeddings
            entity_emb: (np.array): embedding of input entity
        Returns:
            list[tuple]: a list of sorted synonym names, paired with their similarity scores (descending)
        """

        entity_emb = entity_emb.reshape(1, -1)
        synonyms, synonyms_encodings = zip(*[(k, v) for k, v in syn_embs.items()])
        similarity_scores = cosine_similarity(np.array(synonyms_encodings), entity_emb).reshape(-1)

        return sorted([(syn, sim_score) for syn, sim_score in zip(synonyms, similarity_scores)],
                      key=lambda x: x[1], reverse=True  # results in descending scores
                      )

    @staticmethod
    def _process_entity_map(entity_type, entity_map):
        """Loads in the mapping.json file and stores the synonym mappings in a item_map and a
        synonym_map for exact match entity resolution when Elasticsearch is unavailable

        Args:
            entity_type: The entity type associated with this entity resolver
            entity_map: The loaded mapping.json file for the given entity type
        """
        item_map = {}
        syn_map = {}
        seen_ids = []
        for item in entity_map.get("entities"):
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
            for norm_alias in aliases:
                if norm_alias in syn_map:
                    msg = "Synonym %s specified in %s entity map multiple times"
                    logger.debug(msg, cname, entity_type)
                cnames_for_syn = syn_map.get(norm_alias, [])
                cnames_for_syn.append(cname)
                syn_map[norm_alias] = list(set(cnames_for_syn))

        return {"items": item_map, "synonyms": syn_map}

    def _fit(self, clean):
        """
        Fits the resolver model

        Args:
            clean (bool): If ``True``, deletes existing dump of synonym embeddings file
        """

        # load model
        self._sbert_model_name = "bert-base-nli-mean-tokens"
        logger.info(
            "Using {!r} model from sentence-transformers; see https://github.com/UKPLab/sentence-transformers for details",
            self._sbert_model_name
        )
        self._sbert_model = SentenceTransformer(self._sbert_model_name)

        # load mappings.json data
        entity_map = self._resource_loader.get_entity_map(self.type)
        self._exact_match_mapping = self._process_entity_map(
            self.type, entity_map
        )

        # load embeddings for this data
        if clean and os.path.exists(self.cache_path):
            os.remove(self.cache_path)
        if not self._lazy_resolution and os.path.exists(self.cache_path):
            self._load()
            self.dirty = False
        else:
            synonyms = [*self._exact_match_mapping["synonyms"]]
            synonyms_encodings = self._encode(synonyms)
            self._preloaded_mappings_embs = {k: v for k, v in zip(synonyms, synonyms_encodings)}
            self.dirty = True

        if self.dirty and not self._lazy_resolution:
            self._dump()

    def _predict(self, entity):
        """Predicts the resolved value(s) for the given entity using cosine similarity.

        Args:
            entity (Entity, tuple): An entity found in an input query, or a list of n-best entity \
                objects.

        Returns:
            (list): The top 20 resolved values for the provided entity.
        """

        syn_embs = self._preloaded_mappings_embs
        entity = entity[0]  # top_entity
        entity_emb = self._encode(entity.text)[0]

        try:
            sorted_items = self._sort_using_cosine_dist(syn_embs, entity_emb)
            values = []
            for synonym, score in sorted_items:
                cnames = self._exact_match_mapping["synonyms"][synonym]
                for cname in cnames:
                    for item in self._exact_match_mapping["items"][cname]:
                        item_value = copy.copy(item)
                        item_value.pop("whitelist", None)
                        item_value.update({"similarity": score})
                        values.append(item_value)
        except KeyError:
            logger.warning(
                "Failed to resolve entity %r for type %r; "
                "set 'clean=True' for computing embeddings of newly added items in mappings.json",
                entity.text, entity.type
            )
            return None
        except TypeError:
            logger.warning(
                "Failed to resolve entity %r for type %r", entity.text, entity.type
            )
            return None

        return values[0:20]

    def _load(self):
        """Loads embeddings for all synonyms, previously dumped into a .pkl file
        """
        with open(self.cache_path, "rb") as fp:
            self._preloaded_mappings_embs = pickle.load(fp)

    def _dump(self):
        """Dumps embeddings of synonyms into a .pkl file when the .fit() method is called
        """
        if self.dirty:
            folder = os.path.split(self.cache_path)[0]
            if folder and not os.path.exists(folder):
                os.makedirs(folder)
            with open(self.cache_path, "wb") as fp:
                pickle.dump(self._preloaded_mappings_embs, fp)


ENTITY_RESOLVER_MODEL_MAPPINGS = {
    "exact_match": EntityResolverUsingExactMatch,
    "text_relevance": EntityResolverUsingElasticSearch,
    "sbert_cosine_similarity": EntityResolverUsingSentenceBertEmbedder,
}
ENTITY_RESOLVER_MODEL_TYPES = [*ENTITY_RESOLVER_MODEL_MAPPINGS]
