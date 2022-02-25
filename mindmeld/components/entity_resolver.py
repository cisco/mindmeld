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
import json
import logging
import os
import pickle
import re
import uuid
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from string import punctuation

import numpy as np
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import trange

from ._config import (
    get_app_namespace,
    get_classifier_config,
)
from ._util import _is_module_available, _get_module_or_attr as _getattr
from ..core import Entity
from ..exceptions import (
    ElasticsearchConnectionError,
    EntityResolverError
)
from ..models import create_embedder_model
from ..resource_loader import ResourceLoader, Hasher

if _is_module_available("elasticsearch"):
    from ._elasticsearch_helpers import (
        INDEX_TYPE_KB,
        INDEX_TYPE_SYNONYM,
        DOC_TYPE,
        DEFAULT_ES_SYNONYM_MAPPING,
        PHONETIC_ES_SYNONYM_MAPPING,
        create_es_client,
        delete_index,
        does_index_exist,
        get_field_names,
        get_scoped_index_name,
        load_index,
        resolve_es_config_for_version,
    )

logger = logging.getLogger(__name__)

DEFAULT_TOP_N = 20


class EntityResolverFactory:

    @staticmethod
    def _correct_deprecated_er_config(er_config):
        """
        for backwards compatibility
          if `er_config` is supplied in deprecated format, its format is corrected and returned,
          else it is not modified and returned as-is

        deprecated usage
            >>> er_config = {
                    "model_type": "text_relevance",
                    "model_settings": {
                        ...
                    }
                }

        new usage
            >>> er_config = {
                    "model_type": "resolver",
                    "model_settings": {
                        "resolver_type": "text_relevance"
                        ...
                    }
                }
        """

        if not er_config.get("model_settings", {}).get("resolver_type"):
            model_type = er_config.get("model_type")
            if model_type == "resolver":
                raise ValueError(
                    "Could not find `resolver_type` in `model_settings` of entity resolver")
            else:
                msg = "Using deprecated config format for Entity Resolver. " \
                      "See https://www.mindmeld.com/docs/userguide/entity_resolver.html " \
                      "for more details."
                warnings.warn(msg, DeprecationWarning)
                er_config = copy.deepcopy(er_config)
                model_settings = er_config.get("model_settings", {})
                model_settings.update({"resolver_type": model_type})
                er_config["model_settings"] = model_settings
                er_config["model_type"] = "resolver"

        return er_config

    @staticmethod
    def _validate_resolver_type(name):
        if name not in ENTITY_RESOLVER_MODEL_MAPPINGS:
            raise ValueError(f"Expected 'resolver_type' in config of Entity Resolver "
                             f"among {[*ENTITY_RESOLVER_MODEL_MAPPINGS]} but found {name}")
        if name == "sbert_cosine_similarity" and not _is_module_available("sentence_transformers"):
            raise ImportError(
                "Must install the extra [bert] by running `pip install mindmeld[bert]` "
                "to use the built in embedder for entity resolution.")
        if name == "text_relevance" and not _is_module_available("elasticsearch"):
            raise ImportError(
                "Must install the extra [elasticsearch] by running "
                "`pip install mindmeld[elasticsearch]` "
                "to use Elasticsearch based entity resolution.")

    @classmethod
    def create_resolver(cls, app_path, entity_type, config=None, resource_loader=None, **kwargs):
        """
        Identifies appropriate entity resolver based on input config and
            returns it.

        Args:
            app_path (str): The application path.
            entity_type (str): The entity type associated with this entity resolver.
            resource_loader (ResourceLoader): An object which can load resources for the resolver.
            er_config (dict): A classifier config
            es_host (str): The Elasticsearch host server.
            es_client (Elasticsearch): The Elasticsearch client.
        """

        er_config = config or get_classifier_config("entity_resolution", app_path=app_path)
        er_config = cls._correct_deprecated_er_config(er_config)

        resolver_type = er_config["model_settings"]["resolver_type"]
        cls._validate_resolver_type(resolver_type)

        resource_loader = (
            resource_loader or ResourceLoader.create_resource_loader(app_path=app_path)
        )

        return ENTITY_RESOLVER_MODEL_MAPPINGS.get(resolver_type)(
            app_path,
            entity_type,
            config=er_config,
            resource_loader=resource_loader,
            **kwargs)


class BaseEntityResolver(ABC):  # pylint: disable=too-many-instance-attributes
    """
    Base class for Entity Resolvers
    """

    def __init__(self, app_path, entity_type, resource_loader=None, **_kwargs):
        """Initializes an entity resolver

        Args:
            app_path (str): The application path.
            entity_type (str): The entity type associated with this entity resolver.
            resource_loader (ResourceLoader, Optional): A resource loader object for the resolver.
        """
        self.app_path = app_path
        self.type = entity_type
        self._resource_loader = (
            resource_loader or ResourceLoader.create_resource_loader(app_path=self.app_path)
        )

        self._model_settings = {}
        self._is_system_entity = Entity.is_system_entity(self.type)
        self._no_trainable_canonical_entity_map = False
        self.dirty = False  # bool, True if exists any unsaved data/model that can be saved
        self.ready = False  # bool, True if the model is already fitted or loaded
        self.hash = ""

    def __repr__(self):
        msg = "<{} ready: {!r}, dirty: {!r}, app_path: {!r}, entity_type: {!r}>"
        return msg.format(self.__class__.__name__, self.ready, self.dirty, self.app_path, self.type)

    @property
    def resolver_configurations(self):
        return self._model_settings

    @resolver_configurations.setter
    @abstractmethod
    def resolver_configurations(self, model_settings):
        """Sets the configurations for the resolver that are used while creating a dump of configs
        """
        raise NotImplementedError

    def fit(self, clean=False, entity_map=None):
        """Fits the resolver model, if required

        Args:
            clean (bool, optional): If ``True``, deletes and recreates the index from scratch
                with synonyms in the mapping.json.
            entity_map (Dict[str, Union[str, List]]): Entity map if passed in directly instead of
                loading from a file path

        Raises:
            EntityResolverError: if the resolver cannot be fit with the loaded/passed-in data


        Example of a entity_map.json file:
        ---------------------------------
        entity_map = {
            "some_optional_key": "value",
            "entities": [
                {
                    "id": "B01MTUORTQ",
                    "cname": "Seaweed Salad",
                    "whitelist": [...],
                },
                ...
            ],
        }

        """
        msg = f"Fitting {self.__class__.__name__} entity resolver for entity_type {self.type}"
        logger.info(msg)

        if self.ready and not clean:
            return

        if self._is_system_entity:
            self._no_trainable_canonical_entity_map = True
            self.ready = True
            self.dirty = True  # configs need to be saved even for sys entities
            return

        entity_map = entity_map or self._get_entity_map()
        entities_data = entity_map.get("entities", [])
        if not entities_data:
            self._no_trainable_canonical_entity_map = True
            self.ready = True
            self.dirty = True
            return

        # obtain hash
        # hash based on the KB data before any processing
        new_hash = self._get_model_hash(entities_data)

        # see if a model is already available  hash value
        cached_model_path = self._resource_loader.hash_to_model_path.get(new_hash)
        if cached_model_path:
            msg = f"A fit {self.__class__.__name__} model for the found KB data is already " \
                  f"available. Loading the model instead of fitting again. Pass 'clean=True' to " \
                  f"the .fit() method in case you wish to force a re-fitting."
            logger.info(msg)
            self.load(cached_model_path, entity_map=entity_map)
            return

        # reformat (if required) and fit the resolver model
        entity_map["entities"] = self._format_entity_map(entities_data)
        try:
            self._fit(clean, entity_map)
        except Exception as e:
            msg = f"Error in {self.__class__.__name__} while fitting the resolver model with " \
                  f"clean={clean}"
            raise EntityResolverError(msg) from e
        self.hash = new_hash

        self.ready = True
        self.dirty = True

    def predict(self, entity_or_list_of_entities, top_n=DEFAULT_TOP_N, allowed_cnames=None):
        """Predicts the resolved value(s) for the given entity using the loaded entity map or the
        trained entity resolution model.

        Args:
            entity_or_list_of_entities (Entity, tuple[Entity], str, tuple[str]): One or more
                entity query strings or Entity objects that needs to be resolved.
            top_n (int, optional): maximum number of results to populate. If specifically inputted
                as 0 or `None`, results in an unsorted list of results in case of embedder and tfidf
                entity resolvers. This is sometimes helpful when a developer wishes to do some
                wrapper operations on top of unsorted results, such as combining scores from
                multiple resolvers and then sorting, etc.
            allowed_cnames (Iterable, optional): if inputted, predictions will only include objects
                related to these canonical names

        Returns:
            (list): The top n resolved values for the provided entity.

        Raises:
            EntityResolverError: if unable to obtain predictions for the given input
        """

        if not self.ready:
            msg = "Resolver not ready, model must be built (.fit()) or loaded (.load()) first."
            logger.error(msg)

        nbest_entities = entity_or_list_of_entities
        if not isinstance(nbest_entities, (list, tuple)):
            nbest_entities = tuple([nbest_entities])

        nbest_entities = tuple(
            [Entity(e, self.type) if isinstance(e, str) else e for e in nbest_entities]
        )

        if self._is_system_entity:
            # system entities are already resolved
            top_entity = nbest_entities[0]
            return [top_entity.value]

        if self._no_trainable_canonical_entity_map:
            return []

        if allowed_cnames:
            allowed_cnames = set(allowed_cnames)  # order doesn't matter

        # unsorted list in case of tfidf and embedder models; sorted in case of Elasticsearch
        try:
            results = self._predict(nbest_entities, allowed_cnames)
        except Exception as e:
            msg = f"Error in {self.__class__.__name__} while resolving entities for the " \
                  f"input: {entity_or_list_of_entities}"
            raise EntityResolverError(msg) from e

        return self._trim_and_sort_results(results, top_n)

    def dump(self, model_path, incremental_model_path=None):
        """
        Persists the trained classification model to disk. The state for an embedder based model is
        the cached embeddings whereas for text features based resolvers, (if required,) it will
        generally be a serialized pickle of the underlying model/algorithm and the data associated.

        In general, this method leads to creation of the following files:
            - .configs.pkl: pickle of the resolver's configuarble parameters
            - .pkl.hash: a hash string obtained from a combination of KB data and the config params
            - .pkl (optional, for non-ES models): pickle of the underlying model/algo state
            - .embedder_cache.pkl (optional, for embedder models): pickle of underlying embeddings

        Args:
            model_path (str): A .pkl file path where the resolver will be dumped. The model hash
                will be dumped at {path}.hash file path
            incremental_model_path (str, optional): The timestamp folder where the cached
                models are stored.
        """
        msg = f"Dumping {self.__class__.__name__} entity resolver for entity_type {self.type}"
        logger.info(msg)

        if not self.ready:
            msg = "Resolver not ready, model must be built (.fit()) before dumping."
            logger.error(msg)
            raise EntityResolverError(msg)

        for path in [model_path, incremental_model_path]:
            if not path:
                continue

            # underlying resolver model/algorithm/embeddings specific dump
            self._dump(path)

            # save resolver configs
            # in case of classifiers (domain, intent, etc.), dumping configs is handled by the
            # models abstract layer
            head, ext = os.path.splitext(path)
            resolver_config_path = head + ".config" + ext
            os.makedirs(os.path.dirname(resolver_config_path), exist_ok=True)
            with open(resolver_config_path, "wb") as fp:
                pickle.dump(self.resolver_configurations, fp)

            # save data hash
            # this hash is useful for avoiding re-fitting the resolver on unchanged data
            hash_path = path + ".hash"
            os.makedirs(os.path.dirname(hash_path), exist_ok=True)
            with open(hash_path, "w") as hash_file:
                hash_file.write(self.hash)

            if path == model_path:
                self.dirty = False

    def load(self, path, entity_map=None):
        """
        Loads state of the entity resolver as well the KB data.
        The state for embedder model is the cached embeddings whereas for text features based
        resolvers, (if required,) it will generally be a serialized pickle of the underlying
        model/algorithm. There is no state as such for Elasticsearch resolver to be dumped.

        Args:
            path (str): A .pkl file path where the resolver has been dumped
            entity_map (Dict[str, Union[str, List]]): Entity map if passed in directly instead of
                loading from a file path

        Raises:
            EntityResolverError: if the resolver cannot be loaded from the specified path
        """
        msg = f"Loading {self.__class__.__name__} entity resolver for entity_type {self.type}"
        logger.info(msg)

        if self.ready:
            msg = f"The {self.__class__.__name__} entity resolver for entity_type {self.type} is " \
                  f"already loaded. If you wish to do a clean fit, you can call the fit method " \
                  f"as follows: .fit(clean=True)"
            logger.info(msg)
            return

        if self._is_system_entity:
            self._no_trainable_canonical_entity_map = True
            self.ready = True
            self.dirty = False
            return

        entity_map = entity_map or self._get_entity_map()
        entities_data = entity_map.get("entities", [])
        if not entities_data:
            self._no_trainable_canonical_entity_map = True
            self.ready = True
            self.dirty = False
            return

        # obtain hash
        # hash based on the KB data before any processing
        new_hash = self._get_model_hash(entities_data)

        hash_path = path + ".hash"
        with open(hash_path, "r") as hash_file:
            self.hash = hash_file.read()
        if new_hash != self.hash:
            msg = f"Found KB data to have changed when loading {self.__class__.__name__} " \
                  f"resolver ({str(self)}). Please fit using 'clean=True' " \
                  f"before loading a resolver fopr this KB. Found new data hash to be " \
                  f"'{new_hash}' whereas the hash during dumping was '{self.hash}'"
            logger.error(msg)
            raise ValueError(msg)

        # reformat (if required)
        entity_map["entities"] = self._format_entity_map(entities_data)

        # load resolver configs if it exists
        head, ext = os.path.splitext(path)
        resolver_config_path = head + ".config" + ext
        if os.path.exists(resolver_config_path):
            with open(resolver_config_path, "rb") as fp:
                self.resolver_configurations = pickle.load(fp)
        else:
            msg = f"Cannot find a configs path for the resolver while loading the " \
                  f"resolver:{self.__class__.__name__}. This could have happened if you missed " \
                  f"to call the .dump() method of resolver before calling the .load() method."
            logger.debug(msg)
            self.resolver_configurations = {}

        # load underlying resolver model/algorithm/embeddings
        try:
            self._load(path, entity_map=entity_map)
        except Exception as e:
            msg = f"Error in {self.__class__.__name__} while loading the resolver from the " \
                  f"path: {path}"
            raise EntityResolverError(msg) from e

        self.ready = True
        self.dirty = False

    # TODO: method to be removed in a next major release of Mindmeld
    @abstractmethod
    def load_deprecated(self):
        """
        A method to handle the deprecated way of using the .load() method in entity resolvers. This
        ensures backwards compatibility when loading models that were built using an older version
        of Mindmeld i.e a version <=4.4.0. Since no hash pickle file is dumped in the older version
        of MindMeld, using the latest .load() method throws a FileNotFoundError.
        """
        raise NotImplementedError

    def unload(self):
        """
        Unloads the model from memory. This helps reduce memory requirements while
        training other models.
        """
        self._unload()
        self.resolver_configurations = {}
        self.ready = False

    @abstractmethod
    def _fit(self, clean, entity_map):
        """Fits the entity resolver model

        Args:
            clean (bool): If ``True``, deletes and recreates the index from scratch instead of
                            updating the existing index with synonyms in the mapping.json.
            entity_map (json): json data loaded from `mapping.json` file for the entity type
        """
        raise NotImplementedError

    @staticmethod
    def _get_model_hash(entities_data):
        """Returns a hash representing the inputs into the model

        Args:
            entities_data (List[dict]): The entity objects in the KB used to fit this model

        Returns:
            str: The hash
        """
        strings = sorted([json.dumps(ent_obj, sort_keys=True) for ent_obj in entities_data])
        return Hasher(algorithm="sha256").hash_list(strings=[*strings, ])

    def _get_entity_map(self, force_reload=False):
        try:
            return self._resource_loader.get_entity_map(self.type, force_reload=force_reload)
        except Exception as e:
            msg = f"Unable to load entity mapping data for " \
                  f"entity type: {self.type} in app_path: {self.app_path}"
            raise Exception(msg) from e

    @staticmethod
    def _format_entity_map(entities_data):
        """
        Args:
            entities_data (List[dict]): A list of dictionary objects each consisting of a 'cname'
                (canonical name string), 'whitelist' (a list of zero or more synonyms) and 'id' (a
                unique idenfier for the set of cname and whitelist)

        Returns:
            entities_data (List[dict]): A reformatted entities_data list

        Raise:
            valueError: if any object has missing cname as well as whitelist
        """
        all_ids = set()
        for i, ent_object in enumerate(entities_data):
            _id = ent_object.get("id")
            cname = ent_object.get("cname")
            whitelist = list(dict.fromkeys(ent_object.get("whitelist", [])))
            if cname is None and len(whitelist) == 0:
                msg = f"Found no canonical name field 'cname' while processing KB objects. " \
                      f"The observed KB entity object is: {ent_object}"
                raise ValueError(msg)
            elif cname is None and len(whitelist):
                cname = whitelist[0]
                whitelist = whitelist[1:]
            if _id in all_ids:
                msg = f"Found a duplicate id {_id} while formatting data for entity resolution. "
                _id = uuid.uuid4()
                msg += f"Replacing it with a new id: {_id}"
                logger.warning(msg)
            if not _id:
                _id = uuid.uuid4()
                msg = f"Found an entry in entity_map without a corresponding id. " \
                      f"Creating a random new id ({_id}) for this object."
                logger.warning(msg)
            _id = str(_id)
            all_ids.update([_id])
            entities_data[i] = {"id": _id, "cname": cname, "whitelist": whitelist}
        return entities_data

    def _process_entities(
        self, entities, normalizer=None, augment_lower_case=False, augment_title_case=False,
        augment_normalized=False, normalize_aliases=False
    ):
        """
        Loads in the mapping.json file and stores the synonym mappings in a item_map
            and a synonym_map

        Args:
            entities (list[dict]): List of dictionaries with keys `id`, `cname` and `whitelist`
            normalizer (callable): The normalizer to use, if provided, used to normalize synonyms
            augment_lower_case (bool): If to extend the synonyms list with their lower-cased values
            augment_title_case (bool): If to extend the synonyms list with their title-cased values
            augment_normalized (bool): If to extend the synonyms list with their normalized values,
                uses the provided normalizer
        """

        do_mutate_strings = any([augment_lower_case, augment_title_case, augment_normalized])
        if do_mutate_strings:
            msg = "Adding additional form of the whitelist and cnames to list of possible synonyms"
            logger.info(msg)

        item_map = {}
        syn_map = {}
        seen_ids = []
        for item in entities:
            item_id = item.get("id")
            cname = item["cname"]
            if cname in item_map:
                msg = "Canonical name %s specified in %s entity map multiple times"
                logger.debug(msg, cname, self.type)
            if item_id and item_id in seen_ids:
                msg = "Id %s specified in %s entity map multiple times"
                raise ValueError(msg.format(item_id, self.type))
            seen_ids.append(item_id)

            aliases = [cname] + item.pop("whitelist", [])
            if do_mutate_strings:
                new_aliases = []
                if augment_lower_case:
                    new_aliases.extend([string.lower() for string in aliases])
                if augment_title_case:
                    new_aliases.extend([string.title() for string in aliases])
                if augment_normalized and normalizer:
                    new_aliases.extend([normalizer(string) for string in aliases])
                aliases = {*aliases, *new_aliases}
            if normalize_aliases and normalizer:
                aliases = [normalizer(alias) for alias in aliases]

            items_for_cname = item_map.get(cname, [])
            items_for_cname.append(item)
            item_map[cname] = items_for_cname
            for alias in aliases:
                if alias in syn_map:
                    msg = "Synonym %s specified in %s entity map multiple times"
                    logger.debug(msg, cname, self.type)
                cnames_for_syn = syn_map.get(alias, [])
                cnames_for_syn.append(cname)
                syn_map[alias] = list(set(cnames_for_syn))

        return {"items": item_map, "synonyms": syn_map}

    @abstractmethod
    def _predict(self, nbest_entities, allowed_cnames=None):
        """Predicts the resolved value(s) for the given entity using cosine similarity.

        Args:
            nbest_entities (tuple): List of one entity object found in an input query, or a list  \
                of n-best entity objects.
            allowed_cnames (set, optional): if inputted, predictions will only include objects
                related to these canonical names

        Returns:
            (list): The resolved values for the provided entity.
        """
        raise NotImplementedError

    def _trim_and_sort_results(self, results, top_n):
        """
        Trims down the results generated by any ER class, finally populating at max top_n documents

        Args:
            results (list[dict]): Each element in this list is a result dictions with keys such as
                `id` (optional), `cname`, `score` and any others
            top_n (int): Number of top documents required to be populated

        Returns:
            list[dict]: if trimmed, a list similar to `results` but with fewer elements,
                        else, the `results` list as-is is returned
        """

        if not results:
            return []

        if not isinstance(top_n, int) or top_n <= 0:
            msg = f"The value of 'top_n' set to '{top_n}' during predictions in " \
                  f"{self.__class__.__name__}. This will result in an unsorted list of documents. "
            logger.info(msg)
            return results

        # Obtain top scored result for each doc id (only if scores field exist in results)
        best_results = {}
        for result in results:
            if "score" not in result:
                return results
            # use cname as id if no `id` field exist in results
            _id = result["id"] if "id" in result else result["cname"]
            if _id not in best_results or result["score"] > best_results[_id]["score"]:
                best_results[_id] = result
        results = [*best_results.values()]

        # Obtain upto top_n docs and sort them as final result
        n_scores = len(results)
        if n_scores < top_n and top_n != DEFAULT_TOP_N:
            # log only if a value other than default value is specified
            msg = f"Retrieved only {len(results)} entity resolutions instead of asked " \
                  f"number {top_n} for entity type {self.type}"
            logger.info(msg)
        elif n_scores > top_n:
            # select the top_n by using argpartition as it is faster than sorting
            _sim_scores = np.asarray([val["score"] for val in results])
            _top_inds = _sim_scores.argpartition(n_scores - top_n)[-top_n:]
            results = [results[ind] for ind in _top_inds]  # trimmed list of top_n docs

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def _dump(self, path):
        pass

    def _load(self, path, entity_map):
        pass

    def _unload(self):
        pass


class ExactMatchEntityResolver(BaseEntityResolver):
    """
    Resolver class based on exact matching
    """

    def __init__(self, app_path, entity_type, **kwargs):
        """
        Args:
            app_path (str): The application path.
            entity_type (str): The entity type associated with this entity resolver.
            resource_loader (ResourceLoader, Optional): A resource loader object for the resolver.
            config (dict): Configurations can be passed in through `model_settings` field
                `model_settings` (dict): Following keys are configurable:
                    augment_lower_case (bool): to augment lowercased synonyms as whitelist
                    augment_title_case (bool): to augment titlecased synonyms as whitelist
                    augment_normalized (bool): to augment text normalized synonyms as whitelist
        """
        super().__init__(app_path, entity_type, **kwargs)

        self.resolver_configurations = kwargs.get("config", {}).get("model_settings", {})
        self.processed_entity_map = None

    @BaseEntityResolver.resolver_configurations.setter
    def resolver_configurations(self, model_settings):
        self._model_settings = model_settings or {}
        self._aug_lower_case = self._model_settings.get("augment_lower_case", False)
        self._aug_title_case = self._model_settings.get("augment_title_case", False)
        self._aug_normalized = self._model_settings.get("augment_normalized", False)
        self._normalize_aliases = True
        self._model_settings.update({
            "augment_lower_case": self._aug_lower_case,
            "augment_title_case": self._aug_title_case,
            "augment_normalized": self._aug_normalized,
            "normalize_aliases": self._normalize_aliases,
        })

    def get_processed_entity_map(self, entity_map):
        """
        Processes the entity map into a format suitable for indexing and similarity searching

        Args:
            entity_map (Dict[str, Union[str, List]]): Entity map if passed in directly instead of
                loading from a file path

        Returns:
            processed_entity_map (Dict): A processed entity map better suited for indexing and
                querying
        """
        return self._process_entities(
            entity_map.get("entities", []),
            normalizer=self._resource_loader.query_factory.normalize,
            augment_lower_case=self._aug_lower_case,
            augment_title_case=self._aug_title_case,
            augment_normalized=self._aug_normalized,
            normalize_aliases=self._normalize_aliases
        )

    def _fit(self, clean, entity_map):
        self.processed_entity_map = self.get_processed_entity_map(entity_map)

        if clean:
            msg = f"clean=True ignored while fitting {self.__class__.__name__}"
            logger.info(msg)

    def _predict(self, nbest_entities, allowed_cnames=None):
        """Looks for exact name in the synonyms data
        """

        entity = nbest_entities[0]  # top_entity

        normed = self._resource_loader.query_factory.normalize(entity.text)
        try:
            cnames = self.processed_entity_map["synonyms"][normed]
        except (KeyError, TypeError):
            logger.warning(
                "Failed to resolve entity %r for type %r", entity.text, entity.type
            )
            return []

        if len(cnames) > 1:
            logger.info(
                "Multiple possible canonical names for %r entity for type %r",
                entity.text,
                entity.type,
            )

        values = []
        for cname in cnames:
            if allowed_cnames and cname not in allowed_cnames:
                continue
            for item in self.processed_entity_map["items"][cname]:
                item_value = copy.copy(item)
                item_value.pop("whitelist", None)
                values.append(item_value)

        return values

    def _load(self, path, entity_map):
        self.processed_entity_map = self.get_processed_entity_map(entity_map)

    def _unload(self):
        self.processed_entity_map = None

    def load_deprecated(self):
        self.fit()


class ElasticsearchEntityResolver(BaseEntityResolver):
    """
    Resolver class based on Elastic Search
    """

    # prefix for Elasticsearch indices used to store synonyms for entity resolution
    ES_SYNONYM_INDEX_PREFIX = "synonym"
    """The prefix of the ES index."""

    def __init__(self, app_path, entity_type, **kwargs):
        """
        Args:
            app_path (str): The application path.
            entity_type (str): The entity type associated with this entity resolver.
            resource_loader (ResourceLoader, Optional): A resource loader object for the resolver.
            es_host (str): The Elasticsearch host server
            es_client (Elasticsearch): an elastic search client
            config (dict): Configurations can be passed in through `model_settings` field
                `model_settings` (dict): Following keys are configurable:
                    phonetic_match_types (List): a list of phonetic match types that are passed to
                        Elasticsearch. Currently supports only using "double_metaphone" string in
                        the list.
        """
        super().__init__(app_path, entity_type, **kwargs)

        self.resolver_configurations = kwargs.get("config", {}).get("model_settings", {})

        self._es_host = kwargs.get("es_host")
        self._es_config = {"client": kwargs.get("es_client"), "pid": os.getpid()}
        self._app_namespace = get_app_namespace(self.app_path)

    @BaseEntityResolver.resolver_configurations.setter
    def resolver_configurations(self, model_settings):
        self._model_settings = model_settings or {}
        self._use_double_metaphone = "double_metaphone" in (
            self._model_settings.get("phonetic_match_types", [])
        )

    @property
    def _es_index_name(self):
        return f"{ElasticsearchEntityResolver.ES_SYNONYM_INDEX_PREFIX}_{self.type}"

    @property
    def _es_client(self):
        # Lazily connect to Elasticsearch.  Make sure each subprocess gets it's own connection
        if self._es_config["client"] is None or self._es_config["pid"] != os.getpid():
            self._es_config = {"pid": os.getpid(), "client": create_es_client()}
        return self._es_config["client"]

    @staticmethod
    def ingest_synonym(
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

    def _fit(self, clean, entity_map):
        """Loads an entity mapping file to Elasticsearch for text relevance based entity resolution.

        In addition, the synonyms in entity mapping are imported to knowledge base indexes if the
        corresponding knowledge base object index and field name are specified for the entity type.
        The synonym info is then used by Question Answerer for text relevance matches.
        """
        try:
            if clean:
                delete_index(
                    self._app_namespace, self._es_index_name, self._es_host, self._es_client
                )
        except ValueError as e:  # when `clean = True` but no index to delete
            logger.error(e)

        entities = entity_map.get("entities", [])

        # create synonym index and import synonyms
        logger.info("Importing synonym data to synonym index '%s'", self._es_index_name)
        self.ingest_synonym(
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

    def _predict(self, nbest_entities, allowed_cnames=None):
        """Predicts the resolved value(s) for the given entity using the loaded entity map or the
        trained entity resolution model.

        Args:
            nbest_entities (tuple): List of one entity object found in an input query, or a list  \
                of n-best entity objects.

        Returns:
            (list): The resolved values for the provided entity.
        """

        if allowed_cnames:
            msg = f"Cannot set 'allowed_cnames' param for {self.__class__.__name__}."
            raise NotImplementedError(msg)

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
        except _getattr("elasticsearch", "ConnectionError") as ex:
            logger.error(
                "Unable to connect to Elasticsearch: %s details: %s", ex.error, ex.info
            )
            raise ElasticsearchConnectionError(es_host=self._es_client.transport.hosts) from ex
        except _getattr("elasticsearch", "TransportError") as ex:
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
        except _getattr("elasticsearch", "ElasticsearchException") as ex:
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

            return results

    def _load(self, path, entity_map):
        del path

        try:
            scoped_index_name = get_scoped_index_name(
                self._app_namespace, self._es_index_name
            )
            if not self._es_client.indices.exists(index=scoped_index_name):
                self.fit(entity_map=entity_map)
        except _getattr("elasticsearch", "ConnectionError") as e:
            logger.error(
                "Unable to connect to Elasticsearch: %s details: %s", e.error, e.info
            )
            raise ElasticsearchConnectionError(es_host=self._es_client.transport.hosts) from e
        except _getattr("elasticsearch", "TransportError") as e:
            logger.error(
                "Unexpected error occurred when sending requests to Elasticsearch: %s "
                "Status code: %s details: %s",
                e.error,
                e.status_code,
                e.info,
            )
            raise EntityResolverError from e
        except _getattr("elasticsearch", "ElasticsearchException") as e:
            raise EntityResolverError from e

    def load_deprecated(self):
        try:
            scoped_index_name = get_scoped_index_name(
                self._app_namespace, self._es_index_name
            )
            if not self._es_client.indices.exists(index=scoped_index_name):
                self.fit()
        except _getattr("elasticsearch", "ConnectionError") as e:
            logger.error(
                "Unable to connect to Elasticsearch: %s details: %s", e.error, e.info
            )
            raise ElasticsearchConnectionError(es_host=self._es_client.transport.hosts) from e
        except _getattr("elasticsearch", "TransportError") as e:
            logger.error(
                "Unexpected error occurred when sending requests to Elasticsearch: %s "
                "Status code: %s details: %s",
                e.error,
                e.status_code,
                e.info,
            )
            raise EntityResolverError from e
        except _getattr("elasticsearch", "ElasticsearchException") as e:
            raise EntityResolverError from e


class TfIdfSparseCosSimEntityResolver(BaseEntityResolver):
    # pylint: disable=too-many-instance-attributes
    """
    a tf-idf based entity resolver using sparse matrices. ref:
    scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    """

    def __init__(self, app_path, entity_type, **kwargs):
        """
        Args:
            app_path (str): The application path.
            entity_type (str): The entity type associated with this entity resolver.
            resource_loader (ResourceLoader, Optional): A resource loader object for the resolver.
            config (dict): Configurations can be passed in through `model_settings` field
                `model_settings`:
                    augment_lower_case: to augment lowercased synonyms as whitelist
                    augment_title_case: to augment titlecased synonyms as whitelist
                    augment_normalized: to augment text normalized synonyms as whitelist
                    augment_max_synonyms_embeddings: to augment pooled synonyms whose embedding
                        is max-pool of all whitelist's (including above alterations) encodings
        """
        super().__init__(app_path, entity_type, **kwargs)

        self.resolver_configurations = kwargs.get("config", {}).get("model_settings", {})
        self.processed_entity_map = None
        self._analyzer = self._char_ngrams_plus_words_analyzer

        self._unique_synonyms = []
        self._syn_tfidf_matrix = None
        self._vectorizer = None

    @BaseEntityResolver.resolver_configurations.setter
    def resolver_configurations(self, model_settings):
        self._model_settings = model_settings or {}
        self._aug_lower_case = self._model_settings.get("augment_lower_case", True)
        self._aug_title_case = self._model_settings.get("augment_title_case", False)
        self._aug_normalized = self._model_settings.get("augment_normalized", False)
        self._aug_max_syn_embs = self._model_settings.get("augment_max_synonyms_embeddings", True)
        self._normalize_aliases = False
        self.ngram_length = 5  # max number of character ngrams to consider; 3 for elasticsearch
        self._model_settings.update({
            "augment_lower_case": self._aug_lower_case,
            "augment_title_case": self._aug_title_case,
            "augment_normalized": self._aug_normalized,
            "augment_max_synonyms_embeddings": self._aug_max_syn_embs,
            "normalize_aliases": self._normalize_aliases,
            "ngram_length": self.ngram_length,
        })

    def get_processed_entity_map(self, entity_map):
        """
        Processes the entity map into a format suitable for indexing and similarity searching

        Args:
            entity_map (Dict[str, Union[str, List]]): Entity map if passed in directly instead of
                loading from a file path

        Returns:
            processed_entity_map (Dict): A processed entity map better suited for indexing and
                querying
        """

        return self._process_entities(
            entity_map.get("entities", []),
            normalizer=self._resource_loader.query_factory.normalize,
            augment_lower_case=self._aug_lower_case,
            augment_title_case=self._aug_title_case,
            augment_normalized=self._aug_normalized,
            normalize_aliases=self._normalize_aliases
        )

    def _fit(self, clean, entity_map):
        self.processed_entity_map = self.get_processed_entity_map(entity_map)

        if clean:
            msg = f"clean=True ignored while fitting {self.__class__.__name__}"
            logger.info(msg)

        self._vectorizer = TfidfVectorizer(analyzer=self._analyzer, lowercase=False)

        # obtain sparse matrix
        synonyms = {v: k for k, v in
                    dict(enumerate(set(self.processed_entity_map["synonyms"]))).items()}
        synonyms_embs = self._vectorizer.fit_transform([*synonyms.keys()])

        # encode artificial synonyms if required
        if self._aug_max_syn_embs:
            # obtain cnames to synonyms mapping
            synonym2cnames = self.processed_entity_map["synonyms"]
            cname2synonyms = {}
            for syn, cnames in synonym2cnames.items():
                for cname in cnames:
                    items = cname2synonyms.get(cname, [])
                    items.append(syn)
                    cname2synonyms[cname] = items
            pooled_cnames, pooled_cnames_encodings = [], []
            # assert pooled synonyms
            for cname, syns in cname2synonyms.items():
                syns = list(set(syns))
                if len(syns) == 1:
                    continue
                pooled_cname = f"{cname} - SYNONYMS AVERAGE"
                # update synonyms map 'cause such synonyms don't actually exist in mapping.json file
                pooled_cname_aliases = synonym2cnames.get(pooled_cname, [])
                pooled_cname_aliases.append(cname)
                synonym2cnames[pooled_cname] = pooled_cname_aliases
                # check if needs to be encoded
                if pooled_cname in synonyms:
                    continue
                # if required, obtain pooled encoding and update collections
                pooled_encoding = scipy.sparse.csr_matrix(
                    np.max([synonyms_embs[synonyms[syn]].toarray() for syn in syns], axis=0)
                )
                pooled_cnames.append(pooled_cname)
                pooled_cnames_encodings.append(pooled_encoding)
            if pooled_cnames_encodings:
                pooled_cnames_encodings = scipy.sparse.vstack(pooled_cnames_encodings)
            if pooled_cnames:
                synonyms_embs = (
                    pooled_cnames_encodings if not synonyms else scipy.sparse.vstack(
                        [synonyms_embs, pooled_cnames_encodings])
                )
                synonyms.update(
                    OrderedDict(zip(
                        pooled_cnames,
                        np.arange(len(synonyms), len(synonyms) + len(pooled_cnames)))
                    )
                )

        # returns a sparse matrix
        self._unique_synonyms = [*synonyms.keys()]
        self._syn_tfidf_matrix = synonyms_embs

    def _predict(self, nbest_entities, allowed_cnames=None):

        # encode input entity
        top_entity = nbest_entities[0]  # top_entity

        try:
            scored_items = self.find_similarity(top_entity.text, _no_sort=True)
            values = []
            for synonym, score in scored_items:
                cnames = self.processed_entity_map["synonyms"][synonym]
                for cname in cnames:
                    if allowed_cnames and cname not in allowed_cnames:
                        continue
                    for item in self.processed_entity_map["items"][cname]:
                        item_value = copy.copy(item)
                        item_value.pop("whitelist", None)
                        item_value.update({"score": score})
                        item_value.update({"top_synonym": synonym})
                        values.append(item_value)
        except KeyError as e:
            msg = f"Failed to resolve entity {top_entity.text} for type {top_entity.type}; set " \
                  f"'clean=True' for computing TF-IDF of newly added items in mappings.json"
            logger.error(str(e))
            logger.error(msg)
            return []
        except TypeError as f:
            msg = f"Failed to resolve entity {top_entity.text} for type {top_entity.type}"
            logger.error(str(f))
            logger.error(msg)
            return []

        return values

    def _dump(self, path):
        resolver_state = {
            "unique_synonyms": self._unique_synonyms,  # caching unique syns for finding similarity
            "syn_tfidf_matrix": self._syn_tfidf_matrix,  # caching sparse vectors of synonyms
            "vectorizer": self._vectorizer,  # caching vectorizer
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fp:
            pickle.dump(resolver_state, fp)

    def _load(self, path, entity_map):
        self.processed_entity_map = self.get_processed_entity_map(entity_map)
        with open(path, "rb") as fp:
            resolver_state = pickle.load(fp)
        self._unique_synonyms = resolver_state["unique_synonyms"]
        self._syn_tfidf_matrix = resolver_state["syn_tfidf_matrix"]
        self._vectorizer = resolver_state["vectorizer"]

    def _unload(self):
        self.processed_entity_map = None
        self._unique_synonyms = []
        self._syn_tfidf_matrix = None
        self._vectorizer = None

    def _char_ngrams_plus_words_analyzer(self, string):
        """
        Analyzer that accounts for character ngrams as well as individual words in the input
        """
        # get char ngrams
        results = self._char_ngrams_analyzer(string)
        # add individual words
        words = re.split(r'[\s{}]+'.format(re.escape(punctuation)), string.strip())
        results.extend(words)
        return results

    def _char_ngrams_analyzer(self, string):
        """
        Analyzer that only accounts for character ngrams from size 1 to self.ngram_length
        """
        string = string.strip()
        if len(string) == 1:
            return [string]

        results = []
        # give importance to starting and ending characters of a word
        string = f" {string} "
        for n in range(self.ngram_length + 1):
            results.extend([''.join(gram) for gram in zip(*[string[i:] for i in range(n)])])
        results = list(set(results))
        results.remove(' ')
        # adding lowercased single characters might add more noise
        results = [r for r in results if not (len(r) == 1 and r.islower())]
        # returns empty list of an empty string
        return results

    def find_similarity(
        self, src_texts, top_n=DEFAULT_TOP_N, scores_normalizer=None,
        _return_as_dict=False, _no_sort=False
    ):
        """Computes sparse cosine similarity

        Args:
            src_texts (Union[str, list]): string or list of strings to obtain matching scores for.
            top_n (int, optional): maximum number of results to populate. if None, equals length
                of self._syn_tfidf_matrix
            scores_normalizer (str, optional): normalizer type to normalize scores. Allowed values
                are: "min_max_scaler", "standard_scaler"
           _return_as_dict (bool, optional): if the results should be returned as a dictionary of
                target_text name as keys and scores as corresponding values
            _no_sort (bool, optional): If True, results are returned without sorting. This is
                helpful at times when you wish to do additional wrapper operations on top of raw
                results and would like to save computational time without sorting.
        Returns:
            Union[dict, list[tuple]]: if _return_as_dict, returns a dictionary of tgt_texts and
                their scores, else a list of sorted synonym names paired with their
                similarity scores (descending order)

        """

        is_single = False
        if isinstance(src_texts, str):
            is_single = True
            src_texts = [src_texts]

        top_n = self._syn_tfidf_matrix.shape[0] if not top_n else top_n

        results = []
        for src_text in src_texts:
            src_text_vector = self._vectorizer.transform([src_text])

            similarity_scores = self._syn_tfidf_matrix.dot(src_text_vector.T).toarray().reshape(-1)
            # Rounding sometimes helps to bring correct answers on to the
            # top score as other non-correct resolutions
            similarity_scores = np.around(similarity_scores, decimals=4)

            if scores_normalizer:
                if scores_normalizer == "min_max_scaler":
                    _min = np.min(similarity_scores)
                    _max = np.max(similarity_scores)
                    denominator = (_max - _min) if (_max - _min) != 0 else 1.0
                    similarity_scores = (similarity_scores - _min) / denominator
                elif scores_normalizer == "standard_scaler":
                    _mean = np.mean(similarity_scores)
                    _std = np.std(similarity_scores)
                    denominator = _std if _std else 1.0
                    similarity_scores = (similarity_scores - _mean) / denominator
                else:
                    msg = f"Allowed values for `scores_normalizer` are only " \
                          f"{['min_max_scaler', 'standard_scaler']}. Continuing without " \
                          f"normalizing similarity scores."
                    logger.error(msg)

            if _return_as_dict:
                results.append(dict(zip(self._unique_synonyms, similarity_scores)))
            else:
                if not _no_sort:  # sort results in descending scores
                    n_scores = len(similarity_scores)
                    if n_scores > top_n:
                        top_inds = similarity_scores.argpartition(n_scores - top_n)[-top_n:]
                        result = sorted(
                            [(self._unique_synonyms[ii], similarity_scores[ii])
                             for ii in top_inds],
                            key=lambda x: x[1],
                            reverse=True)
                    else:
                        result = sorted(zip(self._unique_synonyms, similarity_scores),
                                        key=lambda x: x[1],
                                        reverse=True)
                    results.append(result)
                else:
                    result = list(zip(self._unique_synonyms, similarity_scores))
                    results.append(result)

        if is_single:
            return results[0]

        return results

    def load_deprecated(self):
        self.fit()


class EmbedderCosSimEntityResolver(BaseEntityResolver):
    """
    Resolver class for embedder models that create dense embeddings
    """

    def __init__(self, app_path, entity_type, **kwargs):
        """
        Args:
            app_path (str): The application path.
            entity_type (str): The entity type associated with this entity resolver.
            resource_loader (ResourceLoader, Optional): A resource loader object for the resolver.
            config (dict): Configurations can be passed in through `model_settings` field
                `model_settings`:
                    embedder_type: the type of embedder picked from embedder_models.py class
                        (eg. 'bert', 'glove', etc. )
                    augment_lower_case: to augment lowercased synonyms as whitelist
                    augment_title_case: to augment titlecased synonyms as whitelist
                    augment_normalized: to augment text normalized synonyms as whitelist
                    augment_average_synonyms_embeddings: to augment pooled synonyms whose embedding
                        is average of all whitelist's (including above alterations) encodings
                    embedder_cache_path (str): A path where the embedder cache can be stored. If it
                        is not specified, an embedder will be instantiated using the app_path
                        information. If specified, it will be used to dump the embeddings cache.
        """
        super().__init__(app_path, entity_type, **kwargs)

        self.resolver_configurations = kwargs.get("config", {}).get("model_settings", {})
        self.processed_entity_map = None
        self._embedder_model = None

    @BaseEntityResolver.resolver_configurations.setter
    def resolver_configurations(self, model_settings):
        self._model_settings = model_settings or {}
        self._aug_lower_case = self._model_settings.get("augment_lower_case", False)
        self._aug_title_case = self._model_settings.get("augment_title_case", False)
        self._aug_normalized = self._model_settings.get("augment_normalized", False)
        self._aug_avg_syn_embs = self._model_settings.get(
            "augment_average_synonyms_embeddings", True)
        self._normalize_aliases = False
        self._model_settings.update({
            "augment_lower_case": self._aug_lower_case,
            "augment_title_case": self._aug_title_case,
            "augment_normalized": self._aug_normalized,
            "normalize_aliases": self._normalize_aliases,
            "augment_max_synonyms_embeddings": self._aug_avg_syn_embs,
        })

    def get_processed_entity_map(self, entity_map):
        """
        Processes the entity map into a format suitable for indexing and similarity searching

        Args:
            entity_map (Dict[str, Union[str, List]]): Entity map if passed in directly instead of
                loading from a file path

        Returns:
            processed_entity_map (Dict): A processed entity map better suited for indexing and
                querying
        """

        return self._process_entities(
            entity_map.get("entities", []),
            normalizer=self._resource_loader.query_factory.normalize,
            augment_lower_case=self._aug_lower_case,
            augment_title_case=self._aug_title_case,
            augment_normalized=self._aug_normalized,
            normalize_aliases=self._normalize_aliases
        )

    def _fit(self, clean, entity_map):
        self.processed_entity_map = self.get_processed_entity_map(entity_map)
        self._embedder_model = create_embedder_model(
            app_path=self.app_path, config=self.resolver_configurations
        )

        if clean:
            msg = f"clean=True ignored while fitting {self.__class__.__name__}"
            logger.info(msg)

        # load embeddings from cache if exists, encode any other synonyms if required
        self._embedder_model.get_encodings([*self.processed_entity_map["synonyms"].keys()])

        # encode artificial synonyms if required
        if self._aug_avg_syn_embs:
            # obtain cnames to synonyms mapping
            cname2synonyms = {}
            for syn, cnames in self.processed_entity_map["synonyms"].items():
                for cname in cnames:
                    cname2synonyms[cname] = cname2synonyms.get(cname, []) + [syn]
            # create and add superficial data
            for cname, syns in cname2synonyms.items():
                syns = list(set(syns))
                if len(syns) == 1:
                    continue
                pooled_cname = f"{cname} - SYNONYMS AVERAGE"
                # update synonyms map 'cause such synonyms don't actually exist in mapping.json file
                if pooled_cname not in self.processed_entity_map["synonyms"]:
                    self.processed_entity_map["synonyms"][pooled_cname] = [cname]
                # obtain encoding and update cache
                # TODO: asumption that embedding cache has __getitem__ can be addressed
                if pooled_cname in self._embedder_model.cache:
                    continue
                pooled_encoding = np.mean(self._embedder_model.get_encodings(syns), axis=0)
                self._embedder_model.add_to_cache({pooled_cname: pooled_encoding})

        # useful for validation while loading
        self._model_settings["embedder_model_id"] = self._embedder_model.model_id

        # snippet for backwards compatibility
        #  even if the .dump() method of resolver isn't called explicitly, the embeddings need to be
        #  cached for fast inference of resolver; however, with the introduction of dump() and
        #  load() methods, this temporary persisting is not necessary and must be removed in future
        #  versions
        self._embedder_model.dump_cache()

    def _predict(self, nbest_entities, allowed_cnames=None):
        """Predicts the resolved value(s) for the given entity using cosine similarity.
        """

        # encode input entity
        top_entity = nbest_entities[0]  # top_entity

        allowed_syns = None
        if allowed_cnames:
            syn2cnames = self.processed_entity_map["synonyms"]
            allowed_syns = [syn for syn, cnames in syn2cnames.items()
                            if any([cname in allowed_cnames for cname in cnames])]

        try:
            scored_items = self._embedder_model.find_similarity(
                top_entity.text, tgt_texts=allowed_syns, _no_sort=True)
            values = []
            for synonym, score in scored_items:
                cnames = self.processed_entity_map["synonyms"][synonym]
                for cname in cnames:
                    if allowed_cnames and cname not in allowed_cnames:
                        continue
                    for item in self.processed_entity_map["items"][cname]:
                        item_value = copy.copy(item)
                        item_value.pop("whitelist", None)
                        item_value.update({"score": score})
                        item_value.update({"top_synonym": synonym})
                        values.append(item_value)
        except KeyError as e:
            msg = f"Failed to resolve entity {top_entity.text} for type {top_entity.type}; set " \
                  f"'clean=True' for computing embeddings of newly added items in mappings.json"
            logger.error(str(e))
            logger.error(msg)
            return []
        except TypeError as f:
            msg = f"Failed to resolve entity {top_entity.text} for type {top_entity.type}"
            logger.error(str(f))
            logger.error(msg)
            return []
        except RuntimeError as r:
            # happens when the input is an empty string and an embedder models fails to embed it
            msg = f"Failed to resolve entity {top_entity.text} for type {top_entity.type}"
            if "mat1 and mat2 shapes cannot be multiplied" in str(r):
                msg += ". This can happen if the input passed to embedder is an empty string!"
            logger.error(str(r))
            logger.error(msg)
            raise RuntimeError(msg) from r

        return values

    def _dump(self, path):
        # kept due to backwards compatibility in _fit(), must be removed in future versions
        self._embedder_model.clear_cache()  # delete the temp cache as .dump() method is now used

        head, ext = os.path.splitext(path)
        embedder_cache_path = head + ".embedder_cache" + ext
        self._embedder_model.dump_cache(cache_path=embedder_cache_path)
        self._model_settings["embedder_cache_path"] = embedder_cache_path

    def _load(self, path, entity_map):
        self.processed_entity_map = self.get_processed_entity_map(entity_map)
        self._embedder_model = create_embedder_model(
            app_path=self.app_path, config=self.resolver_configurations
        )

        # validate model id and load cache
        if self.resolver_configurations["embedder_model_id"] != self._embedder_model.model_id:
            msg = f"Unable to resolve the embedder model configurations. Found mismatched " \
                  f"configuartions between configs in the loaded pickle file and the configs " \
                  f"specified while instantiating {self.__class__.__name__}. Delete the related " \
                  f"model files and re-fit the resolver. Note that embedder models are not " \
                  f"pickled due to their large disk sizes and are only loaded from input configs."
            raise ValueError(msg)
        self._embedder_model.load_cache(
            cache_path=self.resolver_configurations["embedder_cache_path"]
        )

    def _unload(self):
        self.processed_entity_map = None
        self._embedder_model = None

    def _predict_batch(self, nbest_entities_list, batch_size):

        # encode input entity
        top_entity_list = [i[0].text for i in nbest_entities_list]  # top_entity

        try:
            # w/o batch,  [ nsyms x 768*4 ] x [ 1 x 768*4 ] --> [ nsyms x 1 ]
            # w/  batch,  [ nsyms x 768*4 ] x [ k x 768*4 ] --> [ nsyms x k ]
            scored_items_list = []
            for st_idx in trange(0, len(top_entity_list), batch_size, disable=False):
                batch = top_entity_list[st_idx:st_idx + batch_size]
                result = self._embedder_model.find_similarity(batch, _no_sort=True)
                scored_items_list.extend(result)

            values_list = []

            for scored_items in scored_items_list:
                values = []
                for synonym, score in scored_items:
                    cnames = self.processed_entity_map["synonyms"][synonym]
                    for cname in cnames:
                        for item in self.processed_entity_map["items"][cname]:
                            item_value = copy.copy(item)
                            item_value.pop("whitelist", None)
                            item_value.update({"score": score})
                            item_value.update({"top_synonym": synonym})
                            values.append(item_value)
                values_list.append(values)
        except (KeyError, TypeError) as e:
            logger.error(e)
            return None

        return values_list

    def predict_batch(self, entity_list, top_n: int = DEFAULT_TOP_N, batch_size: int = 8):

        if self._no_trainable_canonical_entity_map:
            return [[] for _ in entity_list]

        nbest_entities_list = []
        results_list = []
        for entity in entity_list:

            if isinstance(entity, (list, tuple)):
                top_entity = entity[0]
                nbest_entities = tuple(entity)
            else:
                top_entity = entity
                nbest_entities = tuple([entity])

            nbest_entities_list.append(nbest_entities)

            if self._is_system_entity:
                # system entities are already resolved
                results_list.append(top_entity.value)

        if self._is_system_entity:
            return results_list

        results_list = self._predict_batch(nbest_entities_list, batch_size)

        return [self._trim_and_sort_results(results, top_n) for results in results_list]

    def load_deprecated(self):
        self.fit()


class SentenceBertCosSimEntityResolver(EmbedderCosSimEntityResolver):
    """
    Resolver class for bert models based on the sentence-transformers library
    https://github.com/UKPLab/sentence-transformers
    """

    def __init__(self, app_path, entity_type, **kwargs):
        """
        This wrapper class allows creation of a BERT base embedder class
        (currently based on sentence-transformers)

        Specificall, this wrapper updates er_config in kwargs with
            - any default settings if unavailable in input
            - cache path

        Args:
            app_path (str): App's path to cache embeddings
            er_config (dict): Configurations can be passed in through `model_settings` field
                `model_settings`:
                    embedder_type: the type of embedder picked from embedder_models.py class
                        (eg. 'bert', 'glove', etc. )
                    pretrained_name_or_abspath: the pretrained model for 'bert' embedder
                    bert_output_type: if the output is a sentence mean pool or CLS output
                    quantize_model: if the model needs to be quantized for faster inference time
                        but at a possibly reduced accuracy
                    concat_last_n_layers: if some of the last layers of a BERT model are to be
                        concatenated for better accuracies
                    normalize_token_embs: if the obtained sub-token level encodings are to be
                        normalized
        """

        # default configs useful for reusing model's encodings through a cache path
        defaults = {
            "embedder_type": "bert",
            "pretrained_name_or_abspath": "sentence-transformers/all-mpnet-base-v2",
            "bert_output_type": "mean",
            "quantize_model": True,
            "concat_last_n_layers": 1,
            "normalize_token_embs": False,
        }
        # update er_configs in the kwargs with the defaults if any of the default keys are missing
        kwargs.update({
            "config": {
                **kwargs.get("config", {}),
                "model_settings": {
                    **defaults,
                    **kwargs.get("config", {}).get("model_settings", {}),
                },
            }
        })

        super().__init__(app_path, entity_type, **kwargs)


class EntityResolver:
    """
    Class for backwards compatibility

    deprecated usage
        >>> entity_resolver = EntityResolver(
                app_path, resource_loader, entity_type
            )

    new usage
        >>> entity_resolver = EntityResolverFactory.create_resolver(
                app_path, entity_type
            )
        # or ...
        >>> entity_resolver = EntityResolverFactory.create_resolver(
                app_path, entity_type, resource_loader=resource_loader
            )
    """

    def __new__(cls, app_path, resource_loader, entity_type, **kwargs):
        msg = "Entity Resolver should now be loaded using EntityResolverFactory. " \
              "See https://www.mindmeld.com/docs/userguide/entity_resolver.html for more details."
        warnings.warn(msg, DeprecationWarning)
        return EntityResolverFactory.create_resolver(
            app_path, entity_type, resource_loader=resource_loader, **kwargs
        )


ENTITY_RESOLVER_MODEL_MAPPINGS = {
    "exact_match": ExactMatchEntityResolver,
    "text_relevance": ElasticsearchEntityResolver,
    # TODO: In the newly added resolvers, to support
    #   (1) using all provided entities (i.e all nbest_entities) like elastic search
    #   (2) using kb_index_name and kb_field_name as used by Elasticsearch resolver
    "sbert_cosine_similarity": SentenceBertCosSimEntityResolver,
    "tfidf_cosine_similarity": TfIdfSparseCosSimEntityResolver,
    "embedder_cosine_similarity": EmbedderCosSimEntityResolver,
}
