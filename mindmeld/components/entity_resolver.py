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
from abc import ABC, abstractmethod
from collections import OrderedDict
from string import punctuation

import numpy as np
import scipy
from elasticsearch.exceptions import ConnectionError as EsConnectionError
from elasticsearch.exceptions import ElasticsearchException, TransportError
from sklearn.feature_extraction.text import TfidfVectorizer
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
from ._util import _is_module_available, _get_module_or_attr as _getattr
from .. import path
from ..core import Entity, Bunch
from ..exceptions import EntityResolverConnectionError, EntityResolverError
from ..resource_loader import ResourceLoader, Hasher

logger = logging.getLogger(__name__)


def _correct_deprecated_er_config(er_config):
    """
    for backwards compatability
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

    if not er_config.get("model_settings", {}).get("resolver_type", None):
        model_type = er_config.get("model_type")
        if model_type == "resolver":
            raise Exception("Could not find `resolver_type` in `model_settings` of entity resolver")
        else:
            logger.warning("DeprecationWarning: Use latest format of configs for entity resolver. "
                           "See https://www.mindmeld.com/docs/userguide/entity_resolver.html "
                           "for more details.")
            er_config = copy.deepcopy(er_config)
            model_settings = er_config.get("model_settings", {})
            model_settings.update({"resolver_type": model_type})
            er_config["model_settings"] = model_settings
            er_config["model_type"] = "resolver"

    return er_config


def _torch(op, *args, sub="", **kwargs):
    return _getattr(f"torch{'.' + sub if sub else ''}", op)(*args, **kwargs)


class BertEmbedder:
    """
    Encoder class for bert models based on https://github.com/UKPLab/sentence-transformers
    """

    # class variable to cache bert model(s);
    #   helps to mitigate keeping duplicate sets of large weight matrices
    CACHE_MODELS = {}

    @staticmethod
    def _batch_to_device(batch, target_device):
        """
        send a pytorch batch to a device (CPU/GPU)
        """
        tensor = _getattr("torch", "Tensor")
        for key in batch:
            if isinstance(batch[key], tensor):
                batch[key] = batch[key].to(target_device)
        return batch

    @staticmethod
    def _num_layers(model):
        """
        Finds the number of layers in a given transformers model
        """

        if hasattr(model, "n_layers"):  # eg. xlm
            num_layers = model.n_layers
        elif hasattr(model, "layer"):  # eg. xlnet
            num_layers = len(model.layer)
        elif hasattr(model, "encoder"):  # eg. bert
            num_layers = len(model.encoder.layer)
        elif hasattr(model, "transformer"):  # eg. sentence_transformers models
            num_layers = len(model.transformer.layer)
        else:
            raise ValueError(f"Not supported model {model} to obtain number of layers")

        return num_layers

    @property
    def device(self):
        return "cuda" if _torch("is_available", sub="cuda") else "cpu"

    @staticmethod
    def get_hashid(config):
        string = json.dumps(config, sort_keys=True)
        return Hasher(algorithm="sha1").hash(string=string)

    @staticmethod
    def get_sentence_transformers_encoder(name_or_path,
                                          output_type="mean",
                                          quantize=True,
                                          return_components=False):
        """
        Retrieves a sentence-transformer model and returns it along with its transformer and
        pooling components.

        Args:
            name_or_path: name or path to load a huggingface model
            output_type: type of pooling required
            quantize: if the model needs to be qunatized or not
            return_components: if True, returns the Transformer and Poooling components of the
                                sentence-bert model in a Bunch data type,
                                else just returns the sentence-bert model

        Returns:
            Union[
                sentence_transformers.SentenceTransformer,
                Bunch(sentence_transformers.Transformer,
                      sentence_transformers.Pooling,
                      sentence_transformers.SentenceTransformer)
            ]
        """

        strans_models = _getattr("sentence_transformers.models")
        strans = _getattr("sentence_transformers", "SentenceTransformer")

        transformer_model = strans_models.Transformer(name_or_path,
                                                      model_args={"output_hidden_states": True})
        pooling_model = strans_models.Pooling(transformer_model.get_word_embedding_dimension(),
                                              pooling_mode_cls_token=output_type == "cls",
                                              pooling_mode_max_tokens=False,
                                              pooling_mode_mean_tokens=output_type == "mean",
                                              pooling_mode_mean_sqrt_len_tokens=False)
        sbert_model = strans(modules=[transformer_model, pooling_model])

        if quantize:
            if not _is_module_available("torch"):
                raise ImportError("`torch` library required to quantize models") from None

            torch_qint8 = _getattr("torch", "qint8")
            torch_nn_linear = _getattr("torch.nn", "Linear")
            torch_quantize_dynamic = _getattr("torch.quantization", "quantize_dynamic")

            transformer_model = torch_quantize_dynamic(
                transformer_model, {torch_nn_linear}, dtype=torch_qint8
            ) if transformer_model else None
            pooling_model = torch_quantize_dynamic(
                pooling_model, {torch_nn_linear}, dtype=torch_qint8
            ) if pooling_model else None
            sbert_model = torch_quantize_dynamic(
                sbert_model, {torch_nn_linear}, dtype=torch_qint8
            ) if sbert_model else None

        if return_components:
            return Bunch(
                transformer_model=transformer_model,
                pooling_model=pooling_model,
                sbert_model=sbert_model
            )

        return sbert_model

    def _init_sentence_transformers_encoder(self, model_configs):

        sbert_model = None
        sbert_model_hashid = self.get_hashid(model_configs)
        sbert_model_name = model_configs["pretrained_name_or_abspath"]
        sbert_output_type = model_configs["bert_output_type"]
        sbert_quantize_model = model_configs["quantize_model"]

        if sbert_model_hashid not in BertEmbedder.CACHE_MODELS:

            info_msg = ""
            for name in [f"sentence-transformers/{sbert_model_name}", sbert_model_name]:
                try:
                    sbert_model = (
                        self.get_sentence_transformers_encoder(name,
                                                               output_type=sbert_output_type,
                                                               quantize=sbert_quantize_model,
                                                               return_components=True)
                    )
                    info_msg += f"Successfully initialized name/path `{name}` directly through " \
                                f"huggingface-transformers. "
                except OSError:
                    info_msg += f"Could not initialize name/path `{name}` directly through " \
                                f"huggingface-transformers. "

                if sbert_model:
                    break

            logger.info(info_msg)

            if not sbert_model:
                msg = f"Could not resolve the name/path `{sbert_model_name}`. " \
                      f"Please check the model name and retry."
                raise Exception(msg)

            BertEmbedder.CACHE_MODELS.update({sbert_model_hashid: sbert_model})

        sbert_model = BertEmbedder.CACHE_MODELS.get(sbert_model_hashid)
        self.transformer_model = sbert_model.transformer_model
        self.pooling_model = sbert_model.pooling_model
        self.sbert_model = sbert_model.sbert_model

    def _encode_local(self,
                      sentences,
                      batch_size,
                      show_progress_bar,
                      output_value,
                      convert_to_numpy,
                      convert_to_tensor,
                      device,
                      concat_last_n_layers,
                      normalize_token_embs):
        """
        Computes sentence embeddings (Note: Method largely derived from Sentence Transformers
            library to improve flexibility in encoding and pooling. Notably, `is_pretokenized` and
            `num_workers` are ignored due to deprecation in their library, retrieved 23-Feb-2021)
        """

        if concat_last_n_layers != 1:
            assert 1 <= concat_last_n_layers <= self._num_layers(self.transformer_model.auto_model)

        self.transformer_model.eval()
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or
                logger.getEffectiveLevel() == logging.DEBUG
            )

        if convert_to_tensor:
            convert_to_numpy = False

        input_is_string = isinstance(sentences, str)
        if input_is_string:  # Cast an individual sentence to a list with length 1
            sentences = [sentences]

        self.transformer_model.to(device)
        self.pooling_model.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches",
                                  disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]
            features = self.transformer_model.tokenize(sentences_batch)
            features = self._batch_to_device(features, device)

            with _torch("no_grad"):
                out_features_transformer = self.transformer_model.forward(features)
                token_embeddings = out_features_transformer["token_embeddings"]
                if concat_last_n_layers > 1:
                    _all_layer_embs = out_features_transformer["all_layer_embeddings"]
                    token_embeddings = _torch(
                        "cat", _all_layer_embs[-concat_last_n_layers:], dim=-1)
                if normalize_token_embs:
                    _norm_token_embeddings = _torch(
                        "norm", token_embeddings, sub="linalg", dim=2, keepdim=True)
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

                if convert_to_numpy:
                    embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = _torch("stack", all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_is_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def encode(self, phrases, **kwargs):
        """Encodes input text(s) into embeddings, one vector for each phrase

        Args:
            phrases (str, list[str]): textual inputs that are to be encoded using sentence \
                                        transformers' model
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

        if not phrases:
            return []

        if not isinstance(phrases, (str, list)):
            raise TypeError(f"argument phrases must be of type str or list, not {type(phrases)}")

        batch_size = kwargs.get("batch_size", 16)
        _len_phrases = len(phrases) if isinstance(phrases, list) else 1
        show_progress_bar = kwargs.get("show_progress_bar", False) and _len_phrases > 1
        output_value = kwargs.get("output_value", 'sentence_embedding')
        convert_to_numpy = kwargs.get("convert_to_numpy", True)
        convert_to_tensor = kwargs.get("convert_to_tensor", False)
        device = kwargs.get("device", self.device)
        concat_last_n_layers = kwargs.get("concat_last_n_layers", 1)
        normalize_token_embs = kwargs.get("normalize_token_embs", False)

        # `False` for first call but might not for the subsequent calls
        _use_sbert_model = getattr(self, "_use_sbert_model", False)

        if not _use_sbert_model:
            try:
                # this snippet is to reduce dependency on sentence-transformers library
                #   note that currently, the dependency is not fully eliminated due to backwards
                #   compatability issues in huggingface-transformers between older (python 3.6)
                #   and newer (python >=3.7) versions which needs more conditions to be implemented
                #   in `_encode_local` and hence will be addressed in future work
                # TODO: eliminate depedency on sentence-transformers library
                results = self._encode_local(phrases,
                                             batch_size=batch_size,
                                             show_progress_bar=show_progress_bar,
                                             output_value=output_value,
                                             convert_to_numpy=convert_to_numpy,
                                             convert_to_tensor=convert_to_tensor,
                                             device=device,
                                             concat_last_n_layers=concat_last_n_layers,
                                             normalize_token_embs=normalize_token_embs)
                setattr(self, "_use_sbert_model", False)
            except TypeError as e:
                logger.error(e)
                if concat_last_n_layers != 1 or normalize_token_embs:
                    msg = f"{'concat_last_n_layers,' if concat_last_n_layers != 1 else ''} " \
                          f"{'normalize_token_embs' if normalize_token_embs else ''} " \
                          f"ignored as resorting to using encode methods from sentence-transformers"
                    logger.warning(msg)
                setattr(self, "_use_sbert_model", True)

        if getattr(self, "_use_sbert_model"):
            results = self.sbert_model.encode(phrases,
                                              batch_size=batch_size,
                                              show_progress_bar=show_progress_bar,
                                              output_value=output_value,
                                              convert_to_numpy=convert_to_numpy,
                                              convert_to_tensor=convert_to_tensor,
                                              device=device)

        return results


class EntityResolverFactory:

    @staticmethod
    def _validate_resolver_type(name):
        if name not in ENTITY_RESOLVER_MODEL_TYPES:
            raise Exception(f"Expected 'resolver_type' in ENTITY_RESOLVER_CONFIG "
                            f"among {ENTITY_RESOLVER_MODEL_TYPES}")
        if name == "sbert_cosine_similarity" and not _is_module_available("sentence_transformers"):
            raise ImportError(
                "Must install the extra [bert] by running `pip install mindmeld[bert]` "
                "to use the built in embbedder for entity resolution.")

    @classmethod
    def create_resolver(cls, app_path, entity_type, **kwargs):
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

        er_config = (
            kwargs.pop("er_config", None) or
            get_classifier_config("entity_resolution", app_path=app_path)
        )
        er_config = _correct_deprecated_er_config(er_config)

        resolver_type = er_config["model_settings"]["resolver_type"]
        cls._validate_resolver_type(resolver_type)

        resource_loader = kwargs.pop(
            "resource_loader",
            ResourceLoader.create_resource_loader(app_path=app_path))

        return ENTITY_RESOLVER_MODEL_MAPPINGS.get(resolver_type)(
            app_path, entity_type, er_config, resource_loader, **kwargs
        )


class EntityResolverBase(ABC):
    """
    Base class for Entity Resolvers
    """

    def __init__(self, app_path, entity_type, resource_loader=None):
        """Initializes an entity resolver"""
        self.app_path = app_path
        self.type = entity_type
        self._resource_loader = (
            resource_loader or ResourceLoader.create_resource_loader(app_path=self.app_path)
        )

        self._is_system_entity = Entity.is_system_entity(self.type)
        self._no_trainable_canonical_entity_map = False
        self.dirty = False  # bool, True if exists any unsaved generated data that can be saved
        self.ready = False  # bool, True if the model is fit by calling .fit()

    @staticmethod
    def _process_entity_map(entity_type,
                            entity_map,
                            normalizer=None,
                            augment_lower_case=False,
                            augment_title_case=False,
                            augment_normalized=False,
                            normalize_aliases=False):
        """
        Loads in the mapping.json file and stores the synonym mappings in a item_map
            and a synonym_map

        Args:
            entity_type (str): The entity type associated with this entity resolver
            entity_map (dict): The loaded mapping.json file for the given entity type
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
        for item in entity_map.get("entities", []):
            cname = item["cname"]
            item_id = item.get("id")
            if cname in item_map:
                msg = "Canonical name %s specified in %s entity map multiple times"
                logger.debug(msg, cname, entity_type)
            if item_id and item_id in seen_ids:
                msg = "Item id {!r} specified in {!r} entity map multiple times"
                raise ValueError(msg.format(item_id, entity_type))
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
                aliases = set([*aliases, *new_aliases])
            if normalize_aliases:
                alias_normalizer = normalizer
                aliases = [alias_normalizer(alias) for alias in aliases]

            items_for_cname = item_map.get(cname, [])
            items_for_cname.append(item)
            item_map[cname] = items_for_cname
            for alias in aliases:
                if alias in syn_map:
                    msg = "Synonym %s specified in %s entity map multiple times"
                    logger.debug(msg, cname, entity_type)
                cnames_for_syn = syn_map.get(alias, [])
                cnames_for_syn.append(cname)
                syn_map[alias] = list(set(cnames_for_syn))

        return {"items": item_map, "synonyms": syn_map}

    def _load_entity_map(self, force_reload=False):
        return self._resource_loader.get_entity_map(self.type, force_reload=force_reload)

    @abstractmethod
    def _fit(self, clean, entity_map):
        """Fits the entity resolver model

        Args:
            clean (bool): If ``True``, deletes and recreates the index from scratch instead of
                            updating the existing index with synonyms in the mapping.json.
            entity_map (json): json data loaded from `mapping.json` file for the entity type
        """
        raise NotImplementedError

    def fit(self, clean=False):
        """Fits the resolver model, if required

        Args:
            clean (bool, optional): If ``True``, deletes and recreates the index from scratch
                                    with synonyms in the mapping.json.
        """

        msg = f"Fitting {self.__class__.__name__} entity resolver for entity_type {self.type}"
        logger.info(msg)

        if (not clean) and self.ready:
            return

        if self._is_system_entity:
            self._no_trainable_canonical_entity_map = True
            self.ready = True
            return

        # load data: list of canonical entities and their synonyms
        entity_map = self._load_entity_map()
        if not entity_map.get("entities", []):
            self._no_trainable_canonical_entity_map = True
            self.ready = True
            return

        self._fit(clean, entity_map)
        self.ready = True
        return

    @abstractmethod
    def _predict(self, nbest_entities, top_n):
        raise NotImplementedError

    def predict(self, entity, top_n: int = 20):
        """Predicts the resolved value(s) for the given entity using the loaded entity map or the
        trained entity resolution model.

        Args:
            entity (Entity, tuple[Entity], str, tuple[str]): An entity found in an input query,
                                                                or a list of n-best entity objects.
            top_n (int): maximum number of results to populate

        Returns:
            (list): The top n resolved values for the provided entity.
        """

        if not self.ready:
            msg = "Resolver not ready, model must be built (.fit()) or loaded (.load()) first."
            logger.error(msg)

        nbest_entities = entity
        if not isinstance(nbest_entities, (list, tuple)):
            nbest_entities = tuple([nbest_entities])

        nbest_entities = tuple(
            [Entity(e, self.type) if isinstance(e, str) else e for e in nbest_entities]
        )
        top_entity = nbest_entities[0]

        if self._is_system_entity:
            # system entities are already resolved
            return [top_entity.value]

        if self._no_trainable_canonical_entity_map:
            return []

        results = self._predict(nbest_entities, top_n)

        if not results:
            return None

        results = results[:top_n]
        if len(results) < top_n:
            logger.info(
                "Retrieved only %d entity resolutions instead of asked number %d for "
                "entity %r of type %r",
                len(results), top_n, nbest_entities[0].text, self.type,
            )

        return results

    @abstractmethod
    def _load(self):
        raise NotImplementedError

    def load(self):
        """If available, loads embeddings of synonyms that are previously dumped
        """
        self._load()

    def __repr__(self):
        msg = "<{} ready: {!r}, dirty: {!r}>"
        return msg.format(self.__class__.__name__, self.ready, self.dirty)


class ElasticsearchEntityResolver(EntityResolverBase):
    """
    Resolver class based on Elastic Search
    """

    # prefix for Elasticsearch indices used to store synonyms for entity resolution
    ES_SYNONYM_INDEX_PREFIX = "synonym"
    """The prefix of the ES index."""

    def __init__(self, app_path, entity_type, er_config, resource_loader, **kwargs):
        super().__init__(app_path, entity_type, resource_loader=resource_loader)

        self._es_host = kwargs.get("es_host", None)
        self._es_config = {"client": kwargs.get("es_client", None), "pid": os.getpid()}
        self._use_double_metaphone = "double_metaphone" in (
            er_config.get("model_settings", {}).get("phonetic_match_types", [])
        )

        self._app_namespace = get_app_namespace(self.app_path)

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

    def _predict(self, nbest_entities, top_n):
        """Predicts the resolved value(s) for the given entity using the loaded entity map or the
        trained entity resolution model.

        Args:
            nbest_entities (tuple): List of one entity object found in an input query, or a list  \
                of n-best entity objects.

        Returns:
            (list): The resolved values for the provided entity.
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

            return results

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


class ExactMatchEntityResolver(EntityResolverBase):
    """
    Resolver class based on exact matching
    """

    def __init__(self, app_path, entity_type, er_config, resource_loader, **_kwargs):
        super().__init__(app_path, entity_type, resource_loader=resource_loader)

        self._augment_lower_case = er_config.get(
            "model_settings", {}
        ).get("augment_lower_case", False)
        self._processed_entity_map = None

    def _fit(self, clean, entity_map):

        if clean:
            logger.info(
                "clean=True ignored while fitting ExactMatchEntityResolver"
            )

        self._processed_entity_map = self._process_entity_map(
            self.type,
            entity_map,
            normalizer=self._resource_loader.query_factory.normalize,
            augment_lower_case=self._augment_lower_case,
            normalize_aliases=True
        )

    def _predict(self, nbest_entities, top_n):
        """Looks for exact name in the synonyms data
        """

        entity = nbest_entities[0]  # top_entity

        normed = self._resource_loader.query_factory.normalize(entity.text)
        try:
            cnames = self._processed_entity_map["synonyms"][normed]
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
            for item in self._processed_entity_map["items"][cname]:
                item_value = copy.copy(item)
                item_value.pop("whitelist", None)
                values.append(item_value)

        return values

    def _load(self):
        self.fit()


class SentenceBertCosSimEntityResolver(EntityResolverBase, BertEmbedder):
    """
    Resolver class for bert models as described here:
    https://github.com/UKPLab/sentence-transformers
    """

    def __init__(self, app_path, entity_type, er_config, resource_loader, **_kwargs):
        super().__init__(app_path, entity_type, resource_loader=resource_loader)

        # default configs useful for reusing model's encodings through a cache path
        for key, value in self.default_er_config.get("model_settings", {}).items():
            er_config["model_settings"][key] = value

        self.batch_size = er_config["model_settings"]["batch_size"]
        _model_configs = {
            "pretrained_name_or_abspath": er_config["model_settings"]["pretrained_name_or_abspath"],
            "bert_output_type": er_config["model_settings"]["bert_output_type"],
            "quantize_model": er_config["model_settings"]["quantize_model"],
        }
        self._runtime_configs = {
            "concat_last_n_layers": er_config["model_settings"]["concat_last_n_layers"],
            "normalize_token_embs": er_config["model_settings"]["normalize_token_embs"],
            "augment_lower_case": er_config["model_settings"]["augment_lower_case"],
            "augment_average_synonyms_embeddings":
                er_config["model_settings"]["augment_average_synonyms_embeddings"],
        }
        self.cache_path = self.get_cache_path(
            app_path=self.app_path,
            er_config={**_model_configs, **self._runtime_configs},
            entity_type=self.type
        )

        self._processed_entity_map = None
        self._synonyms = None
        self._synonyms_embs = None

        self._init_sentence_transformers_encoder(_model_configs)

    @property
    def default_er_config(self):
        defaults = {
            "model_settings": {
                "pretrained_name_or_abspath": "distilbert-base-nli-stsb-mean-tokens",
                "batch_size": 16,
                "concat_last_n_layers": 4,
                "normalize_token_embs": True,
                "bert_output_type": "mean",
                "augment_lower_case": False,
                "quantize_model": True,
                "augment_average_synonyms_embeddings": True
            }
        }
        return defaults

    @staticmethod
    def get_cache_path(app_path, er_config, entity_type):
        """Obtains and return a unique cache path for saving synonyms' embeddings

        Args:
               er_config: the er_config dictionary of the reolver class
               entity_type: entity type of the class instance, for unique path identification

        Return:
            str: path with a .pkl extension to cache embeddings
        """
        string = json.dumps(er_config, sort_keys=True)
        hashid = Hasher(algorithm="sha1").hash(string=string)
        hashid = f"{hashid}$synonym_{entity_type}"

        return path.get_entity_resolver_cache_file_path(app_path, hashid)

    @staticmethod
    def _compute_cosine_similarity(synonyms,
                                   synonyms_encodings,
                                   entity_emb,
                                   top_n,
                                   return_as_dict=False):
        """Uses cosine similarity metric on synonym embeddings to sort most relevant ones
            for entity resolution

        Args:
            synonyms (dict): a dict of synonym and its corresponding embedding's row index
                                in synonyms_encodings
            synonyms_encodings (np.array): a 2d array of embedding of the synonyms; an array of
                                            size equal to number of synonyms
            entity_emb (np.array): a 2d array of embedding of the input entity text(s)
            top_n (int): maximum number of results to populate
        Returns:
            Union[dict, list[tuple]]: if return_as_dict, returns a dictionary of synonyms and their
                                        scores, else a list of sorted synonym names, paired with
                                        their similarity scores (descending)
        """

        n_entities = len(entity_emb)

        is_single = n_entities == 1

        # [n_syns, emd_dim] -> [n_entities, n_syns, emd_dim]
        t_syn_enc = _torch("as_tensor", synonyms_encodings)
        t_syn_enc = t_syn_enc.expand([n_entities, *t_syn_enc.shape])

        # [n_entities, emd_dim] -> [n_entities, n_syns, emd_dim]
        t_entity_emb = _torch("as_tensor", entity_emb)
        t_entity_emb = t_entity_emb.unsqueeze(dim=1).expand_as(t_syn_enc)

        # returns -> [n_entities, n_syns]
        similarity_scores_2d = _torch(
            "cosine_similarity", t_syn_enc, t_entity_emb, dim=-1).numpy()

        results = []
        for similarity_scores in similarity_scores_2d:
            similarity_scores = similarity_scores.reshape(-1)
            similarity_scores = np.around(similarity_scores, decimals=2)

            if return_as_dict:
                results.append(dict(zip(synonyms.keys(), similarity_scores)))
            else:
                # results in descending scores
                n_scores = len(similarity_scores)
                if n_scores > top_n:
                    top_inds = similarity_scores.argpartition(n_scores - top_n)[-top_n:]
                    result = sorted(
                        zip(np.asarray([*synonyms.keys()])[top_inds], similarity_scores[top_inds]),
                        key=lambda x: x[1], reverse=True)
                else:
                    result = sorted(zip(synonyms.keys(), similarity_scores), key=lambda x: x[1],
                                    reverse=True)
                results.append(result)

        if is_single:
            return results[0]

        return results

    def _fit(self, clean, entity_map):

        if clean and os.path.exists(self.cache_path):
            os.remove(self.cache_path)

        # load mapping.json data and process it
        augment_lower_case = self._runtime_configs["augment_lower_case"]
        self._processed_entity_map = self._process_entity_map(
            self.type,
            entity_map,
            augment_lower_case=augment_lower_case
        )

        # load embeddings from cache if exists, encode any other synonyms if required
        synonyms, synonyms_embs = OrderedDict(), np.empty(0)
        if os.path.exists(self.cache_path):
            logger.info("Cached embs exists for entity %s. "
                        "Loading existing data from: %s",
                        self.type, self.cache_path)
            cached_data = self._load_embeddings(self.cache_path)
            synonyms, synonyms_embs = cached_data["synonyms"], cached_data["synonyms_embs"]
        new_synonyms_to_encode = [syn for syn in self._processed_entity_map["synonyms"] if
                                  syn not in synonyms]
        if new_synonyms_to_encode:
            new_synonyms_encodings = (
                self.encode(
                    new_synonyms_to_encode,
                    batch_size=self.batch_size,
                    concat_last_n_layers=self._runtime_configs["concat_last_n_layers"],
                    normalize_token_embs=self._runtime_configs["normalize_token_embs"],
                )
            )
            synonyms_embs = new_synonyms_encodings if not synonyms else np.concatenate(
                [synonyms_embs, new_synonyms_encodings])
            synonyms.update(
                OrderedDict(zip(
                    new_synonyms_to_encode,
                    np.arange(len(synonyms), len(synonyms) + len(new_synonyms_to_encode)))
                )
            )

        # encode artificial synonyms if required
        if self._runtime_configs["augment_average_synonyms_embeddings"]:
            # obtain cnames to synonyms mapping
            entity_mapping_synonyms = self._processed_entity_map["synonyms"]
            cnames2synonyms = {}
            for syn, cnames in entity_mapping_synonyms.items():
                for cname in cnames:
                    items = cnames2synonyms.get(cname, [])
                    items.append(syn)
                    cnames2synonyms[cname] = items
            dummy_new_synonyms_to_encode, dummy_new_synonyms_encodings = [], []
            # assert dummy synonyms
            for cname, syns in cnames2synonyms.items():
                dummy_synonym = f"{cname} - SYNONYMS AVERAGE"
                # update synonyms map 'cause such synonyms don't actually exist in mapping.json file
                dummy_synonym_mappings = entity_mapping_synonyms.get(dummy_synonym, [])
                dummy_synonym_mappings.append(cname)
                entity_mapping_synonyms[dummy_synonym] = dummy_synonym_mappings
                # check if needs to be encoded
                if dummy_synonym in synonyms:
                    continue
                # if required, obtain dummy encoding and update collections
                dummy_encoding = np.mean([synonyms_embs[synonyms[syn]] for syn in syns], axis=0)
                dummy_new_synonyms_to_encode.append(dummy_synonym)
                dummy_new_synonyms_encodings.append(dummy_encoding)
            if dummy_new_synonyms_encodings:
                dummy_new_synonyms_encodings = np.vstack(dummy_new_synonyms_encodings)
            if dummy_new_synonyms_to_encode:
                synonyms_embs = dummy_new_synonyms_encodings if not synonyms else np.concatenate(
                    [synonyms_embs, dummy_new_synonyms_encodings])
                synonyms.update(
                    OrderedDict(zip(
                        dummy_new_synonyms_to_encode,
                        np.arange(len(synonyms), len(synonyms) + len(dummy_new_synonyms_to_encode)))
                    )
                )

        # dump embeddings if required
        self._synonyms, self._synonyms_embs = synonyms, synonyms_embs
        do_dump = (
            new_synonyms_to_encode or
            dummy_new_synonyms_to_encode or
            not os.path.exists(self.cache_path)
        )
        if do_dump:
            data_dump = {"synonyms": self._synonyms, "synonyms_embs": self._synonyms_embs}
            self._dump_embeddings(self.cache_path, data_dump)
        self.dirty = False  # never True with the current logic, kept for consistency purpose

    def _predict(self, nbest_entities, top_n):
        """Predicts the resolved value(s) for the given entity using cosine similarity.

        Args:
            nbest_entities (tuple): List of one entity object found in an input query, or a list  \
                of n-best entity objects.

        Returns:
            (list): The resolved values for the provided entity.
        """

        synonyms, synonyms_encodings = self._synonyms, self._synonyms_embs

        # encode input entity
        # TODO: Use all provided entities (i.e all nbest_entities) like elastic search
        top_entity = nbest_entities[0]  # top_entity
        existing_index = synonyms.get(top_entity.text, None)
        if existing_index:
            top_entity_emb = synonyms_encodings[existing_index]
        else:
            top_entity_emb = (
                self.encode(
                    top_entity.text,
                    concat_last_n_layers=self._runtime_configs["concat_last_n_layers"],
                    normalize_token_embs=self._runtime_configs["normalize_token_embs"],
                )
            )
        top_entity_emb = top_entity_emb.reshape(1, -1)

        try:
            sorted_items = self._compute_cosine_similarity(
                synonyms, synonyms_encodings, top_entity_emb, top_n
            )
            values = []
            for synonym, score in sorted_items:
                cnames = self._processed_entity_map["synonyms"][synonym]
                for cname in cnames:
                    for item in self._processed_entity_map["items"][cname]:
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

        return values

    def _load(self):
        self.fit()

    @staticmethod
    def _load_embeddings(cache_path):
        """Loads embeddings for all synonyms, previously dumped into a .pkl file
        """
        with open(cache_path, "rb") as fp:
            _cached_embs = pickle.load(fp)
        return _cached_embs

    def _dump_embeddings(self, cache_path, data):
        """Dumps embeddings of synonyms into a .pkl file when the .fit() method is called
        """
        msg = f"bert embeddings are are being cached for entity_type: `{self.type}` " \
              f"for quicker entity resolution; consumes some disk space"
        logger.info(msg)

        folder = os.path.split(cache_path)[0]
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        with open(cache_path, "wb") as fp:
            pickle.dump(data, fp)

    def _predict_batch(self, nbest_entities_list, batch_size, top_n):
        synonyms, synonyms_encodings = self._synonyms, self._synonyms_embs

        # encode input entity
        top_entity_list = [i[0] for i in nbest_entities_list]  # top_entity
        # called a list but observed as a list
        top_entity_emb_list = []
        for st_idx in trange(0, len(top_entity_list), batch_size, disable=False):
            batch = [top_entity.text for top_entity in top_entity_list[st_idx:st_idx + batch_size]]
            top_entity_emb_list.append(
                self.encode(
                    batch,
                    show_progress_bar=False,
                    batch_size=self.batch_size,
                    concat_last_n_layers=self._runtime_configs["concat_last_n_layers"],
                    normalize_token_embs=self._runtime_configs["normalize_token_embs"],
                )
            )
        top_entity_emb_list = np.vstack(top_entity_emb_list)

        try:
            # w/o batch,  [ nsyms x 768*4 ] x [ 1 x 768*4 ] --> [ nsyms x 1 ]
            # w/  batch,  [ nsyms x 768*4 ] x [ k x 768*4 ] --> [ nsyms x k ]
            sorted_items_list = []
            for st_idx in trange(0, len(top_entity_emb_list), batch_size, disable=False):
                batch = top_entity_emb_list[st_idx:st_idx + batch_size]
                result = self._compute_cosine_similarity(synonyms, synonyms_encodings, batch, top_n)
                # due to way compute similarity returns
                if len(batch) == 1:
                    result = [result]
                sorted_items_list.extend(result)

            values_list = []

            for sorted_items in sorted_items_list:
                values = []
                for synonym, score in sorted_items:
                    cnames = self._processed_entity_map["synonyms"][synonym]
                    for cname in cnames:
                        for item in self._processed_entity_map["items"][cname]:
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

    def predict_batch(self, entity_list, top_n: int = 20, batch_size: int = 16):

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

        results_list = self._predict_batch(nbest_entities_list, batch_size, top_n)

        for i, results in enumerate(results_list):
            if results:
                results_list[i] = results[:top_n]

        return results_list


class TfIdfSparseCosSimEntityResolver(EntityResolverBase):
    """
    a tf-idf based entity resolver using sparse matrices. ref:
    scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    """

    def __init__(self, app_path, entity_type, er_config, resource_loader, **_kwargs):
        super().__init__(app_path, entity_type, resource_loader=resource_loader)

        self._aug_lower_case = er_config.get("model_settings", {}).get("augment_lower_case", True)
        self._aug_title_case = er_config.get("model_settings", {}).get("augment_title_case", False)
        self._aug_normalized = er_config.get("model_settings", {}).get("augment_normalized", False)
        self._aug_max_syn_embs = (
            er_config.get("model_settings", {}).get("augment_max_synonyms_embeddings", True)
        )

        self._processed_entity_map = None
        self.ngram_length = 5  # max number of character ngrams to consider
        self._vectorizer = \
            TfidfVectorizer(analyzer=self._char_ngram_and_word_analyzer, lowercase=False)
        self._syn_tfidf_matrix = None
        self._unique_synonyms = []

    def _char_ngram_and_word_analyzer(self, string):
        results = self._char_ngram_analyzer(string)
        # add words
        words = re.split(r'[\s{}]+'.format(re.escape(punctuation)), string.strip())
        results.extend(words)
        return results

    def _char_ngram_analyzer(self, string):
        results = []
        # give more importance to starting and ending characters of a word
        string = f" {string.strip()} "
        for n in range(self.ngram_length + 1):
            results.extend([''.join(gram) for gram in zip(*[string[i:] for i in range(n)])])
        results = list(set(results))
        results.remove(' ')
        # adding lowercased single characters might add more noise
        results = [r for r in results if not (len(r) == 1 and r.islower())]
        return results

    def _fit(self, clean, entity_map):

        if clean:
            logger.info(
                "clean=True ignored while fitting tf-idf algo for entity resolution"
            )

        # load mappings.json data
        self._processed_entity_map = self._process_entity_map(
            self.type,
            entity_map,
            normalizer=self._resource_loader.query_factory.normalize,
            augment_lower_case=self._aug_lower_case,
            augment_title_case=self._aug_title_case,
            augment_normalized=self._aug_normalized,
        )

        # obtain sparse matrix
        synonyms = {v: k for k, v in
                    dict(enumerate(set(self._processed_entity_map["synonyms"]))).items()}
        synonyms_embs = self._vectorizer.fit_transform([*synonyms.keys()])

        # encode artificial synonyms if required
        if self._aug_max_syn_embs:
            # obtain cnames to synonyms mapping
            entity_mapping_synonyms = self._processed_entity_map["synonyms"]
            cnames2synonyms = {}
            for syn, cnames in entity_mapping_synonyms.items():
                for cname in cnames:
                    items = cnames2synonyms.get(cname, [])
                    items.append(syn)
                    cnames2synonyms[cname] = items
            dummy_new_synonyms_to_encode, dummy_new_synonyms_encodings = [], []
            # assert dummy synonyms
            for cname, syns in cnames2synonyms.items():
                dummy_synonym = f"{cname} - SYNONYMS AVERAGE"
                # update synonyms map 'cause such synonyms don't actually exist in mapping.json file
                dummy_synonym_mappings = entity_mapping_synonyms.get(dummy_synonym, [])
                dummy_synonym_mappings.append(cname)
                entity_mapping_synonyms[dummy_synonym] = dummy_synonym_mappings
                # check if needs to be encoded
                if dummy_synonym in synonyms:
                    continue
                # if required, obtain dummy encoding and update collections
                dummy_encoding = scipy.sparse.csr_matrix(
                    np.max([synonyms_embs[synonyms[syn]].toarray() for syn in syns], axis=0)
                )
                dummy_new_synonyms_to_encode.append(dummy_synonym)
                dummy_new_synonyms_encodings.append(dummy_encoding)
            if dummy_new_synonyms_encodings:
                dummy_new_synonyms_encodings = scipy.sparse.vstack(dummy_new_synonyms_encodings)
            if dummy_new_synonyms_to_encode:
                synonyms_embs = (
                    dummy_new_synonyms_encodings if not synonyms else scipy.sparse.vstack(
                        [synonyms_embs, dummy_new_synonyms_encodings])
                )
                synonyms.update(
                    OrderedDict(zip(
                        dummy_new_synonyms_to_encode,
                        np.arange(len(synonyms), len(synonyms) + len(dummy_new_synonyms_to_encode)))
                    )
                )

        # returns a sparse matrix
        self._unique_synonyms = [*synonyms.keys()]
        self._syn_tfidf_matrix = synonyms_embs

    def _predict(self, nbest_entities, top_n):
        """Predicts the resolved value(s) for the given entity using cosine similarity.

        Args:
            nbest_entities (tuple): List of one entity object found in an input query, or a list  \
                of n-best entity objects.

        Returns:
            (list): The resolved values for the provided entity.
        """

        # encode input entity
        # TODO: Use all provided entities (i.e all nbest_entities) like elastic search
        top_entity = nbest_entities[0]  # top_entity
        top_entity_vector = self._vectorizer.transform([top_entity.text])

        similarity_scores = self._syn_tfidf_matrix.dot(top_entity_vector.T).toarray().reshape(-1)
        # Rounding sometimes helps to bring correct answers on to the top score as other
        # non-correct resolutions
        similarity_scores = np.around(similarity_scores, decimals=4)
        sorted_items = sorted(list(zip(self._unique_synonyms, similarity_scores)),
                              key=lambda x: x[1], reverse=True)

        try:
            values = []
            for synonym, score in sorted_items:
                cnames = self._processed_entity_map["synonyms"][synonym]
                for cname in cnames:
                    for item in self._processed_entity_map["items"][cname]:
                        item_value = copy.copy(item)
                        item_value.pop("whitelist", None)
                        item_value.update({"score": score})
                        item_value.update({"top_synonym": synonym})
                        values.append(item_value)
        except (TypeError, KeyError):
            logger.warning(
                "Failed to resolve entity %r for type %r", top_entity.text, top_entity.type
            )
            return None

        return values

    def _load(self):
        self.fit()


class EntityResolver:
    """
    for backwards compatability

    deprecated usage
        >>> entity_resolver = EntityResolver(
                app_path, self.resource_loader, entity_type
            )

    new usage
        >>> entity_resolver = EntityResolverFactory.create_resolver(
                app_path, entity_type
            )
        # or ...
        >>> entity_resolver = EntityResolverFactory.create_resolver(
                app_path, entity_type, resource_loader=self.resource_loader
            )
    """

    def __new__(cls, app_path, resource_loader, entity_type, es_host=None, es_client=None):
        logger.warning(
            "DeprecationWarning: Entity Resolver should now be loaded using EntityResolverFactory. "
            "See https://www.mindmeld.com/docs/userguide/entity_resolver.html for more details.")
        return EntityResolverFactory.create_resolver(
            app_path, entity_type, resource_loader=resource_loader,
            es_host=es_host, es_client=es_client
        )


ENTITY_RESOLVER_MODEL_MAPPINGS = {
    "exact_match": ExactMatchEntityResolver,
    "text_relevance": ElasticsearchEntityResolver,
    "sbert_cosine_similarity": SentenceBertCosSimEntityResolver,
    "tfidf_cosine_similarity": TfIdfSparseCosSimEntityResolver
}
ENTITY_RESOLVER_MODEL_TYPES = [*ENTITY_RESOLVER_MODEL_MAPPINGS]
