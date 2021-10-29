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
This module contains the embedder model class.
"""
import json
import logging
import os
import pickle
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Any, Callable

import numpy as np
from tqdm.autonotebook import trange

from ._util import _is_module_available, _get_module_or_attr as _getattr, torch_op
from .helpers import register_embedder
from .taggers.embeddings import WordSequenceEmbedding
from .. import path
from ..core import Bunch
from ..resource_loader import Hasher
from ..text_preparation.text_preparation_pipeline import TextPreparationPipelineFactory

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """
    Base class for embedder model
    """

    class EmbeddingsCache:

        def __init__(self, cache_path=None):
            """
            Args:
                cache_path: A .pkl cache path to dump the embeddings cache
            """
            self.reset()
            if cache_path:
                self.load(self._get_cache_path(cache_path=cache_path))

        def reset(self):
            self.data = OrderedDict()

        def load(self, cache_path=None):
            """Loads the cache file."""

            cache_path = self._get_cache_path(cache_path)

            if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
                with open(cache_path, "rb") as fp:
                    data = pickle.load(fp)
                    fp.close()

                if (
                    "_texts" in data
                    and "_texts_embeddings" in data
                    and isinstance(data["_texts"], list)
                ):  # new format;
                    self.data = dict(zip(data["_texts"], data["_texts_embeddings"]))

                elif (
                    "synonyms" in data
                    and "synonyms_embs" in data
                    and isinstance(data["synonyms"], dict)
                ):  # deprecated format; backwards compatible with ER module code
                    self.data = {key: data["synonyms_embs"][j] for key, j in data["synonyms"]}

                else:  # deprecated format; backwards compatible with QA module code
                    if not isinstance(data, dict):
                        msg = "Unknown data format while loading cache embeddings. " \
                              "Ignoring loading ..."
                        logger.error(msg)
                    self.data = data

        def clear(self, cache_path=None):
            """Deletes the cache file."""

            cache_path = self._get_cache_path(cache_path)

            if os.path.exists(cache_path):
                os.remove(cache_path)
                msg = f"Embedder cache cleared at {cache_path}"
                logger.info(msg)

        def dump(self, cache_path=None):
            """Dumps the cache to disk."""

            cache_path = self._get_cache_path(cache_path)

            if self.data:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                data = {
                    "_texts": [*self.data.keys()],
                    "_texts_embeddings": np.array([*self.data.values()])
                }
                with open(cache_path, "wb") as fp:
                    pickle.dump(data, fp)
                    fp.close()

                msg = f"Embedder cache dumped at {cache_path}"
                logger.info(msg)

            else:
                msg = "No embedding data exists to dump. Ignoring dumping."
                logger.warning(msg)

        def get(self, text, default=None):
            return self.__getitem__(text, default)

        def _get_cache_path(self, cache_path):
            if not cache_path:
                msg = f"Invalid cache path '({cache_path})' provided for {self.__class__.__name__}."
                raise ValueError(msg)
            return os.path.abspath(cache_path)

        def __contains__(self, text):
            if text in self.data:
                return True
            return False

        def __getitem__(self, text, default=None):
            return self.data.get(text, default)

        def __setitem__(self, text, encoding):
            self.data[text] = encoding

        def __delitem__(self, text):
            try:
                del self.data[text]
            except KeyError as e:
                logger.error(e)
                pass

        def __iter__(self):
            # zip texts and encodings into a dictionary for iteration
            if self.data:
                return iter(self.data)

        def __len__(self):
            return len(self.data)

    def __init__(self, app_path=None, cache_path=None, **kwargs):
        """
        Initializes an embedder. The instantiated embedder model maintains a cache object that has
        embeddings of inputs observed so far through the .get_encodings() method. This cache can be
        useful especially if obtaining embeddings for the same text input is costlier in time versus
        a lookup.

        Args:
            app_path (str): Path of the app used to create cache folder to dump encodings
            cache_path (str): A .pkl path where the embeddings are to be cached. If provided,
                discards the app_path information.
        """

        # load embedder model
        self.model = self.load()

        # obtain a cache path for creating an embedder cache object
        if cache_path is None:
            if app_path:
                deprecated_cache_path = path.get_embedder_cache_file_path(
                    app_path,
                    kwargs.get("embedder_type", "default"),
                    kwargs.get("model_name", "default")
                )
                if (
                    os.path.exists(deprecated_cache_path) and
                    os.path.getsize(deprecated_cache_path) > 0
                ):
                    # deprecated usage:
                    #   Determine path from `embedder_type` and `model_name`
                    #   Inside a Mindmeld app, this path is generally something like:
                    #       '.generated/indexes/{embedder_type}_{model_name}_cache.pkl'
                    cache_path = deprecated_cache_path
                    msg = f"Found a deprecated cache path at '{cache_path}' that contains " \
                          f"embeddings for a default configuration of embedder models. " \
                          f"If you wish to use mindmeld version greater than 4.3.4 to work with " \
                          f"non-default embedder configurations, consider deleting this cache " \
                          f"path manually and run again."
                    logger.warning(msg)
                else:
                    # new usage:
                    #   Determine cache path for the model using `model_id`
                    #   Implies a previously used path name has no data and hence, is safe to change
                    #   default cache path for this model (backwards compatibility required only for
                    #   loading previously dumped embeddings data).
                    #   Cannot use previous path template because `model_name` alone is not
                    #   sufficient to uniquely identify a bert model (as it can be configured now).
                    #   Inside a Mindmeld app, this path is generally something like:
                    #       '.generated/indexes/{model_id}_cache.pkl'
                    cache_path = path.get_embedder_cache_file_path(
                        app_path,
                        self.model_id
                    )
            else:
                msg = f"{self.__class__.__name__} embedder instantiated without a valid cache " \
                      f"path. This will lead to an error if you try to dump the encodings cache. " \
                      f"To have a valid cache dump location, pass-in 'app_path' or 'cache_path' " \
                      f"argument. Alternatively, the `cache_path` can also be passed to the dump " \
                      f"and load methods directly."
                logger.info(msg)

        # load embedder cache object
        self.cache_path = cache_path
        self.cache = Embedder.EmbeddingsCache(self.cache_path)

    @property
    def model_id(self):
        """Returns a unique hash representation of the embedder model based on its name and configs
        """
        msg = "Embedder models need to have model ids to uniquely identify each model " \
              "associated with a specific configuration. It can be set through the property " \
              "setter 'model_id'. If unspecified, a default value ('default') is used instead."
        logger.warning(msg)
        return "default"

    @abstractmethod
    def load(self, **kwargs):
        """Loads the embedder model

        Returns:
            The model object.
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self, text_list):
        """
        Args:
            text_list (list): A list of text strings for which to generate the embeddings.

        Returns:
            (list): A list of numpy arrays of the embeddings.
        """
        raise NotImplementedError

    def get_encodings(self, text_list, add_to_cache=True) -> List[Any]:
        """
        Fetches the encoded values from the cache, or generates them and adds to cache unless
        add_to_cache is set to False. This method is wrapped around .encode() by maintaining an
        embedding cache.

        Args:
            text_list (list): A list of text strings for which to get the embeddings.
            add_to_cache (bool): If True, adds the encodings to self.cache and returns embeddings

        Returns:
            (list): A list of numpy arrays with the embeddings.
        """

        uniques_text_list, uniques = [], {}
        text_list_to_uniques_text_list_map = []
        for text in text_list:
            if text not in uniques:
                uniques[text] = len(uniques)
                uniques_text_list.append(text)
            text_list_to_uniques_text_list_map.append(uniques[text])

        encoded = [self.cache.get(text, None) for text in uniques_text_list]
        cache_miss_indices = [i for i, vec in enumerate(encoded) if vec is None]
        text_to_encode = [uniques_text_list[i] for i in cache_miss_indices]
        model_encoded_text = self.encode(text_to_encode)

        for i, v in enumerate(cache_miss_indices):
            encoded[v] = model_encoded_text[i]
            if add_to_cache:
                self.cache[text_to_encode[i]] = model_encoded_text[i]

        return [encoded[text_list_to_uniques_text_list_map[i]] for i, text in enumerate(text_list)]

    def add_to_cache(self, mean_or_max_pooled_whitelist_embs):
        """
        Method to add custom embeddings to cache without triggering `.encode()`. Example, one can
        manually add some max-pooled or mean-pooled embeddings to cache. This method is created
        to entertain storing superficial text-encoding pairs (superficial because the encodings are
        not the encodings of the text itself but a combination of encodings of some list of texts
        from the same embedder model). For example, to add superficial entity embeddings as average
        of whitelist embeddings in Entity Resolution.

        Args:
            mean_or_max_pooled_whitelist_embs (dict): texts and their corresponding superficial
                embeddings as a 1D numpy array, having same length as emb_dim of the embedder
        """
        for key, value in mean_or_max_pooled_whitelist_embs.items():
            value = np.asarray(value).reshape(-1)
            known_emb_dim = getattr(self, "emb_dim", None)
            if known_emb_dim and not len(value) == known_emb_dim:
                msg = f"Expected superficial embedding of length {known_emb_dim} but found " \
                      f"{len(value)}. Not adding the embedding for {key} to cache."
                logger.error(msg)
            if key in self.cache:
                msg = f"Overwriting a superficial embedding for {key}"
                logger.warning(msg)
            self.cache[key] = value

    def dump_cache(self, cache_path=None):
        self.cache.dump(cache_path=cache_path or self.cache_path)

    def load_cache(self, cache_path=None):
        self.cache.load(cache_path=cache_path or self.cache_path)

    def clear_cache(self, cache_path=None):
        self.cache.clear(cache_path=cache_path or self.cache_path)

    def find_similarity(
        self,
        src_texts: List[str],
        tgt_texts: List[str] = None,
        top_n: int = 20,
        scores_normalizer: str = None,
        similarity_function: Callable[[List[Any], List[Any]], np.ndarray] = None,
        _return_as_dict=False,
        _no_sort=False
    ):
        """Computes the cosine similarity

        Args:
            src_texts (Union[str, list]): string or list of strings to obtain matching scores for.
            tgt_texts (list, optional): list of strings that will be matched to.
                if None, existing cache is used as target strings
            top_n (int, optional): maximum number of results to populate. if None, equals length
                of tgt_texts
            scores_normalizer (str, optional): normalizer type to normalize scores. Allowed values
                are: "min_max_scaler", "standard_scaler"
            similarity_function (function, optional): if None, defaults to `pytorch_cos_sim`. If
                specified, must take two numpy-array/pytorch-tensor arguments for similarity
                computation with an optional argument to return results as numpy or tensor
            _return_as_dict (bool, optional): if the results should be returned as a dictionary of
                target_text name as keys and scores as corresponding values
            _no_sort (bool, optional): If True, results are returned without sorting. This is
                helpful at times when you wish to do additional wrapper operations on top of raw
                results and would like to save computational time without sorting.
        Returns:
            Union[dict, list[tuple]]: if _return_as_dict, returns a dictionary of tgt_texts and
                their scores, else a list of tuple each consisting of a src_text paired with its
                similarity scores with all tgt_texts as a np array (sorted list in descending order)
        """

        is_single = False
        if isinstance(src_texts, str):
            is_single = True
            src_texts = [src_texts]

        tgt_texts = [*self.cache.data.keys()] if not tgt_texts else tgt_texts
        if not tgt_texts:
            msg = "The list of target texts are empty to compute similarities with the source " \
                  "text(s). This can happen if the embedder cache is empty due to an unloaded " \
                  "index or if passing in an empty list of target texts to find similarity with."
            raise ValueError(msg)
        top_n = len(tgt_texts) if not top_n else top_n
        similarity_function = similarity_function or self.pytorch_cos_sim

        src_vecs = np.asarray(self.get_encodings(list(src_texts), add_to_cache=False))
        tgt_vecs = np.asarray(self.get_encodings(list(tgt_texts), add_to_cache=False))

        similarity_scores_2d = similarity_function(src_vecs, tgt_vecs)

        results = []
        for similarity_scores in similarity_scores_2d:
            similarity_scores = similarity_scores.reshape(-1)
            # Rounding sometimes helps to bring correct answers on to the list of top scored results
            similarity_scores = np.around(similarity_scores, decimals=2)

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
                results.append(dict(zip(tgt_texts, similarity_scores)))
            else:
                if not _no_sort:  # sort results in descending scores
                    n_scores = len(similarity_scores)
                    if n_scores > top_n:
                        top_inds = similarity_scores.argpartition(n_scores - top_n)[-top_n:]
                        result = sorted(
                            [(tgt_texts[ii], similarity_scores[ii]) for ii in top_inds],
                            key=lambda x: x[1],
                            reverse=True)
                    else:
                        result = sorted(zip(tgt_texts, similarity_scores),
                                        key=lambda x: x[1],
                                        reverse=True)
                    results.append(result)
                else:
                    result = list(zip(tgt_texts, similarity_scores))
                    results.append(result)

        if is_single:
            return results[0]

        return results

    @staticmethod
    def pytorch_cos_sim(src_vecs, tgt_vecs, return_tensor=False):
        """Computes the cosine similarity for 2d matrices

        Args:
            src_vecs: a 2d numpy array or pytorch tensor
            tgt_vecs: a 2d numpy array or pytorch tensor
            return_tensor: If False, this method returns the cosine similarity as a numpy 2d array
                instead of tensor, else returns 2d tensor output
        """

        src_vecs = torch_op("as_tensor", src_vecs)
        tgt_vecs = torch_op("as_tensor", tgt_vecs)

        if len(src_vecs.shape) == 1:
            src_vecs = src_vecs.view(1, -1)

        if len(tgt_vecs.shape) == 1:
            tgt_vecs = tgt_vecs.view(1, -1)

        if len(src_vecs.shape) != 2 or len(tgt_vecs.shape) != 2:
            msg = "Only 2-dimensional arrays/tensors are allowed in Embedder.pytorch_cos_sim()"
            raise ValueError(msg)

        # method specific to 2d tensors
        # [n_src, emb_dim] * [n_tgt, emb_dim] -> [n_src, n_tgt]
        a_norm = torch_op("normalize", src_vecs, sub="nn.functional", p=2, dim=1)
        b_norm = torch_op("normalize", tgt_vecs, sub="nn.functional", p=2, dim=1)
        similarity_scores = torch_op("mm", a_norm, b_norm.transpose(0, 1))

        if not return_tensor:
            return similarity_scores.numpy()

        return similarity_scores

    @staticmethod
    def get_hashid(**kwargs):
        string = json.dumps(kwargs, sort_keys=True)
        return Hasher(algorithm="sha256").hash(string=string)

    # deprecated method, same functionality as 'dump_cache' method
    def dump(self, cache_path=None):
        msg = f"DeprecationWarning: Use {self.__class__.__name__}.dump_cache() instead of " \
              f"{self.__class__.__name__}.dump()"
        warnings.warn(msg, DeprecationWarning)
        self.dump_cache(cache_path=cache_path)


class BertEmbedder(Embedder):  # pylint: disable=too-many-instance-attributes
    """
    Encoder class for bert models based on https://github.com/UKPLab/sentence-transformers
    """

    # Class variable to cache bert models: since pretrained transformer models like BERT are
    # generally large in size, it is optimal memory-wise if we do not load one model for each
    # object of this class. This optimization is meaningful only if BERT-like models are used for
    # inference only and not for fine-tuning
    CACHE_MODELS = {}

    def __init__(self, app_path=None, cache_path=None, pretrained_name_or_abspath=None, **kwargs):
        """
        Initializes a BERT based embedder from Huggingface

        Args:
            app_path (str): Path of the app used to create cache folder to dump encodings
            cache_path (str): A .pkl path where the embeddings are to be cached. If provided,
                discards the app_path information.
            pretrained_name_or_abspath (str): name of the BERT model from huggingface models
                repository; arg to be used instead of deprecated arg model_name
            model_name (str, deprecated): name of the BERT model from huggingface models

            Optional keyword args that uniquely identify the embeddings of the model:
                bert_output_type (str): the output of BERT model to use, choices- 'mean', 'cls'
                quantize_model (str): if True, the BERT model is quantized
                concat_last_n_layers (int): num of hidden outputs to concat starting from last layer
                normalize_token_embs (bool): if the (sub-token) embs are to be normalized

            Optional keyword args that are required for run-time:
                device (str): Which torch.device to use for the computation
                batch_size (int): the batch size used for the computation
                output_value (str): Default sentence_embedding, to get sentence embeddings.
                    Can be set to token_embeddings to get wordpiece token embeddings.
                    Choices are `sentence_embedding` and `token_embedding`
                convert_to_numpy (bool): If true, the output is a list of numpy vectors. Else, it
                    is a list of pytorch tensors.
                convert_to_tensor (bool): If true, you get one large tensor as return. Overwrites
                    any setting from convert_to_numpy
        """

        # required libraries check
        if not _is_module_available("sentence_transformers") or not _is_module_available("torch"):
            raise ImportError(
                "Must install the extra [bert] by running `pip install mindmeld[bert]` "
                "to use the built in bert embedder."
            )

        # deprecated configs keys
        model_name = kwargs.get("model_name")
        if model_name:
            msg = "The argument 'model_name' is deprecated and will be removed in future " \
                  "versions. Consider replacing it with 'pretrained_name_or_abspath'"
            warnings.warn(msg, DeprecationWarning)
            if pretrained_name_or_abspath:
                msg = f"Must pass-in only one of 'pretrained_name_or_abspath' and 'model_name' " \
                      f"params while instantiating a {self.__class__.__name__} class."
                raise ValueError(msg)
            pretrained_name_or_abspath = model_name

        # configs that uniquely identify the model, used in model_id
        self.pretrained_name_or_abspath = pretrained_name_or_abspath
        if not self.pretrained_name_or_abspath:
            msg = f"A valid 'pretrained_name_or_abspath' param must be passed " \
                  f"to instantiate {self.__class__.__name__}."
            raise ValueError(msg)
        self.bert_output_type = kwargs.get("bert_output_type", "mean")
        self.quantize_model = kwargs.get("quantize_model", False)
        self.concat_last_n_layers = kwargs.get("concat_last_n_layers", 1)
        self.normalize_token_embs = kwargs.get("normalize_token_embs", False)

        # runtime configs for the embedder model
        self.device = kwargs.get(
            "device", "cuda" if torch_op("is_available", sub="cuda") else "cpu"
        )
        self._batch_size = kwargs.get("batch_size", 8)
        self._output_value = kwargs.get("output_value", 'sentence_embedding')
        self._convert_to_numpy = kwargs.get("convert_to_numpy", True)
        self._convert_to_tensor = kwargs.get("convert_to_tensor", False)
        self._show_progress_bar = (
            logger.getEffectiveLevel() == logging.INFO or
            logger.getEffectiveLevel() == logging.DEBUG
        )

        # unique id for the embedder model based on specified configurations
        self._model_id = str(self.get_hashid(
            pretrained_name_or_abspath=self.pretrained_name_or_abspath,
            bert_output_type=self.bert_output_type,
            quantize_model=self.quantize_model,
            concat_last_n_layers=self.concat_last_n_layers,
            normalize_token_embs=self.normalize_token_embs
        ))

        super().__init__(app_path=app_path, cache_path=cache_path, **kwargs)

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

    @staticmethod
    def _get_sentence_transformers_encoder(name_or_path,
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

        self.transformer_model = self.model.transformer_model
        self.pooling_model = self.model.pooling_model

        if concat_last_n_layers != 1:
            assert 1 <= concat_last_n_layers <= self._num_layers(self.transformer_model.auto_model)

        self.transformer_model.eval()
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
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

            with torch_op("no_grad"):
                out_features_transformer = self.transformer_model.forward(features)
                token_embeddings = out_features_transformer["token_embeddings"]
                if concat_last_n_layers > 1:
                    _all_layer_embs = out_features_transformer["all_layer_embeddings"]
                    token_embeddings = torch_op(
                        "cat", _all_layer_embs[-concat_last_n_layers:], dim=-1)
                if normalize_token_embs:
                    _norm_token_embeddings = torch_op(
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
            all_embeddings = torch_op("stack", all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_is_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def load(self):

        model = BertEmbedder.CACHE_MODELS.get(self._model_id, None)

        if not model:

            info_msg = ""
            for name in [
                self.pretrained_name_or_abspath,
                f"sentence-transformers/{self.pretrained_name_or_abspath}"
            ]:
                try:
                    model = (
                        self._get_sentence_transformers_encoder(name,
                                                                output_type=self.bert_output_type,
                                                                quantize=self.quantize_model,
                                                                return_components=True)
                    )
                    info_msg += f"Successfully initialized name/path `{name}` directly through " \
                                f"huggingface-transformers. "
                except OSError:
                    info_msg += f"Could not initialize name/path `{name}` directly through " \
                                f"huggingface-transformers. "

                if model:
                    break

            logger.info(info_msg)

            if not model:
                msg = f"Could not resolve the name/path `{self.pretrained_name_or_abspath}`. " \
                      f"Please check the model name and retry."
                raise Exception(msg)

            BertEmbedder.CACHE_MODELS.update({self._model_id: model})

        return model

    def encode(self, phrases):
        """Encodes input text(s) into embeddings, one vector for each phrase

        Args:
            phrases (str, list[str]): textual inputs that are to be encoded using sentence \
                                        transformers' model


        Returns:
            (Union[List[Tensor], ndarray, Tensor]): By default, a numpy array is returned.
                If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy
                matrix is returned.
        """

        if not phrases:
            return []

        show_progress_bar = (
            self._show_progress_bar
            and (len(phrases) if isinstance(phrases, list) else 1) > 1
        )

        # `False` for first call but might not for the subsequent calls
        _use_sbert_model = getattr(self, "_use_sbert_model", False)

        results = None
        if not _use_sbert_model:
            try:
                # this snippet is to reduce dependency on sentence-transformers library
                #   note that currently, the dependency is not fully eliminated due to backwards
                #   compatibility issues in huggingface-transformers between older (python 3.6)
                #   and newer (python >=3.7) versions which needs more conditions to be implemented
                #   in `_encode_local` and hence will be addressed in future work
                # TODO: eliminate depedency on sentence-transformers library
                results = self._encode_local(phrases,
                                             batch_size=self._batch_size,
                                             show_progress_bar=show_progress_bar,
                                             output_value=self._output_value,
                                             convert_to_numpy=self._convert_to_numpy,
                                             convert_to_tensor=self._convert_to_tensor,
                                             device=self.device,
                                             concat_last_n_layers=self.concat_last_n_layers,
                                             normalize_token_embs=self.normalize_token_embs)
                setattr(self, "_use_sbert_model", False)
            except TypeError as e:
                logger.error(e)
                if self.concat_last_n_layers != 1 or self.normalize_token_embs:
                    msg = f"{'concat_last_n_layers,' if self.concat_last_n_layers != 1 else ''} " \
                          f"{'normalize_token_embs' if self.normalize_token_embs else ''} " \
                          f"ignored as resorting to using encode methods from sentence-transformers"
                    logger.warning(msg)
                setattr(self, "_use_sbert_model", True)

        if getattr(self, "_use_sbert_model"):
            results = self.model.sbert_model.encode(phrases,
                                                    batch_size=self._batch_size,
                                                    show_progress_bar=show_progress_bar,
                                                    output_value=self._output_value,
                                                    convert_to_numpy=self._convert_to_numpy,
                                                    convert_to_tensor=self._convert_to_tensor,
                                                    device=self.device)

        return results

    @property
    def model_id(self):
        """Returns a unique hash representation of the embedder model based on its name and configs
        """
        return self._model_id


class GloveEmbedder(Embedder):
    """
    Encoder class for GloVe embeddings as described here: https://nlp.stanford.edu/projects/glove/
    """

    DEFAULT_EMBEDDING_DIM = 300

    def __init__(self, app_path=None, cache_path=None, **kwargs):
        """
        Initializes a GloVe embedder.

        Args:
            app_path (str): Path of the app used to create cache folder to dump encodings
            cache_path (str): A .pkl path where the embeddings are to be cached. If provided,
                discards the app_path information.

            Optional keyword args that uniquely identify the embeddings of the model:
                token_embedding_dimension (str): The token dimension of GloVe embedder to load
                token_pretrained_embedding_filepath (str): The path where GloVe embeddings are
                    available. If its None, an appropriate file will be downloaded to
                    mindmeld/data/ folder and used.
        """

        self.token_embedding_dimension = kwargs.get(
            "token_embedding_dimension", self.DEFAULT_EMBEDDING_DIM
        )
        self.token_pretrained_embedding_filepath = kwargs.get("token_pretrained_embedding_filepath")

        # Create a custom pipeline config as the default config for en language eliminates some
        # punctuations that can be required in tasks such as entity resolution.
        pipeline_config = {
            "language": "en",
            "tokenizer": "WhiteSpaceTokenizer",
            "preprocessors": [],
            "normalizers": [],
            "stemmer": None,
            "keep_special_chars": True
        }
        self.text_preparation_pipeline = (
            TextPreparationPipelineFactory.create_text_preparation_pipeline(**pipeline_config)
        )

        # unique id for the embedder model based on specified configurations
        self._model_id = str(self.get_hashid(
            token_embedding_dimension=self.token_embedding_dimension,
            token_pretrained_embedding_filepath=os.path.abspath(
                self.token_pretrained_embedding_filepath
            ) if self.token_pretrained_embedding_filepath else "default",
        ))

        super().__init__(app_path=app_path, cache_path=cache_path, **kwargs)

    def load(self):

        return WordSequenceEmbedding(
            0,
            self.token_embedding_dimension,
            self.token_pretrained_embedding_filepath,
            use_padding=False,
        )

    def encode(self, text_list):
        token_list = [self._tokenize(text) for text in text_list]
        vector_list = [self.model.encode_sequence_of_tokens(tl) for tl in token_list]
        encoded_vecs = []
        for vl in vector_list:
            if len(vl) == 1:
                encoded_vecs.append(vl[0])
            else:
                encoded_vecs.append(np.average(vl, axis=0))
        return encoded_vecs

    def _tokenize(self, text):
        return [
            t["entity"] for t in
            self.text_preparation_pipeline.tokenize_and_normalize(text)
        ]

    def dump(self, cache_path=None):
        """Dumps the cache to disk."""
        super().dump(cache_path=cache_path)
        self.model.save_embeddings()

    @property
    def model_id(self):
        """Returns a unique hash representation of the embedder model based on its name and configs
        """
        return self._model_id


if _is_module_available("sentence_transformers"):
    register_embedder("bert", BertEmbedder)

register_embedder("glove", GloveEmbedder)
