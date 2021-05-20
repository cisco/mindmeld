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
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
from tqdm.autonotebook import trange

from ._util import _is_module_available, _get_module_or_attr as _getattr
from .helpers import register_embedder
from .taggers.embeddings import WordSequenceEmbedding
from .. import path
from ..core import Bunch
from ..resource_loader import Hasher
from ..tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def _torch(op, *args, sub="", **kwargs):
    """
    A safe function caller for torch module
    Usage:
        call ```_torch("normalize", arg1, arg2, sub="nn.functional", p=2, dim=1)```
        instead of ```torch.nn.functional.normalize(arg1, arg2, p=2, dim=1)```
    """

    return _getattr(f"torch{'.' + sub if sub else ''}", op)(*args, **kwargs)


class Embedder(ABC):
    """
    Base class for embedder model
    """

    class EmbeddingsCache:

        def __init__(self, cache_path):
            self.cache_path = cache_path
            self.reset_data()
            self.load()

        def reset_data(self):
            self.data = OrderedDict()

        def load(self):
            """Loads the cache file."""

            if os.path.exists(self.cache_path) and os.path.getsize(self.cache_path) > 0:
                with open(self.cache_path, "rb") as fp:
                    data = pickle.load(fp)

                if (
                    "_texts" in data and
                    "_texts_embeddings" in data and
                    isinstance(data["_texts"], list)
                ):  # new format;
                    self.data = dict(zip(data["_texts"], data["_texts_embeddings"]))

                elif (
                    "synonyms" in data and
                    "synonyms_embs" in data and
                    isinstance(data["synonyms"], dict)
                ):  # deprecated format; backwards compatible with ER module code
                    self.data = {key: data["synonyms_embs"][j] for key, j in data["synonyms"]}

                else:  # deprecated format; backwards compatible with QA module code
                    if not isinstance(data, dict):
                        msg = "Unknown data format while loading cache embeddings. " \
                              "Ignoring loading ..."
                        logger.error(msg)
                    self.data = data

        def clear(self):
            """Deletes the cache file."""

            self.reset_data()

            if os.path.exists(self.cache_path):
                os.remove(self.cache_path)
                msg = f"Embedder cache cleared at {self.cache_path}"
                logger.info(msg)

        def dump(self):
            """Dumps the cache to disk."""

            if self.data:
                folder = os.path.split(self.cache_path)[0]
                if folder and not os.path.exists(folder):
                    os.makedirs(folder)
                data = {
                    "_texts": [*self.data.keys()],
                    "_texts_embeddings": np.array([*self.data.values()])
                }
                with open(self.cache_path, "wb") as fp:
                    pickle.dump(data, fp)
                    fp.close()

                msg = f"Embedder cache dumped at {self.cache_path}"
                logger.info(msg)

            else:
                msg = "No embedding data exists to dump. Ignoring dumping."
                logger.warning(msg)

        def get(self, text, default=None):
            return self.__getitem__(text, default)

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
            len(self.data)

    def __init__(self, app_path, **kwargs):
        """
        Initializes an embedder.

        Args:
            app_path: Path of the app used to create cache folder to dump encodings
        """
        self.model_name = kwargs.get("model_name", "default")
        self.model_id = str(self.model_name)
        self.emb_dim = None

        # load model (change `emb_dim` and if required change `model_id`)
        self.model = self._load(**kwargs)

        # cache path
        cache_path = kwargs.get("cache_path", None)
        if not cache_path:
            # deprecated usage: determine path from `embedder_type` and `model_name` (eg. QA module)
            cache_path = path.get_embedder_cache_file_path(
                app_path, kwargs.get("embedder_type", "default"), self.model_name
            )
            # new usage: determine cache path for the model using model_id (eg. ER module)
            if not (os.path.exists(cache_path) and os.path.getsize(cache_path) > 0):
                # implies a previously used path name has no data and hence, is safe to change
                #   default cache path for this model (backwards compatability in loading data)
                cache_path = path.get_embedder_cache_file_path(app_path, self.model_id)
        self.cache = Embedder.EmbeddingsCache(str(cache_path))

    def __repr__(self):
        return f"{self.__class__.__name__} cache_path: {self.cache.cache_path}"

    @abstractmethod
    def _load(self, **kwargs):
        """Loads the embedder model

        Returns:
            The model object.
        """

        # backwards compatability
        if getattr(self, "load", None):
            # implies `load()` is implemented as an abstract class instead of the new way `_load()`
            getattr(self, "load")(**kwargs)

        raise NotImplementedError

    @abstractmethod
    def _encode(self, text_list):
        """
        Args:
            text_list (list): A list of text strings for which to generate the embeddings.

        Returns:
            (list): A list of numpy arrays of the embeddings.
        """

        # backwards compatability
        if getattr(self, "encode", None):
            # implies `encode()` is implemented as an abstract class instead of new way `_encode()`
            getattr(self, "encode")(text_list)

        raise NotImplementedError

    # deprecated method
    def dump(self):
        msg = f"DeprecationWarning: Use {self.__class__.__name__}.dump_cache() instead of " \
              f"{self.__class__.__name__}.dump()"
        logger.warning(msg)
        self.dump_cache()

    def get_encodings(self, text_list, add_to_cache=True):
        """Fetches the encoded values from the cache, or generates them.
        Args:
            text_list (list): A list of text strings for which to get the embeddings.
            add_to_cache (bool): If True, adds the encodings to self.cache and returns embeddings

        Returns:
            (list): A list of numpy arrays with the embeddings.
        """
        encoded = [self.cache.get(text, None) for text in text_list]
        cache_miss_indices = [i for i, vec in enumerate(encoded) if vec is None]
        text_to_encode = [text_list[i] for i in cache_miss_indices]
        model_encoded_text = self._encode(text_to_encode)

        for i, v in enumerate(cache_miss_indices):
            encoded[v] = model_encoded_text[i]
            if add_to_cache:
                self.cache[text_to_encode[i]] = model_encoded_text[i]

        return encoded

    def add_to_cache(self, mean_or_max_pooled_whitelist_embs):
        """Manually add some max-pooled or mean-pooled embeddings to cache. This method is created
        to entertain storing superficial text-encoding pairs (superficial because the encodings are
        not the encodings of the text itself but a combination of encodings of some list of texts
        from the same embedder model). For example, to add superficial entity embeddings as average
        of whitelist embeddings in Entity Resolution.

        Args:
            mean_or_max_pooled_whitelist_embs (dict): texts and their corresponding superficial
                embeddings as a 1D numpy array, having same length as self.emb_dim
        """
        for key, value in mean_or_max_pooled_whitelist_embs.items():
            value = np.asarray(value).reshape(-1)
            if self.emb_dim and not len(value) == self.emb_dim:
                msg = f"Expected superficial embedding of length {self.emb_dim} but found " \
                      f"{len(value)}. Not adding the embedding for {key} to cache."
                logger.error(msg)
            if key in self.cache:
                msg = f"Overwriting a superficial embedding for {key}"
                logger.warning(msg)
            self.cache[key] = value

    def dump_cache(self):
        self.cache.dump()

    def clear_cache(self):
        self.cache.clear()

    @staticmethod
    def _pytorch_cos_sim(src_vecs, tgt_vecs, as_numpy=False):
        """Computes the cosine similarity for 2d matrices

        Args:
            src_vecs: a 2d numpy array or pytorch tensor
            tgt_vecs: a 2d numpy array or pytorch tensor
            as_numpy: If true, returns the cosine similarity as a numpy 2d array instead of tensor
        """

        src_vecs = _torch("as_tensor", src_vecs)
        tgt_vecs = _torch("as_tensor", tgt_vecs)

        if len(src_vecs.shape) == 1:
            src_vecs = src_vecs.view(1, -1)

        if len(tgt_vecs.shape) == 1:
            tgt_vecs = tgt_vecs.view(1, -1)

        if len(src_vecs.shape) != 2 or len(tgt_vecs.shape) != 2:
            msg = "Only 2-dimensional arrays/tensors are allowed in Embedder._pytorch_cos_sim()"
            raise ValueError(msg)

        # method specific to 2d tensors
        # [n_src, emb_dim] * [n_tgt, emb_dim] -> [n_src, n_tgt]
        a_norm = _torch("normalize", src_vecs, sub="nn.functional", p=2, dim=1)
        b_norm = _torch("normalize", tgt_vecs, sub="nn.functional", p=2, dim=1)
        similarity_scores = _torch("mm", a_norm, b_norm.transpose(0, 1))

        if as_numpy:
            return similarity_scores.numpy()

        return similarity_scores

    def find_similarity(self,
                        src_texts,
                        tgt_texts=None,
                        top_n=20,
                        scores_normalizer=None,
                        _sim_func=None,
                        _return_as_dict=False,
                        _no_sort=False):
        """Computes the cosine similarity

        Args:
            src_texts (Union[str, list]): string or list of strings to obtain matching scores for.
            tgt_texts (list, optional): list of strings that will be matched to.
                if None, existing cache is used as target strings
            top_n (int, optional): maximum number of results to populate. if None, equals length
                of tgt_texts
            scores_normalizer (str, optional): normalizer type to normalize scores. Allowed values
                are: "min_max_scaler", "standard_scaler"
            _sim_func (function, optional): if None, defaults to `_pytorch_cos_sim`. If
                specified, must take two numpy-array/pytorch-tensor arguments for similarity
                computation with an optional argument to return results as numpy or tensor
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

        tgt_texts = [*self.cache.data.keys()] if not tgt_texts else tgt_texts
        top_n = len(tgt_texts) if not top_n else top_n
        _sim_func = _sim_func or self._pytorch_cos_sim

        src_vecs = np.asarray(self.get_encodings(list(src_texts), add_to_cache=False))
        tgt_vecs = np.asarray(self.get_encodings(list(tgt_texts), add_to_cache=False))

        similarity_scores_2d = _sim_func(src_vecs, tgt_vecs, as_numpy=True)  # a 2d numpy array

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


class BertEmbedder(Embedder):
    """
    Encoder class for bert models based on https://github.com/UKPLab/sentence-transformers
    """

    DEFAULT_BERT = "bert-base-nli-mean-tokens"

    # class variable to cache bert model(s);
    #   helps to mitigate keeping duplicate sets of large weight matrices, especially
    #   when creating mutliple BERT based Entity Resolvers
    CACHE_MODELS = {}

    def __init__(self, app_path, **kwargs):
        """
        Args:
            batch_size (int): the batch size used for the computation
            show_progress_bar (bool): Output a progress bar when encode sentences
            output_value (str): Default sentence_embedding, to get sentence embeddings.
                Can be set to token_embeddings to get wordpiece token embeddings.
                Choices are `sentence_embedding` and `token_embedding`
            convert_to_numpy (bool): If true, the output is a list of numpy vectors. Else, it is a
                list of pytorch tensors.
            convert_to_tensor (bool): If true, you get one large tensor as return. Overwrites any
                setting from convert_to_numpy
            device: Which torch.device to use for the computation
            concat_last_n_layers (int): number of hidden outputs to concat starting from last layer
            normalize_token_embs (bool): if the (sub-)token embs are to be individually normalized
        """
        super().__init__(app_path, **kwargs)

        # init runtime configs for the model
        self._batch_size = kwargs.get("batch_size", 8)
        self._show_progress_bar = kwargs.get("show_progress_bar", False)
        self._output_value = kwargs.get("output_value", 'sentence_embedding')
        self._convert_to_numpy = kwargs.get("convert_to_numpy", True)
        self._convert_to_tensor = kwargs.get("convert_to_tensor", False)
        self.device = kwargs.get("device", "cuda" if _torch("is_available", sub="cuda") else "cpu")
        self._concat_last_n_layers = kwargs.get("concat_last_n_layers", 1)
        self._normalize_token_embs = kwargs.get("normalize_token_embs", False)

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
    def get_hashid(**kwargs):
        string = json.dumps(kwargs, sort_keys=True)
        return Hasher(algorithm="sha1").hash(string=string)

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

    def _load(self, **kwargs):

        if not _is_module_available("sentence_transformers"):
            raise ImportError(
                "Must install the extra [bert] by running `pip install mindmeld[bert]` "
                "to use the built in bert embedder.")

        pretrained_name_or_abspath = (
            kwargs.get("pretrained_name_or_abspath", None) or
            BertEmbedder.DEFAULT_BERT if self.model_name == "default" else self.model_name
        )
        bert_output_type = kwargs.get("bert_output_type", "mean")
        quantize_model = kwargs.get("quantize_model", False)

        model_hashid = self.get_hashid(pretrained_name_or_abspath=pretrained_name_or_abspath,
                                       bert_output_type=bert_output_type,
                                       quantize_model=quantize_model)
        model = BertEmbedder.CACHE_MODELS.get(model_hashid, None)

        if not model:

            info_msg = ""
            for name in [f"sentence-transformers/{pretrained_name_or_abspath}",
                         pretrained_name_or_abspath]:
                try:
                    model = (
                        self._get_sentence_transformers_encoder(name,
                                                                output_type=bert_output_type,
                                                                quantize=quantize_model,
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
                msg = f"Could not resolve the name/path `{pretrained_name_or_abspath}`. " \
                      f"Please check the model name and retry."
                raise Exception(msg)

            BertEmbedder.CACHE_MODELS.update({model_hashid: model})

        self.model_id = str(model_hashid)

        return model

    def _encode(self, phrases):
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

        if not isinstance(phrases, (str, list)):
            raise TypeError(f"argument phrases must be of type str or list, not {type(phrases)}")

        show_progress_bar = (
            self._show_progress_bar and
            (len(phrases) if isinstance(phrases, list) else 1) > 1
        )

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
                                             batch_size=self._batch_size,
                                             show_progress_bar=show_progress_bar,
                                             output_value=self._output_value,
                                             convert_to_numpy=self._convert_to_numpy,
                                             convert_to_tensor=self._convert_to_tensor,
                                             device=self.device,
                                             concat_last_n_layers=self._concat_last_n_layers,
                                             normalize_token_embs=self._normalize_token_embs)
                setattr(self, "_use_sbert_model", False)
            except TypeError as e:
                logger.error(e)
                if self._concat_last_n_layers != 1 or self._normalize_token_embs:
                    msg = f"{'concat_last_n_layers,' if self._concat_last_n_layers != 1 else ''} " \
                          f"{'normalize_token_embs' if self._normalize_token_embs else ''} " \
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


class GloveEmbedder(Embedder):
    """
    Encoder class for GloVe embeddings as described here: https://nlp.stanford.edu/projects/glove/
    """

    DEFAULT_EMBEDDING_DIM = 300

    def __init__(self, app_path, **kwargs):
        super().__init__(app_path, **kwargs)
        self.glove_tokenizer = Tokenizer()

    def tokenize(self, text):
        tokens = self.glove_tokenizer.tokenize(text, keep_special_chars=False)
        token_list = [t["entity"] for t in tokens]
        return token_list

    def _load(self, **kwargs):
        token_embedding_dimension = kwargs.get(
            "token_embedding_dimension", self.DEFAULT_EMBEDDING_DIM
        )
        token_pretrained_embedding_filepath = kwargs.get(
            "token_pretrained_embedding_filepath"
        )
        return WordSequenceEmbedding(
            0,
            token_embedding_dimension,
            token_pretrained_embedding_filepath,
            use_padding=False,
        )

    def _encode(self, text_list):
        token_list = [self.tokenize(text) for text in text_list]
        vector_list = [self.model.encode_sequence_of_tokens(tl) for tl in token_list]
        encoded_vecs = []
        for vl in vector_list:
            if len(vl) == 1:
                encoded_vecs.append(vl[0])
            else:
                encoded_vecs.append(np.average(vl, axis=0))
        return encoded_vecs

    def dump(self):
        """Dumps the cache to disk."""
        super().dump()
        self.model.save_embeddings()


if _is_module_available("sentence_transformers"):
    register_embedder("bert", BertEmbedder)

register_embedder("glove", GloveEmbedder)
