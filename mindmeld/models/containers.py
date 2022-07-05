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

import json
import logging
import os
import zipfile
from urllib.request import urlretrieve

import numpy as np
from tqdm import tqdm

from ._util import _is_module_available, _get_module_or_attr as _getattr
from ..constants import EMBEDDINGS_URL
from ..core import Bunch
from ..exceptions import EmbeddingDownloadError
from ..path import EMBEDDINGS_FILE_PATH, EMBEDDINGS_FOLDER_PATH
from ..resource_loader import Hasher

logger = logging.getLogger(__name__)


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """Reports update statistics on the download progress.

        Args:
            b (int): Number of blocks transferred so far [default: 1].
            bsize (int): Size of each block (in tqdm units) [default: 1].
            tsize (int): Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


class GloVeEmbeddingsContainer:
    """This class is responsible for the downloading, extraction and storing of
    word embeddings based on the GloVe format.

    To facilitate not loading the large glove embedding file to memory everytime a new container is
    created, a class-level attribute with a hashmap is created.

    TODO: refactor the call-signature similar to other containers by accepting
    `pretrained_path_or_name` instead of token dimension and filepath. Also deprecate these two
    arguments.
    """
    CONTAINER_LOOKUP = {}

    EMBEDDING_FILE_PATH_TEMPLATE = "glove.6B.{}d.txt"
    ALLOWED_WORD_EMBEDDING_DIMENSIONS = [50, 100, 200, 300]

    def __init__(self, token_dimension=300, token_pretrained_embedding_filepath=None):
        # validations
        if token_dimension not in GloVeEmbeddingsContainer.ALLOWED_WORD_EMBEDDING_DIMENSIONS:
            logger.info(
                "Token dimension %s not supported, "
                "choose from these dimensions: %s. "
                "Selected 300 by default",
                token_dimension,
                str(GloVeEmbeddingsContainer.ALLOWED_WORD_EMBEDDING_DIMENSIONS),
            )
            token_dimension = 300

        self.token_dimension = token_dimension
        self.token_pretrained_embedding_filepath = token_pretrained_embedding_filepath

        # get model name hash
        string_to_hash = json.dumps({"token_dimension": self.token_dimension})
        self.model_id = Hasher(algorithm="sha1").hash(string=string_to_hash)

        self._word_to_embedding = self._extract_embeddings()

    def get_pretrained_word_to_embeddings_dict(self):
        """Returns the word to embedding dict.

        Returns:
            (dict): word to embedding mapping.
        """
        return self._word_to_embedding

    def _download_embeddings_and_return_zip_handle(self):

        logger.info("Downloading embedding from %s", EMBEDDINGS_URL)

        # Make the folder that will contain the embeddings
        if not os.path.exists(EMBEDDINGS_FOLDER_PATH):
            os.makedirs(EMBEDDINGS_FOLDER_PATH)

        with TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc=EMBEDDINGS_URL) as t:

            try:
                urlretrieve(EMBEDDINGS_URL, EMBEDDINGS_FILE_PATH, reporthook=t.update_to)

            except ConnectionError as e:
                logger.error("Error downloading from %s: %s", EMBEDDINGS_URL, e)
                return

            file_name = GloVeEmbeddingsContainer.EMBEDDING_FILE_PATH_TEMPLATE.format(
                self.token_dimension
            )
            zip_file_object = zipfile.ZipFile(EMBEDDINGS_FILE_PATH, "r")

            if file_name not in zip_file_object.namelist():
                logger.info(
                    "Embedding file with %s dimensions not found in %s file path",
                    self.token_dimension,
                    file_name
                )
                return

            return zip_file_object

    @staticmethod
    def _extract_and_map(glove_file):
        word_to_embedding = {}
        for line in glove_file:
            values = line.split()
            word = values[0]
            if not isinstance(word, str):  # can be encoded as byte type
                word = word.decode()
            coefs = np.asarray(values[1:], dtype="float32")
            word_to_embedding[word] = coefs
        return word_to_embedding

    def _extract_embeddings(self):
        if self.model_id not in GloVeEmbeddingsContainer.CONTAINER_LOOKUP:

            file_location = self.token_pretrained_embedding_filepath

            if file_location and os.path.isfile(file_location):
                msg = f"Extracting embeddings from provided file location {str(file_location)}."
                logger.info(msg)
                with open(file_location, "r") as embedding_file:
                    word_to_embedding = self._extract_and_map(embedding_file)
            else:
                if file_location:
                    logger.info("Provided file location %s does not exist.", str(file_location))
                file_name = GloVeEmbeddingsContainer.EMBEDDING_FILE_PATH_TEMPLATE.format(
                    self.token_dimension)
                if os.path.isfile(EMBEDDINGS_FILE_PATH):
                    msg = f"Extracting embeddings from default folder location " \
                          f"{EMBEDDINGS_FILE_PATH}."
                    logger.info(msg)
                    try:
                        zip_file_object = zipfile.ZipFile(EMBEDDINGS_FILE_PATH, "r")
                        with zip_file_object.open(file_name) as embedding_file:
                            word_to_embedding = self._extract_and_map(embedding_file)
                    except zipfile.BadZipFile:
                        logger.warning(
                            "%s is corrupt. Deleting the zip file and attempting to"
                            " download the embedding file again",
                            EMBEDDINGS_FILE_PATH,
                        )
                        os.remove(EMBEDDINGS_FILE_PATH)
                        self._extract_embeddings()
                    except IOError as e:
                        logger.error(
                            "An error occurred when reading %s zip file. The file might"
                            " be corrupt, so try deleting the file and running the program "
                            "again",
                            EMBEDDINGS_FILE_PATH,
                        )
                        raise IOError("Failed to load embeddings.") from e
                else:
                    logger.info("Default folder location %s does not exist.", EMBEDDINGS_FILE_PATH)
                    zip_file_object = self._download_embeddings_and_return_zip_handle()
                    if not zip_file_object:
                        raise EmbeddingDownloadError("Failed to download embeddings.")
                    with zip_file_object.open(file_name) as embedding_file:
                        word_to_embedding = self._extract_and_map(embedding_file)

            GloVeEmbeddingsContainer.CONTAINER_LOOKUP[self.model_id] = word_to_embedding

        return GloVeEmbeddingsContainer.CONTAINER_LOOKUP[self.model_id]


class SentenceTransformersContainer:
    """This class is responsible for the downloading and extraction of sentence transformers models
    based on the https://github.com/UKPLab/sentence-transformers format.

    To facilitate not loading the large glove embedding file to memory everytime a new container is
    created, a class-level attribute with a hashmap is created.
    """
    CONTAINER_LOOKUP = {}

    def __init__(
        self,
        pretrained_name_or_abspath,
        bert_output_type="mean",
        quantize_model=False
    ):
        self.pretrained_name_or_abspath = pretrained_name_or_abspath
        self.bert_output_type = bert_output_type
        self.quantize_model = quantize_model

        # get model name hash
        string_to_hash = json.dumps({
            "pretrained_name_or_abspath": self.pretrained_name_or_abspath,
            "bert_output_type": self.bert_output_type,
            "quantize_model": self.quantize_model,
        }, sort_keys=True)
        self.model_id = Hasher(algorithm="sha1").hash(string=string_to_hash)

        self._model_bunch = None

    def _extract_model(self):
        """
        Looks up the model in the class' lookup dict and returns the found value if available. If
        not found, calls Huggingface APIs to download the model and then loads the model to the
        lookup and returns the model.
        """

        model = SentenceTransformersContainer.CONTAINER_LOOKUP.get(self.model_id)

        if not model:
            info_msg = ""
            for name in [
                self.pretrained_name_or_abspath,
                f"sentence-transformers/{self.pretrained_name_or_abspath}"
            ]:
                try:
                    model = self._get_strans_encoder(
                        name,
                        output_type=self.bert_output_type,
                        quantize=self.quantize_model
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

            SentenceTransformersContainer.CONTAINER_LOOKUP[self.model_id] = model

        return SentenceTransformersContainer.CONTAINER_LOOKUP[self.model_id]

    @staticmethod
    def _get_strans_encoder(
        name_or_path,
        output_type="mean",
        quantize=False
    ):
        """
        Retrieves a sentence-transformer model and returns it along with its transformer and
        pooling components.

        Args:
            name_or_path: name or path to load a huggingface model
            output_type: type of pooling required
            quantize: if the model needs to be qunatized or not

        Returns:
            Bunch(sentence_transformers.Transformer,
                  sentence_transformers.Pooling,
                  sentence_transformers.SentenceTransformer)
        """

        if not _is_module_available("sentence_transformers"):
            msg = "Must install extra [bert] by running " \
                  "'pip install mindmeld[bert]'"
            raise ImportError(msg)

        strans_models = _getattr("sentence_transformers.models")
        strans = _getattr("sentence_transformers", "SentenceTransformer")

        transformer_model = strans_models.Transformer(
            name_or_path,
            model_args={"output_hidden_states": True}
        )
        pooling_model = strans_models.Pooling(
            transformer_model.get_word_embedding_dimension(),
            pooling_mode_cls_token=output_type == "cls",
            pooling_mode_max_tokens=False,
            pooling_mode_mean_tokens=output_type == "mean",
            pooling_mode_mean_sqrt_len_tokens=False
        )
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

        return Bunch(
            transformer_model=transformer_model,
            pooling_model=pooling_model,
            sbert_model=sbert_model
        )

    def get_model_bunch(self):
        # if already looked up, return it
        if self._model_bunch:
            return self._model_bunch
        self._model_bunch = self._extract_model()
        return self._model_bunch

    def get_sbert_model(self):
        return self.get_model_bunch().sbert_model

    def get_pooling_model(self):
        return self.get_model_bunch().pooling_model

    def get_transformer_model(self):
        return self.get_model_bunch().transformer_model


class HuggingfaceTransformersContainer:
    """This class is responsible for the downloading and extraction of transformers models such as
     BERT, Multilingual-BERT, etc. based on the https://github.com/huggingface/transformers format.

    To facilitate not loading the large glove embedding file to memory everytime a new container is
    created, a class-level attribute with a hashmap is created.
    """
    CONTAINER_LOOKUP = {}

    def __init__(
        self,
        pretrained_model_name_or_path,
        quantize_model=False,
        cache_lookup=True,
        from_configs=False
    ):

        if not _is_module_available("transformers"):
            msg = "Must install extra [transformers] by running " \
                  "'pip install mindmeld[transformers]'"
            raise ImportError(msg)

        if from_configs:
            if cache_lookup:
                msg = "Cannot set both 'cache_lookup' and 'from_configs' to True at the same " \
                      "time. Loading from Huggingface model configs returns a model without " \
                      "pretrained weights' initialization."
                raise ValueError(msg)
            if quantize_model:
                msg = "Huggingface model loaded from configs will be quantized instead of a " \
                      "model loaded from pretrained weights"
                logger.warning(msg)

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.quantize_model = quantize_model
        self.cache_lookup = cache_lookup
        self.from_configs = from_configs

        # get model name hash
        string_to_hash = json.dumps({
            "pretrained_model_name_or_path": self.pretrained_model_name_or_path,
            "quantize_model": self.quantize_model
        }, sort_keys=True)
        self.model_id = Hasher(algorithm="sha1").hash(string=string_to_hash)

        self._model_bunch = None

    def _extract_model(self):

        if self.cache_lookup:
            model = HuggingfaceTransformersContainer.CONTAINER_LOOKUP.get(self.model_id)
        else:
            model = None

        if not model:

            try:
                if self.from_configs:
                    config = self.get_transformer_model_config()
                    transformer_model = _getattr("transformers", "AutoModel").from_config(config)
                else:
                    transformer_model = _getattr("transformers", "AutoModel").from_pretrained(
                        self.pretrained_model_name_or_path)
            except OSError as e:
                msg = f"Could not resolve the name/path `{self.pretrained_model_name_or_path}`. " \
                      f"Please check the model name/path and retry."
                raise OSError(msg) from e

            if self.quantize_model:
                if not _is_module_available("torch"):
                    raise ImportError("`torch` library required to quantize models") from None

                torch_qint8 = _getattr("torch", "qint8")
                torch_nn_linear = _getattr("torch.nn", "Linear")
                torch_quantize_dynamic = _getattr("torch.quantization", "quantize_dynamic")

                transformer_model = torch_quantize_dynamic(
                    transformer_model, {torch_nn_linear}, dtype=torch_qint8)

            model = Bunch(
                config=self.get_transformer_model_config(),
                tokenizer=self.get_transformer_model_tokenizer(),
                transformer_model=transformer_model
            )

        # return the model without adding to lookup if the flag is set to False
        if not self.cache_lookup:
            return model

        HuggingfaceTransformersContainer.CONTAINER_LOOKUP[self.model_id] = model
        return HuggingfaceTransformersContainer.CONTAINER_LOOKUP[self.model_id]

    def get_model_bunch(self):
        # if already looked up/loaded, return it
        if self._model_bunch:
            return self._model_bunch
        # either a bunch from lookup or newly loaded bunch if cache_lookup is False
        self._model_bunch = self._extract_model()
        return self._model_bunch

    def get_transformer_model_config(self):
        return _getattr("transformers", "AutoConfig").from_pretrained(
            self.pretrained_model_name_or_path)

    def get_transformer_model_tokenizer(self):
        return _getattr("transformers", "AutoTokenizer").from_pretrained(
            self.pretrained_model_name_or_path)

    def get_transformer_model(self):
        return self.get_model_bunch().transformer_model
