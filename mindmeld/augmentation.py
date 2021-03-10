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

"""This module contains the data augmentation processes for MindMeld."""

import logging
import re

from abc import ABC, abstractmethod
from tqdm import tqdm

from .components._util import _is_module_available, _get_module_or_attr
from .constants import get_pattern
from .models.helpers import register_augmentor

logger = logging.getLogger(__name__)

# pylint: disable=R0201


class Augmentor(ABC):
    """
    Abstract Augmentor class.
    """

    def __init__(self, lang, paths, path_suffix, resource_loader):
        """Initializes an augmentor.

        Args:
            lang (str): The lang code for paraphrasing
            paths (list): Path rules for fetching relevant files to Paraphrase.
            path_suffix (str): Suffix to be added to new augmented files.
            resource_loader (object): Resource Loader object for the application.
        """
        self.lang = lang
        self.paths = paths
        self.path_suffix = path_suffix
        self._resource_loader = resource_loader
        self._check_dependencies()

    def _check_dependencies(self):
        if not _is_module_available("torch"):
            raise ModuleNotFoundError(
                "Library not found: 'torch'. Run 'pip install mindmeld[augment]' to install."
            )

        if not _is_module_available("transformers"):
            raise ModuleNotFoundError(
                "Library not found: 'transformers'. Run 'pip install mindmeld[augment]' to install."
            )

    def augment(self, **kwargs):
        """Augments queries given initial queries in application."""
        filtered_paths = self._get_files(paths=self.paths)

        for path in tqdm(filtered_paths):
            queries = self._read_path_queries(path)
            augmented_queries = self.augment_queries(queries, **kwargs)
            self._write_files(path, augmented_queries, suffix=self.path_suffix)

    @abstractmethod
    def augment_queries(self, queries):
        """Generates augmented data given application queries.

        Args:
            queries (list): List of queries.

        Return:
            augmented_queries (list): List of augmented queries.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _get_files(self, paths=None):
        """Fetches relevant files given the path rules specified in the config.

        Args:
            paths (list): Path rules for fetching relevant files.

        Return:
            filtered_paths (list): List of file paths to be augmeted.
        """
        all_file_paths = self._resource_loader.get_all_file_paths()

        if not paths:
            logger.warning(
                """'paths' field is not configured or misconfigured in the `config.py`.
                 Can't find files to augment."""
            )
            return []

        rules = paths

        filtered_paths = []

        for rule in rules:
            pattern = get_pattern(rule)
            compiled_pattern = re.compile(pattern)
            filtered_paths.extend(
                self._resource_loader.filter_file_paths(
                    compiled_pattern=compiled_pattern, file_paths=all_file_paths
                )
            )
        return filtered_paths

    def _read_path_queries(self, path):
        """Returns all queries in a specified file path.

        Args:
            path (str): File path.

        Return:
            queries (list): List of queries in the file path.
        """
        with open(path, "r") as f:
            queries = f.readlines()
        return queries

    def _write_files(self, path, augmented_queries, suffix):
        """Writes augmented queries to a new file in the path.

        Args:
            path (str): File path to the original file.
            augmented_queries (list): List of augmented queries returned by class.
        """
        write_path = path.strip(".txt") + suffix

        with open(write_path, "w") as outfile:
            for query in augmented_queries:
                outfile.write(query.strip("\n") + "\n")


class EnglishParaphraser(Augmentor):
    """Paraphraser class for generating English paraphrases."""

    def __init__(self, lang, paths, path_suffix, params, resource_loader):
        """Initializes an English paraphraser.

        Args:
            lang (str): The lang code for paraphrasing
            paths (list): Path rules for fetching relevant files to Paraphrase.
            path_suffix (str): Suffix to be added to new augmented files.
            params (dict): Model params.
            resource_loader (object): Resource Loader object for the application.
        """
        super().__init__(
            lang=lang,
            paths=paths,
            path_suffix=path_suffix,
            resource_loader=resource_loader,
        )

        PegasusTokenizer = _get_module_or_attr("transformers", "PegasusTokenizer")
        PegasusForConditionalGeneration = _get_module_or_attr(
            "transformers", "PegasusForConditionalGeneration"
        )

        model_name = "tuner007/pegasus_paraphrase"
        torch_device = (
            "cuda" if _get_module_or_attr("torch.cuda", "is_available")() else "cpu"
        )
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(
            model_name, force_download=False
        ).to(torch_device)

        # Use default model params if not customized for app
        self.params = params or {
            "max_length": 60,
            "num_beams": 10,
            "num_return_sequences": 10,
            "temperature": 1.5,
        }

    def _get_response(self, query):
        """Generates paraphrase responses for given query.

        Args:
            query (str): An application query.

        Return:
            paraphrases (list(str)): List of paraphrased queries.
        """
        batch = self.tokenizer.prepare_seq2seq_batch(
            [query], truncation=True, padding="longest", max_length=60
        )
        translated = self.model.generate(
            **batch,
            **self.params,
        )
        paraphrases = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return paraphrases

    def augment_queries(self, queries, **kwargs):
        augmented_queries = []
        for query in queries:
            augmented_queries.extend(self._get_response(query, **kwargs))
        return augmented_queries


class MultiLingualParaphraser(Augmentor):
    """Paraphraser class for generating paraphrases based on language code of the app
    (currently supports: French, Italian, Spanish, Portuguese, and Romanian).
    """

    def __init__(self, lang, paths, path_suffix, params, resource_loader):
        """Initializes a multi-lingual paraphraser.

        Args:
            lang (str): The lang code for paraphrasing
            paths (list): Path rules for fetching relevant files to Paraphrase.
            path_suffix (str): Suffix to be added to new augmented files.
            params (dict): Model params.
            resource_loader (object): Resource Loader object for the application.
        """
        super().__init__(
            lang=lang,
            paths=paths,
            path_suffix=path_suffix,
            resource_loader=resource_loader,
        )

        self.torch_device = (
            "cuda" if _get_module_or_attr("torch.cuda", "is_available")() else "cpu"
        )

        MarianTokenizer = _get_module_or_attr("transformers", "MarianTokenizer")
        MarianMTModel = _get_module_or_attr("transformers", "MarianMTModel")

        en_model_name = "Helsinki-NLP/opus-mt-ROMANCE-en"
        self.en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
        self.en_model = MarianMTModel.from_pretrained(en_model_name)
        self.en_model.to(self.torch_device)

        target_model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
        self.target_tokenizer = MarianTokenizer.from_pretrained(target_model_name)
        self.target_model = MarianMTModel.from_pretrained(target_model_name).to(
            self.torch_device
        )

        # Use default model params if not customized for app
        self.params = params or {
            "num_beams": 3,
            "num_return_sequences": 3,
            "top_k": 0,
            "temperature": 1.0,
        }

    def _translate(self, *, template, queries, model, tokenizer, **kwargs):
        """The core translation step for forward and back translation.

        Args:
            template (lambda func): Structure input text to model.
            queries (list(str)): List of input queries.
            model: Machine translation model (en-ROMANCE or ROMANCE-en).
            tokenizer: Language tokenizer for input query text.
        """
        queries = [template(query) for query in queries]
        encoded = tokenizer.prepare_seq2seq_batch(queries, return_tensors="pt")
        for key in encoded:
            encoded[key] = encoded[key].to(self.torch_device)

        translated = model.generate(**encoded, **kwargs)
        translated_queries = tokenizer.batch_decode(
            translated, skip_special_tokens=True
        )
        return translated_queries

    def augment_queries(self, queries):
        translated_queries = self._translate(
            template=lambda text: f"{text}",
            queries=queries,
            model=self.en_model,
            tokenizer=self.en_tokenizer,
            **self.params,
        )
        translated_queries = list(set(translated_queries))

        back_translated_queries = self._translate(
            template=lambda text: f">>{self.lang}<< {text}",
            queries=translated_queries,
            model=self.target_model,
            tokenizer=self.target_tokenizer,
            **self.params,
        )
        augmented_queries = list(set(p.lower() for p in back_translated_queries))

        return augmented_queries


register_augmentor("EnglishParaphraser", EnglishParaphraser)
register_augmentor("MultiLingualParaphraser", MultiLingualParaphraser)