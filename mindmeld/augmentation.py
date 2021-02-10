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
import torch

from abc import ABC, abstractmethod

from .components._config import get_augmentation_config
from .constants import _get_pattern
from .models.helpers import register_augmentor
from .resource_loader import ResourceLoader

try:
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer
except ModuleNotFoundError:
    raise ValueError(
        "Library not found: 'transformers'. Run 'pip install mindmeld[augment]'"
        " to install."
    )

logger = logging.getLogger(__name__)


class Augmentor(ABC):
    """
    Abstract Augmentor class.
    """

    def __init__(self, app_path, config=None):
        """Initializes an augmentor.

        Args:
            app_path (str): The location of the MindMeld app
            config (dict, optional): A config object to use. This will
                override the config specified by the app's config.py file.
        """
        self.app_path = app_path
        self.config = config or get_augmentation_config(app_path=app_path)
        self._resource_loader = ResourceLoader.create_resource_loader(app_path)

    def augment(self, config):
        """Augments queries given initial queries in application.

        Args:
            config (dict, optional): App config to use instead of class config.
        """
        config = config or self.config
        filtered_paths = self._get_files(config)

        for path in filtered_paths:
            queries = self._read_path_queries(path)
            augmented_queries = self.augment_queries(queries)
            self._write_files(path, augmented_queries)

    @abstractmethod
    def augment_queries(self, queries):
        """
        Args:
            queries (list): List of queries.

        Return:
            augmented_queries (list): List of augmented queries.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _get_files(self, config):
        """Fetches relevant files given the path rules specified in the config.

        Args:
            config (dict): Application config.

        Return:
            filtered_paths (list): List of file paths to be augmeted.
        """
        all_file_paths = self._resource_loader.get_all_file_paths()

        if not config["paths"]:
            logger.warning(
                """'paths' field is not configured or misconfigured in the `config.py`.
                 Can't find files to augment."""
            )
            return

        rules = config["paths"]

        filtered_paths = []

        for rule in rules:
            pattern = _get_pattern(rule)
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

    def _write_files(self, path, augmented_queries):
        """Writes augmented queries to a new file in the path.

        Args:
            path (str): File path to the original file.
            augmented_queries (list): List of augmented queries returned by class.
        """
        write_path = path.strip(".txt") + ".augmented.txt"

        with open(write_path, "w") as outfile:
            for query in augmented_queries:
                outfile.write(query.strip("\n") + "\n")


class EnglishParaphraser(Augmentor):
    """Paraphraser class for generating English paraphrases."""

    def __init__(self, app_path, config=None):
        """Initializes an English paraphraser.

        Args:
            app_path (str): The location of the MindMeld app
            config (dict, optional): A config object to use. This will
                override the config specified by the app's config.py file.
        """
        super().__init__(app_path=app_path, config=config)
        model_name = "tuner007/pegasus_paraphrase"
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(
            model_name, force_download=False
        ).to(torch_device)

    def _get_response(self, query, num_return_sequences=10, num_beams=10):
        """Generates paraphrase responses for given query.

        Args:
            query (str): An application query.
            num_return_sequences (int, optional): Maximum number of paraphrases to be generated.
            num_beams (int, optional):

        Return:
            paraphrases (list(str)): List of paraphrased queries.
        """
        batch = self.tokenizer.prepare_seq2seq_batch(
            [query], truncation=True, padding="longest", max_length=60
        )
        translated = self.model.generate(
            **batch,
            max_length=60,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            temperature=1.5
        )
        paraphrases = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return paraphrases

    def augment_queries(self, queries):
        augmented_queries = []
        for query in queries:
            augmented_queries.extend(self._get_response(query))
        return augmented_queries


class MultiLingualParaphraser(Augmentor):
    """Paraphraser class for generating paraphrases based on language code of the app
    (for languages other than English).
    """

    def __init__(self, app_path, config=None):
        """Initializes a multi-lingual paraphraser.

        Args:
            app_path (str): The location of the MindMeld app
            config (dict, optional): A config object to use. This will
                override the config specified by the app's config.py file.
        """
        super().__init__(app_path=app_path, config=config)
        pass


register_augmentor("EnglishParaphraser", EnglishParaphraser)
