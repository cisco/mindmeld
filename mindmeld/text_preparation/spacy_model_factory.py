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

"""This module contains a Spacy Model Factory."""
import importlib
import logging
import subprocess
import spacy

from ..constants import (
    SPACY_WEB_TRAINED_LANGUAGES,
    SPACY_SUPPORTED_LANGUAGES,
    SPACY_MODEL_SIZES,
)

logger = logging.getLogger(__name__)


class SpacyModelFactory:
    """Spacy (Language) Model Factory Class"""

    @staticmethod
    def get_spacy_language_model(language, spacy_model_size="lg"):
        """Get a Spacy Language model.

        Args:
            language (str, optional): Language as specified using a 639-1/2 code.
            spacy_model_name (str): Name of the Spacy NER model (Ex: "en_core_web_sm")

        Returns:
            nlp: Spacy language model. (Ex: "spacy.lang.es.Spanish")
        """
        SpacyModelFactory.validate_spacy_language(language)
        SpacyModelFactory.validate_spacy_model_size(spacy_model_size)
        spacy_model_name = SpacyModelFactory._get_spacy_model_name(
            language, spacy_model_size
        )
        return SpacyModelFactory._load_model(spacy_model_name)

    @staticmethod
    def validate_spacy_language(language):
        """Check if the language is valid.

        Args:
            language (str, optional): Language as specified using a 639-1/2 code.
        """
        if language not in SPACY_SUPPORTED_LANGUAGES:
            raise ValueError("Spacy does not currently support: {!r}.".format(language))

    @staticmethod
    def validate_spacy_model_size(spacy_model_size):
        """Check if the model size is valid.

        Args:
            spacy_model_size (str, optional): Size of the Spacy model to use. ("sm", "md", or "lg")
        """
        if spacy_model_size not in SPACY_MODEL_SIZES:
            raise ValueError(
                "{!r} is not a valid model size. Select from: {!r}.".format(
                    spacy_model_size, " ".join(SPACY_MODEL_SIZES)
                )
            )

    @staticmethod
    def _load_model(spacy_model_name):
        """Load Spacy English model. Download if needed.

        Args:
            spacy_model_name (str): Name of the Spacy NER model (Ex: "en_core_web_sm")

        Returns:
            nlp: Spacy language model. (Ex: "spacy.lang.es.Spanish")
        """
        logger.info("Loading Spacy model %s.", spacy_model_name)
        try:
            return spacy.load(spacy_model_name)
        except OSError:
            logger.warning(
                "%s not found on disk. Downloading the model.", spacy_model_name
            )
            SpacyModelFactory._download_spacy_model(spacy_model_name)
            language_module = SpacyModelFactory._import_spacy_model(spacy_model_name)
            return language_module.load()

    @staticmethod
    def _get_spacy_model_name(language, spacy_model_size):
        """Get the name of a Spacy Model.

        Args:
            language (str, optional): Language as specified using a 639-1/2 code.
            spacy_model_size (str, optional): Size of the Spacy model to use. ("sm", "md", or "lg")

        Returns:
            spacy_model_name (str): Name of the Spacy NER model (Ex: "en_core_web_sm")
        """
        model_type = "web" if language in SPACY_WEB_TRAINED_LANGUAGES else "news"
        return f"{language}_core_{model_type}_{spacy_model_size}"

    @staticmethod
    def _download_spacy_model(spacy_model_name):
        """Download Spacy Model.

        Args:
            spacy_model_name (str): Name of the Spacy NER model (Ex: "en_core_web_sm")
        """
        subprocess.run(
            ["python", "-m", "spacy", "download", spacy_model_name], check=True
        )

    @staticmethod
    def _import_spacy_model(spacy_model_name):
        """Attempt to Imort the Spacy Model.

        Args:
            spacy_model_name (str): Name of the Spacy NER model (Ex: "en_core_web_sm")

        Returns:
            language_module (module): Imported language module.
        """
        try:
            return importlib.import_module(spacy_model_name)
        except ModuleNotFoundError as error:
            raise ValueError(
                "Unknown Spacy model name: {!r}.".format(spacy_model_name)
            ) from error
