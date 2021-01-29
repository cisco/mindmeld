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
This module contains translator clients used by the MultiLingual Annotator.
"""
from abc import ABCMeta, abstractmethod
import logging
import os

try:
    from google.cloud import translate_v2
except ModuleNotFoundError:
    raise ValueError(
        "Library not found: 'google-cloud'. Run 'pip install mindmeld[language_annotator]'"
        " to install."
    )

logger = logging.getLogger(__name__)


class Translator(metaclass=ABCMeta):
    """Abstract Translator Base Class for Translators to be used by Mindmeld."""

    def __init__(self):
        """Creates a translation client after finding the credential path."""
        self.translate_client = None

    @abstractmethod
    def get_translate_client(self):
        """
        Args:
            text (str): Input text
        Returns:
            language_code (str): Detected Language Code
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def detect_language(self, text):
        """
        Args:
            text (str): Input text
        Returns:
            language_code (str): Detected Language Code
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def translate(self, text, target_language):
        """
        Args:
            text (str): Input text
            destination_language (str): Language code for target language.
        Returns:
            translated_text (str): Translated text
        """
        raise NotImplementedError("Subclasses must implement this method")


class NoOpTranslator(Translator):
    """No-Ops translator to be used when a Translator is not selected or available."""

    def __init__(self):
        pass

    def get_translate_client(self):
        return

    def detect_language(self, text):
        return

    def translate(self, text, target_language):
        return


class GoogleTranslator(Translator):
    """Class for translation using the Google Translate API."""

    def __init__(self):
        """Initializes the translate_client."""
        self.translate_client = self.get_translate_client()

    def get_translate_client(self):
        """Creates a translation client after finding the credential path."""
        GoogleTranslator._check_credential_exists()
        return translate_v2.Client()

    @staticmethod
    def _check_credential_exists():
        """Searches environment variables for the path to google application credentials.

        Returns:
            credential_path (str): Path to google application credentials.
        """
        try:
            return os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        except KeyError as error:
            raise KeyError(
                "Google credential path not found. Export 'GOOGLE_CREDENTIAL_PATH' as"
                " an environment variable."
            ) from error

    def detect_language(self, text):
        """
        Args:
            text (str): Input text
        Returns:
            language_code (str): Detected Language Code
        """
        return self.translate_client.detect_language(text)["language"]

    def translate(self, text, target_language):
        """
        Args:
            text (str): Input text
            target_language (str): Language code for language to translate the given text to.
        Returns:
            translated_text (str): Translated text
        """
        return self.translate_client.translate(text, target_language=target_language)[
            "translatedText"
        ]


class TranslatorFactory:
    """Translator Factory Class"""

    @staticmethod
    def get_translator(translator):
        """A static method to get a translator

        Args:
            translator (str): Name of the desired translator class
        Returns:
            (Translator): Translator Class
        """
        if translator == "NoOpTranslator":
            return NoOpTranslator()
        if translator == "GoogleTranslator":
            return GoogleTranslator()
        raise AssertionError(
            "Valid 'translator' not found in AUTO_ANNOTATOR_CONFIG."
            " Supported translators include 'NoOpTranslator' and 'GoogleTranslator'."
        )
