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
from abc import ABC, abstractmethod
import json
import logging
import sys
import os
import requests

from mindmeld.components._config import (
    DEFAULT_DUCKLING_URL,
    is_duckling_configured,
    get_system_entity_url_config,
)

from mindmeld.exceptions import MindMeldError

NO_RESPONSE_CODE = -1
SYS_ENTITY_REQUEST_TIMEOUT = os.environ.get("MM_SYS_ENTITY_REQUEST_TIMEOUT", 1.0)
try:
    if float(SYS_ENTITY_REQUEST_TIMEOUT) <= 0.0:
        raise MindMeldError(
            "MM_SYS_ENTITY_REQUEST_TIMEOUT env var has to be > 0.0 seconds."
        )
except ValueError as e:
    raise MindMeldError(
        "MM_SYS_ENTITY_REQUEST_TIMEOUT env var has to be a float value."
    ) from e

logger = logging.getLogger(__name__)


class SystemEntityError(Exception):
    pass


class SystemEntityRecognizer(ABC):
    """SystemEntityRecognizer is the external parsing service used to extract
    system entities. It is intended to be used as a singleton, so it's
    initialized only once during NLP object construction.
    """

    @staticmethod
    def get_instance(app_path):
        """ Static access method.
        In general we want to find the system entity recognizer from the application config.
        If the application configuration is empty, we do not use Duckling.
        Otherwise, we return the Duckling recognizer with the URL defined in the application's
          config, default to the DEFAULT_DUCKLING_URL.

        Args:
            app_path: The application path

        Returns:
            (SystemEntityRecognizer): A SystemEntityRecognizer instance
        """
        if is_duckling_configured(app_path):
            url = get_system_entity_url_config(app_path=app_path)
            return DucklingRecognizer.get_instance(url)
        else:
            return DummySystemEntityRecognizer.get_instance()

    @abstractmethod
    def get_response(self, data):
        pass


class DummySystemEntityRecognizer(SystemEntityRecognizer):
    """
    This is a dummy recognizer which returns empty list and NO_RESPONSE_CODE.
    """

    _instance = None

    def __init__(self):
        if not DummySystemEntityRecognizer._instance:
            DummySystemEntityRecognizer._instance = self
        else:
            raise Exception("DummySystemEntityRecognizer is a singleton")

    @staticmethod
    def get_instance():
        if not DummySystemEntityRecognizer._instance:
            DummySystemEntityRecognizer()

        return DummySystemEntityRecognizer._instance

    def get_response(self, data):
        del data
        return [], NO_RESPONSE_CODE


class DucklingRecognizer(SystemEntityRecognizer):
    _instances = {}

    def __init__(self, url=None):
        """Private constructor for SystemEntityRecognizer. Do not directly
        construct the DucklingRecognizer object. Instead, use the
        static get_instance method.

        Args:
            url (str): Duckling URL
        """
        if url in DucklingRecognizer._instances:
            raise Exception("DucklingRecognizer is a singleton")

        self.url = url
        DucklingRecognizer._instances[url] = self

    @staticmethod
    def get_instance(url=None):
        """ Static access method.
        We get an instance for the Duckling URL. If there is no URL being passed,
          default to DEFAULT_DUCKLING_URL.

        Args:
            url: Duckling URL.

        Returns:
            (DucklingRecognizer): A DucklingRecognizer instance
        """
        url = url or DEFAULT_DUCKLING_URL

        if url not in DucklingRecognizer._instances:
            DucklingRecognizer(url=url)
        return DucklingRecognizer._instances[url]

    def get_response(self, data):
        try:
            response = requests.request(
                "POST", self.url, data=data, timeout=float(SYS_ENTITY_REQUEST_TIMEOUT)
            )

            if response.status_code == requests.codes["ok"]:
                response_json = response.json()

                # Remove the redundant 'values' key in the response['value'] dictionary
                for i, entity_dict in enumerate(response_json):
                    if "values" in entity_dict["value"]:
                        del response_json[i]["value"]["values"]

                return response_json, response.status_code
            else:
                raise SystemEntityError("System entity status code is not 200.")
        except requests.ConnectionError:
            sys.exit(
                "Unable to connect to the system entity recognizer. Make sure it's "
                "running by typing 'mindmeld num-parse' at the command line."
            )
        except Exception as ex:  # pylint: disable=broad-except
            logger.error(
                "Numerical Entity Recognizer Error: %s\nURL: %r\nData: %s",
                ex,
                self.url,
                json.dumps(data),
            )
            sys.exit(
                "\nThe system entity recognizer encountered the following "
                + "error:\n"
                + str(ex)
                + "\nURL: "
                + self.url
                + "\nRaw data: "
                + str(data)
                + "\nPlease check your data and ensure Numerical parsing service is running. "
                "Make sure it's running by typing "
                "'mindmeld num-parse' at the command line."
            )
