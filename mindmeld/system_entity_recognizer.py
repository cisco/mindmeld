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
import sys
import os
import requests

from mindmeld.components._config import (
    get_system_entity_url_config,
    is_duckling_configured,
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


class SystemEntityRecognizer:
    """SystemEntityRecognizer is the external parsing service used to extract
    system entities. It is intended to be used as a singleton, so it's
    initialized only once during NLP object construction.

    TODO: Abstract this class into an interface and implement the duckling
    service as one such service.
    """

    _instance = None

    def __init__(self, app_path=None):
        """Private constructor for SystemEntityRecognizer. Do not directly
        construct the SystemEntityRecognizer object. Instead, use the
        static get_instance method.

        Args:
            app_path (str): A application path
        """
        if SystemEntityRecognizer._instance:
            raise Exception("SystemEntityRecognizer is a singleton")
        else:
            if not app_path:
                # The service is turned on by default
                self._use_duckling_api = True
            else:
                self._use_duckling_api = is_duckling_configured(app_path)

        self.app_path = app_path
        SystemEntityRecognizer._instance = self

    @staticmethod
    def get_instance(app_path=None):
        """ Static access method.

        Args:
            app_path (str): A application path

        Returns:
            (SystemEntityRecognizer): A SystemEntityRecognizer instance
        """
        if not SystemEntityRecognizer._instance:
            SystemEntityRecognizer(app_path)
        return SystemEntityRecognizer._instance

    def get_response(self, data):

        if not self._use_duckling_api:
            return [], NO_RESPONSE_CODE

        url = get_system_entity_url_config(app_path=self.app_path)

        try:
            response = requests.request(
                "POST", url, data=data, timeout=SYS_ENTITY_REQUEST_TIMEOUT
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
                url,
                json.dumps(data),
            )
            sys.exit(
                "\nThe system entity recognizer encountered the following "
                + "error:\n"
                + str(ex)
                + "\nURL: "
                + url
                + "\nRaw data: "
                + str(data)
                + "\nPlease check your data and ensure Numerical parsing service is running. "
                "Make sure it's running by typing "
                "'mindmeld num-parse' at the command line."
            )
