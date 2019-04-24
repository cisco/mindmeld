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
import logging
import sys
import json
import requests
from mindmeld.components._config import get_system_entity_recognizer_config

DUCKLING_URL = "http://localhost:7151"
DUCKLING_ENDPOINT = "parse"

NO_RESPONSE_CODE = -1

logger = logging.getLogger(__name__)


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
                self.is_service_alive = True
            else:
                self.is_service_alive = get_system_entity_recognizer_config(app_path)
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

        if not self.is_service_alive:
            return [], NO_RESPONSE_CODE

        url = '/'.join([DUCKLING_URL, DUCKLING_ENDPOINT])

        try:
            response = requests.request('POST', url, data=data)
            response_json = response.json()

            # Remove the redundant 'values' key in the response['value'] dictionary
            for i, entity_dict in enumerate(response_json):
                if 'values' in entity_dict['value']:
                    del response_json[i]['value']['values']

            return response_json, response.status_code
        except requests.ConnectionError:
            sys.exit("Unable to connect to the system entity recognizer. Make sure it's "
                     "running by typing 'mindmeld num-parse' at the command line.")
        except Exception as ex:  # pylint: disable=broad-except
            logger.error('Numerical Entity Recognizer Error %s\nURL: %r\nData: %s', ex, url,
                         json.dumps(data))
            sys.exit('\nThe system entity recognizer encountered the following ' +
                     'error:\n' + str(ex) + '\nURL: ' + url + '\nRaw data: ' + str(data) +
                     "\nPlease check your data and ensure Numerical parsing service is running. "
                     "Make sure it's running by typing "
                     "'mindmeld num-parse' at the command line.")
