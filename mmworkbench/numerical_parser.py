import requests
import logging
import json
import sys
from mmworkbench.components._config import get_numerical_parser_config

DUCKLING_URL = "http://localhost:7151"
DUCKLING_ENDPOINT = "parse"

SUCCESSFUL_HTTP_CODE = 200
NO_RESPONSE_CODE = -1

logger = logging.getLogger(__name__)


class NumericalParser:

    _instance = None

    def __init__(self, app_path=None):
        """Private constructor for NumericalParser. Do not directly
        construct the NumericalParser object. Instead, use the
        static get_instance method.

        Args:
            app_path (str): A application path
        """
        if NumericalParser._instance:
            raise Exception("Numerical Parser is a singleton")
        else:
            if not app_path:
                self.is_service_alive = False
            else:
                self.is_service_alive = get_numerical_parser_config(app_path)
            NumericalParser._instance = self

    @staticmethod
    def get_instance(app_path=None):
        """ Static access method.

        Args:
            app_path (str): A application path

        Returns:
            (NumericalParser): A NumericalParser instance
        """
        if not NumericalParser._instance:
            NumericalParser(app_path)
        return NumericalParser._instance

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
            logger.error("Unable to connect to the numerical parser. Make sure it's running by typing "
                         "'mmworkbench num-parse' at the command line.")
            return [], NO_RESPONSE_CODE

        except Exception as ex:  # pylint: disable=broad-except
            logger.error('Numerical Entity Recognizer Error %s\nURL: %r\nData: %s', ex, url,
                         json.dumps(data))
            sys.exit('\nThe numerical parser service encountered the following ' +
                     'error:\n' + str(ex) + '\nURL: ' + url + '\nRaw data: ' + str(data) +
                     "\nPlease check your data and ensure Numerical parsing service is running. "
                     "Make sure it's running by typing "
                     "'mmworkbench num-parse' at the command line.")

