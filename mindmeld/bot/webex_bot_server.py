# -*- coding: utf-8 -*-

"""
This module contains the Webex Bot Server component.
"""

import logging
import json
from flask import request
import requests
from ciscosparkapi import CiscoSparkAPI

CISCO_API_URL = 'https://api.ciscospark.com/v1'
ACCESS_TOKEN_WITH_BEARER = 'Bearer '
BAD_REQUEST_NAME = 'BAD REQUEST'
BAD_REQUEST_CODE = 400
APPROVED_REQUEST_NAME = 'OK'
APPROVED_REQUEST_CODE = 200


class WebexBotServer:
    """
    A sample server class for Webex Teams integration with any MindMeld application
    """

    def __init__(self, app, webhook_id, access_token, conv):
        self.app = app
        self.webhook_id = webhook_id
        self.access_token = access_token
        self.conv = conv

        self.logger = logging.getLogger(__name__)

        if not self.webhook_id:
            raise Exception('WEBHOOK_ID not set')
        if not self.access_token:
            raise Exception('BOT_ACCESS_TOKEN not set')

        self.spark_api = CiscoSparkAPI(self.access_token)
        self.access_token_with_bearer = ACCESS_TOKEN_WITH_BEARER + self.access_token

        @self.app.route('/', methods=['POST'])
        def handle_message():
            me = self.spark_api.people.me()
            data = request.get_json()

            for key in ['personId', 'id', 'roomId']:
                if key not in data['data'].keys():
                    payload = {'message': 'personId/id/roomID key not found'}
                    return BAD_REQUEST_NAME, BAD_REQUEST_CODE, payload

            if data['id'] != self.webhook_id:
                self.logger.debug("Retrieved webhook_id {} doesn't match".format(data['id']))
                payload = {'message': 'WEBHOOK_ID mismatch'}
                return BAD_REQUEST_NAME, BAD_REQUEST_CODE, payload

            person_id = data['data']['personId']
            msg_id = data['data']['id']
            txt = self._get_message(msg_id)
            room_id = data['data']['roomId']

            if 'text' not in txt:
                payload = {'message': 'Query not found'}
                return BAD_REQUEST_NAME, BAD_REQUEST_CODE, payload

            # Ignore the bot's own responses, else it would go into an infinite loop
            # of answering it's own questions.
            if person_id == me.id:
                payload = {'message': "Input query is the bot's previous message, \
                            so don't send it to the bot again"}
                return APPROVED_REQUEST_NAME, APPROVED_REQUEST_CODE, payload

            message = str(txt['text']).lower()
            payload = {'message': self._post_message(room_id, self.conv.say(message)[0])}
            return APPROVED_REQUEST_NAME, APPROVED_REQUEST_CODE, payload

    def run(self, host='localhost', port=7150):
        self.app.run(host=host, port=port)

    def _url(self, path):
        return "{0}{1}".format(CISCO_API_URL, path)

    def _get_message(self, msg_id):
        headers = {'Authorization': self.access_token_with_bearer}
        resp = requests.get(self._url('/messages/{0}'.format(msg_id)), headers=headers)
        response = json.loads(resp.text)
        response['status_code'] = str(resp.status_code)
        return response

    def _post_message(self, room_id, text):
        headers = {'Authorization': self.access_token_with_bearer,
                   'content-type': 'application/json'}
        payload = {'roomId': room_id, 'text': text}
        resp = requests.post(url=self._url('/messages'), json=payload, headers=headers)
        response = json.loads(resp.text)
        response['status_code'] = str(resp.status_code)
        return response
