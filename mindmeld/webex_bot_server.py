# -*- coding: utf-8 -*-

"""
This module contains the Webex Bot Server component.
"""

import logging
import json
from flask import request
import requests
from ciscosparkapi import CiscoSparkAPI


class WebexBotServer:

    def __init__(self, app, WEBHOOK_ID, ACCESS_TOKEN, conv):
        self.app = app
        self.WEBHOOK_ID = WEBHOOK_ID
        self.ACCESS_TOKEN = ACCESS_TOKEN
        self.conv = conv

        self.logger = logging.getLogger(__name__)

        if not self.WEBHOOK_ID or not self.ACCESS_TOKEN:
            raise Exception('WEBHOOK_ID and BOT_ACCESS_TOKEN are not set')

        self.spark_api = CiscoSparkAPI(self.ACCESS_TOKEN)
        self.ACCESS_TOKEN_WITH_BEARER = 'Bearer ' + self.ACCESS_TOKEN
        self.CISCO_API_URL = 'https://api.ciscospark.com/v1'

    def run(self, host='localhost', port=7150):
        self.app.run(host=host, port=port)

    def _url(self, path):
        return self.CISCO_API_URL + path

    def _get_message(self, msg_id):
        headers = {'Authorization': self.ACCESS_TOKEN_WITH_BEARER}
        resp = requests.get(self._url('/messages/{0}'.format(msg_id)), headers=headers)
        response = json.loads(resp.text)
        response['status_code'] = str(resp.status_code)
        return response

    def _post_message(self, room_id, text):
        headers = {'Authorization': self.ACCESS_TOKEN_WITH_BEARER, 'content-type': 'application/json'}
        payload = {'roomId': room_id, 'text': text}
        resp = requests.post(url=self._url('/messages'), json=payload, headers=headers)
        response = json.loads(resp.text)
        response['status_code'] = str(resp.status_code)
        return response

    @self.app.route('/', methods=['POST'])
    def handle_message(self):
        me = self.spark_api.people.me()
        data = request.get_json()

        for key in ['personId', 'id', 'roomId']:
            if key not in data['data'].keys():
                return 'OK'

        if data['id'] != self.WEBHOOK_ID:
            self.logger.debug("Retrieved Webhook_id {} doesn't match".format(data['id']))
            return 'OK'

        person_id = data['data']['personId']
        msg_id = data['data']['id']
        txt = self._get_message(msg_id)
        room_id = data['data']['roomId']

        if 'text' not in txt:
            return 'OK'

        message = str(txt['text']).lower()

        # Ignore the bot's own responses, else it would go into an infinite loop
        # of answering it's own questions.
        if person_id != me.id:
            self._post_message(room_id, self.conv.say(message)[0])
        return 'OK'
