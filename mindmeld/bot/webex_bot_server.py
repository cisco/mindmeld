# -*- coding: utf-8 -*-

"""
This module contains the Webex Bot Server component.
"""

import json
import logging

from ciscosparkapi import CiscoSparkAPI
from flask import Flask, request
import requests

from ..components import NaturalLanguageProcessor
from ..components.dialogue import Conversation

CISCO_API_URL = "https://api.ciscospark.com/v1"
ACCESS_TOKEN_WITH_BEARER = "Bearer "
BAD_REQUEST_NAME = "BAD REQUEST"
BAD_REQUEST_CODE = 400
APPROVED_REQUEST_NAME = "OK"
APPROVED_REQUEST_CODE = 200


class WebexBotServerException(Exception):
    pass


class WebexBotServer:
    """
    A sample server class for Webex Teams integration with any MindMeld application
    """

    def __init__(self, name, app_path, nlp=None, webhook_id=None, access_token=None):
        """
        Args:
            name (str): The name of the server.
            app_path (str): The path of the MindMeld application.
            nlp (NaturalLanguageProcessor): MindMeld NLP component, will try to load from app path
              if None.
            webhook_id (str): Webex Team webhook id, will raise exception if not passed.
            access_token (str): Webex Team bot access token, will raise exception if not passed.
        """
        self.app = Flask(name)
        self.webhook_id = webhook_id
        self.access_token = access_token
        if not nlp:
            self.nlp = NaturalLanguageProcessor(app_path)
            self.nlp.load()
        else:
            self.nlp = nlp
        self.conv = Conversation(nlp=self.nlp, app_path=app_path)

        self.logger = logging.getLogger(__name__)

        if not self.webhook_id:
            raise WebexBotServerException("WEBHOOK_ID not set")
        if not self.access_token:
            raise WebexBotServerException("BOT_ACCESS_TOKEN not set")

        self.spark_api = CiscoSparkAPI(self.access_token)
        self.access_token_with_bearer = ACCESS_TOKEN_WITH_BEARER + self.access_token

        @self.app.route("/", methods=["POST"])
        def handle_message():  # pylint: disable=unused-variable
            me = self.spark_api.people.me()
            data = request.get_json()

            for key in ["personId", "id", "roomId"]:
                if key not in data["data"].keys():
                    payload = {"message": "personId/id/roomID key not found"}
                    return BAD_REQUEST_NAME, BAD_REQUEST_CODE, payload

            if data["id"] != self.webhook_id:
                self.logger.debug("Retrieved webhook_id %s doesn't match", data["id"])
                payload = {"message": "WEBHOOK_ID mismatch"}
                return BAD_REQUEST_NAME, BAD_REQUEST_CODE, payload

            person_id = data["data"]["personId"]
            msg_id = data["data"]["id"]
            txt = self._get_message(msg_id)
            room_id = data["data"]["roomId"]

            if "text" not in txt:
                payload = {"message": "Query not found"}
                return BAD_REQUEST_NAME, BAD_REQUEST_CODE, payload

            # Ignore the bot's own responses, else it would go into an infinite loop
            # of answering it's own questions.
            if person_id == me.id:
                payload = {
                    "message": "Input query is the bot's previous message, \
                            so don't send it to the bot again"
                }
                return APPROVED_REQUEST_NAME, APPROVED_REQUEST_CODE, payload

            message = str(txt["text"]).lower()
            payload = {
                "message": self._post_message(room_id, self.conv.say(message)[0])
            }
            return APPROVED_REQUEST_NAME, APPROVED_REQUEST_CODE, payload

    def run(self, host="localhost", port=7150):
        self.app.run(host=host, port=port)

    @staticmethod
    def _url(path):
        return "{0}{1}".format(CISCO_API_URL, path)

    def _get_message(self, msg_id):
        headers = {"Authorization": self.access_token_with_bearer}
        resp = requests.get(self._url("/messages/{0}".format(msg_id)), headers=headers)
        response = json.loads(resp.text)
        response["status_code"] = str(resp.status_code)
        return response

    def _post_message(self, room_id, text):
        headers = {
            "Authorization": self.access_token_with_bearer,
            "content-type": "application/json",
        }
        payload = {"roomId": room_id, "text": text}
        resp = requests.post(url=self._url("/messages"), json=payload, headers=headers)
        response = json.loads(resp.text)
        response["status_code"] = str(resp.status_code)
        return response
