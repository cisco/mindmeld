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
    This module contains the class which serves the MindMeld API.
"""
import json
import logging
import os
import sys
import time
import uuid

from marshmallow.exceptions import ValidationError
from flask import Flask, Request, g, jsonify, request
from flask_cors import CORS

from ._version import current as __version__
from .components.schemas import RequestSchema
from .exceptions import BadMindMeldRequestError

logger = logging.getLogger(__name__)


class MindMeldRequest(Request):  # pylint: disable=too-many-ancestors
    """This class represents requests to the MindMeldServer. It extends
    flask.Request to provide
    custom handling of certain exceptions.
    """

    def on_json_loading_failed(self, exc):
        """Called if decoding of the JSON data failed.

        The return value of this method is used by get_json() when an error
        occurred. The default implementation just raises a BadRequest exception.
        """
        del exc
        raise BadMindMeldRequestError(
            "Malformed request body: {0:s}".format(sys.exc_info()[1])
        )


class MindMeldServer:
    """This class sets up a Flask web server."""

    def __init__(self, app_manager):
        self._app_manager = app_manager
        self._request_schema = RequestSchema(
            context={
                "nlp": self._app_manager.nlp,
                "dialogue_handler_map": self._app_manager.dialogue_manager.handler_map
            }
        )
        self._request_logger = logger.getChild("requests")

        server = Flask("mindmeld")
        CORS(server)

        server.request_class = MindMeldRequest

        server.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 16

        # Set the version for logging purposes
        self._package_version = __version__
        self._app_version = None
        if os.environ.get("MM_APP_VERSION"):
            self._app_version = os.environ.get("MM_APP_VERSION")
        elif os.environ.get("MM_APP_VERSION_FILE"):
            version_file = os.environ.get("MM_APP_VERSION_FILE")
            try:
                with open(version_file, "r") as file:
                    self._app_version = file.readline().strip()
            except (OSError, IOError):
                # failed to set version
                logger.warning("Failed to open app version file: '%s'", version_file)

        # pylint: disable=unused-variable
        @server.route("/parse", methods=["POST"])
        def parse():
            """The main endpoint for the MindMeld API"""
            request_json = request.get_json()
            if request_json is None:
                msg = "Invalid Content-Type: Only 'application/json' is supported."
                raise BadMindMeldRequestError(msg, status_code=415)

            try:
                validated_request = self._request_schema.load(request_json)
                response = self._app_manager.parse(
                    text=validated_request.get('text'),
                    params=validated_request.get('params'),
                    context=validated_request.get('context'),
                    frame=validated_request.get('frame'),
                    history=validated_request.get('history'),
                    form=validated_request.get('form'),
                    verbose=validated_request.get('verbose', False)
                )
                # add request id to response
                # use the passed in id if any
                response = dict(response)
                request_id = validated_request.get("request_id", str(uuid.uuid4()))
                response['request_id'] = request_id
                return jsonify(response)
            except (ValidationError, ValueError, KeyError) as e:
                err_message = "Bad request {} caused error {}".format(request_json, e)
                logger.error(err_message)
                raise BadMindMeldRequestError(err_message, status_code=400) from e

        @server.before_request
        def _before_request():
            g.start_time = time.time()

        @server.after_request
        def _after_request(response):
            g.response_time = time.time() - g.start_time
            g.response = response

            # add response time to response
            try:
                data = json.loads(response.get_data(as_text=True))
                data["response_time"] = g.response_time
                data["version"] = "2.0"
                response.set_data(json.dumps(data))
            except json.JSONDecodeError:
                pass

            return response

        @server.teardown_request
        def _teardown_request(error):
            del error
            if hasattr(g, "log_this_request") and g.log_this_request:
                response = g.get("response", None)
                self._log_request(request, response)

        # handle exceptions
        @server.errorhandler(BadMindMeldRequestError)
        def handle_bad_request(error):
            response = jsonify(error.to_dict())
            response.status_code = error.status_code
            logger.error(json.dumps(error.to_dict()))
            return response

        @server.errorhandler(500)
        def handle_server_error(error):
            response_data = {"error": error.message}
            response = jsonify(response_data)
            response.status_code = 500
            logger.error(json.dumps(response_data))
            return response

        @server.route("/_status", methods=["GET"])
        def status_check():
            body = {"status": "OK", "package_version": self._package_version}
            if self._app_version:
                body["app_version"] = self._app_version
            return jsonify(body)

        self._server = server

    def run(self, **kwargs):
        """Starts the flask server."""
        self._server.run(**kwargs)

    def _log_request(self, req, response):
        try:
            response_data = json.loads(response.data)

            if req.headers.getlist("X-Forwarded-For"):
                ip_address = req.headers.getlist("X-Forwarded-For")[0]
            else:
                ip_address = req.remote_addr

            if req.method == "GET":
                request_data = req.args.to_dict()
            else:
                request_data = req.get_json()

            # TODO add hook for app to modify logged info

        except (KeyError, ValueError, AttributeError) as exc:
            logger.warning("Error occured while logging request")
            logger.debug("Response: %s\nerror: %s", response, exc)
            return

        log_request_data = {}
        log_request_data["response"] = response_data
        log_request_data["request"] = request_data
        if hasattr(g, "response_time"):
            log_request_data["response_time"] = g.response_time
        if hasattr(g, "app_name"):
            log_request_data["app_name"] = g.app_name

        log_request_data["ip"] = ip_address
        log_request_data["platform"] = req.user_agent.platform
        log_request_data["url_root"] = req.url_root
        log_request_data["base_url"] = req.base_url
        log_request_data["source"] = {"type": "app"}
        log_request_data["source"]["package_version"] = self._package_version
        if os.environ.get("MM_NODE_NAME"):
            log_request_data["source"]["node_name"] = os.environ.get("MM_NODE_NAME")
        if self._app_version:
            log_request_data["source"]["app_version"] = self._app_version

        self._request_logger.info(json.dumps(log_request_data))
