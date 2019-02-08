# -*- coding: utf-8 -*-
"""
    This module contains the class which serves the workbench API.
"""
import logging
import json
import os
import sys
import time
import uuid

from flask import Flask, Request, request, jsonify, g
from flask_cors import CORS

from ._version import current as __version__
from .exceptions import BadWorkbenchRequestError
from .components.dialogue import DialogueResponder

logger = logging.getLogger(__name__)


class WorkbenchRequest(Request):
    """This class represents requests to the WorkbenchServer. It extends
    flask.Request to provide
    custom handling of certain exceptions.
    """

    def on_json_loading_failed(self, exc):
        """Called if decoding of the JSON data failed.

        The return value of this method is used by get_json() when an error
        occurred. The default implementation just raises a BadRequest exception.
        """
        raise BadWorkbenchRequestError("Malformed request body: {0:s}".format(sys.exc_info()[1]))


class WorkbenchServer:
    """This class sets up a Flask web server."""

    def __init__(self, app_manager):
        self._app_manager = app_manager
        self._request_logger = logger.getChild('requests')

        server = Flask('workbench')
        CORS(server)

        server.request_class = WorkbenchRequest

        server.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 16

        # Set the version for logging purposes
        self._package_version = __version__
        self._app_version = None
        if os.environ.get('MM_APP_VERSION'):
            self._app_version = os.environ.get('MM_APP_VERSION')
        elif os.environ.get('MM_APP_VERSION_FILE'):
            version_file = os.environ.get('MM_APP_VERSION_FILE')
            try:
                with open(version_file, 'r') as file:
                    self._app_version = file.readline().strip()
            except (OSError, IOError):
                # failed to set version
                logger.warning("Failed to open app version file: '{}'".format(version_file))

        # pylint: disable=locally-disabled,unused-variable
        @server.route('/parse', methods=['POST'])
        def parse():
            """The main endpoint for the workbench API"""
            request_json = request.get_json()
            if request_json is None:
                msg = "Invalid Content-Type: Only 'application/json' is supported."
                raise BadWorkbenchRequestError(msg, status_code=415)

            safe_request = {}
            for key in ['text', 'params', 'context', 'frame', 'history', 'verbose']:
                if key in request_json:
                    safe_request[key] = request_json[key]
            response = self._app_manager.parse(**safe_request)
            # add request id to response
            # use the passed in id if any
            request_id = request_json.get('request_id', str(uuid.uuid4()))
            response.request_id = request_id
            return jsonify(DialogueResponder.to_json(response))

        @server.before_request
        def _before_request(*args, **kwargs):
            g.start_time = time.time()

        @server.after_request
        def _after_request(response):
            g.response_time = time.time() - g.start_time
            g.response = response

            # add response time to response
            try:
                data = json.loads(response.get_data(as_text=True))
                data['response_time'] = g.response_time
                data['version'] = '2.0'
                response.set_data(json.dumps(data))
            except json.JSONDecodeError:
                pass

            return response

        @server.teardown_request
        def _teardown_request(*args, **kwargs):
            if hasattr(g, 'log_this_request') and g.log_this_request:
                response = g.get('response', None)
                self._log_request(request, response)

        # handle exceptions
        @server.errorhandler(BadWorkbenchRequestError)
        def handle_bad_request(error):
            response = jsonify(error.to_dict())
            response.status_code = error.status_code
            # TODO: should this be in the request log?
            logger.error(json.dumps(error.to_dict()))
            return response

        @server.errorhandler(500)
        def handle_server_error(error):
            # TODO: only expose this when verbose param is true
            response_data = {'error': error.message}
            response = jsonify(response_data)
            response.status_code = 500
            # TODO: should this be in the request log?
            logger.error(json.dumps(response_data))
            return response

        @server.route('/_status', methods=['GET'])
        def status_check():
            body = {'status': 'OK', 'package_version': self._package_version}
            if self._app_version:
                body['app_version'] = self._app_version
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

            if req.method == 'GET':
                request_data = req.args.to_dict()
            else:
                request_data = req.get_json()

            # TODO add hook for app to modify logged info

        except (KeyError, ValueError, AttributeError) as exc:
            logger.warning('Error occured while logging request')
            logger.debug('Response: %s\nerror: %s', response, exc)
            return

        log_request_data = {}
        log_request_data['response'] = response_data
        log_request_data['request'] = request_data
        if hasattr(g, 'response_time'):
            log_request_data['response_time'] = g.response_time
        if hasattr(g, 'app_name'):
            log_request_data['app_name'] = g.app_name

        log_request_data['ip'] = ip_address
        log_request_data['platform'] = req.user_agent.platform
        log_request_data['url_root'] = req.url_root
        log_request_data['base_url'] = req.base_url
        log_request_data['source'] = {'type': 'app'}
        log_request_data['source']['package_version'] = self._package_version
        if os.environ.get('MM_NODE_NAME'):
            log_request_data['source']['node_name'] = os.environ.get('MM_NODE_NAME')
        if self._app_version:
            log_request_data['source']['app_version'] = self._app_version

        self._request_logger.info(json.dumps(log_request_data))
