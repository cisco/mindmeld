import logging

import ssl
import aiohttp
import requests

from .request import Params


logger = logging.getLogger(__name__)


RESPONSE_FIELDS = ["frame", "directives", "params", "slots"]


class CustomActionException(Exception):
    pass


class CustomAction:
    """
    This class allows the client to send Request and Responder to another server and return the
      the directives and frame in the response.
    """

    def __init__(self, name: str, config: dict, merge: bool = True):
        self._name = name
        self._config = config or {}
        self.url = self._config.get("url")
        self._cert = self._config.get("cert")
        self._public_key = self._config.get("public_key")
        self._private_key = self._config.get("private_key")
        self.merge = merge

    def get_json_payload(self, request, responder):
        request_json = dict(request)
        responder_json = dict(responder)
        return {
            "request": request_json,
            "responder": {field: responder_json[field] for field in RESPONSE_FIELDS},
            "action": self._name,
        }

    def invoke(self, request, responder, async_mode=False):
        """Invoke the custom action with Request and Responder and return True if the action is
        executed successfully, False otherwise. Upon successful execution, we update the Frame
        and Directives of the Responder object.

        Args:
            request (Request)
            responder (DialogueResponder)
            async_mode (bool)

        Returns:
            (bool)
        """
        if not self.url:
            raise CustomActionException(
                "No URL is given for custom action {}.".format(self._name)
            )

        json_data = self.get_json_payload(request, responder)

        try:
            if async_mode:
                # returning the coroutine to be awaited elsewhere
                return self._process_async(json_data, responder)
            else:
                return self._process(json_data, responder)
        except ConnectionError:
            logger.error(
                "Connection error trying to reach custom action server %s.", self.url
            )
            return False

    async def invoke_async(self, request, responder):
        """Asynchronously invoke the custom action with Request and Responder and return True if
        the action is executed successfully, False otherwise. Upon successful execution, we update
        the Frame and Directives of the Responder object.

        Args:
            request (Request)
            responder (DialogueResponder)

        Returns:
            (bool)
        """
        return await self.invoke(request, responder, async_mode=True)

    def _process(self, json_data, responder):
        status_code, result_json = self.post(json_data)
        return self._process_post_response(status_code, result_json, responder)

    async def _process_async(self, json_data, responder):
        status_code, result_json = await self.post_async(json_data)
        return self._process_post_response(status_code, result_json, responder)

    def _process_post_response(self, status_code, result_json, responder):
        if status_code == 200:
            for field in RESPONSE_FIELDS:
                if field not in result_json:
                    logger.info(
                        "`%s` not in the response of custom action %s.",
                        field,
                        self._name,
                    )
            if self.merge:
                responder.frame.update(result_json.get("frame", {}))
                responder.directives.extend(result_json.get("directives", []))
                responder.slots.update(result_json.get("slots", {}))
                params = Params(**result_json.get("params", {}))
                responder.params.allowed_intents += tuple(params.allowed_intents)
                responder.params.dynamic_resource.update(params.dynamic_resource)
                responder.params.time_zone = params.time_zone
                responder.params.language = params.language
                responder.params.locale = params.locale
                responder.params.target_dialogue_state = params.target_dialogue_state
                responder.params.timestamp = params.timestamp
            else:
                responder.frame = result_json.get("frame", {})
                responder.directives = result_json.get("directives", [])
                responder.slots = result_json.get("slots", {})
                responder.params = Params(**result_json.get("params", {}))
            return True
        else:
            logger.error(
                "Error %s trying to reach custom action server %s.",
                status_code,
                self.url,
            )
            return False

    def post(self, json_data):
        if self._public_key and self._private_key:
            result = requests.post(
                url=self.url, json=json_data, cert=(self._public_key, self._private_key)
            )
        elif self._public_key:
            result = requests.post(url=self.url, json=json_data, cert=self._public_key)
        else:
            result = requests.post(url=self.url, json=json_data)
        if result.status_code == 200:
            return 200, result.json()
        else:
            return result.status_code, {}

    async def post_async(self, json_data):
        ssl_context = None
        if self._cert:
            ssl_context = ssl.create_default_context(cafile=self._cert)

        if self._public_key and self._private_key:
            ssl_context.load_cert_chain(self._public_key, self._private_key)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.url, json=json_data, ssl=ssl_context
            ) as response:
                if response.status == 200:
                    return 200, await response.json()
                else:
                    return response.status, {}

    def __repr__(self):
        return self._name

    def __str__(self):
        return self._name


class CustomActionSequence:
    """
    This class implements a sequence of custom actions
    """

    def __init__(self, actions, config, merge=True):
        self.actions = [CustomAction(action, config, merge=merge) for action in actions]

    def invoke(self, request, responder):
        for action in self.actions:
            result = action.invoke(request, responder)
            if not result:
                logger.warning("Failed to invoke action %s.", action)
                return False
        return True

    async def invoke_async(self, request, responder):
        for action in self.actions:
            result = await action.invoke_async(request, responder)
            if not result:
                logger.warning("Failed to invoke action %s.", action)
                return False
        return True

    def __repr__(self):
        return str(self.actions)

    def __str__(self):
        return "action_seq=" + str(self.actions)


def invoke_custom_action(name, config, request, responder, merge=True):
    return CustomAction(name, config, merge=merge).invoke(request, responder)


async def invoke_custom_action_async(name, config, request, responder, merge=True):
    return await CustomAction(name, config, merge=merge).invoke_async(
        request, responder
    )
