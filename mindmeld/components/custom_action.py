import logging

import aiohttp
import requests

from .dialogue import DialogueResponder


logger = logging.getLogger(__name__)


RESPONSE_FIELDS = ["frame", "directives"]


class CustomActionException(Exception):
    pass


class CustomAction:
    """
    This class allows the client to send Request and Responder to another server and return the
      the directives and frame in the response.
    """

    def __init__(self, name: str, config: dict, overwrite: bool = False):
        self._name = name
        self._config = config or {}
        self.url = self._config.get("url")
        self.overwrite = overwrite

    def get_json_payload(self, request, responder):
        request_json = {
            "text": request.text,
            "domain": request.domain,
            "intent": request.intent,
            "context": dict(request.context),
            "params": request.params.to_dict(),
            "frame": dict(request.frame),
        }
        responder_json = DialogueResponder.to_json(responder)
        return {
            "request": request_json,
            "responder": {field: responder_json[field] for field in RESPONSE_FIELDS},
            "action": self._name,
        }

    def invoke(self, request, responder):
        """Invoke the custom action with Request and Responder and return True if the action is
        executed successfully, False otherwise. Upon successful execution, we update the Frame
        and Directives of the Responder object.

        Args:
            request (Request)
            responder (DialogueResponder)

        Returns:
            (bool)
        """
        if not self.url:
            raise CustomActionException(
                "No URL is given for custom action {}.".format(self._name)
            )

        json_data = self.get_json_payload(request, responder)

        try:
            status_code, result_json = self.post(json_data)

            return self._process_response(status_code, result_json, responder)
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
        if not self.url:
            raise CustomActionException(
                "No URL is given for custom action {}.".format(self._name)
            )

        json_data = self.get_json_payload(request, responder)

        try:
            status_code, result_json = await self.post_async(json_data)

            return self._process_response(status_code, result_json, responder)
        except ConnectionError:
            logger.error(
                "Connection error trying to reach custom action server %s.", self.url
            )
            return False

    def _process_response(self, status_code, result_json, responder):
        if status_code == 200:
            for field in RESPONSE_FIELDS:
                if field not in result_json:
                    logger.warning(
                        "`%s` not in the response of custom action %s",
                        field,
                        self._name,
                    )
            if self.overwrite:
                responder.frame = result_json.get("frame", {})
                responder.directives = result_json.get("directives", [])
            else:
                responder.frame.update(result_json.get("frame", {}))
                responder.directives.extend(result_json.get("directives", []))
            return True
        else:
            logger.error(
                "Error %s trying to reach custom action server %s.",
                status_code,
                self.url,
            )
            return False

    def post(self, json_data):
        result = requests.post(url=self.url, json=json_data)
        if result.status_code == 200:
            return 200, result.json()
        else:
            return result.status_code, {}

    async def post_async(self, json_data):
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, json=json_data) as response:
                if response.status == 200:
                    json_response = await response.json()
                    return 200, json_response
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

    def __init__(self, actions, config, overwrite=None):
        self.actions = [
            CustomAction(action, config, overwrite=overwrite) for action in actions
        ]

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


def invoke_custom_action(name, config, request, responder, overwrite=False):
    return CustomAction(name, config=config, overwrite=overwrite).invoke(
        request, responder
    )


async def invoke_custom_action_async(name, config, request, responder, overwrite=False):
    return await CustomAction(name, config=config, overwrite=overwrite).invoke_async(
        request, responder
    )
