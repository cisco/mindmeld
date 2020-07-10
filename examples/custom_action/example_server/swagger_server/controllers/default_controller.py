import connexion

from ..models.data import Data
from ..models.responder import Responder
from ..models.directive import Directive


def invoke_action(body):
    """Invoke an action

    This API accepts the serialized MindMeld Request and Responder and returns a serialized Responder

    :param body:
    :type body: dict | bytes

    :rtype: Responder
    """
    directives = []
    if connexion.request.is_json:
        data = Data.from_dict(body)

        msg = "Invoking {action} on custom server.".format(action=data.action)

        reply = Directive(name="reply", payload={"text": msg}, type="view")
        directives.append(reply)
    responder = Responder(directives=directives, frame={})
    return responder
