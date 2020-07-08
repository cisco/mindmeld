import connexion

from ..models.data import Data
from ..models.responder import Responder
from ..models.directive import Directive


def invoke_action(data):
    """Invoke an action 

    :param data: MindMeld Data
    :type data: dict | bytes

    :rtype: Responder
    """
    if connexion.request.is_json:
        data = Data.from_dict(connexion.request.get_json())

    action = data.action
    msg = None

    if action == "action_restart":
        msg = "Restarting device..."
    elif action == "action_check_in":
        msg = "Checking you in now!"
    elif action == "action_check_out":
        msg = "You have been checked out!"
    elif action == "action_call_people":
        msg = "Perform call action on device."
    else:
        msg = "Invoking {action} on custom server.".format(action=data.action)

    reply = Directive(name="reply", payload={"text": msg}, type="view")
    responder = (Responder(directives=[reply], frame={}),)
    return responder
