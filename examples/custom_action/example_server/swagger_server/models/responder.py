# coding: utf-8

from __future__ import absolute_import

from typing import List

from .base_model_ import Model
from .directive import Directive
from .params import Params
from .. import util


class Responder(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    def __init__(
        self,
        directives: List[Directive] = None,
        frame: object = None,
        params: Params = None,
        slots: object = None,
    ):
        """Responder - a model defined in Swagger

        :param directives: The directives of this Responder.
        :type directives: List[Directive]
        :param frame: The frame of this Responder.
        :type frame: object
        :param params: The params of this Responder.
        :type params: Params
        :param slots: The slots of this Responder.
        :type slots: object
        """
        self.swagger_types = {
            "directives": List[Directive],
            "frame": object,
            "params": Params,
            "slots": object,
        }

        self.attribute_map = {
            "directives": "directives",
            "frame": "frame",
            "params": "params",
            "slots": "slots",
        }
        self._directives = directives
        self._frame = frame
        self._params = params
        self._slots = slots

    @classmethod
    def from_dict(cls, dikt) -> "Responder":
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The Responder of this Responder.
        :rtype: Responder
        """
        return util.deserialize_model(dikt, cls)

    @property
    def directives(self) -> List[Directive]:
        """Gets the directives of this Responder.

        The list of directives (such as replies) to be sent to the user

        :return: The directives of this Responder.
        :rtype: List[Directive]
        """
        return self._directives

    @directives.setter
    def directives(self, directives: List[Directive]):
        """Sets the directives of this Responder.

        The list of directives (such as replies) to be sent to the user

        :param directives: The directives of this Responder.
        :type directives: List[Directive]
        """

        self._directives = directives

    @property
    def frame(self) -> object:
        """Gets the frame of this Responder.

        The frame object of the responder, which contains key-value pairs to send to the application

        :return: The frame of this Responder.
        :rtype: object
        """
        return self._frame

    @frame.setter
    def frame(self, frame: object):
        """Sets the frame of this Responder.

        The frame object of the responder, which contains key-value pairs to send to the application

        :param frame: The frame of this Responder.
        :type frame: object
        """

        self._frame = frame

    @property
    def params(self) -> Params:
        """Gets the params of this Responder.


        :return: The params of this Responder.
        :rtype: Params
        """
        return self._params

    @params.setter
    def params(self, params: Params):
        """Sets the params of this Responder.


        :param params: The params of this Responder.
        :type params: Params
        """

        self._params = params

    @property
    def slots(self) -> object:
        """Gets the slots of this Responder.

        The slots object of the responder, which contains key-value pairs that can be used to render the natural language responses

        :return: The slots of this Responder.
        :rtype: object
        """
        return self._slots

    @slots.setter
    def slots(self, slots: object):
        """Sets the slots of this Responder.

        The slots object of the responder, which contains key-value pairs that can be used to render the natural language responses

        :param slots: The slots of this Responder.
        :type slots: object
        """

        self._slots = slots
