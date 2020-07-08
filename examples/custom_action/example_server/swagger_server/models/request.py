# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from .base_model_ import Model
from .entity import Entity
from .. import util


class Request(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    def __init__(
        self,
        text: str = None,
        domain: str = None,
        intent: str = None,
        entities: List[Entity] = None,
        context: object = None,
        params: object = None,
    ):  # noqa: E501
        """Request - a model defined in Swagger

        :param text: The text of this Request.  # noqa: E501
        :type text: str
        :param domain: The domain of this Request.  # noqa: E501
        :type domain: str
        :param intent: The intent of this Request.  # noqa: E501
        :type intent: str
        :param entities: The entities of this Request.  # noqa: E501
        :type entities: List[Entity]
        :param context: The context of this Request.  # noqa: E501
        :type context: object
        :param params: The params of this Request.  # noqa: E501
        :type params: object
        """
        self.swagger_types = {
            "text": str,
            "domain": str,
            "intent": str,
            "entities": List[Entity],
            "context": object,
            "params": object,
        }

        self.attribute_map = {
            "text": "text",
            "domain": "domain",
            "intent": "intent",
            "entities": "entities",
            "context": "context",
            "params": "params",
        }

        self._text = text
        self._domain = domain
        self._intent = intent
        self._entities = entities
        self._context = context
        self._params = params

    @classmethod
    def from_dict(cls, dikt) -> "Request":
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The Request of this Request.  # noqa: E501
        :rtype: Request
        """
        return util.deserialize_model(dikt, cls)

    @property
    def text(self) -> str:
        """Gets the text of this Request.

        The query text  # noqa: E501

        :return: The text of this Request.
        :rtype: str
        """
        return self._text

    @text.setter
    def text(self, text: str):
        """Sets the text of this Request.

        The query text  # noqa: E501

        :param text: The text of this Request.
        :type text: str
        """

        self._text = text

    @property
    def domain(self) -> str:
        """Gets the domain of this Request.

        Domain of the current query.  # noqa: E501

        :return: The domain of this Request.
        :rtype: str
        """
        return self._domain

    @domain.setter
    def domain(self, domain: str):
        """Sets the domain of this Request.

        Domain of the current query.  # noqa: E501

        :param domain: The domain of this Request.
        :type domain: str
        """

        self._domain = domain

    @property
    def intent(self) -> str:
        """Gets the intent of this Request.

        Intent of the current query.  # noqa: E501

        :return: The intent of this Request.
        :rtype: str
        """
        return self._intent

    @intent.setter
    def intent(self, intent: str):
        """Sets the intent of this Request.

        Intent of the current query.  # noqa: E501

        :param intent: The intent of this Request.
        :type intent: str
        """

        self._intent = intent

    @property
    def entities(self) -> List[Entity]:
        """Gets the entities of this Request.

        A list of entities in the current query.  # noqa: E501

        :return: The entities of this Request.
        :rtype: List[Entity]
        """
        return self._entities

    @entities.setter
    def entities(self, entities: List[Entity]):
        """Sets the entities of this Request.

        A list of entities in the current query.  # noqa: E501

        :param entities: The entities of this Request.
        :type entities: List[Entity]
        """

        self._entities = entities

    @property
    def context(self) -> object:
        """Gets the context of this Request.

        Map containing front-end client state that is passed to the application from the client in the request.  # noqa: E501

        :return: The context of this Request.
        :rtype: object
        """
        return self._context

    @context.setter
    def context(self, context: object):
        """Sets the context of this Request.

        Map containing front-end client state that is passed to the application from the client in the request.  # noqa: E501

        :param context: The context of this Request.
        :type context: object
        """

        self._context = context

    @property
    def params(self) -> object:
        """Gets the params of this Request.

        Map of stored data across multiple dialogue turns.  # noqa: E501

        :return: The params of this Request.
        :rtype: object
        """
        return self._params

    @params.setter
    def params(self, params: object):
        """Sets the params of this Request.

        Map of stored data across multiple dialogue turns.  # noqa: E501

        :param params: The params of this Request.
        :type params: object
        """

        self._params = params
