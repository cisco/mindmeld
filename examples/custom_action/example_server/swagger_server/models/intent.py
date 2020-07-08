# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from .base_model_ import Model
from .. import util


class Intent(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    def __init__(self, value: str = None, confidence: float = None):
        """Intent - a model defined in Swagger

        :param value: The value of this Intent.
        :type value: str
        :param confidence: The confidence of this Intent.
        :type confidence: float
        """
        self.swagger_types = {"value": str, "confidence": float}

        self.attribute_map = {"value": "value", "confidence": "confidence"}

        self._value = value
        self._confidence = confidence

    @classmethod
    def from_dict(cls, dikt) -> "Intent":
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The Intent of this Intent.
        :rtype: Intent
        """
        return util.deserialize_model(dikt, cls)

    @property
    def value(self) -> str:
        """Gets the value of this Intent.


        :return: The value of this Intent.
        :rtype: str
        """
        return self._value

    @value.setter
    def value(self, value: str):
        """Sets the value of this Intent.


        :param value: The value of this Intent.
        :type value: str
        """

        self._value = value

    @property
    def confidence(self) -> float:
        """Gets the confidence of this Intent.


        :return: The confidence of this Intent.
        :rtype: float
        """
        return self._confidence

    @confidence.setter
    def confidence(self, confidence: float):
        """Sets the confidence of this Intent.


        :param confidence: The confidence of this Intent.
        :type confidence: float
        """

        self._confidence = confidence
