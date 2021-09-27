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
This module contains a preprocessor base class.
"""
from abc import ABC, abstractmethod


class Preprocessor(ABC):
    """
    Base class for Preprocessor object
    """

    @abstractmethod
    def process(self, text):
        """
        Args:
            text (str)

        Returns:
            (str)
        """
        pass

    def tojson(self):
        """
        Method defined to obtain recursive JSON representation of a TextPreparationPipeline.

        Args:
            None.

        Returns:
            JSON representation of Preprocessor (dict) .
        """
        return {self.__class__.__name__: None}


class NoOpPreprocessor(Preprocessor):
    """
    NoOpPreprocessor object
    """

    def process(self, text):
        """
        Args:
            text (str)

        Returns:
            (str)
        """
        return text


class PreprocessorFactory:
    """Preprocessor Factory Class"""

    @staticmethod
    def get_preprocessor(preprocessor: str):
        """A static method to get a Preprocessor

        Args:
            preprocessor (str): Name of the desired Preprocessor class
        Returns:
            (Preprocessor): Preprocessor Class
        """
        if preprocessor == NoOpPreprocessor.__name__:
            return NoOpPreprocessor()
        raise TypeError(f"{preprocessor} is not a valid Preprocessor type.")
