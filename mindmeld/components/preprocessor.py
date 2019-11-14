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

    @abstractmethod
    def get_char_index_map(self, raw_text, processed_text):
        """
        Generates character index mapping from processed query to raw query.

        See the Tokenizer class for a similar implementation.

        Args:
            raw_text (str)
            processed_text (str)

        Returns:
            (dict, dict): A tuple consisting of two maps, forward and backward
        """
        pass
