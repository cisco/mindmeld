# -*- coding: utf-8 -*-
"""
This module contains a preprocessor base class.
"""
from abc import ABCMeta, abstractmethod


class Preprocessor(metaclass=ABCMeta):
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
