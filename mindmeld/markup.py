# -*- coding: utf-8 -*-
"""The markup module contains functions for interacting with the MindMeld Markup language for
representing annotations of query text inline.
"""
from __future__ import unicode_literals


def create_parsed_query_for_markup(markup, tokenizer, preprocessor, is_gold=False):
    """Creates a parsed query object from marked up query text. The

    Args:
        markup (str): The marked up query text
        tokenizer (str): The tokenizer which should be used to tokenize the query text
        preprocessor (str): The preprocessor which should be used to process the query text
        is_gold (bool): True if the markup passed in is a reference, human-labeled example

    Returns:
        ParsedQuery: a parsed
    """
    pass


def create_markup_for_parsed_query(parsed_query):
    """Converts a parsed query into marked up query text.

    Args:
        parsed_query (ParsedQuery): The query to convert

    Returns:
        str: A marked up representation of the query
    """
    pass


def validate_markup(markup):
    """Checks whether the markup text is well-formed.

    Args:
        markup (str): Description

    Returns:
        bool: True if the markup is valid
    """
    pass
