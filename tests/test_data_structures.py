#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_markup
----------------------------------

Tests for `markup` module.
"""
from __future__ import unicode_literals

import pytest

from mmworkbench.tokenizer import Tokenizer

from mmworkbench.data_structures import (Query, TEXT_FORM_RAW, TEXT_FORM_PROCESSED,
                                         TEXT_FORM_NORMALIZED)


@pytest.fixture
def tokenizer():
    return Tokenizer()


@pytest.fixture
def query(tokenizer):
    return Query('Test: One. 2. 3.', tokenizer)


def test_query(tokenizer):
    text = 'Test: 1. 2. 3.'
    query = Query(text, tokenizer)

    assert query.raw_text == text
    assert query.processed_text == text
    assert query.normalized_text == 'test 1 2 3'


def test_transform_index_forward(query):
    raw_index = 6
    raw_char = query.raw_text[raw_index]

    proc_index = query.transform_index(raw_index, TEXT_FORM_RAW, TEXT_FORM_PROCESSED)
    proc_char = query.processed_text[proc_index]

    norm_index = query.transform_index(raw_index, TEXT_FORM_RAW, TEXT_FORM_NORMALIZED)
    norm_char = query.normalized_text[norm_index]

    assert raw_char == 'O'

    assert proc_index == raw_index
    assert proc_char == raw_char

    assert norm_index == 5
    assert norm_char == 'o'


def test_transform_index_backward(query):
    norm_index = 5
    norm_char = query.normalized_text[norm_index]

    proc_index = query.transform_index(norm_index, TEXT_FORM_NORMALIZED, TEXT_FORM_PROCESSED)
    proc_char = query.processed_text[proc_index]

    raw_index = query.transform_index(norm_index, TEXT_FORM_NORMALIZED, TEXT_FORM_RAW)
    raw_char = query.raw_text[raw_index]

    assert norm_char == 'o'

    assert proc_index == raw_index
    assert proc_char == raw_char

    assert raw_index == 6
    assert raw_char == 'O'


def test_transform_index_backward_2(query):
    norm_index = 7
    norm_char = query.normalized_text[norm_index]

    proc_index = query.transform_index(norm_index, TEXT_FORM_NORMALIZED, TEXT_FORM_PROCESSED)
    proc_char = query.processed_text[proc_index]

    raw_index = query.transform_index(norm_index, TEXT_FORM_NORMALIZED, TEXT_FORM_RAW)
    raw_char = query.raw_text[raw_index]

    assert norm_char == 'e'

    assert proc_index == raw_index
    assert proc_char == raw_char

    assert raw_index == 8
    assert raw_char == 'e'


def test_transform_index_backward_3(query):
    norm_index = 8
    norm_char = query.normalized_text[norm_index]

    proc_index = query.transform_index(norm_index, TEXT_FORM_NORMALIZED, TEXT_FORM_PROCESSED)
    proc_char = query.processed_text[proc_index]

    raw_index = query.transform_index(norm_index, TEXT_FORM_NORMALIZED, TEXT_FORM_RAW)
    raw_char = query.raw_text[raw_index]

    assert norm_char == ' '

    assert proc_index == raw_index
    assert proc_char == raw_char

    assert raw_index == 10
    assert raw_char == ' '


def test_transform_range_forward(query):
    raw_span = (0, 10)
    raw_text = query.raw_text[raw_span[0]:raw_span[1]]

    proc_span = query.transform_range(raw_span, TEXT_FORM_RAW, TEXT_FORM_PROCESSED)
    proc_text = query.processed_text[proc_span[0]:proc_span[1]]

    norm_span = query.transform_range(raw_span, TEXT_FORM_RAW, TEXT_FORM_NORMALIZED)
    norm_text = query.normalized_text[norm_span[0]:norm_span[1]]

    assert raw_text == 'Test: One.'

    assert proc_span == raw_span
    assert proc_text == raw_text

    assert norm_span == (0, 8)
    assert norm_text == 'test one'


def test_transform_range_backward(query):
    norm_span = (0, 8)
    norm_text = query.normalized_text[norm_span[0]:norm_span[1]]

    proc_span = query.transform_range(norm_span, TEXT_FORM_NORMALIZED, TEXT_FORM_PROCESSED)
    proc_text = query.processed_text[proc_span[0]:proc_span[1]]

    raw_span = query.transform_range(norm_span, TEXT_FORM_NORMALIZED, TEXT_FORM_RAW)
    raw_text = query.raw_text[raw_span[0]:raw_span[1]]

    assert norm_text == 'test one'

    assert proc_span == raw_span
    assert proc_text == raw_text

    assert raw_span == (0, 10)
    assert raw_text == 'Test: One.'
