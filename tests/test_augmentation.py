#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_augmentation
----------------------------------

Tests the ``augmentation`` module.
"""

import pytest

from mindmeld.augmentation import EnglishParaphraser


@pytest.fixture(scope="module")
def english_paraphraser(kwik_e_mart_app_path):
    return EnglishParaphraser(kwik_e_mart_app_path)


@pytest.mark.parametrize(
    "query, num_paraphrases",
    [
        ("some text", 10),
        ("another text", 5),
        ("yet another text", 2),
    ],
)
def test_num_paraphrases(english_paraphraser, query, num_paraphrases):
    paraphrases = english_paraphraser.augment_queries(
        [query], num_return_sequences=num_paraphrases
    )
    assert num_paraphrases == len(paraphrases)
