# -*- coding: utf-8 -*-

"""
conftest
----------------------------------

Configurations for tests. Include shared fixtures here.
"""
from __future__ import unicode_literals

import pytest

from mmworkbench.tokenizer import Tokenizer
from mmworkbench.ser import SystemEntityRecognizer

from mmworkbench.core import QueryFactory


@pytest.fixture
def tokenizer():
    return Tokenizer()


@pytest.fixture
def sys_ent_rec():
    return SystemEntityRecognizer()


@pytest.fixture
def query_factory(sys_ent_rec, tokenizer):
    return QueryFactory(sys_ent_rec, tokenizer)
