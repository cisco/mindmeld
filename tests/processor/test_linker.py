#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_linker
----------------------------------

Tests for `linker` module.
"""
# pylint: disable=locally-disabled,redefined-outer-name
from __future__ import unicode_literals

import pytest

from mmworkbench.core import Entity

from mmworkbench.processor.linker import EntityLinker

ENTITY_TYPE = 'location'


@pytest.fixture
def linker(resource_loader, tokenizer):
    """An entity linker for 'location' on the Kwik-E-Mart app"""
    return EntityLinker(resource_loader, ENTITY_TYPE, tokenizer.normalize)


def test_canonical(linker):
    """Tests that entity linking works for a canonical entity in the map"""
    expected = {'id': '2', 'cname': 'Pine and Market'}
    assert linker.predict(Entity('Pine and Market', ENTITY_TYPE)) == expected


def test_synonym(linker):
    """Tests that entity linking works for an entity synonym in the map"""
    expected = {'id': '2', 'cname': 'Pine and Market'}
    assert linker.predict(Entity('Pine St', ENTITY_TYPE)) == expected
