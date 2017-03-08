#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_entity_resolver
----------------------------------

Tests for `entity_resolver` module.
"""
# pylint: disable=locally-disabled,redefined-outer-name
from __future__ import unicode_literals

import pytest

from mmworkbench.core import Entity

from mmworkbench.processor.entity_resolver import EntityResolver

ENTITY_TYPE = 'location'


@pytest.fixture
def resolver(resource_loader):
    """An entity resolver for 'location' on the Kwik-E-Mart app"""
    return EntityResolver(resource_loader, ENTITY_TYPE)


def test_canonical(resolver):
    """Tests that entity resolution works for a canonical entity in the map"""
    expected = {'id': '2', 'cname': 'Pine and Market'}
    assert resolver.predict(Entity('Pine and Market', ENTITY_TYPE)) == expected


def test_synonym(resolver):
    """Tests that entity resolution works for an entity synonym in the map"""
    expected = {'id': '2', 'cname': 'Pine and Market'}
    assert resolver.predict(Entity('Pine St', ENTITY_TYPE)) == expected
