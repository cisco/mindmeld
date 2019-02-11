#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_dialogue
----------------------------------

Tests for app_manager module.

These tests apply only when async/await are supported.
"""
# pylint: disable=locally-disabled,redefined-outer-name
import pytest

from mmworkbench.app_manager import ApplicationManager


@pytest.fixture
def app_manager(kwik_e_mart_app_path, kwik_e_mart_nlp):
    return ApplicationManager(kwik_e_mart_app_path, nlp=kwik_e_mart_nlp)


def test_parse(app_manager):
    response = app_manager.parse('hello')

    fields = {'params', 'request', 'dialogue_state', 'directives', 'history'}
    for field in fields:
        assert field in vars(response).keys()
