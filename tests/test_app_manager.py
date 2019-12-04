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

from mock import patch

from mindmeld.app_manager import ApplicationManager, freeze_params
from mindmeld.components.request import FrozenParams, Params


@pytest.fixture
def app_manager(kwik_e_mart_app_path, kwik_e_mart_nlp):
    return ApplicationManager(kwik_e_mart_app_path, nlp=kwik_e_mart_nlp)


def test_freeze_params():
    params = freeze_params({"target_dialogue_state": "some-state"})
    assert params.__class__ == FrozenParams

    input_params = Params()
    input_params.target_dialogue_state = "some-state-2"
    params = freeze_params(input_params)
    assert params.__class__ == FrozenParams

    params = freeze_params(params)
    assert params.__class__ == FrozenParams

    with pytest.raises(TypeError):
        freeze_params([1, 2, 3])


def test_parse(app_manager):
    response = app_manager.parse("hello")

    fields = {"params", "request", "dialogue_state", "directives", "history"}
    for field in fields:
        assert field in vars(response).keys()


def test_language(kwik_e_mart_app_path, kwik_e_mart_nlp):
    app_manager = ApplicationManager(
        kwik_e_mart_app_path, kwik_e_mart_nlp, language="vi"
    )

    nlp_result = kwik_e_mart_nlp.process("hi")

    # first we test for passing language param into app manager's construction

    with patch.object(
        kwik_e_mart_nlp, "process", autospec=True, return_value=nlp_result
    ) as mock:
        app_manager.parse("hi")

    assert mock.call_args[1]["query_text"] == "hi"
    assert mock.call_args[1]["language"] == "vi"
    assert mock.call_args[1]["locale"] is None

    # we test for manually overriding language or locale using params

    with patch.object(
        kwik_e_mart_nlp, "process", autospec=True, return_value=nlp_result
    ) as mock:
        app_manager.parse("hi", params={"language": "en"})

    assert mock.call_args[1]["query_text"] == "hi"
    assert mock.call_args[1]["language"] == "en"
    assert mock.call_args[1]["locale"] is None

    with patch.object(
        kwik_e_mart_nlp, "process", autospec=True, return_value=nlp_result
    ) as mock:
        app_manager.parse("hi", params={"locale": "en_us"})

    assert mock.call_args[1]["query_text"] == "hi"
    assert mock.call_args[1]["locale"] == "en_US"
    assert mock.call_args[1]["language"] is None


def test_locale(kwik_e_mart_app_path, kwik_e_mart_nlp):
    app_manager = ApplicationManager(
        kwik_e_mart_app_path, kwik_e_mart_nlp, locale="vi_vi"
    )

    nlp_result = kwik_e_mart_nlp.process("hi")

    # first we test for passing locale param into app manager's construction

    with patch.object(
        kwik_e_mart_nlp, "process", autospec=True, return_value=nlp_result
    ) as mock:
        app_manager.parse("hi")

    assert mock.call_args[1]["query_text"] == "hi"
    assert mock.call_args[1]["language"] is None
    assert mock.call_args[1]["locale"] == "vi_VI"

    # we test for manually overriding language or locale using params

    with patch.object(
        kwik_e_mart_nlp, "process", autospec=True, return_value=nlp_result
    ) as mock:
        app_manager.parse("hi", params={"language": "en"})

    assert mock.call_args[1]["query_text"] == "hi"
    assert mock.call_args[1]["language"] == "en"
    assert mock.call_args[1]["locale"] is None

    with patch.object(
        kwik_e_mart_nlp, "process", autospec=True, return_value=nlp_result
    ) as mock:
        app_manager.parse("hi", params={"locale": "en_us"})

    assert mock.call_args[1]["query_text"] == "hi"
    assert mock.call_args[1]["locale"] == "en_US"
    assert mock.call_args[1]["language"] is None
