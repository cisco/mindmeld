#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_dialogue
----------------------------------

Tests for app_manager module.

These tests apply only when async/await are supported.
"""
import asyncio
import pytest

from mock import patch


@pytest.fixture(scope="function")
def async_app_manager(kwik_e_mart_nlp):
    from .kwik_e_mart import app_async

    app = app_async.app
    app.lazy_init(kwik_e_mart_nlp)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(app.app_manager.load())

    return app.app_manager


@pytest.mark.asyncio
async def test_parse(async_kwik_e_mart_app):
    response = await async_kwik_e_mart_app.app_manager.parse("hello")
    fields = {"params", "request", "dialogue_state", "directives", "history"}
    nested_fields = {"domain", "intent", "entities"}
    for field in fields:
        assert field in vars(response).keys()
    for field in nested_fields:
        assert field in vars(response.request).keys()


@pytest.mark.asyncio
async def test_language(async_app_manager):
    kwik_e_mart_nlp = async_app_manager.nlp

    nlp_result = kwik_e_mart_nlp.process("hi")

    # first we test for passing language param into app manager
    async_app_manager.language = "vi"
    async_app_manager.locale = None

    with patch.object(
        kwik_e_mart_nlp, "process", autospec=True, return_value=nlp_result
    ) as mock:
        await async_app_manager.parse("hi")

    assert mock.call_args[1]["query_text"] == "hi"
    assert mock.call_args[1]["language"] == "vi"
    assert mock.call_args[1]["locale"] is None

    # we test for manually overriding language or locale using params

    with patch.object(
        kwik_e_mart_nlp, "process", autospec=True, return_value=nlp_result
    ) as mock:
        await async_app_manager.parse("hi", params={"language": "en"})

    assert mock.call_args[1]["query_text"] == "hi"
    assert mock.call_args[1]["language"] == "en"
    assert mock.call_args[1]["locale"] is None

    with patch.object(
        kwik_e_mart_nlp, "process", autospec=True, return_value=nlp_result
    ) as mock:
        await async_app_manager.parse("hi", params={"locale": "en_us"})

    assert mock.call_args[1]["query_text"] == "hi"
    assert mock.call_args[1]["locale"] == "en_US"
    assert mock.call_args[1]["language"] is None


@pytest.mark.asyncio
async def test_locale(async_app_manager):
    kwik_e_mart_nlp = async_app_manager.nlp

    nlp_result = kwik_e_mart_nlp.process("hi")

    # first we test for passing locale param into app manager's construction
    # not sure why fixture is not resetting properly
    async_app_manager.language = None
    async_app_manager.locale = "vi_VI"

    with patch.object(
        kwik_e_mart_nlp, "process", autospec=True, return_value=nlp_result
    ) as mock:
        await async_app_manager.parse("hi")

    assert mock.call_args[1]["query_text"] == "hi"
    assert mock.call_args[1]["language"] is None
    assert mock.call_args[1]["locale"] == "vi_VI"

    # we test for manually overriding language or locale using params

    with patch.object(
        kwik_e_mart_nlp, "process", autospec=True, return_value=nlp_result
    ) as mock:
        await async_app_manager.parse("hi", params={"language": "en"})

    assert mock.call_args[1]["query_text"] == "hi"
    assert mock.call_args[1]["language"] == "en"
    assert mock.call_args[1]["locale"] is None

    with patch.object(
        kwik_e_mart_nlp, "process", autospec=True, return_value=nlp_result
    ) as mock:
        await async_app_manager.parse("hi", params={"locale": "en_us"})

    assert mock.call_args[1]["query_text"] == "hi"
    assert mock.call_args[1]["locale"] == "en_US"
    assert mock.call_args[1]["language"] is None
