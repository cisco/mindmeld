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
async def test_language_locale(async_app_manager):
    kwik_e_mart_nlp = async_app_manager.nlp

    nlp_result = kwik_e_mart_nlp.process("hi")

    # we test for manually overriding language or locale using params

    with patch.object(
        kwik_e_mart_nlp, "process", autospec=True, return_value=nlp_result
    ) as mock:
        await async_app_manager.parse("hi", params={"language": "vi"})

    assert mock.call_args[1]["query_text"] == "hi"
    assert mock.call_args[1]["language"] == "vi"
    assert mock.call_args[1]["locale"] is None

    with patch.object(
        kwik_e_mart_nlp, "process", autospec=True, return_value=nlp_result
    ) as mock:
        await async_app_manager.parse("hi", params={"locale": "vi_vi"})

    assert mock.call_args[1]["query_text"] == "hi"
    assert mock.call_args[1]["locale"] == "vi_VI"
    assert mock.call_args[1]["language"] is None
