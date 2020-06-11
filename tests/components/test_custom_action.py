#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_custom_action
----------------------------------

Tests for custom actions.
"""
import pytest
from unittest.mock import Mock, patch

from mindmeld.components import (
    CustomAction,
    invoke_custom_action,
    invoke_custom_action_async,
)
from mindmeld.components.dialogue import DialogueResponder
from mindmeld.components.request import Request


def test_custom_action_config(kwik_e_mart_app):
    assert kwik_e_mart_app.custom_action_config is not None
    assert "url" in kwik_e_mart_app.custom_action_config
    assert kwik_e_mart_app.custom_action_config["url"] == "http://0.0.0.0:8080/"


def test_custom_action():
    action_config = {"url": "http://localhost:8080/v2/action"}
    action = CustomAction(name="action_call_people", config=action_config)

    with patch("requests.post") as mock_object:
        mock_object.return_value = Mock()
        mock_object.return_value.status_code = 200
        mock_object.return_value.json.return_value = {}

        request = Request(
            text="sing a song", domain="some domain", intent="some intent"
        )
        responder = DialogueResponder()
        assert action.invoke(request, responder)
        assert mock_object.call_args[1]["url"] == action_config["url"]
        assert "request" in mock_object.call_args[1]["json"]
        assert "responder" in mock_object.call_args[1]["json"]
        assert mock_object.call_args[1]["json"]["action"] == "action_call_people"


def test_invoke_custom_action():
    action_config = {"url": "http://localhost:8080/v2/action"}

    with patch("requests.post") as mock_object:
        mock_object.return_value = Mock()
        mock_object.return_value.status_code = 200
        mock_object.return_value.json.return_value = {}

        request = Request(
            text="sing a song", domain="some domain", intent="some intent"
        )
        responder = DialogueResponder()
        assert invoke_custom_action(
            "action_call_people", action_config, request, responder
        )
        assert mock_object.call_args[1]["url"] == action_config["url"]
        assert "request" in mock_object.call_args[1]["json"]
        assert "responder" in mock_object.call_args[1]["json"]
        assert mock_object.call_args[1]["json"]["action"] == "action_call_people"


@pytest.mark.asyncio
async def test_custom_action_async():
    action_config = {"url": "http://localhost:8080/v2/action"}
    action = CustomAction(name="action_call_people", config=action_config)

    with patch("mindmeld.components.CustomAction.post_async") as mock_object:

        async def mock_coroutine():
            return 200, {}

        mock_object.return_value = mock_coroutine()
        request = Request(
            text="sing a song", domain="some domain", intent="some intent"
        )
        responder = DialogueResponder()
        assert await action.invoke_async(request, responder)
        call_args = mock_object.call_args_list[0][0][0]
        assert "request" in call_args
        assert "responder" in call_args
        assert call_args["action"] == "action_call_people"


@pytest.mark.asyncio
async def test_invoke_custom_action_async():
    action_config = {"url": "http://localhost:8080/v2/action"}

    with patch("mindmeld.components.CustomAction.post_async") as mock_object:

        async def mock_coroutine():
            return 200, {}

        mock_object.return_value = mock_coroutine()
        request = Request(
            text="sing a song", domain="some domain", intent="some intent"
        )
        responder = DialogueResponder()
        assert await invoke_custom_action_async(
            "action_call_people", action_config, request, responder
        )
        call_args = mock_object.call_args_list[0][0][0]
        assert "request" in call_args
        assert "responder" in call_args
        assert call_args["action"] == "action_call_people"
