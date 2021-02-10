#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_custom_action
----------------------------------

Tests for custom actions.
"""
import pytest
from unittest.mock import Mock, patch

from mindmeld import Application
from mindmeld.components import (
    CustomAction,
    invoke_custom_action,
    invoke_custom_action_async,
)
from mindmeld.components.dialogue import DialogueResponder
from mindmeld.components.request import Request


def test_custom_action_config(kwik_e_mart_app):
    """Test to get custom action config from app"""
    assert kwik_e_mart_app.custom_action_config is not None
    assert "url" in kwik_e_mart_app.custom_action_config
    assert kwik_e_mart_app.custom_action_config["url"] == "http://0.0.0.0:8080/"


def test_custom_action():
    """Test CustomAction.invoke to ensure that the parameters of the JSON body are correct"""
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


def test_custom_action_merge():
    "Test `merge=True` for custom actions"
    action_config = {"url": "http://localhost:8080/v2/action"}
    action = CustomAction(name="action_call_people", config=action_config)

    with patch("requests.post") as mock_object:
        mock_object.return_value = Mock()
        mock_object.return_value.status_code = 200
        mock_object.return_value.json.return_value = {
            "directives": [{"payload": "directive3"}, {"payload": "directive4"}],
            "frame": {"k2": "v2"},
            "slots": {"s2": "v2"},
            "params": {
                "allowed_intents": ["intent3", "intent4"],
                "dynamic_resource": {"r2": "v2"},
                "language": "some-language",
                "locale": "some-locale",
                "time_zone": "some-time-zone",
                "target_dialogue_state": "some-state",
                "timestamp": "some-timestamp",
            },
        }

        request = Request(
            text="sing a song", domain="some domain", intent="some intent"
        )
        responder = DialogueResponder()
        responder.directives = [{"payload": "directive1"}, {"payload": "directive2"}]
        responder.frame = {"k1": "v1"}
        responder.slots = {"s1": "v1"}
        responder.params.allowed_intents = ("intent1", "intent2")
        responder.params.dynamic_resource = {"r1": "v1"}
        assert action.invoke(request, responder)
        assert responder.directives == [
            {"payload": "directive1"},
            {"payload": "directive2"},
            {"payload": "directive3"},
            {"payload": "directive4"},
        ]
        assert responder.frame == {"k1": "v1", "k2": "v2"}
        assert responder.slots == {"s1": "v1", "s2": "v2"}
        assert responder.params.allowed_intents == (
            "intent1",
            "intent2",
            "intent3",
            "intent4",
        )
        assert responder.params.dynamic_resource == {"r1": "v1", "r2": "v2"}
        assert responder.params.target_dialogue_state == "some-state"
        assert responder.params.language == "some-language"
        assert responder.params.locale == "some-locale"
        assert responder.params.time_zone == "some-time-zone"
        assert responder.params.timestamp == "some-timestamp"


def test_custom_action_no_merge():
    "Test `merge=False` for custom actions"
    action_config = {"url": "http://localhost:8080/v2/action"}
    action = CustomAction(name="action_call_people", config=action_config, merge=False)

    with patch("requests.post") as mock_object:
        mock_object.return_value = Mock()
        mock_object.return_value.status_code = 200
        mock_object.return_value.json.return_value = {
            "directives": [{"payload": "directive3"}, {"payload": "directive4"}],
            "frame": {"k2": "v2"},
            "slots": {"s2": "v2"},
            "params": {
                "allowed_intents": ["intent3", "intent4"],
                "dynamic_resource": {"r2": "v2"},
                "language": "some-language",
                "locale": "some-locale",
                "time_zone": "some-time-zone",
                "target_dialogue_state": "some-state",
                "timestamp": "some-timestamp",
            },
        }

        request = Request(
            text="sing a song", domain="some domain", intent="some intent"
        )
        responder = DialogueResponder()
        responder.directives = ["directive1", "directive2"]
        responder.frame = {"k1": "v1"}
        responder.slots = {"s1": "v1"}
        responder.params.allowed_intents = ("intent1", "intent2")
        responder.params.dynamic_resource = {"r1": "v1"}
        assert action.invoke(request, responder)
        assert responder.directives == [{"payload": "directive3"},
                                        {"payload": "directive4"}]
        assert responder.frame == {"k2": "v2"}
        assert responder.slots == {"s2": "v2"}
        assert tuple(responder.params.allowed_intents) == (
            "intent3",
            "intent4",
        )
        assert responder.params.dynamic_resource == {"r2": "v2"}
        assert responder.params.target_dialogue_state == "some-state"
        assert responder.params.language == "some-language"
        assert responder.params.locale == "some-locale"
        assert responder.params.time_zone == "some-time-zone"
        assert responder.params.timestamp == "some-timestamp"


def test_invoke_custom_action():
    """Test invoke_custom_action to ensure that the parameters of the JSON body are correct"""
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
    """Test CustomAction.invoke_async to ensure that the parameters of the JSON body are correct"""
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
    """Test invoke_custom_action_async to ensure that the parameters of the JSON body are correct"""
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


def test_custom_action_handler(home_assistant_nlp):
    """Test Application.custom_action handle"""
    app = Application("home_assistant")
    app.lazy_init(home_assistant_nlp)
    app.custom_action_config = {"url": "some-url"}
    app.custom_action(intent="set_thermostat", action="set-thermostat")
    app.custom_action(default=True, action="times-and-dates")

    with patch("requests.post") as mock_object:
        mock_object.return_value = Mock()
        mock_object.return_value.status_code = 200
        mock_object.return_value.json.return_value = {
            "directives": [{"payload": "set-thermostat-action"}]
        }
        # invoke set thermostat intent
        res = app.app_manager.parse("turn it to 70 degrees")
        assert res.directives == [{"payload": "set-thermostat-action"}]
        assert mock_object.call_args[1]["url"] == "some-url"
        assert mock_object.call_args[1]["json"]["action"] == "set-thermostat"

        mock_object.return_value.json.return_value = {
            "directives": [{"payload": "time-and-dates-action"}]
        }
        # invoke time & dates intent
        res = app.app_manager.parse("change my alarm to 9")
        assert res.directives == [{"payload": "time-and-dates-action"}]
        assert mock_object.call_args[1]["url"] == "some-url"
        assert mock_object.call_args[1]["json"]["action"] == "times-and-dates"


def test_custom_action_sequence(home_assistant_nlp):
    """Test Application.custom_action handle for a sequence of actions"""
    app = Application("home_assistant")
    app.lazy_init(home_assistant_nlp)
    app.custom_action_config = {"url": "some-url"}
    app.custom_action(
        intent="set_thermostat", actions=["set-thermostat", "clear-thermostat"]
    )

    with patch("requests.post") as mock_object:
        mock_object.return_value = Mock()
        mock_object.return_value.status_code = 200
        mock_object.return_value.json.return_value = {"directives": [{"payload": "some-directive"}]}
        # invoke set thermostat intent and we should expect two directives
        res = app.app_manager.parse("turn it to 70 degrees")
        assert res.directives == [{"payload": "some-directive"}, {"payload": "some-directive"}]
        assert mock_object.call_args[1]["url"] == "some-url"


@pytest.mark.asyncio
async def test_custom_action_handler_async(home_assistant_nlp):
    """Test Application.custom_action handle with async mode"""
    app = Application("home_assistant", async_mode=True)
    app.lazy_init(home_assistant_nlp)
    app.custom_action_config = {"url": "some-url"}
    app.custom_action(intent="set_thermostat", action="set-thermostat", async_mode=True)
    app.custom_action(default=True, action="times-and-dates", async_mode=True)

    with patch("mindmeld.components.CustomAction.post_async") as mock_object:

        async def mock_coroutine():
            return 200, {"directives": [{"payload": "set-thermostat-action"}]}

        mock_object.return_value = mock_coroutine()

        # invoke set thermostat intent
        res = await app.app_manager.parse("turn it to 70 degrees")
        assert res.directives == [{"payload": "set-thermostat-action"}]
