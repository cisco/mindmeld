import pytest

from mindmeld.components import Conversation
from .test_dialogue_flow import assert_target_dialogue_state, assert_reply


@pytest.mark.conversation
def test_auto_fill_happy_path(kwik_e_mart_app, kwik_e_mart_app_path, qa_kwik_e_mart):
    """Tests that the params are cleared in one trip from app to mm."""
    convo = Conversation(
        app=kwik_e_mart_app, app_path=kwik_e_mart_app_path, force_sync=True
    )
    directives = convo.process("What's the store phone number?").directives
    assert_target_dialogue_state(convo, "send_store_phone")
    assert_reply(directives, "Which store would you like to know about?")

    directives = convo.process("elm street").directives
    assert_target_dialogue_state(convo, None)


@pytest.mark.conversation
def test_auto_fill_retry(kwik_e_mart_app, kwik_e_mart_app_path, qa_kwik_e_mart):
    """Tests that the params are cleared in one trip from app to mm."""
    convo = Conversation(
        app=kwik_e_mart_app, app_path=kwik_e_mart_app_path, force_sync=True
    )
    directives = convo.process("What's the store phone number?").directives
    assert_target_dialogue_state(convo, "send_store_phone")
    assert_reply(directives, "Which store would you like to know about?")

    directives = convo.process("Some store").directives
    assert_target_dialogue_state(convo, "send_store_phone")
    assert_reply(
        directives,
        "Sorry, I did not get you. " "Which store would you like to know about?",
    )

    directives = convo.process("elm street").directives
    assert_target_dialogue_state(convo, None)


@pytest.mark.conversation
def test_auto_fill_exit_flow(kwik_e_mart_app, kwik_e_mart_app_path):
    """Tests that the params are cleared in one trip from app to mm."""
    convo = Conversation(
        app=kwik_e_mart_app, app_path=kwik_e_mart_app_path, force_sync=True
    )
    directives = convo.process("What's the store phone number?").directives
    assert_target_dialogue_state(convo, "send_store_phone")
    assert_reply(directives, "Which store would you like to know about?")

    directives = convo.process("exit").directives
    assert_reply(directives, "Sorry I cannot help you. Please try again.")
    assert_target_dialogue_state(convo, None)
