import pytest

from mindmeld.components import Conversation
from .test_dialogue_flow import assert_target_dialogue_state, assert_reply


@pytest.mark.conversation
def test_auto_fill_happy_path(kwik_e_mart_app, kwik_e_mart_app_path, qa_kwik_e_mart):
    """Tests a happy path for the app."""
    convo = Conversation(app=kwik_e_mart_app, app_path=kwik_e_mart_app_path)
    directives = convo.process("What's the store phone number?").directives
    assert_target_dialogue_state(convo, "send_store_phone")
    assert_reply(directives, "Which store would you like to know about?")

    convo.process("elm street")
    assert_target_dialogue_state(convo, None)


@pytest.mark.conversation
def test_auto_fill_happy_path_validation(
    kwik_e_mart_app, kwik_e_mart_app_path, qa_kwik_e_mart
):
    """Tests a happy path for the app with gazetter validation."""
    convo = Conversation(app=kwik_e_mart_app, app_path=kwik_e_mart_app_path)
    directives = convo.process("What's the store phone number?").directives
    assert_target_dialogue_state(convo, "send_store_phone")
    assert_reply(directives, "Which store would you like to know about?")

    convo.process("the store on elm street")
    assert_target_dialogue_state(convo, None)


@pytest.mark.conversation
def test_auto_fill_retry(kwik_e_mart_app, kwik_e_mart_app_path, qa_kwik_e_mart):
    """Tests that the retry logic for slots/entities."""
    convo = Conversation(app=kwik_e_mart_app, app_path=kwik_e_mart_app_path)
    directives = convo.process("What's the store phone number?").directives
    assert_target_dialogue_state(convo, "send_store_phone")
    assert_reply(directives, "Which store would you like to know about?")

    directives = convo.process("Some store").directives
    assert_target_dialogue_state(convo, "send_store_phone")
    assert_reply(
        directives,
        "Sorry, I did not get you. " "Which store would you like to know about?",
    )

    convo.process("elm street")
    assert_target_dialogue_state(convo, None)


@pytest.mark.conversation
def test_auto_fill_exit_flow(kwik_e_mart_app, kwik_e_mart_app_path):
    """Tests slot-fill exit flow logic."""
    convo = Conversation(app=kwik_e_mart_app, app_path=kwik_e_mart_app_path)
    directives = convo.process("What's the store phone number?").directives
    assert_target_dialogue_state(convo, "send_store_phone")
    assert_reply(directives, "Which store would you like to know about?")

    directives = convo.process("exit").directives
    assert_reply(directives, "Sorry I cannot help you. Please try again.")
    assert_target_dialogue_state(convo, None)


@pytest.mark.conversation
@pytest.mark.asyncio
async def test_auto_fill_async(async_kwik_e_mart_app, kwik_e_mart_app_path):
    """Tests auto fill for async apps."""
    convo = Conversation(app=async_kwik_e_mart_app, app_path=kwik_e_mart_app_path)
    response = await convo.process("What's the store phone number?")
    directives = response.directives
    assert_target_dialogue_state(convo, "send_store_phone")
    assert_reply(directives, "Which store would you like to know about?")

    response = await convo.process("exit")
    directives = response.directives
    assert_reply(directives, "Sorry I cannot help you. Please try again.")
    assert_target_dialogue_state(convo, None)


@pytest.mark.conversation
@pytest.mark.asyncio
async def test_auto_fill_async_happy_path(
    async_kwik_e_mart_app, kwik_e_mart_app_path, qa_kwik_e_mart
):
    """Tests auto fill for async apps."""
    convo = Conversation(app=async_kwik_e_mart_app, app_path=kwik_e_mart_app_path)
    response = await convo.process("What's the store phone number?")
    directives = response.directives
    assert_target_dialogue_state(convo, "send_store_phone")
    assert_reply(directives, "Which store would you like to know about?")

    response = await convo.process("elm street")
    assert_target_dialogue_state(convo, None)
