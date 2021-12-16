import pytest
from mock import MagicMock
from mindmeld.components import Conversation
from mindmeld.components.dialogue import DialogueResponder, AutoEntityFilling
from mindmeld.components.request import Request
from mindmeld.core import FormEntity
from .test_dialogue_flow import assert_target_dialogue_state, assert_reply


@pytest.mark.conversation
def test_auto_fill_happy_path(kwik_e_mart_app, qa_kwik_e_mart):
    """Tests a happy path for the app."""
    convo = Conversation(app=kwik_e_mart_app)
    directives = convo.process("What's the store phone number?").directives
    assert_target_dialogue_state(convo, "send_store_phone")
    assert_reply(directives, "Which store would you like to know about?")

    convo.process("elm street")
    assert_target_dialogue_state(convo, None)


@pytest.mark.conversation
def test_auto_fill_happy_path_validation(kwik_e_mart_app, qa_kwik_e_mart):
    """Tests a happy path for the app with gazetter validation."""
    convo = Conversation(app=kwik_e_mart_app)
    directives = convo.process("What's the store phone number?").directives
    assert_target_dialogue_state(convo, "send_store_phone")
    assert_reply(directives, "Which store would you like to know about?")

    convo.process("the store on elm street")
    assert_target_dialogue_state(convo, None)


@pytest.mark.conversation
def test_auto_fill_retry(kwik_e_mart_app, qa_kwik_e_mart):
    """Tests that the retry logic for slots/entities."""
    convo = Conversation(app=kwik_e_mart_app)
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
def test_auto_fill_exit_flow(kwik_e_mart_app, qa_kwik_e_mart):
    """Tests slot-fill exit flow logic."""
    convo = Conversation(app=kwik_e_mart_app)
    directives = convo.process("What's the store phone number?").directives
    assert_target_dialogue_state(convo, "send_store_phone")
    assert_reply(directives, "Which store would you like to know about?")

    directives = convo.process("exit").directives
    assert_reply(directives, "Sorry I cannot help you. Please try again.")
    assert_target_dialogue_state(convo, None)


@pytest.mark.conversation
def test_auto_fill_switch_flow(kwik_e_mart_app, qa_kwik_e_mart):
    """Tests flow switching from inside slot filling to another intent when
    the number of retry attempts are exceeded."""
    convo = Conversation(app=kwik_e_mart_app)
    directives = convo.process("What's the store phone number?").directives
    assert_target_dialogue_state(convo, "send_store_phone")
    assert_reply(directives, "Which store would you like to know about?")

    directives = convo.process("goodbye").directives
    assert_reply(
        directives,
        "Sorry, I did not get you. " "Which store would you like to know about?",
    )

    directives = convo.process("goodbye").directives
    assert_reply(directives, ["Bye", "Goodbye", "Have a nice day."])


@pytest.mark.conversation
def test_auto_fill_validation_missing_entities(kwik_e_mart_app, qa_kwik_e_mart):
    """Tests default validation when user input has no entities.
    Check is to see that flow doesn't break."""
    convo = Conversation(app=kwik_e_mart_app)
    directives = convo.process("What's the store phone number?").directives
    assert_target_dialogue_state(convo, "send_store_phone")
    assert_reply(directives, "Which store would you like to know about?")

    directives = convo.process("123").directives
    assert_reply(
        directives,
        "Sorry, I did not get you. " "Which store would you like to know about?",
    )


@pytest.mark.conversation
def test_auto_fill_invoke(kwik_e_mart_app):
    """Tests slot-filling invoke functionality"""
    app = kwik_e_mart_app
    request = Request(
        text="elm street",
        domain="store_info",
        intent="get_store_number",
        entities=[
            {"type": "store_name", "value": [{"cname": "23 Elm Street"}], "role": None}
        ],
    )
    responder = DialogueResponder()

    # custom eval func
    @app.register_func()
    def test_custom_eval(r):
        # entity already passed in, this is to check custom eval flow.
        del r
        return True

    # mock handler for invoke
    handler_sub = MagicMock()
    handler_sub.__name__ = "handler_sub"

    form = {
        "entities": [
            FormEntity(
                entity="store_name",
                value="23 Elm Street",
                default_eval=False,
                custom_eval="test_custom_eval",
            )
        ],
    }

    @app.handle(domain="store_info", intent="get_store_number")
    def handler_main(request, responder):
        AutoEntityFilling(handler_sub, form, app).invoke(request, responder)

    handler_main(request, responder)

    # check whether the sub handler was invoked.
    handler_sub.assert_called_once()

    # check whether new rule has been added for sub handler.
    assert any(
        [
            rule.dialogue_state == handler_sub.__name__
            for rule in list(app.app_manager.dialogue_manager.rules)
        ]
    )


@pytest.mark.conversation
def test_auto_fill_custom_validation_resolution(kwik_e_mart_app):
    """Tests slot-filling's custom validation with custom resolution"""
    app = kwik_e_mart_app
    request = Request(
        text="what is the sum of 5 and 15?",
        domain="some domain",
        intent="some intent",
    )
    responder = DialogueResponder()

    # custom eval func
    @app.register_func()
    def test_custom_eval(r):
        return 5 + 15

    form = {
        "entities": [
            FormEntity(
                entity="sys_number",
                default_eval=False,
                custom_eval="test_custom_eval",
            )
        ],
    }

    def handler_sub(request, responder):
        entity = next((e for e in request.entities if e["type"] == "sys_number"), None)

        # Check custom resolution validity
        assert entity
        assert entity["value"][0]["value"] == 20

    @app.handle(domain="some domain", intent="some intent")
    def handler(request, responder):
        AutoEntityFilling(handler_sub, form, app).invoke(request, responder)

    handler(request, responder)


@pytest.mark.conversation
@pytest.mark.asyncio
async def test_auto_exit_flow_async(async_kwik_e_mart_app, qa_kwik_e_mart):
    """Tests auto fill for async apps."""
    convo = Conversation(app=async_kwik_e_mart_app)
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
async def test_auto_fill_happy_path_async(async_kwik_e_mart_app, qa_kwik_e_mart):
    """Tests auto fill happy path for async apps."""
    convo = Conversation(app=async_kwik_e_mart_app)
    response = await convo.process("What's the store phone number?")
    directives = response.directives
    assert_target_dialogue_state(convo, "send_store_phone")
    assert_reply(directives, "Which store would you like to know about?")

    await convo.process("elm street")
    assert_target_dialogue_state(convo, None)


@pytest.mark.conversation
@pytest.mark.asyncio
async def test_auto_fill_validation_async(async_kwik_e_mart_app, qa_kwik_e_mart):
    """Tests a happy path for the async app with gazetter validation."""
    convo = Conversation(app=async_kwik_e_mart_app)
    response = await convo.process("What's the store phone number?")
    directives = response.directives
    assert_target_dialogue_state(convo, "send_store_phone")
    assert_reply(directives, "Which store would you like to know about?")

    await convo.process("elm street")
    assert_target_dialogue_state(convo, None)


@pytest.mark.conversation
@pytest.mark.asyncio
async def test_auto_fill_retry_async(async_kwik_e_mart_app, qa_kwik_e_mart):
    """Tests that the retry logic for slots/entities in async app."""
    convo = Conversation(app=async_kwik_e_mart_app)
    response = await convo.process("What's the store phone number?")
    directives = response.directives
    assert_target_dialogue_state(convo, "send_store_phone")
    assert_reply(directives, "Which store would you like to know about?")

    response = await convo.process("Some store")
    directives = response.directives
    assert_target_dialogue_state(convo, "send_store_phone")
    assert_reply(
        directives,
        "Sorry, I did not get you. " "Which store would you like to know about?",
    )

    await convo.process("elm street")
    assert_target_dialogue_state(convo, None)
