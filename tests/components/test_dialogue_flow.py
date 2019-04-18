import pytest
from mindmeld.components import Conversation


def assert_reply(directives, templates, *, start_index=0, slots=None):
    """Asserts that the provided directives contain the specified reply

    Args:
        directives (list[dict[str, dict]]): list of directives returned by application
        templates (Union[str, Set[str]]): The reply must be a member of this set.
        start_index (int, optional): The index of the first client action associated
            with this reply.
        slots (dict, optional): The slots to fill the templates
    """
    slots = slots or {}
    if isinstance(templates, str):
        templates = [templates]

    texts = set(map(lambda x: x.format(**slots), templates))

    assert len(directives) >= start_index + 1
    assert directives[start_index]['name'] == 'reply'
    assert directives[start_index]['payload']['text'] in texts


def assert_target_dialogue_state(convo, target_dialogue_state):
    assert convo.params.target_dialogue_state == target_dialogue_state


@pytest.mark.conversation
def test_default_handler(async_kwik_e_mart_app, kwik_e_mart_app_path):
    """Tests that the params are cleared in one trip from app to mm."""
    convo = Conversation(app=async_kwik_e_mart_app, app_path=kwik_e_mart_app_path, force_sync=True)
    convo.process('When does that open?')
    assert_target_dialogue_state(convo, 'send_store_hours_flow')
    directives = convo.process('are there any stores near me?').directives
    assert_target_dialogue_state(convo, 'send_store_hours_flow')
    assert_reply(directives,
                 templates='Sorry, I did not get you. Which store would you like to know about?')


@pytest.mark.conversation
def test_repeated_flow(async_kwik_e_mart_app, kwik_e_mart_app_path):
    """Tests that the params are cleared in one trip from app to mm."""
    convo = Conversation(app=async_kwik_e_mart_app, app_path=kwik_e_mart_app_path, force_sync=True)
    convo.process('When does that open?')
    assert_target_dialogue_state(convo, 'send_store_hours_flow')
    for i in range(2):
        directives = convo.process('When does that open?').directives
        assert_reply(directives, 'Which store would you like to know about?')
        assert_target_dialogue_state(convo, 'send_store_hours_flow')
    directives = convo.process('When does that open?').directives
    assert_reply(directives, 'Sorry I cannot help you. Please try again.')
    assert_target_dialogue_state(convo, None)


@pytest.mark.conversation
def test_intent_handler_and_exit_flow(async_kwik_e_mart_app, kwik_e_mart_app_path):
    """Tests that the params are cleared in one trip from app to mm."""
    convo = Conversation(app=async_kwik_e_mart_app, app_path=kwik_e_mart_app_path, force_sync=True)
    convo.process('When does that open?')
    assert_target_dialogue_state(convo, 'send_store_hours_flow')
    directives = convo.process('exit').directives
    assert_target_dialogue_state(convo, None)
    assert_reply(directives, templates=['Bye', 'Goodbye', 'Have a nice day.'])


def assert_dialogue_state(dm, dialogue_state):
    for rule in dm.rules:
        if rule.dialogue_state == dialogue_state:
            return True
    return False


def test_dialogue_flow_async(async_kwik_e_mart_app):
    @async_kwik_e_mart_app.dialogue_flow(domain='some_domain', intent='some_intent')
    async def some_handler(context, responder):
        pass

    assert some_handler.flow_state == 'some_handler_flow'
    assert 'some_handler' in some_handler.all_flows

    dm = some_handler.dialogue_manager
    assert_dialogue_state(dm, 'some_handler')
    assert_dialogue_state(dm, 'some_handler_flow')

    assert len(some_handler.rules) == 0

    @some_handler.handle(intent='some_intent')
    async def some_flow_handler(context, responder):
        pass

    assert len(some_handler.rules) == 1

    @some_handler.handle(intent='some_intent_2', exit_flow=True)
    async def some_flow_handler_2(context, responder):
        pass

    assert len(some_handler.rules) == 2
    assert 'some_flow_handler_2' in some_handler.exit_flow_states


def test_dialogue_flow(kwik_e_mart_app):
    @kwik_e_mart_app.dialogue_flow(domain='some_domain', intent='some_intent')
    def some_handler(context, responder):
        pass

    assert some_handler.flow_state == 'some_handler_flow'
    assert 'some_handler' in some_handler.all_flows

    dm = some_handler.dialogue_manager
    assert_dialogue_state(dm, 'some_handler')
    assert_dialogue_state(dm, 'some_handler_flow')

    assert len(some_handler.rules) == 0

    @some_handler.handle(intent='some_intent')
    def some_flow_handler(context, responder):
        pass

    assert len(some_handler.rules) == 1

    @some_handler.handle(intent='some_intent_2', exit_flow=True)
    def some_flow_handler_2(context, responder):
        pass

    assert len(some_handler.rules) == 2
    assert 'some_flow_handler_2' in some_handler.exit_flow_states
