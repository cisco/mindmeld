import pytest

from mmworkbench.components.dialogue import Conversation


@pytest.fixture
def convo(kwik_e_mart_app_path, kwik_e_mart_nlp):
    return Conversation(nlp=kwik_e_mart_nlp, app_path=kwik_e_mart_app_path)


def test_params_are_cleared(convo):
    """Tests that the params are cleared in one trip from app to wb."""
    convo.params = {
        'allowed_intents': ['store_info.find_nearest_store'],
        'target_dialogue_state': 'greeting'
    }
    convo.say('close door')

    assert convo.params == {}
