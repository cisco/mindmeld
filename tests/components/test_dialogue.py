#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_dialogue
----------------------------------

Tests for dialogue module.

These tests apply regardless of async/await support.
"""
# pylint: disable=locally-disabled,redefined-outer-name
import pytest

from mmworkbench.components import Conversation, DialogueManager, DialogueResponder
from mmworkbench.components.request import Request, Params, FrozenParams


def create_request(domain, intent, entities=None):
    """Creates a request object for use by the dialogue manager"""
    entities = entities or ()
    return Request(domain=domain, intent=intent, entities=entities, text='')


def create_responder(request):
    """Creates a response object for use by the dialogue manager"""
    return DialogueResponder(request=request)


@pytest.fixture
def dm():
    dm = DialogueManager()
    dm.add_dialogue_rule('domain', lambda x, y: None, domain='domain')
    dm.add_dialogue_rule('intent', lambda x, y: None, intent='intent')
    dm.add_dialogue_rule('domain_intent', lambda x, y: None,
                         domain='domain', intent='intent')
    dm.add_dialogue_rule('intent_entity_1', lambda x, y: None,
                         intent='intent', has_entity='entity_1')
    dm.add_dialogue_rule('intent_entity_2', lambda x, y: None,
                         intent='intent', has_entity='entity_2')
    dm.add_dialogue_rule('intent_entities', lambda x, y: None,
                         intent='intent', has_entities=('entity_1', 'entity_2', 'entity_3'))

    dm.add_dialogue_rule('targeted_only', lambda x, y: None, targeted_only=True)
    dm.add_dialogue_rule('dummy_ruleless', lambda x, y: None)  # Defined to test default use
    dm.add_dialogue_rule('default', lambda x, y: None, default=True)

    return dm


class TestDialogueManager:
    """Tests for the dialogue manager"""

    def test_default(self, dm):
        """Default dialogue state when no rules match
           This will select the rule with default=True"""
        request = create_request('other', 'other')
        response = create_responder(request)
        result = dm.apply_handler(request, response)
        assert result.dialogue_state == 'default'

    def test_default_uniqueness(self, dm):
        with pytest.raises(AssertionError):
            dm.add_dialogue_rule('default2', lambda x, y: None, default=True)

    def test_default_kwarg_exclusion(self, dm):
        with pytest.raises(ValueError):
            dm.add_dialogue_rule('default3', lambda x, y: None,
                                 intent='intent', default=True)

    def test_domain(self, dm):
        """Correct dialogue state is found for a domain"""
        request = create_request('domain', 'other')
        response = create_responder(request)
        result = dm.apply_handler(request, response)
        assert result.dialogue_state == 'domain'

    def test_domain_intent(self, dm):
        """Correct state should be found for domain and intent"""
        request = create_request('domain', 'intent')
        response = create_responder(request)
        result = dm.apply_handler(request, response)
        assert result.dialogue_state == 'domain_intent'

    def test_intent(self, dm):
        """Correct state should be found for intent"""
        request = create_request('other', 'intent')
        response = create_responder(request)
        result = dm.apply_handler(request, response)
        assert result.dialogue_state == 'intent'

    def test_intent_entity(self, dm):
        """Correctly match intent and entity"""
        request = create_request('domain', 'intent', [{'type': 'entity_2'}])
        response = create_responder(request)
        result = dm.apply_handler(request, response)
        assert result.dialogue_state == 'intent_entity_2'

    def test_intent_entity_tiebreak(self, dm):
        """Correctly break ties between rules of equal complexity"""
        request = create_request('domain', 'intent', [{'type': 'entity_1'}, {'type': 'entity_2'}])
        response = create_responder(request)
        result = dm.apply_handler(request, response)
        assert result.dialogue_state == 'intent_entity_1'

    def test_intent_entities(self, dm):
        """Correctly break ties between rules of equal complexity"""
        request = create_request('domain', 'intent', [{'type': 'entity_1'}, {'type': 'entity_2'},
                                                      {'type': 'entity_3'}])
        response = create_responder(request)
        result = dm.apply_handler(request, response)
        assert result.dialogue_state == 'intent_entities'

    def test_target_dialogue_state_management(self, dm):
        """Correctly sets the dialogue state based on the target_dialogue_state"""
        request = create_request('domain', 'intent')
        response = create_responder(request)
        result = dm.apply_handler(request, response, target_dialogue_state='intent_entity_2')
        assert result.dialogue_state == 'intent_entity_2'

    def test_targeted_only_kwarg_exclusion(self, dm):
        with pytest.raises(ValueError):
            dm.add_dialogue_rule('targeted_only2', lambda x, y: None,
                                 intent='intent', targeted_only=True)

    def test_middleware_single(self, dm):
        """Adding a single middleware works"""
        def _middle(request, responder, handler):
            responder.flag = True
            handler(request, responder)

        def _handler(request, responder):
            assert responder.flag

        dm.add_middleware(_middle)
        dm.add_dialogue_rule('middleware_test', _handler, intent='middle')

        request = create_request('domain', 'middle')
        response = create_responder(request)

        result = dm.apply_handler(request, response)
        assert result.dialogue_state == 'middleware_test'

    def test_middleware_multiple(self, dm):
        """Adding multiple middleware works"""
        def _first(request, responder, handler):
            responder.middles = vars(responder).get('middles', []) + ['first']
            handler(request, responder)

        def _second(request, responder, handler):
            responder.middles = vars(responder).get('middles', []) + ['second']
            handler(request, responder)

        def _handler(request, responder):
            # '_first' should have been called first, then '_second'
            assert responder.middles == ['first', 'second']

        dm.add_middleware(_first)
        dm.add_middleware(_second)
        dm.add_dialogue_rule('middleware_test', _handler, intent='middle')

        request = create_request('domain', 'middle')
        response = create_responder(request)

        result = dm.apply_handler(request, response)
        assert result.dialogue_state == 'middleware_test'


def test_convo_params_are_cleared(kwik_e_mart_nlp, kwik_e_mart_app_path):
    """Tests that the params are cleared in one trip from app to wb."""
    convo = Conversation(nlp=kwik_e_mart_nlp, app_path=kwik_e_mart_app_path)
    convo.params = Params(allowed_intents=['store_info.find_nearest_store'],
                          target_dialogue_state='greeting')
    convo.say('close door')
    assert convo.params == Params(previous_params=FrozenParams(
        allowed_intents=('store_info.find_nearest_store',), target_dialogue_state='greeting'),
        target_dialogue_state='send_store_hours_flow')
