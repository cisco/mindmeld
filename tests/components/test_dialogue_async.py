#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_dialogue
----------------------------------

Tests for dialogue module.

These tests apply only when async/await are supported.
"""
# pylint: disable=locally-disabled,redefined-outer-name
import asyncio

import pytest
from mmworkbench.components import Conversation, DialogueManager
from mmworkbench.components.dialogue import DialogueResponder
from mmworkbench.components.request import Params, FrozenParams

from .test_dialogue import create_request, create_responder


@pytest.fixture
def dm():
    dialogue_manager = DialogueManager(async_mode=True)
    dialogue_manager.called_async_handler = False

    @dialogue_manager.handle(domain='domain')
    async def domain(ctx, handler):
        pass

    @dialogue_manager.handle(intent='intent')
    async def intent(ctx, handler):
        pass

    @dialogue_manager.handle(domain='domain', intent='intent')
    async def domain_intent(ctx, handler):
        pass

    @dialogue_manager.handle(intent='intent', has_entity='entity_1')
    async def intent_entity_1(ctx, handler):
        pass

    @dialogue_manager.handle(intent='intent', has_entity='entity_2')
    async def intent_entity_2(ctx, handler):
        pass

    @dialogue_manager.handle(intent='intent', has_entities=('entity_1', 'entity_2', 'entity_3'))
    async def intent_entities(ctx, handler):
        pass

    @dialogue_manager.handle(targeted_only=True)
    async def targeted_only(ctx, handler):
        pass

    # Defined to test default use
    @dialogue_manager.handle()
    async def dummy_ruleless(ctx, handler):
        pass

    @dialogue_manager.handle(default=True)
    async def default(ctx, handler):
        pass

    @dialogue_manager.handle(intent='async')
    async def async_handler(_, responder):
        await asyncio.sleep(0.050)
        dialogue_manager.called_async_handler = True
        responder.reply('this is the async handler')

    return dialogue_manager


class TestDialogueManager:
    """Tests for the dialogue manager"""

    @pytest.mark.asyncio
    async def test_default(self, dm):
        """Default dialogue state when no rules match
           This will select the rule with default=True"""
        result = await dm.apply_handler(create_request('other', 'other'))
        assert result['dialogue_state'] == 'default'

    def test_default_uniqueness(self, dm):
        with pytest.raises(AssertionError):
            @dm.handle(default=True)
            async def default2(x, y):
                pass

    def test_default_kwarg_exclusion(self, dm):
        with pytest.raises(ValueError):
            @dm.handle(intent='intent', default=True)
            async def default3(x, y):
                pass

    def test_sync_handler(self, dm):
        with pytest.raises(TypeError):
            @dm.handle(intent='sync')
            def sync_handler(x, y):
                pass

    def test_sync_middleware(self, dm):
        with pytest.raises(TypeError):
            @dm.middleware
            def middleware(x, y, z):
                pass

    @pytest.mark.asyncio
    async def test_domain(self, dm):
        """Correct dialogue state is found for a domain"""
        result = await dm.apply_handler(create_request('domain', 'other'))
        assert result['dialogue_state'] == 'domain'

    @pytest.mark.asyncio
    async def test_domain_intent(self, dm):
        """Correct state should be found for domain and intent"""
        result = await dm.apply_handler(create_request('domain', 'intent'))
        assert result['dialogue_state'] == 'domain_intent'

    @pytest.mark.asyncio
    async def test_intent(self, dm):
        """Correct state should be found for intent"""
        result = await dm.apply_handler(create_request('other', 'intent'))
        assert result['dialogue_state'] == 'intent'

    @pytest.mark.asyncio
    async def test_intent_entity(self, dm):
        """Correctly match intent and entity"""
        result = await dm.apply_handler(create_request('domain', 'intent', [{'type': 'entity_2'}]))
        assert result['dialogue_state'] == 'intent_entity_2'

    @pytest.mark.asyncio
    async def test_intent_entity_tiebreak(self, dm):
        """Correctly break ties between rules of equal complexity"""
        context = create_request('domain', 'intent', [{'type': 'entity_1'}, {'type': 'entity_2'}])
        result = await dm.apply_handler(context)
        assert result['dialogue_state'] == 'intent_entity_1'

    @pytest.mark.asyncio
    async def test_intent_entities(self, dm):
        """Correctly break ties between rules of equal complexity"""
        context = create_request('domain', 'intent', [{'type': 'entity_1'}, {'type': 'entity_2'},
                                                      {'type': 'entity_3'}])
        result = await dm.apply_handler(context)
        assert result['dialogue_state'] == 'intent_entities'

    @pytest.mark.asyncio
    async def test_target_dialogue_state_management(self, dm):
        """Correctly sets the dialogue state based on the target_dialogue_state"""
        context = create_request('domain', 'intent')
        result = await dm.apply_handler(context, target_dialogue_state='intent_entity_2')
        assert result['dialogue_state'] == 'intent_entity_2'

    def test_targeted_only_kwarg_exclusion(self, dm):
        with pytest.raises(ValueError):
            @dm.handle(intent='intent', targeted_only=True)
            async def targeted_only2(x, y):
                pass

    @pytest.mark.asyncio
    async def test_middleware_single(self, dm):
        """Adding a single middleware works"""
        async def _middle(request, responder, handler):
            responder.middle = True
            await handler(request, responder)

        async def _handler(request, responder):
            assert responder.middle
            responder.handler = True

        dm.add_middleware(_middle)
        dm.add_dialogue_rule(
            'middleware_test', _handler, intent='middle')
        request = create_request('domain', 'middle')
        response = create_responder(request)
        result = await dm.apply_handler(request, response)
        assert result.dialogue_state == 'middleware_test'
        assert result.handler

    @pytest.mark.asyncio
    async def test_middleware_multiple(self, dm):
        """Adding multiple middleware works"""
        async def _first(ctx, responder, handler):
            ctx['middles'] = ctx.get('middles', []) + ['first']
            await handler(ctx, responder)

        async def _second(ctx, responder, handler):
            ctx['middles'] = ctx.get('middles', []) + ['second']
            await handler(ctx, responder)

        async def _handler(ctx, responder):
            # '_first' should have been called first, then '_second'
            assert ctx['middles'] == ['first', 'second']
            ctx['handler'] = True

        dm.add_middleware(_first)
        dm.add_middleware(_second)
        dm.add_dialogue_rule('middleware_test', _handler, intent='middle')
        ctx = create_request('domain', 'middle')
        result = await dm.apply_handler(ctx)
        assert result['dialogue_state'] == 'middleware_test'
        assert ctx['handler']


@pytest.mark.asyncio
async def test_async_handler(dm):
    """Test asynchronous dialogue state handler works correctly"""
    assert not dm.called_async_handler
    request = create_request('domain', 'async')
    response = create_responder(request)
    result = await dm.apply_handler(request, response)
    assert dm.called_async_handler
    assert result.dialogue_state == 'async_handler'
    assert len(result.directives) == 1
    assert result.directives[0]['name'] == 'reply'
    assert result.directives[0]['payload'] == {'text': 'this is the async handler'}


@pytest.mark.asyncio
async def test_async_middleware(dm):
    """Adding a single async middleware works"""
    async def _middle(request, responder, handler):
        responder.middle = True
        await handler(request, responder)

    async def _handler(request, responder):
        assert responder.middle
        responder.handler = True

    dm.add_middleware(_middle)
    dm.add_dialogue_rule('middleware_test', _handler, intent='middle')
    request = create_request('domain', 'middle')
    response = create_responder(request)
    result = await dm.apply_handler(request, response)
    dm.apply_handler(request, response)
    assert result.dialogue_state == 'middleware_test'
    assert result.handler


@pytest.mark.conversation
@pytest.mark.asyncio
async def test_convo_params_are_cleared(async_kwik_e_mart_app, kwik_e_mart_app_path):
    """Tests that the params are cleared in one trip from app to wb."""
    convo = Conversation(app=async_kwik_e_mart_app, app_path=kwik_e_mart_app_path)
    convo.params = Params(
        allowed_intents=['store_info.find_nearest_store'],
        target_dialogue_state='welcome')
    await convo.say('close door')
    assert convo.params == FrozenParams(
        previous_params=FrozenParams(allowed_intents=['store_info.find_nearest_store'],
                                     target_dialogue_state='welcome'))


@pytest.mark.conversation
def test_convo_force_sync_creation(async_kwik_e_mart_app, kwik_e_mart_app_path):
    """Tests that force sync kwarg works correctly when passed to convo
    at creation.
    """
    convo = Conversation(app=async_kwik_e_mart_app,
                         app_path=kwik_e_mart_app_path,
                         force_sync=True)

    response = convo.process('close door')

    assert isinstance(response, DialogueResponder)


@pytest.mark.conversation
def test_convo_force_sync_invocation(async_kwik_e_mart_app, kwik_e_mart_app_path):
    """Tests that force sync kwarg works correctly when passed to convo
    at invocation.
    """
    convo = Conversation(app=async_kwik_e_mart_app, app_path=kwik_e_mart_app_path)

    response = convo.process('close door', force_sync=True)

    assert isinstance(response, DialogueResponder)
