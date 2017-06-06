#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_parser
----------------------------------

Tests for parser module.
"""
# pylint: disable=locally-disabled,redefined-outer-name
from __future__ import unicode_literals

from mmworkbench.components.dialogue import DialogueManager


def create_context(domain, intent, entities=None):
    """Creates a context object for use by the dialogue manager"""
    entities = entities or ()
    return {'domain': domain, 'intent': intent, 'entities': entities}


class TestDialogueManager:
    """Tests for the dialogue manager"""

    @classmethod
    def setup_class(cls):
        """Sets up test by creating a dialogue manager with some simple rules"""
        cls.dm = DialogueManager()
        cls.dm.add_dialogue_rule('domain', lambda x, y, z: None, domain='domain')
        cls.dm.add_dialogue_rule('intent', lambda x, y, z: None, intent='intent')
        cls.dm.add_dialogue_rule('domain_intent', lambda x, y, z: None,
                                 domain='domain', intent='intent')
        cls.dm.add_dialogue_rule('intent_entity_1', lambda x, y, z: None,
                                 intent='intent', entity='entity_1')
        cls.dm.add_dialogue_rule('intent_entity_2', lambda x, y, z: None,
                                 intent='intent', entity='entity_2')
        cls.dm.add_dialogue_rule('intent_entities', lambda x, y, z: None,
                                 intent='intent', entities=('entity_1', 'entity_2', 'entity_3'))

        cls.dm.add_dialogue_rule('default', lambda x, y, z: None)

    def test_default(self):
        """Default dialogue state when no rules match"""
        result = self.dm.apply_handler(create_context('other', 'other'))
        assert result['dialogue_state'] == 'default'

    def test_domain(self):
        """Correct dialogue state is found for a domain"""
        result = self.dm.apply_handler(create_context('domain', 'other'))
        assert result['dialogue_state'] == 'domain'

    def test_domain_intent(self):
        """Correct state should be found for domain and intent"""
        result = self.dm.apply_handler(create_context('domain', 'intent'))
        assert result['dialogue_state'] == 'domain_intent'

    def test_intent(self):
        """Correct state should be found for intent"""
        result = self.dm.apply_handler(create_context('other', 'intent'))
        assert result['dialogue_state'] == 'intent'

    def test_intent_entity(self):
        """Correctly match intent and entity"""
        result = self.dm.apply_handler(create_context('domain', 'intent', [{'type': 'entity_2'}]))
        assert result['dialogue_state'] == 'intent_entity_2'

    def test_intent_entity_tiebreak(self):
        """Correctly break ties between rules of equal complexity"""
        context = create_context('domain', 'intent', [{'type': 'entity_1'}, {'type': 'entity_2'}])
        result = self.dm.apply_handler(context)
        assert result['dialogue_state'] == 'intent_entity_1'

    def test_intent_entities(self):
        """Correctly break ties between rules of equal complexity"""
        context = create_context('domain', 'intent', [{'type': 'entity_1'}, {'type': 'entity_2'},
                                                      {'type': 'entity_3'}])
        result = self.dm.apply_handler(context)
        assert result['dialogue_state'] == 'intent_entities'
