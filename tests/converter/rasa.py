"""This module tests the convertion tool that is used to convert a rasa or diaglogflow
project into a mindmeld project"""

import pytest

from mindmeld import NaturalLanguageProcessor
from mindmeld.converter.rasa_converter import RasaConverter


def test_rasa_converter(rasa_project_path, mindmeld_project_path):
    converter = RasaConverter(rasa_project_path, mindmeld_project_path)
    converter.convert_project()


@pytest.fixture
def rasa_nlp(mindmeld_project_path):
    nlp = NaturalLanguageProcessor(mindmeld_project_path)
    nlp.build()
    return nlp


intent_test_data = [
    ('hi there', 'greet'),
    ('My name is Josh', 'name'),
    ('thank you', 'thanks'),
    ('See you later', 'goodbye'),
    ('I am Josh Williams', 'lastname'),
    ('No, I don\'t want that', 'deny')
]


@pytest.mark.parametrize("query1,expected_intent", intent_test_data)
def test_nlp_correct_intent(rasa_nlp, query1, expected_intent):
    assert rasa_nlp.process(query1)['intent'] == '{}'.format(expected_intent)


def test_nlp_name(rasa_nlp):
    query1 = 'My name is Josh'
    assert rasa_nlp.process(query1)['intent'] == 'name'
    entity = rasa_nlp.process(query1)['entities']
    assert entity[0]['text'] == 'Josh'


def test_nlp_lastname(rasa_nlp):
    query1 = 'I am Josh Williams'
    assert rasa_nlp.process(query1)['intent'] == 'lastname'
    entity = rasa_nlp.process(query1)['entities']
    assert entity[0]['text'] == 'Josh'
    assert entity[1]['text'] == 'Williams'
