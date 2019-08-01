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


def test_nlp_greet(rasa_nlp):
    query1 = 'hi there'
    assert rasa_nlp.process(query1)['intent'] == 'greet'


def test_nlp_name(rasa_nlp):
    query1 = 'My name is Josh'
    assert rasa_nlp.process(query1)['intent'] == 'name'
    entity = rasa_nlp.process(query1)['entities']
    assert entity[0]['text'] == 'Josh'


def test_nlp_thanks(rasa_nlp):
    query1 = 'thank you'
    assert rasa_nlp.process(query1)['intent'] == 'thanks'


def test_nlp_goodbye(rasa_nlp):
    query1 = 'See you later'
    assert rasa_nlp.process(query1)['intent'] == 'goodbye'


def test_nlp_lastname(rasa_nlp):
    query1 = 'I am Josh Williams'
    assert rasa_nlp.process(query1)['intent'] == 'lastname'
    entity = rasa_nlp.process(query1)['entities']
    assert entity[0]['text'] == 'Josh'
    assert entity[1]['text'] == 'Williams'


def test_nlp_deny(rasa_nlp):
    query1 = 'No, I don\'t want that'
    assert rasa_nlp.process(query1)['intent'] == 'deny'
