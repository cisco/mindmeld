"""This module tests the convertion tool that is used to convert a rasa or diaglogflow
project into a mindmeld project"""

from mindmeld import NaturalLanguageProcessor
from mindmeld.converter.rasa_converter import RasaConverter


def test_rasa_converter(rasa_project_path, mindmeld_project_path):
    converter = RasaConverter(rasa_project_path, mindmeld_project_path)
    converter.convert_project()


def test_nlp_greet(mindmeld_project_path):
    nlp = NaturalLanguageProcessor(mindmeld_project_path)
    nlp.build()
    query1 = 'hi there'
    assert nlp.process(query1)['intent'] == 'greet'


def test_nlp_name(mindmeld_project_path):
    nlp = NaturalLanguageProcessor(mindmeld_project_path)
    nlp.build()
    query1 = 'My name is Josh'
    assert nlp.process(query1)['intent'] == 'name'
    entity = nlp.process(query1)['entities']
    assert entity[0]['text'] == 'Josh'


def test_nlp_thanks(mindmeld_project_path):
    nlp = NaturalLanguageProcessor(mindmeld_project_path)
    nlp.build()
    query1 = 'thank you'
    assert nlp.process(query1)['intent'] == 'thanks'


def test_nlp_goodbye(mindmeld_project_path):
    nlp = NaturalLanguageProcessor(mindmeld_project_path)
    nlp.build()
    query1 = 'See you later'
    assert nlp.process(query1)['intent'] == 'goodbye'


def test_nlp_lastname(mindmeld_project_path):
    nlp = NaturalLanguageProcessor(mindmeld_project_path)
    nlp.build()
    query1 = 'I am Josh Williams'
    assert nlp.process(query1)['intent'] == 'lastname'
    entity = nlp.process(query1)['entities']
    assert entity[0]['text'] == 'Josh'
    assert entity[1]['text'] == 'Williams'


def test_nlp_deny(mindmeld_project_path):
    nlp = NaturalLanguageProcessor(mindmeld_project_path)
    nlp.build()
    query1 = 'No, I don\'t want that'
    assert nlp.process(query1)['intent'] == 'deny'
