"""This module tests the convertion tool that is used to convert a rasa or diaglogflow
project into a mindmeld project"""

from mindmeld import NaturalLanguageProcessor
from mindmeld.converter.dialogflowconverter import DialogFlowConverter


def test_dialogflow_converter(dialogflow_project_path, mindmeld_project_path):
    converter = DialogflowConverter(dialogflow_project_path, mindmeld_project_path)
    converter.convert_project()


def test_nlp_greet(mindmeld_project_path):
    nlp = NaturalLanguageProcessor(mindmeld_project_path)
    nlp.build()
    query1 = 'hi there'
    assert nlp.process(query1)['intent'] == 'default_welcome'


def test_nlp_eat_out_search(mindmeld_project_path):
    nlp = NaturalLanguageProcessor(mindmeld_project_path)
    nlp.build()
    query1 = 'Find me food'
    assert nlp.process(query1)['intent'] == 'venues.eating_out.search'
    entity = nlp.process(query1)['entities']
    assert entity[0]['text'] == 'Josh'


def test_nlp_venue_search(mindmeld_project_path):
    nlp = NaturalLanguageProcessor(mindmeld_project_path)
    nlp.build()
    query1 = 'thank you'
    assert nlp.process(query1)['intent'] == 'venues.search-location'
