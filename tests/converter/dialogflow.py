"""This module tests the convertion tool that is used to convert a rasa or diaglogflow
project into a mindmeld project"""

from mindmeld import NaturalLanguageProcessor
from mindmeld.converter.dialogflow import DialogFlowConverter


def test_dialogflow_converter(dialogflow_project_path, mindmeld_project_path2):
    converter = DialogFlowConverter(dialogflow_project_path, mindmeld_project_path2)
    converter.convert_project()

def test_nlp_greet(mindmeld_project_path2):
    nlp = NaturalLanguageProcessor(mindmeld_project_path2)
    nlp.build()
    query1 = 'hi there'
    assert nlp.process(query1)['intent'] == 'default_welcome_intent_usersays_en'

def test_nlp_i_can_code_in_en(mindmeld_project_path2):
    nlp = NaturalLanguageProcessor(mindmeld_project_path2)
    nlp.build()
    query1 = 'I know Python'
    assert nlp.process(query1)['intent'] == 'i_can_code_in_a_language_usersays_en'
    entity = nlp.process(query1)['entities']
    assert entity[0]['text'] == 'Python'

def test_nlp_i_can_code_in_es(mindmeld_project_path2):
    nlp = NaturalLanguageProcessor(mindmeld_project_path2)
    nlp.build()
    query1 = 'Puedo codificar en Java'
    assert nlp.process(query1)['intent'] == 'i_can_code_in_a_language_usersays_es'
    entity = nlp.process(query1)['entities']
    assert entity[0]['text'] == 'Java'

