"""This module tests the convertion tool that is used to convert a rasa or diaglogflow
project into a mindmeld project"""

from mindmeld import NaturalLanguageProcessor
from mindmeld.converter.rasa_converter import RasaConverter


def test_rasa_converter(rasa_project_path, mindmeld_project_path):
    converter = RasaConverter(rasa_project_path, mindmeld_project_path)
    converter.convert_project()

def test_nlp(mindmeld_project_path):
    nlp = NaturalLanguageProcessor(mindmeld_project_path)
    nlp.build()
    query1 = 'hi there'
    assert nlp.process(query1)['intent'] == 'greet'
