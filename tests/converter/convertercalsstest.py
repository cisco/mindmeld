"""This module tests the convertion tool that is used to convert a rasa or diaglogflow
project into a mindmeld project"""

from mindmeld import rasaconverter


# Location of the rasa project folder
RASA_PROJECT = "sample_rasa_project"
# Location of where you want to create mindmeld project folder
MINDMELD_PROJECT = "mindmeld_project"
converter = rasaconverter.RasaConverter(RASA_PROJECT, MINDMELD_PROJECT)
converter.convert_project()
