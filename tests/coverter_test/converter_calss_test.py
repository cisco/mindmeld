"""This module tests the convertion tool that is used to convert a rasa or diaglogflow
project into a mindmeld project"""

import os

from rasaconverter import RasaConverter


cwd = os.getcwd()
# Location of the rasa project folder
RASA_PROJECT = cwd + "/rasa_sample_project"
# Location of where you want to create mindmeld project folder
MINDMELD_PROJECT = cwd + "/mindmeld_project"
converter = RasaConverter(RASA_PROJECT, MINDMELD_PROJECT)
converter.convert_project()
