Food Ordering
=============

This page documents the Workbench blueprint for a conversational application for a smart home that allows users to control different devices and appliances.


Quick Start
-----------

|

1. Download
^^^^^^^^^^^

Open a python shell and type the following commands to download and set up the home assistant blueprint application.

.. code:: python

   >>> import mmworkbench as wb
   >>> wb.configure_logs()
   >>> wb.blueprint('home_assistant')


2. Build
^^^^^^^^

Build the Natural Language Processing models that power the app.

.. code:: python

   >>> from mmworkbench.components import NaturalLanguageProcessor
   >>> nlp = NaturalLanguageProcessor('home_assistant')
   >>> nlp.build()


3. Run
^^^^^^

Interact with the app in the python shell using the commands below. Try out the examples shown here as well as some queries of your own.

.. code:: python

   >>> from mmworkbench.components.dialogue import Conversation
   >>> conv = Conversation(nlp=nlp, app_path='home_asssitant')
   >>> conv.say('Hi')
   ['Hi, I am your home assistant. I can help you to check weather, set temperature and control the lights and other appliances.']
   >>> conv.say('What is the weather today?')
   ['The weather forecast in San Francisco is haze with a min of 66.2 F and a max of 89.6 F']
   >>> conv.say('Set the temperature to 72')
   ['The thermostat temperature in the home is now 72 degrees F.']

Home Assistant uses Open Weather Map for retrieving weather forecast. You will need to register for a key online and set the environment variable:

.. code::bash
  >>>> export OPEN_WEATHER_KEY=[YOUR-KEY]


Deep Dive
---------

|

1. The Use Case
^^^^^^^^^^^^^^^

This application provides a conversational interface for users to check weather, set alarms and timer, and control the lights, doors, thermostat and different appliances in the house.

2. Example Dialogue Interactions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
