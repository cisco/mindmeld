Mindmeld Conversion Tool
========================

Introduction
------------

The MindMeld conversion tool is designed to help make the migration from other conversational AI platforms to MindMeld seamless.
Currently, we offer conversions for Rasa and Dialogflow projects.

How does this tool work?
------------------------

This tool will generate a basic scaffold for a MindMeld project from a given Rasa/Dialogflow project, and will use the
entities and intents of the existing project to populate the entities folders and create training files in MindMeld.
Dialogue state handlers will be created for each intent to handle any necessary logic behind responses, and a default model
configurations file will also be created, which may be changed as needed. If you are unfamiliar with the structure of
MindMeld projects, code examples can be found in our `Key Concepts <https://www.mindmeld.com/docs/intro/key_concepts.html>`_ page.

Usage
-----

For Rasa project conversion, use the following command:

.. code:: console

    mindmeld convert --rs from_rasa_project_location to_mindmeld_project_location


For Dialogflow project conversion, use the following command:

.. code:: console

    mindmeld convert --df from_dialogflow_project_location to_mindmeld_project_location
