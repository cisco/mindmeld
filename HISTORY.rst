Package History
===============

4.0.0 (Unreleased)
------------------

Enhancements
^^^^^^^^^^^^

- Add support for asynchronous dialogue state handlers

Changes
^^^^^^^

- ``session`` parameter for ``NaturalLanguageProcessor.parse()`` and ``Conversation()`` renamed to ``context``

3.4.0 (Unreleased)
------------------

Enhancements
^^^^^^^^^^^^

- Standardized feature names for taggers and classifiers

- Add the ability to add dialogue middleware to dialogue managers

- Add ability to denote a handler as only reachable via target_dialogue_state

- Add ability to explicitly denote default handler

- Add CLI command to generate predicted markup for queries

3.3.1 (2018-06-20)
------------------

Bug Fixes
~~~~~~~~~

- ``mmworkbench blueprint`` has been temporarily disabled. Blueprints can manually downloaded from https://devcenter.mindmeld.com/bp/{blueprint_name}/app.tar.gz.


3.3.0 (2018-05-10)
------------------

Enhancements
^^^^^^^^^^^^

- Add the ability to inspect learned feature weights for certain kinds of text classifiers

- Add character n-gram features to domain, intent, and entity models

- Add support for better management of multiple datasets within the same project

- Add the ability to override global classifier configurations with custom settings per domain, intent, or entity type

- Add the ability for incremental NLP model building to reduce overall training time

- Add the ability to specify the time zone and timestamp associated with each query to inform NLP predictions

- Add the ability to define custom preprocessors that can make arbitrary transformations on the input query before sending it to the NLP pipeline

Bug Fixes
^^^^^^^^^

- The help messages for Workbench command line tools (``python app.py`` and ``mmworkbench``) should show the correct list of compatible commands

- Various fixes to improve the numerical parser's robustness and logging


3.2.0 (2017-10-23)
------------------

Enhancements
^^^^^^^^^^^^

- Add Long Short Term Memory (LSTM) network as a model option for the entity recognizer

- Add support for TensorFlow-based deep learning models in Workbench

- Add the ability to evaluate all NLP models at once with a single method/command

- Add functionality to specify a target dialogue state or a set of allowable intents for the next turn

- Add in-built support for conversational history management instead of relying on the client to preserve history across turns

- Improve interfaces for constructing responses within the dialogue state handlers (see **Compatibility Notes** below)


Compatibility Notes
^^^^^^^^^^^^^^^^^^^

- The ``prompt()`` and ``respond()`` methods of the ``DialogueResponder`` object are deprecated in Workbench 3.2. See :doc:`Working with the Dialogue Manager <../userguide/dm>` to learn how to use new ``DialogueResponder`` methods in your dialogue state handlers.

3.1.0 (2017-09-20)
------------------

Enhancements
^^^^^^^^^^^^

- Add linear-chain conditional random field (CRF) as a model option for the entity recognizer

- Allow the role classifier to be trained with other text models (e.g. SVM, decision tree, etc.) in addition to logistic regression

- Make model configuration format for all classifiers consistent [See "Compatibility Notes" below]

- Add new metrics for better error analysis of entity recognition performance

- Add support for modularizing dialogue state handling logic by allowing arbitrary module imports in ``app.py`` (see **Compatibility Notes** below)

- Make blueprints check the current Workbench package version to validate compatibility

- Only load NLP resources that are needed by active feature extractors (as defined in the model config) to improve runtime performance

Bug Fixes
^^^^^^^^^

- Correctly compute entity spans in queries with special characters

- Warn the developer and proceed with model training (if possible) when entity mapping or gazetteer files are missing

Compatibility Notes
^^^^^^^^^^^^^^^^^^^

- To make the interfaces for NLP classifiers consistent, the model configuration formats for the entity recognizer and the role classifier have been updated to be in line with the domain and intent classifiers. The model configurations for entity recognizer and role classifier from Workbench 3.0 **will not** work with Workbench 3.1. Refer to the user guide for those components to learn how to rewrite your 3.0 configs in the new 3.1 format.

- To support modular organization of dialogue state handling logic by allowing arbitrary package/module imports in the application container (``app.py``), Workbench now needs to load the project folder as a Python package. Every project in Workbench 3.1 must hence have an empty ``__init__.py`` file at its root level. Projects created for Workbench 3.0 **will not** work with Workbench 3.1 unless an ``__init__.py`` file is added. Refer to the user guide for the dialogue manager to learn how to use imports in your application container.


3.0.0 (2017-08-14)
------------------

* First release of the MindMeld Workbench conversational AI toolkit
