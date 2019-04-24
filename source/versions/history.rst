Package History
===============


4.1.0 (2019-04-22)
------------------

Major Features and Improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- De-couple Duckling, the numerical parser, from the core MindMeld platform

- Configure the MindMeld project to support Apache 2.0 open-source license


Breaking Changes
^^^^^^^^^^^^^^^^

- Replace all instances of the term ``mmworkbench`` to ``mindmeld``. Older pickled models that refer to the old term will not load, so delete the ``.generated`` folder in application folder, and rebuild all classifiers.


Bug Fixes and Other Changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- The language parser component correctly handles role types

- Add documentation for Webex Teams Integration for MindMeld

- Add extensive documentation to all methods in the MindMeld project for API documentation viewing


4.0.0 (2019-02-25)
------------------

Major Features and Improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Improved entity resolution for voice-based apps using n-best ASR transcripts

- Add support for user-defined custom feature extractors

- Replace wit-ai/duckling_old (deprecated numerical parser) with facebook/duckling (actively maintained library)

- Add support for Dialogue flows, an improved methodology for authoring constrained multi-turn dialogues

- Add support for entity recognition on n-best ASR transcripts


Breaking Changes
^^^^^^^^^^^^^^^^

- Refactor dialogue state handlers by adding a read-only request object and a writable responder object

- New modular project structure for MindMeld apps

- New model format using consistent internal feature representations



Bug Fixes and Other Changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Implement the ``predict_proba()`` method for role classifiers and entity classifiers

- Add support for viewing feature weights for model introspection

- Improve support for MindMeld applications as python packages

- Add support for query stemming during feature extraction

- Add support for numerical parser features for domain and intent classification

- Add support for word shape features for domain and intent classification

- Add support for dynamic gazetteers, which are online gazetteers that bias the natural language processor's prediction for the current turn

- Add support for asynchronous dialogue state handlers

- Refactor classifier features names to make them more consistent

- Re-enable automatic blueprint loading using the ``blueprint`` command

- Deprecate the ``session`` object and replace it with the ``context`` object

- Add support for parallel processing of entity recognition for n-best ASR transcripts

- MindMeld version compatibility checks are warnings, not exceptions


3.4.0 (2018-08-20)
------------------

Enhancements
^^^^^^^^^^^^

- Add the ability to add dialogue middleware to dialogue managers

- Add the ability to denote a handler as only reachable via target_dialogue_state

- Add the ability to explicitly denote a default handler

- Add the ability to specify different custom datasets for different NLP models

- Add support for frequency-based thresholding of n-gram features

- Add CLI command to generate predicted markup for queries

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

- The help messages for MindMeld command line tools (``python app.py`` and ``mmworkbench``) should show the correct list of compatible commands

- Various fixes to improve the numerical parser's robustness and logging


3.2.0 (2017-10-23)
------------------

Enhancements
^^^^^^^^^^^^

- Add Long Short Term Memory (LSTM) network as a model option for the entity recognizer

- Add support for TensorFlow-based deep learning models in MindMeld

- Add the ability to evaluate all NLP models at once with a single method/command

- Add functionality to specify a target dialogue state or a set of allowable intents for the next turn

- Add in-built support for conversational history management instead of relying on the client to preserve history across turns

- Improve interfaces for constructing responses within the dialogue state handlers (see **Compatibility Notes** below)


Compatibility Notes
^^^^^^^^^^^^^^^^^^^

- The ``prompt()`` and ``respond()`` methods of the ``DialogueResponder`` object are deprecated in MindMeld 3.2. See :doc:`Working with the Dialogue Manager <../userguide/dm>` to learn how to use new ``DialogueResponder`` methods in your dialogue state handlers.

3.1.0 (2017-09-20)
------------------

Enhancements
^^^^^^^^^^^^

- Add linear-chain conditional random field (CRF) as a model option for the entity recognizer

- Allow the role classifier to be trained with other text models (e.g. SVM, decision tree, etc.) in addition to logistic regression

- Make model configuration format for all classifiers consistent [See "Compatibility Notes" below]

- Add new metrics for better error analysis of entity recognition performance

- Add support for modularizing dialogue state handling logic by allowing arbitrary module imports in ``app.py`` (see **Compatibility Notes** below)

- Make blueprints check the current MindMeld package version to validate compatibility

- Only load NLP resources that are needed by active feature extractors (as defined in the model config) to improve runtime performance

Bug Fixes
^^^^^^^^^

- Correctly compute entity spans in queries with special characters

- Warn the developer and proceed with model training (if possible) when entity mapping or gazetteer files are missing

Compatibility Notes
^^^^^^^^^^^^^^^^^^^

- To make the interfaces for NLP classifiers consistent, the model configuration formats for the entity recognizer and the role classifier have been updated to be in line with the domain and intent classifiers. The model configurations for entity recognizer and role classifier from MindMeld 3.0 **will not** work with MindMeld 3.1. Refer to the user guide for those components to learn how to rewrite your 3.0 configs in the new 3.1 format.

- To support modular organization of dialogue state handling logic by allowing arbitrary package/module imports in the application container (``app.py``), MindMeld now needs to load the project folder as a Python package. Every project in MindMeld 3.1 must hence have an empty ``__init__.py`` file at its root level. Projects created for MindMeld 3.0 **will not** work with MindMeld 3.1 unless an ``__init__.py`` file is added. Refer to the user guide for the dialogue manager to learn how to use imports in your application container.


3.0.0 (2017-08-14)
------------------

* First release of the MindMeld conversational AI toolkit
