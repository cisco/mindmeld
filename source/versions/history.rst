Package History
===============

4.5.0 (2022-01-10)
------------------

Major Features and Improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Implemented support for deep neural networks and `Huggingface <https://huggingface.co/models>`_ models in our current NLP pipeline.

- Introduce active learning support for entity recognition.

- Improved the performance and speed of the `Tokenization pipeline <https://github.com/cisco/mindmeld/pull/398>`_.

- Updated Duckling dependency with the latest upstream changes.

Breaking Changes
^^^^^^^^^^^^^^^^

- Fixed a `bug <https://github.com/cisco/mindmeld/pull/405>`__ with the EntityLabelEncoder by refactoring the System Entity Recognizer attribute.
- Fixed a `bug <https://github.com/cisco/mindmeld/pull/387>`__ that ensures all model cache files are saved properly during an incremental build.

After these 2 fixes, older pickled models that refer to the previous versions will not load, so delete the ``.generated`` folder in application folder, and rebuild all classifiers.


Bug Fixes and Other Changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Fixed an `issue <https://github.com/cisco/mindmeld/pull/400>`__ with the build of docker containers.
- Set `default tokenizer <https://github.com/cisco/mindmeld/pull/397>`_ for all languages to Whitespace.
- Refactor how the resource loader is `initialized in NativeQA <https://github.com/cisco/mindmeld/pull/391>`_
- Fix a `bug <https://github.com/cisco/mindmeld/pull/390>`__ related to updating the MM version.
- Remove a redundant warning in `TextPreparationPipeline creation <https://github.com/cisco/mindmeld/pull/385>`_.
- Fixed a `bug <https://github.com/cisco/mindmeld/pull/408>`__ with the Letter Tokenizer that showed incorrect behavior when a mix of Latin/non-Latin characters appeared.

4.4.0 (2021-10-18)
------------------

Major Features and Improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Implemented a text preparation pipeline that supports :ref:`multilingual tokenization <tokenization>` and makes our Preprocessors, Normalizers, Tokenizers, and Stemmers configurable.

- Implemented :ref:`non-elasticsearch Question Answering capability <non_elasticsearch_question_answerer>` that works similar to Elasticsearch based Question Answerer.

- Implemented `deny_nlp` functionality for the `nlp.process` `API <https://github.com/cisco/mindmeld/pull/311/>`_.

- Added a :doc:`new Spanish blueprint <../blueprints/screening_app>` for non-English MindMeld applications.

- Implemented consistent input validation of APIs using Marshmallow

- Implemented automatic annotation tools to bootstrap training data in MindMeld

- Implemented multilingual annotation tools for automatically annotating non-English MindMeld applications

- Implemented :doc:`active learning pipeline <../walkthroughs/wt_active_learning>` for MindMeld

- Implemented multilingual :doc:`paraphrasing functionality <../userguide/augmentation>` to make it easier to bootstrap small conversational applications with data augmentation.

- Implemented query caching using sqlite db to reduce training time speed

- Integrated with DagsHub to provide experiments tracking and intuitive UIs to track model performance

- Updated Duckling dependency with the latest upstream changes


Bug fixes
^^^^^^^^^

- Fixed custom validation `bug <https://github.com/cisco/mindmeld/issues/352>`__ in Automatic Slotfilling
- Fixed input validation `bug <https://github.com/cisco/mindmeld/issues/363>`__ for certain queries
- Fixed `serialization issues <https://github.com/cisco/mindmeld/issues/270>`__ with responder object
- Fixed `bug <https://github.com/cisco/mindmeld/issues/274>`__ where duckling was not returning any entity candidates
- Fixed RASA to MindMeld conversion `bug <https://github.com/cisco/mindmeld/pull/277>`__
- Fixed a `path loading issue <https://github.com/cisco/mindmeld/issues/307>`_ with Windows environments
- Fixed a `memory leak issue <https://github.com/cisco/mindmeld/pull/296>`_ when loading multiple MindMeld apps in parallel
- Fixed a `sys_candidate` value `bug <https://github.com/cisco/mindmeld/pull/317>`__
- Fixed a `bug <https://github.com/cisco/mindmeld/pull/318>`__ with the Conditional Random Field model
- Fixed a feature extraction `bug <https://github.com/cisco/mindmeld/pull/323>`__ in MindMeld
- Fixed Question Answerer `issue <https://github.com/cisco/mindmeld/issues/220>`__ when using a compact json format for Knowledge Base and tightened up its interface to `avoid conflicting usage <https://github.com/cisco/mindmeld/issues/219>`_ of app_namespace and app_path arguments.


Legacy
^^^^^^

- Removed support for Python 3.5
- Refactored our legacy tokenizer functionality to improve configurability and add new functionality
- Refactored our Question Answerer component and improved it's interface



4.3.2 (2020-10-15)
------------------

Major Features and Improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Add auto entity annotating tool which leverages pre-trained NLP models to add entity annotations to training queries

- Add model tracking through DVC integration

- Add progress bars for classifier training

- Add sentiment features for classifiers

- Add support for custom resolution in custom evaluation function for Automatic Slotfilling

- Allow detailed entity resolution from Duckling

- Allow the MindMeld tokenizer to preserve special characters

- Allow the MindMeld app to configure the max history length that they should keep

- Allow the role classifier to process a single label

- Expose Elasticsearch scoring in QA responses


Bug fixes
^^^^^^^^^

- Fixed issue were entities were not immutable in the request object

- Fixed issue were the system entity recognizer would be loaded without being initialized

- Fixed token mismatch issue in the system entity feature extractor for queries with special characters. Retraining entity recognition models that use sys-candidates-seq feature is recommended.


Legacy
^^^^^^

- Add log warnings for Python 3.5; we will officially remove support in the next release


4.3.1 (2020-06-17)
------------------

Major Features and Improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Custom Actions provide the ability for applications to integrate external dialogue management logic with MindMeld applications

- Question Answerer can now leverage deep-learning based semantic embeddings (BERT, Glove) to produce more relevant answers to queries (available for Elasticsearch 7 and above)

- Automatic slot filing allows an intuitive way for developers to automatically prompt users for missing slots to fulfill an intent

- A new banking blueprint for enterprise use-cases

- WhatsApp Bot Integration with MindMeld

- Docker setup update to Elasticsearch 7

- MindMeld application can configure language and locale settings in the application config file


Bug fixes
^^^^^^^^^

- Addressed an issue which caused MindMeld to not detect system entities with no surrounding context (for example: "december 21st")

- Previously, MindMeld applications called Elasticsearch even if the application did not functionally use it (i.e have no entity to resolve); This has been fixed in :doc:`Managing Dependencies <../userguide/optional_dependency>`

- MindMeld had a dependency on Pandas which increased the overall library footprint and is removed in MindMeld 4.3


4.2.0 (2019-09-16)
------------------

Major Features and Improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- MindMeld UI is a sample web-based chat client interface to interact with any MindMeld application

- A built-in Question-Answering (QA) component for unstructured data using Elasticsearch

- A new Human Resources blueprint for enterprise use-cases

- Webex Teams Bot Integration

- MindMeld now supports internationalization through language and locale codes

- New built-in Spanish and English stemmers

- An improvement to DialogueFlow where the user can exit the current flow and return to the main dialogue flow

- Docker setup update that makes getting started with MindMeld much easier by removing the Elasticsearch dependency


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
