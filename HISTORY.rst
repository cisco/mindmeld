Package History
===============

3.1.0 (2017-09-18)
------------------

Enhancements
^^^^^^^^^^^^

- Add linear-chain conditional random field (CRF) as a model option for the entity recognizer

- Allow the role classifier to be trained with other text models (e.g. SVM, decision tree, etc.) in addition to logistic regression

- Make model configuration format for all classifiers consistent [See "Compatibility Notes" below]

- Add new metrics for better error analysis of entity recognition performance

- Add support for modularizing dialogue state handling logic by allowing arbitrary module imports in ``app.py`` [See "Compatibility Notes" below]

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
