Recent Changes
==============

Workbench 3.1
-------------

.. warning::

   Upgrading some existing Workbench 3.0 projects to Workbench 3.1 will fail unless modified as described below.

Workbench 3.1 has improved natural language processing and application logic management capabilities, along with enhancements and bug fixes. This section provides highlights; see :doc:`history` for the full release notes.

**1. Consistent configuration format for NLP classifiers**

The classifier configuration formats for the entity recognizer and the role classifier have been updated to be consistent with the domain and intent classifiers. See the relevant sections on :ref:`entity recognizer training <train_entity_model>` and :ref:`role classifier training <train_role_model>` for the new format.

*For existing Workbench 3.0 apps:*

 - If custom classifier configurations for the entity and role models are defined in the application configuration file (``config.py``), you must manually update those configurations to the 3.1 format.

 - If the app is based on a Workbench blueprint, you can use the :ref:`blueprint <getting_started_blueprint>` command to upgrade to the 3.1 format. Running this command will download the version of the blueprint that is compatible with the latest stable Workbench release and overwrite your local copy. This means that if you have modified the blueprint, your modifications will be lost, so you should consider saving the modifications outside of your project and manually adding them back in after upgrading.

**2. Support for modular dialogue state handling logic**

Relative imports of arbitrary modules and packages are now supported within the application container file (``app.py``). This means that all application logic required for dialogue state handling need not be contained within a single Python file (``app.py``), as was the case with Workbench 3.0. Because Workbench loads each project as a Python package to support this new capability, every project folder must now have an empty ``__init__.py`` file at root level.

*For existing Workbench 3.0 apps:*

 - Manually add an empty ``__init__.py`` file at the root of your project folder to ensure compatibility with Workbench 3.1. You can use the :ref:`blueprint <getting_started_blueprint>` command to overwrite previously-downloaded blueprints with the new 3.1-compatible versions.

To learn more about support for relative imports, see the :ref:`application container <app_container>` section in Step 4 of the Step-by-Step Guide.

**3. CRF for entity recognition**

You now have the option of training your entity recognizers using a linear-chain conditional random field (CRF) instead of the default maximum entropy Markov model (MEMM). See :ref:`entity recognizer training <train_entity_model>`.

**4. More models for role classification**

You now have the option of training your role classifiers using any of the text models (namely, SVM, Decision Tree, and so on) instead of the default maximum entropy model. See :ref:`role classifier training <train_role_model>`.

**5. New metrics for entity recognition**

Entity recognizer evaluation now exposes new metrics called *segment-level errors*. These make it easier to interpret and understand the model's sequence tagging performance. See :ref:`entity recognizer evaluation <entity_evaluation>`.


