Recent Changes
==============

Workbench 3.1
-------------

.. warning::

   We highly recommend that you read this section before upgrading to Workbench 3.1. In particular, note the changes that are required for making Workbench 3.0 projects work with Workbench 3.1.

Workbench 3.1 adds improved capabilities for natural language processing and application logic management, in addition to a few other enhancements and bug fixes. See :doc:`history` for the full release notes. Here are some of the highlights.

**1. Consistent configuration format for NLP classifiers**

The classifier configuration formats for the entity recognizer and the role classifier have been updated to be consistent with the domain and intent classifiers. See the relevant sections on :ref:`entity recognizer training <train_entity_model>` and :ref:`role classifier training <train_role_model>` for the new format.

If you have a Workbench 3.0 app where custom classifier configurations for the entity and role models are defined in the application configuration file (``config.py``), you have to manually update those configurations to the 3.1 format. If you have simply been using one of the Workbench blueprints without any modifications (or are fine with losing those local edits), you can upgrade your blueprint to the 3.1 format by using :ref:`Workbench's blueprint <getting_started_blueprint>` command. Running the ``blueprint`` command will download the version of the blueprint that is compatible with the latest stable Workbench release and overwrite your local copy.


**2. Support for modular dialogue state handling logic**

Relative imports of arbitrary modules and packages are now supported within the application container file (``app.py``). This means that all the application logic required for dialogue state handling need not necessarily be contained within a single Python file (``app.py``), as was the limitation with Workbench 3.0. In order support this capability, Workbench has to load each project as a Python package, which means that every project folder must now have an empty ``__init__.py`` file at its root level.

If you have an existing Workbench 3.0 app, manually add an empty ``__init__.py`` file at the root of your project folder to ensure compatibility with Workbench 3.1. Previously downloaded blueprints can be overwritten with the new 3.1-compatible versions by using :ref:`Workbench's blueprint <getting_started_blueprint>` command.

To learn more about support for relative imports, see the :ref:`application container <app_container>` section in Step 4 of the step-by-step guide.


**3. CRF for entity recognition**

Entity recognizers can now optionally be trained using a linear-chain conditional random field (CRF) instead of the default maximum entropy markov model. See :ref:`entity recognizer training <train_entity_model>`.


**4. More models for role classification** 

Role classifiers can now optionally be trained using any of the text models (e.g. SVM, Decision Tree, etc.) instead of the default maximum entropy model. See :ref:`role classifier training <train_role_model>`.


**5. New metrics for entity recognition**

Entity recognizer evaluation now exposes an additional set of metrics called segment-level errors that are easier to interpret and aid in understanding the model's sequence tagging performance better. See :ref:`entity recognizer evaluation <entity_evaluation>`.


