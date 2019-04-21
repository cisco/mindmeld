Recent Changes
==============

MindMeld 4.1
-------------

.. warning::

   This release includes breaking changes. See below
   for instructions on migrating your apps from MindMeld 4.0 to MindMeld 4.1


MindMeld 4.1 allows the package to be open-sourced by complying to the Apache 2.0 license standard.

**1. De-coupled Duckling from MindMeld**

Duckling, the numerical parser used to detect system entities, is now a configurable option, so an application can
disable it if it doesn't need it. See :ref:`configuring system entities section <configuring-system-entities>` for more details.

**2. Added extensive API documentation for the MindMeld library**

The API reference for the MindMeld package can be found here: :doc:`../internal/api_reference`.

**3. Replaced all instances of the term mmworkbench to mindmeld**

All instances of the term ``mmworkbench`` in the codebase have been replaced to ``mindmeld`` to be consistent with the new open-source package name.
Due to this change, older saved models will no longer load in 4.1. Please make sure to delete the ``.generated`` folder in
the top level of the application and re-build the application.


MindMeld 4.0
-------------

.. warning::

   This is a major release that includes breaking changes. Refer to the changes numbered 6, 9, and
   10 below for instructions on migrating your apps from MindMeld 3 to MindMeld 4.

MindMeld 4 is a major update to the MindMeld conversational AI platform, adding a
number of new features to the natural language processor and dialogue manager components. This
section provides highlights; see :doc:`history` for the full release notes.

**1. Robustness to ASR errors**

Conversational applications that support voice inputs use an automatic speech recognition (ASR)
system to convert the input speech into text and then send the resulting transcript to the
MindMeld NLP pipeline. ASRs often make errors, especially on domain-specific vocabulary and
proper nouns which can in turn adversely affect the accuracy of the NLP classifiers. MindMeld 4
introduces a couple of new techniques to make the entity processing steps (recognition and
resolution) more resilient to ASR errors. Read the new chapter on :doc:`../userguide/voice` for more details.


**2. Improved recognition of numerical entities**

MindMeld 4 uses the actively maintained `Duckling library <https://github.com/facebook/duckling>`_
for recognizing numerical entities. The new Haskell-based version is faster and more robust than
the deprecated `Java-based version <https://github.com/wit-ai/duckling_old>`_ that was used in
MindMeld 3. There are minor changes to the MindMeld system entity recognizer's
:meth:`parse_numerics` method as a result. See the
:ref:`system entities section <system-entities>`.


**3. Dynamic gazetteers**

Gazetteer-based features have a significant impact on NLP accuracy since they provide a very
strong signal to the classification models. This is especially true for entity recognition. In
addition to the static gazetteers used by the NLP classifiers at training time, MindMeld 4
introduces the ability to dynamically inject new entries into the gazetteers at runtime to further
aid the model in making the right prediction. The section on
:ref:`dynamic gazetteers <dynamic_gaz>` in the dialogue manager chapter describes when and how to
use this new functionality.


**4. New features for text classification**

MindMeld 4 adds three new feature extractors for the domain and intent classifiers:

- The ``'word-shape'`` feature encodes information about the presence of capitalization, numerals,
  punctuation, etc. in the input query.

- The ``'sys-candidates'`` feature indicates the presence of system entities in the query.
  This feature extractor was only available to the entity recognizer in previous versions.

- The ``'enable-stemming'`` feature extracts stemmed versions of the query tokens in
  addition to the regular bag-of-words features.

Refer to the "Feature Extraction Settings" section of the domain and intent classifier chapters for
more details.


**5. Support for user-defined features**

If the standard set of available features for the various classifiers isn't adequate for your use
case, MindMeld now allows you to define your own custom feature extractors and use them with the
NLP models. See the new chapter on :doc:`../userguide/custom_features`.


**6. Improvements to model debugging**

The :meth:`predict_proba` method is now available for the entity recognizer and the role
classifier as well. The entity recognizer's :meth:`predict_proba` method outputs a confidence score
for each detected entity. The role classifier's :meth:`predict_proba` method returns a probability
distribution across all the possible role labels for a given entity. See the relevant sections in
the :ref:`entity recognizer <predict_entities>` and :ref:`role classifier <predict_roles>`
chapters.

While training a new model or investigating classification errors, it is useful to view the
features used by the model to make sure they are being extracted correctly. To enable this, each
classifier in the MindMeld NLP hierarchy now exposes a :meth:`view_extracted_features` method that
dumps all the features extracted from a given query. See the section titled "Viewing features
extracted for classification" for each NLP classifier.

To make MindMeld's model inspection capabilities more user-friendly, the internal representation
of all extracted features has been modified to make the output of :meth:`nlp.inspect` and
:meth:`view_extracted_features` methods easier to comprehend. Due to this change, models trained
and saved using MindMeld 3 cannot be loaded in MindMeld 4. You need to train your models afresh
on MindMeld 4.

.. warning::

   NLP models trained on MindMeld 3 cannot be loaded by MindMeld 4.

.. tip::

   After installing MindMeld 4, follow these steps to upgrade your old project:

   - Modify your app's project structure to comply with the newly introduced
     :ref:`modular project structure <new_project_structure>`.
   - Clear all the previously trained models by running ``python -m APP_NAME clean``.
   - Rebuild all models by running ``python -m APP_NAME build`` or running :meth:`nlp.build` in a
     Python shell.


**7. Dialogue flows**

MindMeld 4 introduces a new construct called *Dialogue Flow* for easily structuring conversation
flows where the user needs to be directed towards a specific end goal in a focused manner. See the
new :ref:`dialogue_flow` section in the Dialogue Manager chapter.


**8. Asynchronous dialogue state handlers and middleware**

To improve the performance and scalability of complex applications that depend on remote services,
MindMeld 4 supports asynchronous execution of dialogue state handling logic. Read the section on
:ref:`async_dialogue` for more information.


**9. New dialogue state handler interface**

MindMeld 4 introduces a new dialogue state handler interface that makes an explicit mutability distinction between the data
being passed into the dialogue manager from the client and the natural language processor (immutable) and the
output data written by the dialogue state handlers and sent back to the client (mutable). This distinction is useful in
cases where a single request is handled by multiple dialogue state handlers in sequence, and it's important to keep track of both
the original data passed into the dialogue manager and the new data being generated by the dialogue state handling logic. Here is
an example of the new interface, where the ``request`` object is the immutable data passed into the handler and the
``responder`` object is the carrier of the mutable data written to by the handler:

.. code:: python

   @app.handle(intent='greet')
   def welcome(request, responder):
      username = request.context.get('username', 'World')
      responder.reply('Hello ' + username)
      responder.frame['message'] = 'Hello ' + username

See the :ref:`updated section <dialogue_state_handlers>` in the dialogue manager chapter for more details on the ``request`` and ``responder`` objects.

.. warning::

   The new dialogue state handler interface is incompatible with MindMeld 3 applications.

.. tip::

   Previously, the application used the ``context`` and ``responder`` objects in its dialogue state handlers, e.g. ``def welcome(context, responder)``.

   The ``context`` object has now been replaced by the immutable ``request`` object which cannot be written to. You can only perform write operations on the corresponding properties in the mutable ``responder`` object. You should write all your data to the appropriate ``responder`` object property instead of the ``context`` dictionary.

   See the :ref:`examples <dialogue_example>` in the user guide and the blueprints.

.. _new_project_structure:

**10. New project structure**

Previously, MindMeld required all application logic to be in a single file, ``app.py``. As an application grows in complexity, this approach is not scalable.
MindMeld 4 allows the application logic to be shared across multiple files. The :ref:`home assistant <home_assistant>` blueprint is an example of this modularized approach,
where the ``times_and_dates.py`` file handles all the logic for the time and date-related functionality.

In the new project structure, we introduce two files: ``__init__.py`` where you register all the application files as imports and ``__main__.py`` where you register the application command line interface.
Read the updated section in the :ref:`Step-by-Step Guide <app_container>` for more information.

.. warning::

   The new project structure is incompatible with MindMeld 3 applications.

.. tip::

   - In the new modular application project structure, we require two files: ``__init__.py`` where you register all the application files as imports, and ``__main__.py`` where you register the application command line interface. You can still keep all the application logic in a single file (``__init__.py``); this is how we organize most of our blueprint applications except for Home Assistant.

   - If the app has all the dialogue state logic in ``app.py``, rename the file to ``__init__.py``. Add a new file called ``__main__.py``, similar to ``__main__.py`` in :ref:`Home Assistant <home_assistant>`.

   - To build and run the application, use the commands ``python -m my_app build`` and ``python -m my_app run`` from outside the application directory.


MindMeld 3.4
-------------

MindMeld 3.4 brings new functionality to the dialogue manager along with some improvements to the natural language processing pipeline. This section provides highlights; see :doc:`history` for the full release notes.

**1. Dialogue middleware**

MindMeld 3.4 provides a useful mechanism for changing the behavior of many or all dialogue states via middleware. Middleware are developer-defined functions that get called for every request before the matched dialogue state handler. The :ref:`Dialogue Middleware <dialogue_middleware>` section describes potential use cases for the middleware functionality and details on how to implement them.

**2. Targeted-only and default dialogue state handlers**

MindMeld 3.2 introduced the ability to skip NLP classification and pre-select a :ref:`target dialogue state <target_dialogue_state_release_note>` for the next conversational turn. In 3.4, you can further mark certain dialogue states as ``targeted_only`` to exclude them from consideration in regular non-targeted turns.

Additionally, you can now also explicitly denote a dialogue state handler as the default handler without worrying about where it appears in ``app.py``. See the updated :doc:`Dialogue Manager <../userguide/dm>` chapter for more details.

**3. Different datasets for different NLP models**

It is now possible to specify different sets of labeled query files for training or testing different classifiers in the NLP pipeline. This addresses a big limitation in the earlier versions of MindMeld. For instance, previously, you couldn't add data files under an intent folder and use them only for training the entity recognizer without also affecting the domain or intent models. MindMeld 3.4 gives you the flexibility to do so and hence have a finer control over the behavior of your individual classification models. Read more about the newly added `Custom Train/Test Settings` in the "Classifier configuration" section for each NLP classifier.

**4. Frequency-based thresholding for n-gram features**

MindMeld 3.4 allows you to specify a frequency threshold for n-gram feature extractors such as ``bag-of-words`` and ``char-ngrams`` to prevent rare n-grams from being used as features in your classification model. See `Feature Extraction Settings` under the "Classifier configuration" section for each NLP classifier.

**5. Batch predictions**

The :ref:`MindMeld CLI <cli>` has been updated with a new ``predict`` command that runs NLP predictions on a given set of queries using your app's trained models. The command is useful when you want to run your NLP models in batch on a dataset of queries or bootstrap expected labels in new queries for training. For instance, consider the case where you are preparing additional training data to improve your entity recognizer's performance. It is a lot easier to annotate your new training queries with your existing entity model and then manually correct any errors, than go through every new query and annotate the ground truth entities by hand from scratch.


MindMeld 3.3
-------------

MindMeld 3.3 contains many useful enhancements aimed at reducing the amount of time it takes to iterate on ML experiments and giving developers a finer-grained control over certain aspects of the application behavior. This section provides highlights; see :doc:`history` for the full release notes.

**1. New feature types and inspection capabilities for NLP models**

In addition to word n-grams, you can now use character n-grams as features for the :doc:`domain classifier <../userguide/domain_classifier>`, :doc:`intent classifier <../userguide/intent_classifier>` and :doc:`entity recognizer <../userguide/entity_recognizer>`. Refer to the "Feature Extraction Settings" section of each classifier for more details.

For the domain and intent classifiers, you can also use the newly-introduced feature inspection capability in MindMeld to view the learned feature weights for your trained models. See the section titled "Inspect features and their importance" for each classifier.

**2. Improvements to NLP model training**

**Overriding global configuration:** Depending on the characteristics and distribution of your training data across domains and intents, you might want to train a different kind of model for each domain, intent, or entity type in your application. This was not possible previously as you could only specify one global configuration for each classifier type in your NLP pipeline. Refer to the updated section on :ref:`custom configurations <custom_configs>` to see how MindMeld 3.3 allows you to override these global settings on a model-by-model basis.

..

**Incremental builds:** Till version 3.2, every call to the :meth:`NaturalLanguageProcessor.build` method kicked off a full build where MindMeld trained/retrained every NLP component from scratch across every domain, intent, and entity type in the project. From version 3.3 onwards, you can do an incremental build where the :class:`NaturalLanguageProcessor` only trains those subset of models that have been affected by changes to the training data and associated resources. This significantly reduces the time to rebuild the NLP pipeline after small changes to the data. See :ref:`building models incrementally <incremental_builds>`.

**3. Custom datasets**

You can now create your own arbitrarily-named custom datasets in addition to the default ``'train'`` and ``'test'`` sets recognized by MindMeld. This allows you to store multiple datasets for your ML experiments and select the relevant dataset for use with each round of training or testing. See :ref:`select data for experiments <custom_datasets>`.

**4. Improved support for dates and times**

For applications dealing with temporal events, you can now specify the time zone and timestamp associated with each query to the :class:`NaturalLanguageProcessor` to ensure accurate prediction of time-based :ref:`system entities <system-entities>`. See :ref:`specifying request timestamp and time zone <specify_timestamp>`.

**5. Preprocessor**

The preprocessor is a new component that has been added to MindMeld in version 3.3. It allows developers to define any custom preprocessing logic that must be applied on each query before being processed by the NLP pipeline. Read more in the new user guide chapter on :doc:`../userguide/preprocessor`.


MindMeld 3.2
-------------

MindMeld 3.2 brings deep learning models to the MindMeld platform for the first time. This release also improves natural language processing and enhances dialogue management capabilities. This section provides highlights; see :doc:`history` for the full release notes.

**1. Deep Learning for Entity Recognition (Beta)**

You can now opt to train your entity recognizers with a Long Short Term Memory (LSTM) network build in TensorFlow. See :ref:`Train an entity recognizer <train_entity_model>`.

.. _target_dialogue_state_release_note:

**2. Support for targeted dialogue state handling**

The dialogue manager now offers finer-grained control over the dialogue flow logic. You can specify rules that override or bias the output of the NLP classifiers to ensure that you reach a pre-determined dialogue state in the next conversational turn. See :ref:`Targeted Dialogue State Handling <targeted_dialogue>`.

**3. Improved dialogue state handler interfaces**

In version 3.2, the term *directives* replaces the term *client actions* found in previous versions. Also, the ``DialogueResponder`` class used in dialogue state handlers has been refactored to make its functions more intuitive. See :ref:`responder <responder>`.

*For existing MindMeld 3.1 apps:*

 - If the app used the ``responder.prompt()`` construct, change that to ``responder.reply()`` followed by a ``responder.listen()``.

 - If the app used the ``responder.respond()`` construct, change that to ``responder.direct()``.

**4. Easy evaluation interface**

The ``NaturalLanguageProcessor`` class now has an ``evaluate()`` method that runs model evaluation for all the components in the NLP pipeline. The :ref:`MindMeld CLI <cli>` has a corresponding ``evaluate`` command.

**5. Conversational History Management**

The ``history`` field of the ``context`` object used by dialogue state handlers is now maintained by MindMeld. Prior to 3.2, MindMeld assumed that the client would manage the conversational history by appending the necessary information to the ``history`` after each turn.


MindMeld 3.1
-------------

.. warning::

   Upgrading some existing MindMeld 3.0 projects to MindMeld 3.1 will fail unless modified as described below.

MindMeld 3.1 has improved natural language processing and application logic management capabilities, along with enhancements and bug fixes. This section provides highlights; see :doc:`history` for the full release notes.

**1. Consistent configuration format for NLP classifiers**

The classifier configuration formats for the entity recognizer and the role classifier have been updated to be consistent with the domain and intent classifiers. See the relevant sections on :ref:`entity recognizer training <train_entity_model>` and :ref:`role classifier training <train_role_model>` for the new format.

*For existing MindMeld 3.0 apps:*

 - If custom classifier configurations for the entity and role models are defined in the application configuration file (``config.py``), you must manually update those configurations to the 3.1 format.

 - If the app is based on a MindMeld blueprint, you can use the :ref:`blueprint <getting_started_blueprint>` command to upgrade to the 3.1 format. Running this command will download the version of the blueprint that is compatible with the latest stable MindMeld release and overwrite your local copy. This means that if you have modified the blueprint, your modifications will be lost, so you should consider saving the modifications outside of your project and manually adding them back in after upgrading.

**2. Support for modular dialogue state handling logic**

Relative imports of arbitrary modules and packages are now supported within the application container file (``app.py``). This means that all application logic required for dialogue state handling need not be contained within a single Python file (``app.py``), as was the case with MindMeld 3.0. Because MindMeld loads each project as a Python package to support this new capability, every project folder must now have an empty ``__init__.py`` file at root level.

*For existing MindMeld 3.0 apps:*

 - Manually add an empty ``__init__.py`` file at the root of your project folder to ensure compatibility with MindMeld 3.1. You can use the :ref:`blueprint <getting_started_blueprint>` command to overwrite previously-downloaded blueprints with the new 3.1-compatible versions.

To learn more about support for relative imports, see the :ref:`application container <app_container>` section in Step 4 of the Step-by-Step Guide.

**3. CRF for entity recognition**

You now have the option of training your entity recognizers using a linear-chain conditional random field (CRF) instead of the default maximum entropy Markov model (MEMM). See :ref:`entity recognizer training <train_entity_model>`.

**4. More models for role classification**

You now have the option of training your role classifiers using any of the text models (namely, SVM, Decision Tree, and so on) instead of the default maximum entropy model. See :ref:`role classifier training <train_role_model>`.

**5. New metrics for entity recognition**

Entity recognizer evaluation now exposes new metrics called *segment-level errors*. These make it easier to interpret and understand the model's sequence tagging performance. See :ref:`entity recognizer evaluation <entity_evaluation>`.


