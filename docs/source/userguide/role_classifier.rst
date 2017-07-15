.. meta::
    :scope: private

Role Classifier
===============

The Role Classifier is run as the fourth step in the natural language processing pipeline to determine the target roles for entities in a given query. It is a machine-learned `classification <https://en.wikipedia.org/wiki/Statistical_classification>`_ model that is trained using all the labeled queries for a given intent. Labels are derived from the role types annotated within the training queries. See :doc:`Step 6 <../quickstart/06_generate_representative_training_data>` for more details on training data preparation. Role classification models are trained per entity type. A Workbench app hence has one role classifier for every entity type with associated roles.

.. note::

   For a quick introduction, refer to :ref:`Step 7 <role_classification>` of the Step-By-Step Guide.

   Recommended prior reading: :doc:`Natural Language Processor <nlp>` chapter of the User Guide.


Access the role classifier
--------------------------

Before using any of the NLP componenets, you need to generate the necessary training data for your app by following the guidelines in :doc:`Step 6 <../quickstart/06_generate_representative_training_data>`. You can then start by :ref:`instantiating an object <instantiate_nlp>` of the :class:`NaturalLanguageProcessor` (NLP) class.

.. code-block:: python

  >>> from mmworkbench.components.nlp import NaturalLanguageProcessor
  >>> nlp = NaturalLanguageProcessor(app_path='home_assistant')
  >>> nlp
  <NaturalLanguageProcessor 'home_assistant' ready: False, dirty: False>

Next, verify that the NLP has correctly identified all the domains and intents for your app.

.. code-block:: python

   >>> nlp.domains
   {
    'smart_home': <DomainProcessor 'smart_home' ready: False, dirty: False>,
    'times_and_dates': <DomainProcessor 'times_and_dates' ready: False, dirty: False>,
    'unknown': <DomainProcessor 'unknown' ready: False, dirty: False>,
    'weather': <DomainProcessor 'weather' ready: False, dirty: False>
   }
   ...
   >>> nlp.domains['times_and_dates'].intents
   {
   	'change_alarm': <IntentProcessor 'change_alarm' ready: True, dirty: True>,
 	'check_alarm': <IntentProcessor 'check_alarm' ready: False, dirty: False>,
 	'remove_alarm': <IntentProcessor 'remove_alarm' ready: False, dirty: False>,
 	'set_alarm': <IntentProcessor 'set_alarm' ready: True, dirty: True>,
 	'start_timer': <IntentProcessor 'start_timer' ready: True, dirty: True>,
 	'stop_timer': <IntentProcessor 'stop_timer' ready: False, dirty: False>
   }
   ...
   >>> nlp.domains['weather'].intents
   {
   	'check-weather': <IntentProcessor 'check-weather' ready: False, dirty: False>
   }

Call the :meth:`build` method for the intent you are interested in and inspect the identified entity types. The :meth:`build` operation can take several minutes if the number of training queries for the chosen intent is large.

.. code-block:: python

   >>> nlp.domains['times_and_dates'].intents['change_alarm'].build()
   >>> nlp.domains['times_and_dates'].intents['change_alarm'].entities
   {
   	'time': <EntityProcessor 'time' ready: True, dirty: True>
   }

The :class:`RoleClassifier` for each entity type can then be accessed using the :attr:`role_classifier` attribute of the corresponding entity.

.. code-block:: python

   >>> rc = nlp.domains['times_and_dates'].intents['change_alarm'].entities['time'].role_classifier
   >>> rc
   <RoleClassifier ready: True, dirty: True>


Training a baseline role classifier
-----------------------------------




Introduce the general ML techniques and methodology common to all NLP classifiers:
Getting the right kind of training data using in-house data generation and crowdsourcing, QAing and analyzing the data
Training a Workbench classifier, using k-fold cross-validation for hyperparameter selection
Training with default settings
Training with different classifier configurations (varying the model type, features or hyperparameter selection settings)
Testing a Workbench classifier on a held-out validation set
Doing error analysis on the validation set, retraining based on observations from error analysis by adding more training examples or feature tweaks
Getting final evaluation numbers on an unseen “blind” test set
Saving models for production use 

===


Role Classification is the task of identifying predicates and predicate arguments. A **semantic role** in language is the relationship that a syntactic constituent has with a predicate. In Conversational NLU, a **role** represents the semantic theme a given entity can take. It can also be used to define how a named entity should be used for fulfilling a query intent. For example, in the query :red:`"Play Black Sabbath by Black Sabbath"`, the **title** entity :green:`"Black Sabbath"` has different semantic themes - **song** and **artist** respectively.

Treating Named Entity Recognition (NER) and Semantic Role Labeling (SRL) as separate tasks has a few advantages -

* NER models are hurt by splitting examples across fairly similar categories. Grouping entities with significantly overlapping entities and similar surrounding natural language will lead to better parsing and let us use more powerful models.
* Joint NER & SRL needs global dependencies, but fast & good NER models only do local. NER models (MEMM, CRF) quickly become intractable with long-distance dependencies. Separating NER from SRL let us use local dependencies for NER and long-distance dependencies in SRL.
* Role labeling might be a multi-label problem. With multi-label roles, we can use the same entity to query multiple fields.

===