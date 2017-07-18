.. meta::
    :scope: private

Entity Recognizer
=================

The Entity Recognizer is run as the third step in the natural language processing pipeline to detect all the relevant entities in a given query. It is a `sequence labeling <https://en.wikipedia.org/wiki/Sequence_labeling>`_ model that is trained using all the labeled queries for a given intent. Labels are derived from the entity types annotated within the training queries. See :doc:`Step 6 <../quickstart/06_generate_representative_training_data>` for more details on training data preparation. Entity recognition models are trained per intent. A Workbench app hence has one entity recognizer for every intent that requires entity detection.

.. note::

   For a quick introduction, refer to :ref:`Step 7 <entity_recognition>` of the Step-By-Step Guide.

   Recommended prior reading: :doc:`Natural Language Processor <nlp>` chapter of the User Guide.


Access the entity recognizer
----------------------------

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

Each intent has its own :class:`EntityRecognizer` which can be accessed using the :attr:`entity_recognizer` attribute of the corresponding intent.

.. code-block:: python

   >>> # Entity recognizer for the 'change_alarm' intent in the 'times_and_dates' domain:
   >>> er = nlp.domains['times_and_dates'].intents['change_alarm'].entity_recognizer
   >>> er
   <EntityRecognizer ready: False, dirty: False>
   ...
   >>> # Entity recognizer for the 'check_weather' intent in the 'weather' domain:
   >>> er = nlp.domains['weather'].intents['check-weather'].entity_recognizer
   >>> er
   <EntityRecognizer ready: False, dirty: False>


.. _train_entity_model:

Training a baseline entity recognizer
-------------------------------------



Introduce the general ML techniques and methodology common to all NLP classifiers:
Getting the right kind of training data using in-house data generation and crowdsourcing, QAing and analyzing the data
Training a Workbench classifier, using k-fold cross-validation for hyperparameter selection
Training with default settings
Training with different classifier configurations (varying the model type, features or hyperparameter selection settings)
Testing a Workbench classifier on a held-out validation set
Doing error analysis on the validation set, retraining based on observations from error analysis by adding more training examples or feature tweaks
Getting final evaluation numbers on an unseen “blind” test set
Saving models for production use 

====


The MindMeld Entity Recognizer is a generalized version of a `Named Entity Recognizer <https://en.wikipedia.org/wiki/Named-entity_recognition>`_, which is common in NLP academic literature and research. This allows for detection of custom entities relevant to the application domain and not just standard `named entities <https://en.wikipedia.org/wiki/Named_entity>`_ like persons, locations and organizations.  


The Entity Recognizer's job is to analyze the user input and extract all the entities relevant to the current intent. In NLP literature, this problem is commonly referred to as 

The problem essentially consists of two parts:

The Entity Recognizer uses a machine-learned `sequence labeling <https://en.wikipedia.org/wiki/Sequence_labeling>`_ model to scan each word in

The problem consists of two parts:

1. Detect which spans of words within the input text correspond to entities of interest
2. Classify those detected text spans into a pre-determined set of entity types

The Entity Recognizer uses a Machine-Learned Sequence Labeling model to look at each word in the input query sequentially and assign a label to it. It is trained using labeled training data where queries are annotated to mark entity spans along with their corresponding entity types. We train a separate entity recognition model for each user intent since the types of entities required to satisfy the end goal vary from intent to intent. We will get into the details of build entity recognition models in :doc:`Entity Recognizer`.



At runtime, the Entity Recognizer loads the appropriate model, based on the predicted intent for the query. Once this step is done and we have extracted the relevant entities, we will finally have all the raw ingredients required to make sense out of the user input. The next step would be to put those together to build a semantic representation that encapsulates all the information necessary to execute the user's intended action.



====