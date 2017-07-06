.. meta::
    :scope: private

Natural Language Processor
==========================

The Natural Language Processor (NLP) understands the user's natural language input and outputs a representation that captures all the salient information in the user query. It uses a pipeline of six components that sequentially analyze the query, with each one building upon the output of the previous ones. These individual components are covered in depth in the following chapters. This chapter focuses on the :class:`NaturalLanguageProcessor` class in Workbench, which is a higher level abstraction encapsulating the full natural language processing pipeline. 

.. note::

   For a quick introduction, refer to :doc:`Step 7 <../quickstart/07_train_the_natural_language_processing_classifiers>` of the Step-By-Step Guide.

Instantiate the NLP class
-------------------------

Before you can use the Natural Language Processor, you need to generate the necessary training data for your app by following the guidelines in :doc:`Step 6 <../quickstart/06_generate_representative_training_data>`. You can then get started by importing the :class:`NaturalLanguageProcessor` class from Workbench's :mod:`nlp` module and instantiating an object with the path to your Workbench project.

.. code-block:: python

  >>> from mmworkbench.components.nlp import NaturalLanguageProcessor
  >>> nlp = NaturalLanguageProcessor(app_path='home_assistant')

The Natural Language Processor automatically infers the domain-intent-entity-role hierarchy for your app based on the project structure. Inspect the :attr:`domains` attribute of the :obj:`nlp` object to view the list of domains it identified.

.. code-block:: python

   >>> nlp.domains
   {
    'smart_home': <DomainProcessor 'smart_home' ready: False, dirty: False>,
    'times_and_dates': <DomainProcessor 'times_and_dates' ready: False, dirty: False>,
    'unknown': <DomainProcessor 'unknown' ready: False, dirty: False>,
    'weather': <DomainProcessor 'weather' ready: False, dirty: False>
   }

You can similarly view the list of :attr:`intents` for each of the :attr:`domains`.

.. code-block:: python

   >>> nlp.domains['times_and_dates'].intents 
   {
    'change_alarm': <IntentProcessor 'change_alarm' ready: False, dirty: False>,
    'check_alarm': <IntentProcessor 'check_alarm' ready: False, dirty: False>,
    'remove_alarm': <IntentProcessor 'remove_alarm' ready: False, dirty: False>,
     'set_alarm': <IntentProcessor 'set_alarm' ready: False, dirty: False>,
    'start_timer': <IntentProcessor 'start_timer' ready: False, dirty: False>,
    'stop_timer': <IntentProcessor 'stop_timer' ready: False, dirty: False>
   }
   >>> nlp.domains['weather'].intents
   {'check-weather': <IntentProcessor 'check-weather' ready: False, dirty: False>}

There are two other useful attributes that indicate the current status of an NLP object. First, the :attr:`ready` flag indicates if the NLP instance is ready for processing user input. The value of this attribute is ``True`` only if all the NLP classification models have been trained and can be used for making predictions on new queries. 

.. code-block:: python

   >>> nlp.ready
   False

The :attr:`dirty` flag indicates if the NLP object has changed since it was last loaded from, or written to disk. The value of this attribute is ``True`` if the models have been retrained since the last disk I/O operation.

.. code-block:: python

   >>> nlp.dirty
   False

The values of both these attributes are currently ``False`` since we have merely initialized an NLP object so far. We are yet to do something interesting with it. 


Build NLP models
----------------

As described in :doc:`Step 7 <../quickstart/07_train_the_natural_language_processing_classifiers>`, the fastest way to train a baseline natural language processor is by using the :meth:`NaturalLanguageProcessor.build` method. Depending on the complexity of your Workbench project and the size of the training data, this can take anywhere from a few seconds to several minutes to finish. If the logging level is set to ``INFO`` or below, you should see the build progress in the console along with the cross-validation accuracies for each of the classifiers.

.. code-block:: python

   >>> from mmworkbench import configure_logs; configure_logs()
   >>> from mmworkbench.components.nlp import NaturalLanguageProcessor
   >>> nlp = NaturalLanguageProcessor(app_path='food_ordering')
   >>> nlp.build()
   Fitting intent classifier: domain='ordering'
   Loading queries from file ordering/build_order/train.txt
   Loading queries from file ordering/exit/train.txt
   Loading queries from file ordering/greet/train.txt
   Loading queries from file ordering/help/train.txt
   Loading queries from file ordering/place_order/train.txt
   Loading queries from file ordering/start_over/train.txt
   Loading queries from file ordering/unsupported/train.txt
   Selecting hyperparameters using k-fold cross validation with 10 splits
   Best accuracy: 98.25%, params: {'C': 100, 'class_weight': {0: 1.5061564059900165, 1: 3.0562737642585551, 2: 0.9076278290025146, 3: 4.5641176470588229, 4: 2.5373456790123461, 5: 1.7793877551020409, 6: 0.47226711026615975}, 'fit_intercept': True}
   ...
   Fitting entity recognizer: domain='ordering', intent='build_order'
   Selecting hyperparameters using k-fold cross validation with 5 splits
   Best accuracy: 92.82%, params: {'C': 100, 'penalty': 'l1'}
   ...

The :meth:`build` method loads all the training queries, checks them for annotation errors, and then proceeds to build all the necessary NLP components using the machine learning settings defined in the app's configuration file (``config.py``). If settings have not been specified for a particular component, it uses Workbench's preset configuration for that component.

The :meth:`build` method thus accomplishes the following:

    - Calls the :meth:`fit` method on each of the classifiers in the domain-intent-entity-role hierarchy to train them using the provided model, feature and hyperparameter configurations.

    - Builds the Entity Resolver using the provided entity mapping file.

    - Configures the Language Parser using the provided parser configuration file.

You will learn more about each of these steps in the upcoming chapters which will also describe the default settings for each component and methods to override them with your own custom configurations. For experimentation, it is recommended that you train, tune and test each classifier individually to identify the ideal configuration for each. The best machine learning settings should then be stored in your application configuration file, ``config.py``, so the :meth:`build` method can use them instead of the Workbench defaults.

Here's an example of a ``config.py`` file where the default configurations for the domain and intent classifiers are being overridden by custom settings that have been optimized for the app.

.. code-block:: python

   DOMAIN_MODEL_CONFIG = {
       'model_type': 'text',
       'model_settings': {
           'classifier_type': 'logreg'
       },
       'params': {
           'C': 10,
       },
       'features': {
           "bag-of-words": {
               "lengths": [1, 2]
           },
           "edge-ngrams": {"lengths": [1, 2]},
           "in-gaz": {},
           "exact": {"scaling": 10},
           "gaz-freq": {},
           "freq": {"bins": 5}
       }
   }

   INTENT_MODEL_CONFIG = {
       'model_type': 'text',
       'model_settings': {
           'classifier_type': 'logreg'
       },
       'params': {
           'C': 10,
           "class_bias": 0.3
       },
       'features': {
           "bag-of-words": {
               "lengths": [1, 2]
           },
           "edge-ngrams": {"lengths": [1, 2]},
           "in-gaz": {},
           "exact": {"scaling": 10},
           "gaz-freq": {},
           "freq": {"bins": 5}
       }
   }
   
Refer to the chapters on the individual NLP components for details on the different configuration options.


Run NLP models
--------------

A trained NLP pipeline can be run on a test query using the :meth:`NaturalLanguageProcessor.process` method. The :meth:`process` method sends the query for sequential processing by each component in the NLP pipeline and returns the aggregated output from all of them.

.. code:: python

   >>> nlp.process("I'd like a mujaddara wrap and two chicken kebab from palmyra")
   {
    'domain': 'ordering',
    'entities': [
      {
        'role': None,
        'span': {'end': 24, 'start': 11},
        'text': 'mujaddara wrap',
        'type': 'dish',
        'value': [{'cname': 'Mujaddara Wrap', 'id': 'B01DEFNIRY'}]
      },
      {
        'role': None,
        'span': {'end': 32, 'start': 30},  
        'text': 'two',
        'type': 'sys_number',
        'value': {'value': 2}
      },
      {
        'children': [
          {
            'role': None,
            'span': {'end': 32, 'start': 30},
            'text': 'two',    
            'type': 'sys_number',
            'value': {'value': 2}
          }
        ],
        'role': None,
        'span': {'end': 46, 'start': 34},
        'text': 'chicken kebab',
        'type': 'dish',
        'value': [{'cname': 'Chicken Kebab', 'id': 'B01DEFMUSW'}]
      },
      {
        'role': None,
        'span': {'end': 59, 'start': 53},
        'text': 'palmyra',
        'type': 'restaurant',
        'value': [{'cname': 'Palmyra', 'id': 'B01DEFLJIO'}]
      }
    ],
    'intent': 'build_order',
    'text': "I'd like a mujaddara wrap and two chicken kebab from palmyra"
   }

The return value is a dictionary with the following fields:

+----------+--------------------------------------------------------------------------+-----------------------------------------------+
| Key      | Value                                                                    | Component(s) Responsible                      |
+==========+==========================================================================+===============================================+
| domain   | The predicted domain label for the query                                 | :doc:`Domain Classifier <domain_classifier>`  |
+----------+--------------------------------------------------------------------------+-----------------------------------------------+
|          | A list of the entities recognized in the query, with each entity         | :doc:`Entity Recognizer <entity_recognizer>`, |
| entities | represented as a dictionary containing entity-specific properties        | :doc:`Role Classifer <role_classifier>`,      |
|          | like detected text span, entity type, role type, resolved value,         | :doc:`Entity Resolver <entity_resolver>`,     |
|          | children (dependents), etc.                                              | :doc:`Language Parser <parser>`               |
+----------+--------------------------------------------------------------------------+-----------------------------------------------+
| intent   | The predicted intent label for the query                                 | :doc:`Intent Classifier <intent_classifier>`  |
+----------+--------------------------------------------------------------------------+-----------------------------------------------+
| text     | The input query text                                                     |                                               |
+----------+--------------------------------------------------------------------------+-----------------------------------------------+

The :meth:`process` method executes the following steps:

    - Calls the :meth:`predict` (or equivalent) method for each of the classifiers in the domain-intent-entity-role hierarchy to detect the domain, intent, entities and roles in the query

    - Calls the Entity Resolver's :meth:`predict` method to resolve all detected entities to their canonical forms

    - Calls the Language Parser's :meth:`parse_entities` method to cluster all the resolved entities

    - Returns the detailed output from each component

The chapters on the individual NLP components provide more details on the above steps, along with documentation on their outputs and advanced features like batch testing and evaluation.


Save Models for Future Use
--------------------------

Once you have trained an NLP pipeline and are satisfied with its accuracy, you can save it to disk using the :meth:`NaturalLanguageProcessor.dump` method. The :meth:`dump` method saves all the trained models to a Workbench-generated cache folder within your project.

.. code:: python

   >>> nlp.dump()
   Saving intent classifier: domain='ordering'
   Saving entity recognizer: domain='ordering', intent='build_order'
   ...

 The saved models can then be loaded anytime in the future using the :meth:`NaturalLanguageProcess.load` method.

.. code:: python

   >>> nlp.load()
   Loading intent classifier: domain='ordering'
   ...

Refer to the chapters


