.. meta::
    :scope: private

Working with the Natural Language Processor
===========================================

We have seen how the :ref:`Natural Language Processor (NLP) <arch_nlp>` uses a pipeline of components to analyze the query. Workbench encapsulates this pipeline in a higher-level abstraction, in the form of the :class:`NaturalLanguageProcessor` Python class, or NLP class. This chapter focuses on the NLP class,  while subsequent chapters examine each individual component of the pipeline.

.. note::

   :doc:`Step 7 <../quickstart/07_train_the_natural_language_processing_classifiers>` of the Step-By-Step Guide is a pre-requisite for this chapter.

.. _instantiate_nlp:

Instantiate the NLP class
-------------------------

To use the natural language processor, you must first generate the training data for your app. See :doc:`Step 6 <../quickstart/06_generate_representative_training_data>`. Once you have training data, import the :class:`NaturalLanguageProcessor` class from the Workbench :mod:`nlp` module and instantiate an object with the path to your Workbench project.

.. code-block:: python

   >>> from mmworkbench.components.nlp import NaturalLanguageProcessor
   >>> nlp = NaturalLanguageProcessor(app_path='home_assistant')
   >>> nlp
   <NaturalLanguageProcessor 'home_assistant' ready: False, dirty: False>

The natural language processor automatically infers the domain-intent-entity-role hierarchy for your app based on the project structure. Inspect the :attr:`domains` attribute of the :obj:`nlp` object to view the list of domains it identified.

.. code-block:: python

   >>> nlp.domains
   {
    'smart_home': <DomainProcessor 'smart_home' ready: False, dirty: False>,
    'times_and_dates': <DomainProcessor 'times_and_dates' ready: False, dirty: False>,
    'unknown': <DomainProcessor 'unknown' ready: False, dirty: False>,
    'weather': <DomainProcessor 'weather' ready: False, dirty: False>
   }

View the list of :attr:`intents` for each of the :attr:`domains`.

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
   ...
   >>> nlp.domains['weather'].intents
   {'check_weather': <IntentProcessor 'check_weather' ready: False, dirty: False>}

Upon initialization, the natural language processor merely scans the directory structure of your project, but does not read in the training data files. At this point in our tutorial, it has no knowledge of the entities associated with each intent.

.. code-block:: python

   >>> nlp.domains['weather'].intents['check_weather'].entities
   {}

The NLP learns about the entities when labeled queries are loaded at model training time. Once training is finished, you can use the :attr:`entities` attribute to view the entity types identified for each intent. The code snippet below introduces the :meth:`NaturalLanguageProcessor.build` method for model training. This method can take several minutes to run.

.. code-block:: python

   >>> nlp.build()
   >>> nlp.domains['weather'].intents['check_weather'].entities
   {
    'city': <EntityProcessor 'city' ready: True, dirty: True>,
    'sys_interval': <EntityProcessor 'sys_interval' ready: True, dirty: True>,
    'sys_time': <EntityProcessor 'sys_time' ready: True, dirty: True>,
    'unit': <EntityProcessor 'unit' ready: True, dirty: True>
   }

The :attr:`ready` and :attr:`dirty` attributes further describe the status of an NLP object.

The :attr:`ready` flag indicates whether the NLP instance is ready to process user input. Its value is ``True`` only if all the NLP classification models have been trained and can be used for making predictions on new queries.

.. code-block:: python

   >>> nlp.ready
   False

The :attr:`dirty` flag indicates whether the NLP object has changed since last loaded from or written to disk. Its value is ``True`` if the models have been retrained since the last disk I/O operation.

.. code-block:: python

   >>> nlp.dirty
   False

So far in our tutorial, the NLP object has been initialized but has not yet been trained, so :attr:`ready` and :attr:`dirty` are both false.


.. _build_nlp:

Train the NLP pipeline
----------------------

As described in :doc:`Step 7 <../quickstart/07_train_the_natural_language_processing_classifiers>`, the :meth:`NaturalLanguageProcessor.build` method is the fastest way to train a baseline natural language processor. Depending on the complexity of your Workbench project and the size of its training data, this can take anywhere from a few seconds to several minutes. With logging level set to ``INFO`` or below, you should see the build progress in the console along with cross-validation accuracies for the classifiers.

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

The :meth:`build` method loads all the training queries, checks them for annotation errors, then proceeds to build all the necessary NLP components using the machine learning settings defined in ``config.py``, the app's configuration file. The method applies Workbench's preset configuration for any component whose settings have not been specified.

In so doing, the :meth:`build` method:

    - Calls the :meth:`fit` method on the classifiers in the domain-intent-entity-role hierarchy to train them using the provided model, feature, and hyperparameter settings

    - Builds the Entity Resolver using the provided entity mapping file

    - Configures the Language Parser using the provided parser configuration file

.. _build_nlp_with_config:

These steps are described further in upcoming chapters, along with default settings for each component, and methods to override them with your own custom configurations.

To identify the optimal configuration for each classifier, you should experiment by training, tuning and testing. Then, store the best machine learning settings in ``config.py``, for the :meth:`build` method to use instead of the Workbench defaults.

Here's an example of a ``config.py`` file where custom settings optimized for the app override the default configurations for the domain and intent classifiers.

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

You will learn more about classifier configuration later in this chapter.

.. _build_partial_nlp:

Training at different levels of the NLP hierarchy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While calling the :meth:`build` method on the :obj:`nlp` object is the easiest way to build or rebuild all the classifiers, it can be time-consuming. Sometimes it is more efficient to only rebuild a subset of your classifiers. To do this, call the :meth:`build` method at the appropriate level in the domain-intent-entity-role hierarchy.

For instance, the code below rebuilds the NLP models for one selected domain only, namely the ``times_and_dates`` domain of the ``home_assistant`` app.

.. code-block:: python

   >>> from mmworkbench import configure_logs; configure_logs()
   >>> from mmworkbench.components.nlp import NaturalLanguageProcessor
   >>> nlp = NaturalLanguageProcessor(app_path='home_assistant')
   >>> nlp.domains['times_and_dates'].build()
   Fitting intent classifier: domain='times_and_dates'
   Loading queries from file times_and_dates/change_alarm/train.txt
   Loading queries from file times_and_dates/check_alarm/train.txt
   Loading queries from file times_and_dates/remove_alarm/train.txt
   Loading queries from file times_and_dates/set_alarm/train.txt
   Loading queries from file times_and_dates/start_timer/train.txt
   Loading queries from file times_and_dates/stop_timer/train.txt
   Selecting hyperparameters using k-fold cross validation with 10 splits
   Best accuracy: 99.33%, params: {'C': 100, 'class_weight': {0: 1.0848387096774192, 1: 1.2278761061946901, 2: 0.8924193548387096, 3: 0.81719056974459714, 4: 1.3213541666666666, 5: 6.665}, 'fit_intercept': False}
   Fitting entity recognizer: domain='times_and_dates', intent='set_alarm'
   Selecting hyperparameters using k-fold cross validation with 5 splits
   Best accuracy: 98.08%, params: {'C': 1000000, 'penalty': 'l2'}
   Fitting entity recognizer: domain='times_and_dates', intent='change_alarm'
   Selecting hyperparameters using k-fold cross validation with 5 splits
   Best accuracy: 97.23%, params: {'C': 100, 'penalty': 'l2'}
   Fitting entity recognizer: domain='times_and_dates', intent='start_timer'
   Selecting hyperparameters using k-fold cross validation with 5 splits
   Best accuracy: 98.95%, params: {'C': 100, 'penalty': 'l1'}
   Fitting entity recognizer: domain='times_and_dates', intent='check_alarm'
   Selecting hyperparameters using k-fold cross validation with 5 splits
   Best accuracy: 97.18%, params: {'C': 1000000, 'penalty': 'l1'}

To specify a level in the domain-intent-entity-role when invoking the :meth:`build` method, choose one of the following patterns:

1. :meth:`nlp.build`

  | Trains all the classifiers in the NLP pipeline.

2. :meth:`nlp.domains['d_name'].build`

  | Trains the intent classifier for the ``d_name`` domain, the entity recognizers for all the intents under ``d_name``, and the role classifiers for all the entity types contained within those intents.

3. :meth:`nlp.domains['d_name'].intents['i_name'].build`

  | Trains the entity recognizer for the ``i_name`` intent, and the role classifiers for all the entity types in this intent.

4. :meth:`nlp.domains['d_name'].intents['i_name'].entities['e_name'].build`

  | Trains the role classifier for ``e_name`` entity type.

More about fine-grained access to individual classifiers appears in the subsequent chapters.

.. _config:

Classifier configurations
^^^^^^^^^^^^^^^^^^^^^^^^^

We have seen how the natural language processor's :meth:`build` method and the individual classifiers' :meth:`fit` methods use configurations to train models.

To be more precise, a classifier configuration defines the `machine learning algorithm <https://en.wikipedia.org/wiki/Supervised_learning#Approaches_and_algorithms>`_ to use, the `features <https://en.wikipedia.org/wiki/Feature_(machine_learning)>`_ to be extracted from the input data, and the methodology to use for `hyperparameter selection <https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)>`_.

Workbench domain, intent, entity, and role classifiers all use a *configuration dictionary* to define the machine learning settings for model training.

This section describes the structure and format of the configuration dictionary. Detailed explanation of configurable options for each type of classifier appears in subsequent chapters.

Anatomy of a classifier configuration
"""""""""""""""""""""""""""""""""""""

A classifier configuration has three sections: **Model Settings**, **Feature Extraction Settings**, and **Hyperparameter Settings**.

1. **Model Settings** - The `machine learning algorithm <https://en.wikipedia.org/wiki/Supervised_learning#Approaches_and_algorithms>`_  or modeling approach to use, along with any algorithm-specific settings.

This snippet from a domain classifier configuration specifies a '`text classifier <https://en.wikipedia.org/wiki/Text_classification>`_' to be trained using a '`logistic regression <https://en.wikipedia.org/wiki/Logistic_regression>`_' model.

.. code:: python

   'model_type': 'text',
   'model_settings': {
      'classifier_type': 'logreg',
   },
   ...

This example, also from entity recognition, specifies '`maximum entropy markov model <https://en.wikipedia.org/wiki/Maximum-entropy_Markov_model>`_' as the machine learning algorithm and the '`Inside-Outside-Beginning <https://en.wikipedia.org/wiki/Inside_Outside_Beginning>`_' format as the tagging scheme. It further specifies the ':sk_api:`maximum absolute scaling <sklearn.preprocessing.MaxAbsScaler>`' feature transformation operation as a preprocessing step.

.. code:: python

   'model_type': 'memm',
   'model_settings': {
      'tag_scheme': 'IOB',
      'feature_scaler': 'max-abs'
   },
   ...

2. **Feature Extraction Settings** - The `features <https://en.wikipedia.org/wiki/Feature_(machine_learning)>`_ to extract from the input query, along with any configurable settings for each feature group.

These feature extraction settings are from a domain classifier configuration.

.. code:: python

   ...
   'features': {
      'bag-of-words': {'lengths': [1]},
      'in-gaz': {},
      'freq': {'bins': 5},
      'length': {}
   }
   ...

The above configuration instructs Workbench to extract four different groups of features for each input query:

  a. ':sk_guide:`Bag of n-grams <feature_extraction#the-bag-of-words-representation>`' of length 1 (also called 'bag of words')
  b. `Gazetteer <https://gate.ac.uk/sale/tao/splitch13.html#x18-32600013.1>`_-derived features
  c. Token frequency-based features, quantized into 5 `bins <https://en.wikipedia.org/wiki/Data_binning>`_
  d. Features derived from the query length

3. **Hyperparameter Settings** - The `hyperparameters <https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)>`_ to use during model training, or the settings for choosing optimal hyperparameters.

This role classifier configuration defines hyperparameters for its `maximum entropy classification model <https://en.wikipedia.org/wiki/Maximum_entropy_classifier>`_. It specifies a value of 100 for the ':sk_guide:`C <linear_model#logistic-regression>`' parameter and ':sk_guide:`L1 <linear_model#logistic-regression>`' as the norm to be used for `regularization <https://en.wikipedia.org/wiki/Regularization_%28mathematics%29#Use_of_regularization_in_classification>`_.

.. code:: python

   ...
   'params': {
      'C': 100,
      'penalty': 'l1'
   }

You can also provide a hyperparameter grid instead of exact values and let Workbench search for optimal settings. This type of configuration must specify both the hyperparameter search grid and settings for the selection methodology, as shown below.

.. code:: python

   ...
   'param_selection': {
      'type': 'k-fold',
      'k': 10,
      'grid': {
        'C': [10, 100, 1000, 10000, 100000],
        'penalty': ['l1', 'l2']
      },
    }

The above configuration defines a grid with five potential values for the 'C' parameter and two possible values for the 'penalty' parameter. It also specifies that optimal values need to be found using a 10-fold cross-validated grid search over the provided parameter grid.

Using custom configurations
"""""""""""""""""""""""""""

There are two ways to override Workbench's preset configurations for NLP classifiers.

The first method, as described :ref:`earlier <build_nlp_with_config>`, is to define the classifier settings in your application configuration file, ``config.py``. The classifier configuration must be defined as a dictionary with one of the following names to override the corresponding classifier's default settings.

  - :data:`DOMAIN_MODEL_CONFIG`
  - :data:`INTENT_MODEL_CONFIG`
  - :data:`ENTITY_MODEL_CONFIG`
  - :data:`ROLE_MODEL_CONFIG`

Alternatively, you could pass configuration settings (like model type, features, and so on) as arguments to the :meth:`fit` method of the appropriate classifier. Arguments passed to :meth:`fit` take precedence over both Workbench defaults and settings defined in ``config.py``. See individual classifier chapters for more about the :meth:`fit` method.


Configuring rest of the pipeline
""""""""""""""""""""""""""""""""

Since neither the entity resolver nor the language parser are supervised classifiers, they are configured differently from the rest of the NLP pipeline. See `Working with the Entity Resolver <entity_resolver>`_ and `Working with the Language Parser <parser>`_, respectively, to learn how to configure these components.

.. _run_nlp:

Run the NLP pipeline
--------------------

Run the trained NLP pipeline on a test query using the :meth:`NaturalLanguageProcessor.process` method. The :meth:`process` method sends the query for sequential processing by each component in the NLP pipeline and returns the aggregated output from all of them.

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

The return value is the dictionary described in the table below.

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

    - Call the :meth:`predict` (or equivalent) method for each classifier in the domain-intent-entity-role hierarchy to detect the domain, intent, entities and roles in the query

    - Call the Entity Resolver's :meth:`predict` method to resolve all detected entities to their canonical forms

    - Call the Language Parser's :meth:`parse_entities` method to cluster the resolved entities

    - Return the detailed output from each component

For more about the above steps, including outputs and methods for batch testing and evaluation, see the chapters on individual NLP components.

.. _evaluate_nlp:

Evaluate NLP performance
------------------------

The cross-validation accuracies for each classifier, reported during model training, can be good initial indicators of your NLP pipeline's performance. However, the true measure of a machine-learned system's real-world performance is its accuracy on previously unseen test data. The test data is a set of labeled queries prepared in :doc:`the same manner <../quickstart/06_generate_representative_training_data>` as the training data. Names of files containing test queries have the prefix ``test``. These files placed within the intent subfolders, alongside the training data files.

.. image:: /images/food_ordering_directory2.png
    :width: 350px
    :align: center

While training data is used for training and tuning the models, test data is used solely for model evaluation. Ideally, the test data should have no queries in common with the training data and be representative of the real-world usage of the app. During evaluation, the ground truth annotations are stripped away from the test queries and the unlabeled queries are passed in to a trained classifier. The classifier's output predictions are then compared against the ground truth labels to measure the model's prediction accuracy. A successful production-grade conversational app must achieve test accuracies greater than 90% for all the classification models in its NLP pipeline.

For more about how evaluation works for each individual classifier, see the `evaluation` sections of the respective chapters.


Optimize the NLP models
-----------------------

The typical experimentation flow for ML-based systems looks like this:

  - Gather representative labeled data

  - Train a baseline model

  - Measure model performance using `cross-validation <https://en.wikipedia.org/wiki/Cross-validation_(statistics)>`_ or `heldout dataset <https://en.wikipedia.org/wiki/Test_set#Validation_set>`_

  - Perform error analysis on incorrect model predictions

  - Apply insights from the analysis to improve model performance by appropriately updating the machine learning setup

In practice, optimizing the NLP models to production-level accuracies demands several iterations of this flow. During each round of experimentation, there are two primary ways to improve the model performance.

  1. **Adding more training data**: In most cases, model accuracy can be improved simply by adding more representative training data. Error analysis can help identify a relevant set of training queries to add. This helps the model generalize better and make more accurate predictions on the misclassified examples. Filling in gaps in the training data and improving the overall quality of labeled queries should always be the first step when debugging classifier performance.

..

  2. **Optimizing the classifier configuration**: Accuracy can also be improved by selecting a classifier configuration that is better suited for your training data. The natural language processor's :meth:`build` method uses a default configuration for each classifier to train the NLP models. While these baseline models provide a reasonable starting point for your NLP pipeline, experimenting with different model types, features, etc., could help identify alternate configurations that produce more accurate models. However, unlike training data augmentation, this more advanced approach requires expertise in applied machine learning to run meaningful experiments and identify optimal classifier settings. For details about configuration options available for each NLP classifier, see the respective chapters.


Save models for future use
--------------------------

Once you have trained an NLP pipeline and are satisfied with its accuracy, you can save it to disk using the :meth:`NaturalLanguageProcessor.dump` method. The :meth:`dump` method saves all the trained models to a cache folder within your Workbench project.

.. code:: python

   >>> nlp.dump()
   Saving intent classifier: domain='ordering'
   Saving entity recognizer: domain='ordering', intent='build_order'
   ...

The saved models can then be loaded anytime using the :meth:`NaturalLanguageProcess.load` method.

.. code:: python

   >>> nlp.load()
   Loading intent classifier: domain='ordering'
   ...

Another option is to save just one specific NLP model, which is useful when you are actively experimenting with individual classifiers and want to checkpoint your work or save multiple model versions for comparison. This is done using the :meth:`dump` and :meth:`load` methods exposed by each classifier. Refer to the chapter for the appropriate classifier to learn more.
