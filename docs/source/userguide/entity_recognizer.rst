Working with the Entity Recognizer
==================================

The :ref:`Entity Recognizer <arch_entity_model>`

 - is run as the third step in the :ref:`natural language processing pipeline <instantiate_nlp>`
 - is a `sequence labeling <https://en.wikipedia.org/wiki/Sequence_labeling>`_ model that detects all the relevant :term:`entities <entity>` in a given query
 - is trained per intent, using all the labeled queries for a given intent, with labels derived from the entity types annotated within the training queries

Every Workbench app has one entity recognizer for every intent that requires entity detection.

.. note::

    This is an in-depth tutorial to work through from start to finish. Before you begin, read the :ref:`Step-by-Step Guide <quickstart>`, paying special attention to the :ref:`Entity Recognition <entity_recognition>` section.


System entities and custom entities
-----------------------------------

Entities in Workbench are categorized into two types:

**System Entities**
  Generic entities that are application-agnostic and are automatically detected by Workbench. Examples include numbers, time expressions, email addresses, URLs and measured quantities like distance, volume, currency and temperature. See :ref:`system-entities` below.

**Custom Entities**
  Application-specific entities that can only be detected by an entity recognizer that uses statistical models trained with deep domain knowledge. These are generally `named entities <https://en.wikipedia.org/wiki/Named_entity>`_, like 'San Bernardino,' a proper name that could be a ``location`` entity. Custom entities that are *not* based on proper nouns (and therefore are not named entities) are also possible.

This chapter focuses on training entity recognition models for detecting all the custom entities used by your app.

Access the entity recognizer
----------------------------

Working with any natural language processor component falls into two broad phases:

 - First, generate the training data for your app. App performance largely depends on having sufficient quantity and quality of training data. See :doc:`Step 6 <../quickstart/06_generate_representative_training_data>`.
 - Then, conduct experimentation in the Python shell.

When you are ready to begin experimenting, import the :class:`NaturalLanguageProcessor` class from the Workbench :mod:`nlp` module and instantiate an object with the path to your Workbench project.

.. code-block:: python

   >>> from mmworkbench.components.nlp import NaturalLanguageProcessor
   >>> nlp = NaturalLanguageProcessor(app_path='home_assistant')
   >>> nlp
   <NaturalLanguageProcessor 'home_assistant' ready: False, dirty: False>

Verify that the NLP has correctly identified all the domains and intents for your app.

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
    'check_weather': <IntentProcessor 'check_weather' ready: False, dirty: False>
   }

Access the :class:`EntityRecognizer` an intent of your choice, using the :attr:`entity_recognizer` attribute of the desired intent.

.. code-block:: python

   >>> # Entity recognizer for the 'change_alarm' intent in the 'times_and_dates' domain:
   >>> er = nlp.domains['times_and_dates'].intents['change_alarm'].entity_recognizer
   >>> er
   <EntityRecognizer ready: False, dirty: False>
   ...
   >>> # Entity recognizer for the 'check_weather' intent in the 'weather' domain:
   >>> er = nlp.domains['weather'].intents['check_weather'].entity_recognizer
   >>> er
   <EntityRecognizer ready: False, dirty: False>


.. _train_entity_model:

Train an entity recognizer
--------------------------

Use the :meth:`EntityRecognizer.fit` method to train an entity recognition model. Depending on the size of the training data, this can take anywhere from a few seconds to several minutes. With logging level set to ``INFO`` or below, you should see the build progress in the console along with cross-validation accuracy of the trained model.

.. _baseline_entity_fit:

.. code-block:: python

   >>> from mmworkbench import configure_logs; configure_logs()
   >>> er = nlp.domains['weather'].intents['check_weather'].entity_recognizer
   >>> er.fit()
   Fitting entity recognizer: domain='weather', intent='check_weather'
   No app configuration file found. Using default entity model configuration
   Loading queries from file weather/check_weather/train.txt
   Selecting hyperparameters using k-fold cross validation with 5 splits
   Best accuracy: 99.14%, params: {'C': 10000, 'penalty': 'l2'}

The :meth:`fit` method loads all necessary training queries and trains an entity recognition model. When called with no arguments (as in the example above), the method uses the settings from ``config.py``, the :ref:`app's configuration file <build_nlp_with_config>`. If ``config.py`` is not defined, the method uses the Workbench preset :ref:`classifier configuration <config>`.

Using default settings is the recommended (and quickest) way to get started with any of the NLP classifiers. The resulting baseline classifier should provide a reasonable starting point from which to bootstrap your machine learning experimentation. You can then try alternate settings as you seek to identify the optimal classifier configuration for your app.


Classifier configuration
^^^^^^^^^^^^^^^^^^^^^^^^

Use the :attr:`config` attribute of a trained classifier to view the :ref:`configuration <config>` that the classifier is using. Here's an  example where we view the configuration of a entity recognizer trained using default settings:

.. code-block:: python

   >>> er.config.to_dict()
   {
     'features': {
       'bag-of-words-seq': {
         'ngram_lengths_to_start_positions': {
            1: [-2, -1, 0, 1, 2],
            2: [-2, -1, 0, 1]
         }
       },
       'in-gaz-span-seq': {},
       'sys-candidates-seq': {
         'start_positions': [-1, 0, 1]
       }
     },
     'model_settings': {'feature_scaler': 'max-abs', 'tag_scheme': 'IOB'},
     'model_type': 'memm',
     'param_selection': {
       'grid': {
         'C': [0.01, 1, 100, 10000, 1000000, 100000000],
         'penalty': ['l1', 'l2']
       },
      'k': 5,
      'scoring': 'accuracy',
      'type': 'k-fold'
     },
     'params': None
   }

Let's take a look at the allowed values for each setting in an entity recognizer configuration.

1. **Model Settings**

``'model_type'`` (:class:`str`)
  |

  Always ``'memm'``, since the `maximum entropy markov model (MEMM) <https://en.wikipedia.org/wiki/Maximum-entropy_Markov_model>`_ is currently the only supported model for training entity recognizers in Workbench.

``'model_settings'`` (:class:`dict`)
  |

  A dictionary containing model-specific machine learning settings. The allowed keys are:

  +-----------------------+-------------------------------------------------------------------------------------------------------------------+
  | Key                   | Value                                                                                                             |
  +=======================+===================================================================================================================+
  | ``'feature_scaler'``  | The :sk_guide:`methodology <preprocessing.html#standardization-or-mean-removal-and-variance-scaling>` for         |
  |                       | scaling raw feature values.                                                                                       |
  |                       |                                                                                                                   |
  |                       | Allowed values are:                                                                                               |
  |                       |                                                                                                                   |
  |                       | - ``'none'``: No scaling, i.e., use raw feature values.                                                           |
  |                       |                                                                                                                   |
  |                       | - ``'std-dev'``: Standardize features by removing the mean and scaling to unit variance. See                      |
  |                       |   :sk_api:`StandardScaler <sklearn.preprocessing.StandardScaler>`.                                                |
  |                       |                                                                                                                   |
  |                       | - ``'max-abs'``: Scale each feature by its maximum absolute value. See                                            |
  |                       |   :sk_api:`MaxAbsScaler <sklearn.preprocessing.MaxAbsScaler>`.                                                    |
  +-----------------------+-------------------------------------------------------------------------------------------------------------------+
  | ``'tag_scheme'``      | The tagging scheme for generating per-token labels.                                                               |
  |                       |                                                                                                                   |
  |                       | Allowed values are:                                                                                               |
  |                       |                                                                                                                   |
  |                       | - ``'IOB'``: The `Inside-Outside-Beginning <https://en.wikipedia.org/wiki/Inside_Outside_Beginning>`_ tagging     |
  |                       |   format.                                                                                                         |
  |                       |                                                                                                                   |
  |                       | - ``'IOBES'``: An extension to IOB where 'E' represents the ending token in an entity span,                       |
  |                       |   and 'S' represents a single-token entity.                                                                       |
  +-----------------------+-------------------------------------------------------------------------------------------------------------------+

2. **Feature Extraction Settings**

``'features'`` (:class:`dict`)
  |

  A dictionary whose keys are names of feature groups to extract. The corresponding values are dictionaries representing the feature extraction settings for each group. The table below enumerates the features that can be used for entity recognition.

.. _entity_features:

  +---------------------------+------------------------------------------------------------------------------------------------------------+
  | Group Name                | Description                                                                                                |
  +===========================+============================================================================================================+
  | ``'bag-of-words-seq'``    | Generates n-grams of specified lengths from the query text surrounding the current token.                  |
  |                           |                                                                                                            |
  |                           | Supported settings:                                                                                        |
  |                           | A dictionary with n-gram lengths as keys and a list of different starting positions as values.             |
  |                           | Each starting position is a token index, relative to the the current token.                                |
  |                           |                                                                                                            |
  |                           | E.g.,``'ngram_lengths_to_start_positions': {1: [0], 2: [0]}`` will extract all words (unigrams) and bigrams|
  |                           | starting with the current token. To additionally include unigrams and bigrams starting from the words      |
  |                           | before and after the current token, the settings can be modified to                                        |
  |                           | ``'ngram_lengths_to_start_positions': {1: [-1, 0, 1], 2: [-1, 0, 1]}``.                                    |
  |                           |                                                                                                            |
  |                           | Suppose the query is "weather in {San Francisco|location} {next week|sys_time}" and the classifier is      |
  |                           | extracting features for the token "Francisco". Then,                                                       |
  |                           |                                                                                                            |
  |                           | - ``{1: [-1, 0, 1]}`` would extract "San", "Francisco", and "next"                                         |
  |                           | - ``{2: [-1, 0, 1]}`` would extract "in San", "San Francisco",  and "Francisco next"                       |
  +---------------------------+------------------------------------------------------------------------------------------------------------+
  | ``'in-gaz-span-seq'``     | Generates a set of features indicating the presence of the current token in different entity gazetteers,   |
  |                           | along with popularity information (as defined in the gazetteer).                                           |
  +---------------------------+------------------------------------------------------------------------------------------------------------+
  | ``'sys-candidates-seq'``  | Generates a set of features indicating the presence of system entities in the query text surrounding the   |
  |                           | current token.                                                                                             |
  |                           |                                                                                                            |
  |                           | Supported settings:                                                                                        |
  |                           | A dictionary with a single key named ``'start_positions'`` and a list of different starting positions      |
  |                           | as its value. As in the ``'bag-of-words-seq'`` feature, each starting position is a token index, relative  |
  |                           | to the the current token.                                                                                  |
  |                           |                                                                                                            |
  |                           | E.g.,``'start_positions': [-1, 0, 1]`` will extract features indicating whether the current token or its   |
  |                           | immediate neigbors are system entities.                                                                    |
  +---------------------------+------------------------------------------------------------------------------------------------------------+

.. _entity_tuning:

3. **Hyperparameter Settings**

``'params'`` (:class:`dict`)
  |

  A dictionary of values to use for model hyperparameters during training. These include inverse of regularization strength as ``'C'``, the norm used in penalization as ``'penalty'``, and so on. The hyperparameters for the MEMM model are the same as those for a `maximum entropy model (MaxEnt) <https://en.wikipedia.org/wiki/Multinomial_logistic_regression>`_. The list of allowed hyperparameters is :sk_api:`here <sklearn.linear_model.LogisticRegression.html>`.

``'param_selection'`` (:class:`dict`)
  |

  A dictionary of settings for :sk_guide:`hyperparameter selection <grid_search>`. Provides an alternative to the ``'params'`` dictionary above if the ideal hyperparameters for the model are not already known and need to be estimated.

  To estimate parameters, Workbench needs two pieces of information from the developer:

  #. The parameter space to search, as the value for the ``'grid'`` key
  #. The strategy for splitting the labeled data into training and validation sets, as the value for the ``'type'`` key

  Depending on the splitting scheme selected, the :data:`param_selection` dictionary can contain other keys that define additional settings. The table below enumerates the allowable keys.

  +-----------------------+-------------------------------------------------------------------------------------------------------------------+
  | Key                   | Value                                                                                                             |
  +=======================+===================================================================================================================+
  | ``'grid'``            | A dictionary which maps each hyperparameter to a list of potential values to search.                              |
  |                       | Here is an example for a :sk_api:`logistic regression <sklearn.linear_model.LogisticRegression>` model:           |
  |                       |                                                                                                                   |
  |                       | .. code-block:: python                                                                                            |
  |                       |                                                                                                                   |
  |                       |    {                                                                                                              |
  |                       |      'penalty': ['l1', 'l2'],                                                                                     |
  |                       |      'C': [10, 100, 1000, 10000, 100000],                                                                         |
  |                       |       'fit_intercept': [True, False]                                                                              |
  |                       |    }                                                                                                              |
  |                       |                                                                                                                   |
  |                       | See the full list of allowed hyperparameters :sk_api:`here <sklearn.linear_model.LogisticRegression.html>`.       |
  +-----------------------+-------------------------------------------------------------------------------------------------------------------+
  | ``'type'``            | The :sk_guide:`cross-validation <cross_validation>` methodology to use. One of:                                   |
  |                       |                                                                                                                   |
  |                       | - ``'k-fold'``: :sk_api:`K-folds <sklearn.model_selection.KFold>`                                                 |
  |                       | - ``'shuffle'``: :sk_api:`Randomized folds <sklearn.model_selection.ShuffleSplit>`                                |
  |                       | - ``'group-k-fold'``: :sk_api:`K-folds with non-overlapping groups <sklearn.model_selection.GroupKFold>`          |
  |                       | - ``'group-shuffle'``: :sk_api:`Group-aware randomized folds <sklearn.model_selection.GroupShuffleSplit>`         |
  |                       | - ``'stratified-k-fold'``: :sk_api:`Stratified k-folds <sklearn.model_selection.StratifiedKFold>`                 |
  |                       | - ``'stratified-shuffle'``: :sk_api:`Stratified randomized folds <sklearn.model_selection.StratifiedShuffleSplit>`|
  |                       |                                                                                                                   |
  +-----------------------+-------------------------------------------------------------------------------------------------------------------+
  | ``'k'``               | Number of folds (splits)                                                                                          |
  +-----------------------+-------------------------------------------------------------------------------------------------------------------+
  | ``'scoring'``         | The metric to use for evaluating model performance. One of:                                                       |
  |                       |                                                                                                                   |
  |                       | - ``'accuracy'``: :sk_guide:`Accuracy score <model_evaluation.html#accuracy-score>`                               |
  |                       | - ``'log_loss'``: :sk_api:`Log loss (cross-entropy loss) <model_evaluation.html#log-loss>`                        |
  +-----------------------+-------------------------------------------------------------------------------------------------------------------+

  To identify the parameters that give the highest accuracy, the :meth:`fit` method does an :sk_guide:`exhaustive grid search <grid_search.html#exhaustive-grid-search>` over the parameter space, evaluating candidate models using the specified cross-validation strategy. Subsequent calls to :meth:`fit` can use these optimal parameters and skip the parameter selection process.

.. _build_entity_with_config:

Training with custom configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To override Workbench's default entity recognizer configuration with custom settings, you can either edit the app configuration file, or, you can call the :meth:`fit` method with appropriate arguments.


1. Application configuration file
"""""""""""""""""""""""""""""""""

When you define custom classifier settings in ``config.py``, the :meth:`EntityRecognizer.fit` and :meth:`NaturalLanguageProcessor.build` methods use those settings instead of Workbench's defaults. To do this, define a dictionary of your custom settings, named :data:`ENTITY_MODEL_CONFIG`.

Here's an example of a ``config.py`` file where custom settings optimized for the app override the preset configuration for the entity recognizer.

.. code-block:: python

   ENTITY_MODEL_CONFIG = {
       'model_type': 'memm',
       'model_settings': {
           'tag_scheme': 'IOBES',
           'feature_scaler': 'max-abs'
       },
       'param_selection': {
           'type': 'k-fold',
           'k': 5,
           'scoring': 'log_loss',
           'grid': {
               'penalty': ['l1', 'l2'],
               'C': [0.01, 1, 100, 10000]
           },
       },
       'features': {
           'bag-of-words-seq': {
               'ngram_lengths_to_start_positions': {
                   1: [-2, -1, 0, 1, 2],
                   2: [-1, 0, 1]
               }
           },
           'in-gaz-span-seq': {},
           'sys-candidates-seq': {
             'start_positions': [-1, 0, 1]
           }
       }
   }

This method is recommended for storing your optimal classifier settings once you have identified them through experimentation. Then the classifier training methods will use the optimized configuration to rebuild the models. A common use case is retraining models on newly-acquired training data, without retuning the underlying model settings.

Since this method requires updating a file each time you modify a setting, it's less suitable for rapid prototyping than the method described next.


2. Arguments to the :meth:`fit` method
""""""""""""""""""""""""""""""""""""""

For experimenting with an entity recognizer, the recommended method is to use arguments to the :meth:`fit` method. The main areas for exploration are feature extraction, hyperparameter tuning, and model selection.

**Feature extraction**

Let's start with the baseline classifier that was trained :ref:`above <baseline_entity_fit>`. Here's how you get the default feature set used by the classifer.

.. code-block:: python

   >>> my_features = er.config.features
   >>> my_features
   {
     'bag-of-words-seq': {
       'ngram_lengths_to_start_positions': {
         1: [-2, -1, 0, 1, 2],
         2: [-2, -1, 0, 1]
       }
     },
     'in-gaz-span-seq': {},
     'sys-candidates-seq': {
       'start_positions': [-1, 0, 1]
     }
   }

Notice that the ``'ngram_lengths_to_start_positions'`` settings tell the classifier to extract n-grams within a context window of two tokens or less around the token of interest — that is, just words in the immediate vicinity.

Let's have the classifier look at a larger context window — extract n-grams starting from tokens that are further away. We'll see whether that provides better information than the smaller default window. To do so, change the ``'ngram_lengths_to_start_positions'`` settings to extract all the unigrams and bigrams in a window of three tokens around the current token, as shown below.

.. code-block:: python

   >>> my_features['bag-of-words-seq']['ngram_lengths_to_start_positions'] = {
   ...     1: [-3, -2, -1, 0, 1, 2, 3],
   ...     2: [-3, -2, -1, 0, 1, 2]
   ... }
   >>> my_features
   {
     'bag-of-words-seq': {
       'ngram_lengths_to_start_positions': {
         1: [-3, -2, -1, 0, 1, 2, 3],
         2: [-3, -2, -1, 0, 1, 2]
       }
     },
     'in-gaz-span-seq': {},
     'sys-candidates-seq': {
       'start_positions': [-1, 0, 1]
     }
   }

Suppose w\ :sub:`i` represents the word at the *ith* index in the query, where the index is calculated relative to the current token. Then, the above feature configuration should extract the following n-grams (w\ :sub:`0` being the current token).

  - Unigrams: { w\ :sub:`-3`, w\ :sub:`-2`, w\ :sub:`-1`, w\ :sub:`0`, w\ :sub:`1`, w\ :sub:`2`, w\ :sub:`3` }

  - Bigrams: { w\ :sub:`-3`\ w\ :sub:`-2`, w\ :sub:`-2`\ w\ :sub:`-1`, w\ :sub:`-1`\ w\ :sub:`0`,  w\ :sub:`0`\ w\ :sub:`1`, w\ :sub:`1`\ w\ :sub:`2`, w\ :sub:`2`\ w\ :sub:`3` }

To retrain the classifier with the updated feature set, pass in the :data:`my_features` dictionary as an argument to the :data:`features` parameter of the :meth:`fit` method. This trains the entity recognition model using our new feature extraction settings, while continuing to use Workbench defaults for model type (MEMM) and hyperparameter selection.

.. code-block:: python

   >>> er.fit(features=my_features)
   Fitting entity recognizer: domain='weather', intent='check_weather'
   No app configuration file found. Using default entity model configuration
   Selecting hyperparameters using k-fold cross-validation with 5 splits
   Best accuracy: 99.04%, params: {'C': 10000, 'penalty': 'l2'}

**Hyperparameter tuning**

View the model's :ref:`hyperparameters <entity_tuning>`, keeping in mind the hyperparameters for the MEMM model in Workbench. These include: ``'C'``, the inverse of regularization strength; and, ``'fit_intercept'``, which determines whether to add an intercept term to the decision function. The ``'fit_intercept'`` parameter is not shown in the response but defaults to ``'True'``.

.. code-block:: python

   >>> my_param_settings = er.config.param_selection
   >>> my_param_settings
   {
     'grid': {
       'C': [0.01, 1, 100, 10000, 1000000, 100000000],
       'penalty': ['l1', 'l2']
     },
    'k': 5,
    'scoring': 'accuracy',
    'type': 'k-fold'
   }

Let's reduce the range of values to search for ``'C'``, and allow the hyperparameter estimation process to choose whether to add an intercept term to the decision function.

Pass the updated settings to :meth:`fit` as an argument to the :data:`param_selection` parameter. The :meth:`fit` method then searches over the updated parameter grid, and prints the hyperparameter values for the model whose cross-validation accuracy is highest.

.. code-block:: python

   >>> my_param_settings['grid']['C'] = [0.01, 1, 100, 10000]
   >>> my_param_settings['grid']['fit_intercept'] = ['True', 'False']
   >>> my_param_settings
   {
     'grid': {
       'C': [0.01, 1, 100, 10000],
       'fit_intercept': ['True', 'False'],
       'penalty': ['l1', 'l2']
     },
    'k': 5,
    'scoring': 'accuracy',
    'type': 'k-fold'
   }
   >>> er.fit(param_selection=my_param_settings)
   Fitting entity recognizer: domain='weather', intent='check_weather'
   No app configuration file found. Using default entity model configuration
   Selecting hyperparameters using k-fold cross-validation with 5 splits
   Best accuracy: 99.09%, params: {'C': 100, 'fit_intercept': 'False', 'penalty': 'l1'}

Finally, we'll try a new cross-validation strategy of randomized folds, replacing the default of k-fold. We'll keep the default of five folds. To do this, we modify the values of the   ``'type'`` key in :data:`my_param_settings`:

.. code-block:: python

   >>> my_param_settings['type'] = 'shuffle'
   >>> my_param_settings
   {
     'grid': {
       'C': [0.01, 1, 100, 10000],
       'fit_intercept': ['True', 'False'],
       'penalty': ['l1', 'l2']
     },
    'k': 5,
    'scoring': 'accuracy',
    'type': 'shuffle'
   }
   >>> er.fit(param_selection=my_param_settings)
   Fitting entity recognizer: domain='weather', intent='check_weather'
   No app configuration file found. Using default entity model configuration
   Selecting hyperparameters using shuffle cross-validation with 5 splits
   Best accuracy: 99.39%, params: {'C': 100, 'fit_intercept': 'False', 'penalty': 'l1'}

For a list of configurable hyperparameters for each model, along with available cross-validation methods, see :ref:`hyperparameter settings <entity_tuning>`.

**Model settings**

To vary the model training settings, start by inspecting the current settings:

.. code-block:: python

   >>> my_model_settings = er.config.model_settings
   >>> my_model_settings
   {'feature_scaler': 'max-abs', 'tag_scheme': 'IOB'}

For an example experiment, we'll turn off feature scaling and change the tagging scheme to IOBES, while leaving defaults in place for feature extraction and hyperparameter selection.

Retrain the entity recognition model with our updated settings:

.. code-block:: python

   >>> my_model_settings['feature_scaler'] = None
   >>> my_model_settings['tag_scheme'] = 'IOBES'
   >>> {'feature_scaler': None, 'tag_scheme': 'IOBES'}
   >>> er.fit(model_settings=my_model_settings)
   Fitting entity recognizer: domain='weather', intent='check_weather'
   No app configuration file found. Using default entity model configuration
   Selecting hyperparameters using k-fold cross-validation with 5 splits
   Best accuracy: 98.78%, params: {'C': 10000, 'penalty': 'l2'}

Run the entity recognizer
-------------------------

Entity recognition takes place in two steps:

  #. The trained sequence labeling model predicts the output tag (in IOB or IOBES format) with the highest probability for each token in the input query.

  #. The predicted tags are then processed to extract the span and type of each entity in the query.

Run the trained entity recognizer on a test query using the :meth:`EntityRecognizer.predict` method, which returns a list of detected entities in the query.

.. code-block:: python

   >>> er.predict('Weather in San Francisco next week')
   (<QueryEntity 'San Francisco' ('city') char: [11-23], tok: [2-3]>,
    <QueryEntity 'next week' ('sys_time') char: [25-33], tok: [4-5]>)

.. note::

   At runtime, the natural language processor's :meth:`process` method calls :meth:`predict` to recognize all the entities in an incoming query.

The :meth:`predict` takes one query at a time. Next, we'll see how to test a trained model on a batch of labeled test queries.

Evaluate classifier performance
-------------------------------

Before you can evaluate the accuracy of your trained domain classifier, you must first create labeled test data and place it in your Workbench project as described in the :ref:`Natural Language Processor <evaluate_nlp>` chapter.

Then, when you are ready, use the :meth:`EntityRecognizer.evaluate` method, which

 - strips away all ground truth annotations from the test queries,
 - passes the resulting unlabeled queries to the trained entity recognizer for prediction, and
 - compares the classifier's output predictions against the ground truth labels to compute the model's prediction accuracy.

In the example below, the model gets 33 out of 37 test queries correct, resulting in an accuracy of about 89%.

.. code-block:: python

   >>> er.evaluate()
   Loading queries from file weather/check_weather/test.txt
   <EntityModelEvaluation score: 89.19%, 33 of 37 examples correct>

Note that this is *query-level* accuracy. A prediction on a query can only be graded as "correct" when all the entities detected by the entity recognizer exactly match exactly the annotated entities in the test query.

The aggregate accuracy score we see above is only the beginning, because the :meth:`evaluate` method returns a rich object containing overall statistics, statistics by class, a confusion matrix, and sequence statistics.

Print all the model performance statistics reported by the :meth:`evaluate` method:

.. code-block:: python

   >>> eval = er.evaluate()
   >>> eval.print_stats()
   Overall Statistics:

       accuracy f1_weighted          TP          TN          FP          FN    f1_macro    f1_micro
          0.971       0.970         201        1443           6           6       0.959       0.971



   Statistics by Class:

                  class      f_beta   precision      recall     support          TP          TN          FP          FN
                  O||O|       0.984       0.969       1.000         155         155          47           5           0
              S|city|O|       0.939       0.958       0.920          25          23         181           1           2
              B|city|O|       0.875       1.000       0.778           9           7         198           0           2
              I|city|O|       1.000       1.000       1.000           2           2         205           0           0
              E|city|O|       0.875       1.000       0.778           9           7         198           0           2
          O||B|sys_time       1.000       1.000       1.000           3           3         204           0           0
          O||E|sys_time       1.000       1.000       1.000           3           3         204           0           0
          O||S|sys_time       1.000       1.000       1.000           1           1         206           0           0



   Confusion Matrix:

                            O||O|      S|city|O|      B|city|O|      I|city|O|      E|city|O|   O||B|sys_t..   O||E|sys_t..   O||S|sys_t..
             O||O|            155              0              0              0              0              0              0              0
         S|city|O|              2             23              0              0              0              0              0              0
         B|city|O|              1              1              7              0              0              0              0              0
         I|city|O|              0              0              0              2              0              0              0              0
         E|city|O|              2              0              0              0              7              0              0              0
      O||B|sys_t..              0              0              0              0              0              3              0              0
      O||E|sys_t..              0              0              0              0              0              0              3              0
      O||S|sys_t..              0              0              0              0              0              0              0              1



   Sequence Statistics:

    sequence_accuracy
                0.892

Let's decipher the statistics output by the :meth:`evaluate` method.

**Overall Statistics**
  |

  Aggregate token-level stats measured across the entire test set:

  ===========  ===
  accuracy     :sk_guide:`Classification accuracy score <model_evaluation.html#accuracy-score>`
  f1_weighted  :sk_api:`Class-weighted average f1 score <sklearn.metrics.f1_score.html>`
  TP           Number of `true positives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  TN           Number of `true negatives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  FP           Number of `false positives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  FN           Number of `false negatives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  f1_macro     :sk_api:`Macro-averaged f1 score <sklearn.metrics.f1_score.html>`
  f1_micro     :sk_api:`Micro-averaged f1 score <sklearn.metrics.f1_score.html>`
  ===========  ===

  Here are some basic guidelines on how to interpret these statistics. Note that this is not meant to be an exhaustive list, but includes some possibilities to consider if your app and evaluation results fall into one of these cases:

  - **Classes are balanced**: When the number of annotated entities for each entity type are comparable and each entity type is equally important, focusing on the accuracy metric is usually good enough.

  - **Classes are imbalanced**: When classes are imbalanced it is important to take the F1 scores into account.

  - **All F1 and accuracy scores are low**: Entity recognition is performing poorly across all entity types. You may not have enough training data for the model to learn or you may need to tune your model hyperparameters.

  - **F1 weighted is higher than F1 macro**: Your entity types with fewer evaluation examples are performing poorly. You may need to add more data to entity types that have fewer examples.

  - **F1 macro is higher than F1 weighted**: Your entity types with more evaluation examples are performing poorly. Verify that the number of evaluation examples reflects the class distribution of your training examples.

  - **F1 micro is higher than F1 macro**: Certain entity types are being misclassified more often than others. Check the class-wise statistics below to identify these entity types. Some entity types may be too similar to another entity type or you may need to add more training data.

  - **Some classes are more important than others**: If some entities are more important than others for your use case, it is good to focus more on the class-wise statistics described below.

**Class-wise Statistics**
  |

  Stats computed at a per-class level:

  ===========  ===
  class        Entity tag (in IOB or IOBES format)
  f_beta       :sk_api:`F-beta score <sklearn.metrics.fbeta_score>`
  precision    `Precision <https://en.wikipedia.org/wiki/Precision_and_recall#Precision>`_
  recall       `Recall <https://en.wikipedia.org/wiki/Precision_and_recall#Recall>`_
  support      Number of test entities with this entity type (based on ground truth)
  TP           Number of `true positives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  TN           Number of `true negatives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  FP           Number of `false positives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  FN           Number of `false negatives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  ===========  ===

**Confusion Matrix**
  |

  A `confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`_ where each row represents the number of instances in an actual class and each column represents the number of instances in a predicted class. This reveals whether the classifier tends to confuse two classes, i.e., mislabel one class as another. In the above example, the entity recognizer wrongly classified two instances of ``S|city|O|`` tokens as ``O||O|``.


**Sequence Statistics**
  |

  Sequence-level accuracy that tracks the fraction of queries for which the entity recognizer successfully identified **all** the expected entities.

Now we have a wealth of information about the performance of our classifier. Let's go further and inspect the classifier's predictions at the level of individual queries, to better understand error patterns.

View the classifier predictions for the entire test set using the :attr:`results` attribute of the returned :obj:`eval` object. Each result is an instance of the :class:`EvaluatedExample` class which contains information about the original input query, the expected ground truth label, the predicted label, and the predicted probability distribution over all the class labels.

.. code-block:: python

   >>> eval.results
   [
     EvaluatedExample(example=<Query 'check temperature outside'>, expected=(), predicted=(), probas=None, label_type='entities'),
     EvaluatedExample(example=<Query 'check temperature in miami'>, expected=(<QueryEntity 'miami' ('city') char: [21-25], tok: [3-3]>,), predicted=(<QueryEntity 'miami' ('city') char: [21-25], tok: [3-3]>,), probas=None, label_type='entities'),
     ...
   ]

Next, we look selectively at just the correct or incorrect predictions.

.. code-block:: python

   >>> list(eval.correct_results())
   [
     EvaluatedExample(example=<Query 'check temperature outside'>, expected=(), predicted=(), probas=None, label_type='entities'),
     EvaluatedExample(example=<Query 'check temperature in miami'>, expected=(<QueryEntity 'miami' ('city') char: [21-25], tok: [3-3]>,), predicted=(<QueryEntity 'miami' ('city') char: [21-25], tok: [3-3]>,), probas=None, label_type='entities'),
     ...
   ]
   >>> list(eval.incorrect_results())
   [
     EvaluatedExample(example=<Query 'taipei current temperature'>, expected=(<QueryEntity 'taipei' ('city') char: [0-5], tok: [0-0]>,), predicted=(), probas=None, label_type='entities'),
     EvaluatedExample(example=<Query 'london weather'>, expected=(<QueryEntity 'london' ('city') char: [0-5], tok: [0-0]>,), predicted=(), probas=None, label_type='entities'),
     ...
   ]

Slicing and dicing these results for error analysis is easily done with `list comprehensions <https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions>`_.

A simple example of this is to inspect incorrect predictions where the query's first entity is supposed to be of a particular type. For the ``city`` type, we get:

.. code-block:: python

   >>> [(r.example, r.expected, r.predicted) for r in eval.incorrect_results()
   ...  if r.expected and r.expected[0].entity.type == 'city']
   [
     (
       <Query 'taipei current temperature'>,
       (<QueryEntity 'taipei' ('city') char: [0-5], tok: [0-0]>,),
       ()
     ),
     (
       <Query 'london weather'>,
       (<QueryEntity 'london' ('city') char: [0-5], tok: [0-0]>,),
       ()
     ),
     (
       <Query 'temperature in san fran'>,
       (<QueryEntity 'san fran' ('city') char: [15-22], tok: [2-3]>,),
       (<QueryEntity 'san' ('city') char: [15-17], tok: [2-2]>,)
     ),
     (
       <Query "how's the weather in the big apple">,
       (<QueryEntity 'big apple' ('city') char: [25-33], tok: [5-6]>,),
       ()
     )
   ]

The entity recognizer was unable to correctly detect the full ``city`` entity in *any* of the above queries. This is usually a sign that the training data lacks coverage for queries with language patterns or entities like those in the examples above. It could also mean that the gazetteer for this entity type is not comprehensive enough.

Start by looking for similar queries in the :doc:`training data <../blueprints/home_assistant>`. You should discover that the ``check_weather`` intent does indeed lack labeled training queries like the first two queries above.

To solve this problem, you could try adding more queries annotated with the ``city`` entity to the ``check_weather`` intent's training data. Then, the recognition model should be able to generalize better.


The last two misclassified queries feature nicknames (``'san fran'`` and ``'the big apple'``) rather than formal city names. Noticing this, the logical step is to inspect the :doc:`gazetteer data <../blueprints/home_assistant>`. You should discover that this gazetteer does indeed lack slang terms and nicknames for cities.

To mitigate this, try expanding the ``city`` gazetteer to contain entries like "San Fran", "Big Apple" and other popular synonyms for location names that are relevant to the ``weather`` domain.

Error analysis on the results of the :meth:`evaluate` method can inform your experimentation and help in building better models. Augmenting training data and adding gazetteer entries should be the first steps, as in the above example. Beyond that, you can experiment with different model types, features, and hyperparameters, as described :ref:`earlier <build_entity_with_config>` in this chapter.

Save model for future use
-------------------------

Save the trained entity recognizer for later use by calling the :meth:`EntityRecognizer.dump` method. The :meth:`dump` method serializes the trained model as a `pickle file <https://docs.python.org/3/library/pickle.html>`_ and saves it to the specified location on disk.

.. code:: python

   >>> er.dump(model_path='experiments/entity_recognizer.memm.20170701.pkl')
   Saving entity recognizer: domain='weather', intent='check_weather'

You can load the saved model anytime using the :meth:`EntityRecognizer.load` method.

.. code:: python

   >>> er.load(model_path='experiments/entity_recognizer.memm.20170701.pkl')
   Loading entity recognizer: domain='weather', intent='check_weather'

.. _system-entities:

More about system entities
--------------------------

System entities are generic application-agnostic entities that all Workbench applications detect automatically. There is no need to train models to learn system entities; they just work.

Supported system entities are enumerated in the table below.

+-----------------+------------------------------------------------------------+
| System Entity   | Examples                                                   |
+=================+============================================================+
| sys_time        | "today" , "Tuesday, Feb 18" , "last week" , "Mother’s      |
|                 | day"                                                       |
+-----------------+------------------------------------------------------------+
| sys_interval    | "tomorrow morning" , "from 9:30 - 11:00 on tuesday" ,      |
|                 | "Friday 13th evening"                                      |
+-----------------+------------------------------------------------------------+
| sys_temperature | "64°F" , "71° Fahrenheit" , "twenty seven celsius"         |
+-----------------+------------------------------------------------------------+
| sys_number      | "fifteen" , "0.62" , "500k" , "66"                         |
+-----------------+------------------------------------------------------------+
| sys_ordinal     | "3rd" , "fourth" , "first"                                 |
+-----------------+------------------------------------------------------------+
| sys_distance    | "10 miles" , "2feet" , "0.2 inches" , "3’’ "5km" ,"12cm"   |
+-----------------+------------------------------------------------------------+
| sys_volume      | "500 ml" , "5liters" , "2 gallons"                         |
+-----------------+------------------------------------------------------------+
| sys_currency    | "forty dollars" , "9 bucks" , "$30"                        |
+-----------------+------------------------------------------------------------+
| sys_email       | "help@cisco.com"                                           |
+-----------------+------------------------------------------------------------+
| sys_url         | "washpo.com/info" , "foo.com/path/path?ext=%23&foo=bla" ,  |
|                 | "localhost"                                                |
+-----------------+------------------------------------------------------------+
| sys_phone-number| "+91 736 124 1231" , "+33 4 76095663" , "(626)-756-4757    |
|                 | ext 900"                                                   |
+-----------------+------------------------------------------------------------+

Workbench does not assume that any of the system entities are needed in your app. It is the system entities *that you annotate in your training data* that Workbench knows are needed.

.. note::
   Workbench defines ``sys_time`` and ``sys_interval`` as subtly different entities.

  |
   The ``sys_time`` entity connotes a *value of a single unit of time*, where the unit can be a date, an hour, a week, and so on. For example, "tomorrow" is a ``sys_time`` entity because it corresponds to a single (unit) date, like "2017-07-08."
  |
  |
   The ``sys_interval`` entity connotes a *time interval* that *spans several units* of time. For example, "tomorrow morning" is a ``sys_interval`` entity because "morning" corresponds to the span of hours from 4 am to 12 pm.

Custom entities, system entities, and training set size
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Any application's training set must focus on capturing all the entity variations and language patterns for the *custom entities* that the app uses. By contrast, the part of the training set concerned with *system entities* can be relatively minimal, because Workbench does not need to train an entity recognition model to recognize system entities.

Annotating system entities
^^^^^^^^^^^^^^^^^^^^^^^^^^

Assuming that you have defined the :ref:`domain-intent-entity-role hierarchy <model_hierarchy>` for your app, you know

 - which system entities your app needs to use
 - what roles (if any) apply to those system entities

Use this knowledge to guide you in annotating any system entities in your training data.

These examples of annotated system entities come from the Home Assistant blueprint application:

.. code-block:: text

    - adjust the temperature to {65|sys_temperature}
    - {in the morning|sys_interval} set the temperature to {72|sys_temperature}
    - change my {6:45|sys_time|old_time} alarm to {7 am|sys_time|new_time}
    - move my {6 am|sys_time|old_time} alarm to {3pm in the afternoon|sys_time|new_time}
    - what's the forecast for {tomorrow afternoon|sys_interval}

For more examples, see the training data for any of the blueprint apps.

Inspecting how Workbench detects system entities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To see which token spans in a query are detected as system entities, and what system entities Workbench thinks they are, use the :func:`parse_numerics` function:

.. code-block:: python

    >>> from mmWorkbench.ser import parse_numerics
    >>> parse_numerics("tomorrow morning at 9am")
    {'data': [{'dimension': 'number',
       'entity': {'end': 21, 'start': 20, 'text': '9'},
       'likelihood': -0.11895194286136536,
       'operator': 'equals',
       'rule_count': 1,
       'value': [9]},
        .
        .
      {'dimension': 'time',
       'entity': {'end': 23, 'start': 0, 'text': 'tomorrow morning at 9am'},
       'grain': 'hour',
       'likelihood': -23.558523074887038,
       'operator': 'equals',
       'rule_count': 8,
       'value': ['2017-07-08T09:00:00.000-07:00']}],
     'status': '200'}

The :func:`parse_numerics` function returns a dictionary where the key is ``'data'`` and the value is a list of dictionaries. Each dictionary in this list represents a token span that Workbench has detected as a system entity.

Significant keys and values within these inner dictionaries are shown in the table below.

+-----------+--------------------------------------------+-------------------------------------------------+
| Key       | Value                                      | Meaning or content                              |
+===========+============================================+=================================================+
| entity    | A dictionary whose keys are                | Where the entity starts and ends, and its text  |
|           | ``start`` , ``end`` , and ``text``         |                                                 |
+-----------+--------------------------------------------+-------------------------------------------------+
| dimension | ``time`` , ``number`` , or another label   | What type of numeric entity this is             |
+-----------+--------------------------------------------+-------------------------------------------------+
| grain     | ``hour``, ``minute``, or another label     | The entity's unit of time;                      |
|           |                                            | only present when dimension is ``time``         |
+-----------+--------------------------------------------+-------------------------------------------------+
| value     | A list of values (numeric or text);        | The real-world value                            |
|           | usually a single value                     | that the entity represents                      |
+-----------+--------------------------------------------+-------------------------------------------------+

This output is especially useful when debugging system entity behavior.

When Workbench is unable to resolve a system entity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two common mistakes when working with system entities are: annotating an entity as the wrong type, and, labeling an unsupported token as an entity. In these cases, Workbench will be unable to resolve the system entity.

**Annotating a system entity as the wrong type**

Because ``sys_interval`` and ``sys_time`` are so close in meaning, developers or annotation scripts sometimes use one in place of the other.

In the example below, both entities should be annotated as ``sys_time``, but one was mislabeled as ``sys_interval``:

.. code-block:: text

    change my {6:45|sys_interval|old_time} alarm to {7 am|sys_time|new_time}

Workbench prints the following error during training:

.. code-block:: text

    Unable to load query: Unable to resolve system entity of type 'sys_interval' for '6:45'. Entities found for the following types ['sys_time']

The solution is to change the first entity to ``{6:45|sys_time|old_time}``.

**Unsupported tokens in system entities**

Not all reasonable-sounding tokens are actually supported by a Workbench system entity.

In the example below, the token "daily" is annotated as a ``sys_time`` entity:

.. code-block:: text

    set my alarm {daily|sys_time}

Workbench prints the following error during training:

.. code-block:: text

    Unable to load query: Unable to resolve system entity of type 'sys_time' for 'daily'.

Possible solutions:

#. Add a custom entity that supports the token in question. For example, a ``recurrence`` custom entity could support tokens like "daily", "weekly", and so on. The correctly-annotated query would be "set my alarm {daily|recurrence}".

#. Remove the entity label from tokens like "daily" and see if the app satisfactorily handles the queries anyway.

#. Remove all queries that contain unsupported tokens like "daily" entirely from the training data.



