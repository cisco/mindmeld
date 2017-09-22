Working with the Entity Recognizer
==================================

The :ref:`Entity Recognizer <arch_entity_model>`

 - is run as the third step in the :ref:`natural language processing pipeline <instantiate_nlp>`
 - is a `sequence labeling <https://en.wikipedia.org/wiki/Sequence_labeling>`_ or tagging model that detects all the relevant :term:`entities <entity>` in a given query
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

Use the :meth:`EntityRecognizer.fit` method to train an entity recognition model. Depending on the size of the training data and the selected model, this can take anywhere from a few seconds to several minutes. With logging level set to ``INFO`` or below, you should see the build progress in the console along with cross-validation accuracy of the trained model.

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

Use the :attr:`config` attribute of a trained classifier to view the :ref:`configuration <config>` that the classifier is using. Here's an example where we view the configuration of an entity recognizer trained using default settings:

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
     'model_settings': {
       'classifier_type': 'memm',
       'feature_scaler': 'max-abs',
       'tag_scheme': 'IOB'
     },
     'model_type': 'tagger',
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

  Always ``'tagger'``, since the entity recognizer is a tagger model. `Tagging, sequence tagging, or sequence labeling <https://en.wikipedia.org/wiki/Sequence_labeling>`_ are common terms used in NLP literature for models that generate a tag for each token in a sequence. Taggers are most commonly used for part-of-speech tagging or named entity recognition.


``'model_settings'`` (:class:`dict`)
  |

  A dictionary containing model-specific machine learning settings. The key ``'classifier_type'``, whose value specifies the machine learning model to use, is required. Allowed values are shown in the table below.

  .. _er_models:

  =============== ============================================================================================ ==========================================
  Value           Classifier                                                                                   Reference for configurable hyperparameters
  =============== ============================================================================================ ==========================================
  ``'memm'``      `Maximum Entropy Markov Model <https://en.wikipedia.org/wiki/Maximum-entropy_Markov_model>`_ :sk_api:`sklearn.linear_model.LogisticRegression <sklearn.linear_model.LogisticRegression.html>`
  ``'crf'``       `Conditional Random Field <https://en.wikipedia.org/wiki/Conditional_random_field>`_         `sklearn-crfsuite <https://sklearn-crfsuite.readthedocs.io/en/latest/api.html>`_
  =============== ============================================================================================ ==========================================

  Tagger models allow you to specify the additional model settings shown below.

  +-----------------------+-------------------------------------------------------------------------------------------------------------------+
  | Key                   | Value                                                                                                             |
  +=======================+===================================================================================================================+
  | ``'feature_scaler'``  | The :sk_guide:`methodology <preprocessing.html#standardization-or-mean-removal-and-variance-scaling>` for         |
  |                       | scaling raw feature values. Applicable to the MEMM model only.                                                    |
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
  |                       | - ``'IOBES'``: An extension to IOB where ``'E'`` represents the ending token in an entity span,                   |
  |                       |   and ``'S'`` represents a single-token entity.                                                                   |
  +-----------------------+-------------------------------------------------------------------------------------------------------------------+

2. **Feature Extraction Settings**

``'features'`` (:class:`dict`)
  |

  A dictionary whose keys are names of feature groups to extract. The corresponding values are dictionaries representing the feature extraction settings for each group. The table below enumerates the features that can be used for entity recognition.

  .. _entity_features:

  +---------------------------+------------------------------------------------------------------------------------------------------------+
  | Group Name                | Description                                                                                                |
  +===========================+============================================================================================================+
  | ``'bag-of-words-seq'``    | Generates n-grams of specified lengths from the query text                                                 |
  |                           | surrounding the current token.                                                                             |
  |                           |                                                                                                            |
  |                           | Settings:                                                                                                  |
  |                           |                                                                                                            |
  |                           | A dictionary with n-gram lengths as keys                                                                   |
  |                           | and a list of starting positions as values.                                                                |
  |                           | Each starting position is a token index,                                                                   |
  |                           | relative to the current token.                                                                             |
  |                           |                                                                                                            |
  |                           | Examples:                                                                                                  |
  |                           |                                                                                                            |
  |                           | ``'ngram_lengths_to_start_positions': {1: [0], 2: [0]}``                                                   |
  |                           |  - extracts all words (unigrams) and bigrams starting with the current token                               |
  |                           |                                                                                                            |
  |                           | ``'ngram_lengths_to_start_positions': {1: [-1, 0, 1], 2: [-1, 0, 1]}``                                     |
  |                           |  - additionally includes unigrams and bigrams starting from the words before and after the current token   |
  |                           |                                                                                                            |
  |                           | Given the query "weather in {San Francisco|location} {next week|sys_time}"                                 |
  |                           | and a classifier extracting features for the token "Francisco":                                            |
  |                           |                                                                                                            |
  |                           | ``{1: [-1, 0, 1]}``                                                                                        |
  |                           |  - extracts "San", "Francisco", and "next"                                                                 |
  |                           |                                                                                                            |
  |                           | ``{2: [-1, 0, 1]}``                                                                                        |
  |                           |  - extracts "in San", "San Francisco", and "Francisco next"                                                |
  +---------------------------+------------------------------------------------------------------------------------------------------------+
  | ``'in-gaz-span-seq'``     | Generates a set of features indicating the presence of the current token in different entity gazetteers,   |
  |                           | along with popularity information (as defined in the gazetteer).                                           |
  +---------------------------+------------------------------------------------------------------------------------------------------------+
  | ``'sys-candidates-seq'``  | Generates a set of features indicating the presence of system entities in the query text surrounding the   |
  |                           | current token.                                                                                             |
  |                           |                                                                                                            |
  |                           | Settings:                                                                                                  |
  |                           |                                                                                                            |
  |                           | A dictionary with a single key named ``'start_positions'`` and a list of different starting positions      |
  |                           | as its value. As in the ``'bag-of-words-seq'`` feature, each starting position is a token index, relative  |
  |                           | to the the current token.                                                                                  |
  |                           |                                                                                                            |
  |                           | Example:                                                                                                   |
  |                           |                                                                                                            |
  |                           | ``'start_positions': [-1, 0, 1]``                                                                          |
  |                           |  - extracts features indicating whether the current token or its immediate neigbors are system entities    |
  +---------------------------+------------------------------------------------------------------------------------------------------------+

.. _entity_tuning:

3. **Hyperparameter Settings**

``'params'`` (:class:`dict`)
  |

  A dictionary of values to be used for model hyperparameters during training. Examples include the norm used in penalization as ``'penalty'`` for MEMM, the coefficients for L1 and L2 regularization ``'c1'`` and ``'c2'`` for CRF, and so on. The list of allowable hyperparameters depends on the model selected. See the :ref:`reference links <er_models>` above for parameter lists.

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
  |                       | See the :ref:`reference links <er_models>` above for details on the hyperparameters available for each model.     |
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
  |                       | - ``'accuracy'``: Accuracy score at a tag level                                                                   |
  |                       | - ``'seq_accuracy'``: Accuracy score at a full sequence level (not available for MEMM)                            |
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
       'model_type': 'tagger',
       'model_settings': {
           'classifier_type': 'memm',
           'tag_scheme': 'IOBES',
           'feature_scaler': 'max-abs'
       },
       'param_selection': {
           'type': 'k-fold',
           'k': 5,
           'scoring': 'accuracy',
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

.. _entity_evaluation:

Evaluate classifier performance
-------------------------------

Before you can evaluate the accuracy of your trained domain classifier, you must first create labeled test data and place it in your Workbench project as described in the :ref:`Natural Language Processor <evaluate_nlp>` chapter.

Then, when you are ready, use the :meth:`EntityRecognizer.evaluate` method, which

 - strips away all ground truth annotations from the test queries,
 - passes the resulting unlabeled queries to the trained entity recognizer for prediction, and
 - compares the classifier's output predictions against the ground truth labels to compute the model's prediction accuracy.

In the example below, the model gets 35 out of 37 test queries correct, resulting in an accuracy of about 94.6%.

.. code-block:: python

   >>> er.evaluate()
   Loading queries from file weather/check_weather/test.txt
   <EntityModelEvaluation score: 94.59%, 35 of 37 examples correct>

Note that this is *query-level* accuracy. A prediction on a query can only be graded as "correct" when all the entities detected by the entity recognizer exactly match exactly the annotated entities in the test query.

The aggregate accuracy score we see above is only the beginning, because the :meth:`evaluate` method returns a rich object containing overall statistics, statistics by class, a confusion matrix, and sequence statistics.

Print all the model performance statistics reported by the :meth:`evaluate` method:

.. code-block:: python

   >>> eval = er.evaluate()
   >>> eval.print_stats()
   Overall tag-level statistics:

      accuracy f1_weighted          tp          tn          fp          fn    f1_macro    f1_micro
         0.986       0.985         204         825           3           3       0.975       0.986



   Tag-level statistics by class:

                 class      f_beta   precision      recall     support          tp          tn          fp          fn
                    O|       0.990       0.981       1.000         155         155          49           3           0
                B|city       0.985       1.000       0.971          34          33         173           0           1
            B|sys_time       1.000       1.000       1.000           4           4         203           0           0
            I|sys_time       1.000       1.000       1.000           3           3         204           0           0
                I|city       0.900       1.000       0.818          11           9         196           0           2



   Confusion matrix:

                              O|         B|city     B|sys_time     I|sys_time         I|city
               O|            155              0              0              0              0
           B|city              1             33              0              0              0
       B|sys_time              0              0              4              0              0
       I|sys_time              0              0              0              3              0
           I|city              2              0              0              0              9



   Segment-level statistics:

            le          be         lbe          tp          tn          fp          fn
             0           1           0          36          42           0           1



   Sequence-level statistics:

     sequence_accuracy
                 0.946


The :meth:`eval.get_stats()` method returns all the above statistics in a structured dictionary without printing them to the console.

Let's decipher the statistics output by the :meth:`evaluate` method.

**Overall tag-level statistics**
  |

  Aggregate IOB or IOBES tag-level stats measured across the entire test set:

  ===========  ===
  accuracy     :sk_guide:`Classification accuracy score <model_evaluation.html#accuracy-score>`
  f1_weighted  :sk_api:`Class-weighted average f1 score <sklearn.metrics.f1_score.html>`
  tp           Number of `true positives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  tn           Number of `true negatives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  fp           Number of `false positives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  fn           Number of `false negatives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  f1_macro     :sk_api:`Macro-averaged f1 score <sklearn.metrics.f1_score.html>`
  f1_micro     :sk_api:`Micro-averaged f1 score <sklearn.metrics.f1_score.html>`
  ===========  ===

  When interpreting these statistics, consider whether your app and evaluation results fall into one of the cases below, and if so, apply the accompanying guideline. This list is basic, not exhaustive, but should get you started.

  - **Classes are balanced** – When the number of annotated entities for each entity type are comparable and each entity type is equally important, focusing on the accuracy metric is usually good enough. For entity recognition it is very unlikely that your data would fall into this category, since the O tag (used for words that are not part of an entity) usually occurs much more often than the I/B/E/S tags (for words that are part of an entity).

  - **Classes are imbalanced** — In this case, it's important to take the f1 scores into account. For entity recognition it is also important to consider the segment level statistics described below. By primarily optimizing for f1, your model will tend to predict no entity rather than predict one that is uncertain about. See `this blog post <https://nlpers.blogspot.com/2006/08/doing-named-entity-recognition-dont.html>`_.

  - **All f1 and accuracy scores are low** — When entity recognition is performing poorly across all entity types, either of the following may be the problem: 1) You do not have enough training data for the model to learn, or 2) you need to tune your model hyperparameters. Look at segment-level statistics for a more intuitive breakdown of where the model is making errors.

  - **f1 weighted is higher than f1 macro** — This means that entity types with fewer evaluation examples are performing poorly. Try adding more data to these entity types. This entails adding more training queries with labeled entities, specifically entities of the type that are performing the worst as indicated in the tag-level statistics table.

  - **f1 macro is higher than f1 weighted** — This means that entity types with more evaluation examples are performing poorly. Verify that the number of evaluation examples reflects the class distribution of your training examples.

  - **f1 micro is higher than f1 macro** — This means that certain entity types are being misclassified more often than others. Identify the problematic entity types by checking the tag-level class-wise statistics below. Some entity types may be too similar to others, or you may need to add more training data.

  - **Some classes are more important than others** — If some entities are more important than others for your use case, it is best to focus especially on the tag-level class-wise statistics below.

**Tag-level statistics by class**
  |

  Tag-level (IOB or IOBES) statistics that are calculated for each class:

  ===========  ===
  class        Entity tag (in IOB or IOBES format)
  f_beta       :sk_api:`F-beta score <sklearn.metrics.fbeta_score>`
  precision    `Precision <https://en.wikipedia.org/wiki/Precision_and_recall#Precision>`_
  recall       `Recall <https://en.wikipedia.org/wiki/Precision_and_recall#Recall>`_
  support      Number of test entities with this entity tag (based on ground truth)
  tp           Number of `true positives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  tn           Number of `true negatives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  fp           Number of `false positives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  fn           Number of `false negatives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  ===========  ===


**Confusion matrix**
  |

  A `confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`_ where each row represents the number of instances in an actual class and each column represents the number of instances in a predicted class. This reveals whether the classifier tends to confuse two classes, i.e., mislabel one tag as another.


**Segment-level statistics**
  |

  .. note::

     Currently, segment-level statistics cannot be generated for the IOBES tag scheme. They are only available for IOB.

  Although it is useful to analyze tag-level statistics, they don't tell the full story for entity recognition in an intuitive way. It helps to think of the entity recognizer as performing two tasks: 1) identifying the span of words that should be part of an entity, and 2) selecting the label for the identified entity. When the recognizer makes a mistake, it misidentifies either the label, the span boundary, or both.

  Segment-level statistics capture the distribution of these error types across all the segments in a query.

  A segment is either:

    - A continuous span of non-entity tokens, or
    - A continuous span of tokens that represents a single entity

  For example, the query "I’ll have an {eggplant parm|dish} and some {breadsticks|dish} please" has five segments: "I'll have an", "eggplant parm", "and some", "breadsticks", and "please".

  The table below describes the segment-level statistics available in Workbench.

  ============  =========================  ===
  Abbreviation  Statistic                  Description
  ------------  -------------------------  ---
  le            **Label error**            The classifier correctly predicts the existence of an entity and the span of that entity, but chooses the wrong label. For example, the classifier recognizes that 'pad thai' is an entity in the query 'Order some pad thai', but labels it as a restaurant entity instead of a dish entity.
  be            **Boundary error**         The classifier correctly predicts the existence of an entity and its label but misclassifies its span. For example, the classifier predicts that 'some pad thai' is a dish entity instead of just 'pad thai' in the query 'Order some pad thai'.
  lbe           **Label-boundary error**   The classifier correctly predicts the existence of an entity, but gets both the label and the span wrong. For example, the classifier labels 'some pad thai' as an option in the query 'Order some pad thai'. The option label is wrong (dish is correct), and, the boundary is misplaced (because it includes the word 'some' which does not belong in the entity).
  tp            **True positive**          The classifier correctly predicts an entity, its label, and its span.
  tn            **True negative**          The classifier correctly predicts that that a segment contains no entities. For example, the classifier predicts that the query 'Hi there' has no entities.
  fp            **False positive**         The classifier predicts the existence of an entity that is not there. For example, the classifier predicts that 'there' is a dish entity in the query 'Hi there'.
  fn            **False negative**         The classifier fails to predict an entity that *is* present. For example,  the classifier predicts no entity in the query 'Order some pad thai'.
  ============  =========================  ===

  Note that the true positive, true negative, false positive, and false negative values are different when calculated at a segment level rather than a tag level. To illustrate this difference consider the following example:

  ::

             I’ll  have  an      eggplant  parm    please
    Exp:     O.    O     O       B|dish    I|dish  O
    Pred:    O.    O.    B|dish  I|dish.   O.      O

  In the traditional tag-level statistics, predicting ``B|dish`` instead of ``O`` and predicting ``I|dish`` instead of ``B|dish`` would both be `false positives`. There would also be `3 true negatives` for correctly predicting ``O``.

  At the segment level, however, this would be just `2 true negatives` (one for the segment 'I'll have' and one for the segment 'please'), and `1 label-boundary error` (for the segment 'an eggplant parm').

  Considering errors at a segment level is often more intuitive and may even provide better metrics to optimize against, as described `here <https://nlpers.blogspot.com/2006/08/doing-named-entity-recognition-dont.html>`_.


**Sequence-level Statistics**
  |

  In Workbench, we define *sequence-level accuracy* as the fraction of queries for which the entity recognizer successfully identified **all** the expected entities.

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
     EvaluatedExample(example=<Query 'london weather'>, expected=(<QueryEntity 'london' ('city') char: [0-5], tok: [0-0]>,), predicted=(), probas=None, label_type='entities')
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
| sys_time        | “today” , “Tuesday, Feb 18” , “last week” , “Mother’s      |
|                 | day”                                                       |
+-----------------+------------------------------------------------------------+
| sys_interval    | “tomorrow morning” , “from 9:30 - 11:00 on tuesday” ,      |
|                 | “Friday 13th evening”                                      |
+-----------------+------------------------------------------------------------+
| sys_temperature | “64°F” , “71° Fahrenheit” , “twenty seven celsius”         |
+-----------------+------------------------------------------------------------+
| sys_number      | “fifteen” , “0.62” , “500k” , “66”                         |
+-----------------+------------------------------------------------------------+
| sys_ordinal     | “3rd” , “fourth” , “first”                                 |
+-----------------+------------------------------------------------------------+
| sys_distance    | “10 miles” , “2feet” , “0.2 inches” , “3’’ “5km” ,”12cm”   |
+-----------------+------------------------------------------------------------+
| sys_volume      | “500 ml” , “5liters” , “2 gallons”                         |
+-----------------+------------------------------------------------------------+
| sys_currency    | “forty dollars” , “9 bucks” , “$30”                        |
+-----------------+------------------------------------------------------------+
| sys_email       | “help@cisco.com”                                           |
+-----------------+------------------------------------------------------------+
| sys_url         | “washpo.com/info” , “foo.com/path/path?ext=%23&foo=bla” ,  |
|                 | “localhost”                                                |
+-----------------+------------------------------------------------------------+
| sys_phone-number| “+91 736 124 1231” , “+33 4 76095663” , “(626)-756-4757    |
|                 | ext 900”                                                   |
+-----------------+------------------------------------------------------------+

Workbench does not assume that any of the system entities are needed in your app. It is the system entities *that you annotate in your training data* that Workbench knows are needed.

.. note::
   Workbench defines ``sys_time`` and ``sys_interval`` as subtly different entities.

  |
   The ``sys_time`` entity connotes a *value of a single unit of time*, where the unit can be a date, an hour, a week, and so on. For example, “tomorrow” is a ``sys_time`` entity because it corresponds to a single (unit) date, like "2017-07-08."
  |
  |
   The ``sys_interval`` entity connotes a *time interval* that *spans several units* of time. For example, “tomorrow morning” is a ``sys_interval`` entity because “morning” corresponds to the span of hours from 4 am to 12 pm.

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



