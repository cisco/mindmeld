.. meta::
    :scope: private

Entity Recognizer
=================

The :ref:`Entity Recognizer <arch_entity_model>` is run as the third step in the natural language processing pipeline to detect all the relevant :term:`entities <entity>` in a given query. It is a `sequence labeling <https://en.wikipedia.org/wiki/Sequence_labeling>`_ model that is trained using all the labeled queries for a given intent. Labels are derived from the entity types annotated within the training queries. See :doc:`Step 6 <../quickstart/06_generate_representative_training_data>` for more details on training data preparation. Entity recognition models are trained per intent. A Workbench app hence has one entity recognizer for every intent that requires entity detection.

.. note::

   **Recommended prior reading:**

   - :ref:`Step 7: Train the Natural Language Processing Classifiers <entity_recognition>` (Step-By-Step Guide)
   - :doc:`Natural Language Processor <nlp>` (User Guide)


System entities and custom entities
-----------------------------------

Entities in Workbench are categorized into two types:

**System Entities**
  Generic entities that are application-agnostic and are automatically detected by Workbench. Examples include numbers, time expressions, email addresses, URLs and measured quantities like distance, volume, currency and temperature. Read more in the :doc:`System Entities <system_entities>` chapter. 

**Custom Entities**
  Application-specific entities that need to be detected using a trained entity recognizer. These are generally `named entities <https://en.wikipedia.org/wiki/Named_entity>`_ which can only be recognized using statistical models that have been trained with deep domain knowledge.

This chapter focuses on training entity recognition models for detecting all the custom entities used by your app.


Access the entity recognizer
----------------------------

Before using any of the NLP components, you need to generate the necessary training data for your app by following the guidelines in :doc:`Step 6 <../quickstart/06_generate_representative_training_data>`. You can then start by :ref:`instantiating an object <instantiate_nlp>` of the :class:`NaturalLanguageProcessor` (NLP) class.

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
    'check_weather': <IntentProcessor 'check_weather' ready: False, dirty: False>
   }

Each intent has its own :class:`EntityRecognizer` which can be accessed using the :attr:`entity_recognizer` attribute of the corresponding intent.

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

To train an entity recognition model for a specific intent, use the :meth:`EntityRecognizer.fit` method. Depending on the size of the training data, this can take anywhere from a few seconds to several minutes to finish. If the logging level is set to ``INFO`` or below, you should see the build progress in the console and the cross-validation accuracy of the trained model.

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

The :meth:`fit` method loads all the necessary training queries and trains an entity recognition model using the provided machine learning settings. When the method is called without any parameters (as in the example above), it uses the settings from the :ref:`app's configuration file <build_nlp_with_config>` (``config.py``), if defined, or Workbench's preset :ref:`classifier configuration <config>`.

The quickest and recommended way to get started with any of the NLP classifiers is by using Workbench's default settings. The resulting baseline classifier should provide a reasonable starting point to bootstrap your machine learning experimentation from. You can then experiment with alternate settings to identify the optimal classifier configuration for your app.


Classifier configuration
^^^^^^^^^^^^^^^^^^^^^^^^

To view the current :ref:`configuration <config>` being used by a trained classifier, use its :attr:`config` attribute. For example, here is the configuration being used by a baseline entity recognizer trained using Workbench's default settings.

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

  Is always ``'memm'``, since Workbench currently only supports training a `maximum entropy markov model (MEMM) <https://en.wikipedia.org/wiki/Maximum-entropy_Markov_model>`_ for entity recognition.

``'model_settings'`` (:class:`dict`)
  |

  Is a dictionary containing model-specific machine learning settings. The allowed keys are:

  +-----------------------+-------------------------------------------------------------------------------------------------------------------+
  | Key                   | Value                                                                                                             |
  +=======================+===================================================================================================================+
  | ``'feature_scaler'``  | The :sk_guide:`methodology <preprocessing.html#standardization-or-mean-removal-and-variance-scaling>` to use      |
  |                       | for scaling raw feature values.                                                                                   |
  |                       |                                                                                                                   |
  |                       | Allowed values are:                                                                                               |
  |                       |                                                                                                                   |
  |                       | - ``'none'``: No scaling, i.e. use raw feature values.                                                            |
  |                       |                                                                                                                   |
  |                       | - ``'std-dev'``: Standardize features by removing the mean and scaling to unit variance. See                      |
  |                       |   :sk_api:`StandardScaler <sklearn.preprocessing.StandardScaler>`.                                                |
  |                       |                                                                                                                   |
  |                       | - ``'max-abs'``: Scale each feature by its maximum absolute value. See                                            |
  |                       |   :sk_api:`MaxAbsScaler <sklearn.preprocessing.MaxAbsScaler>`.                                                    |
  +-----------------------+-------------------------------------------------------------------------------------------------------------------+
  | ``'tag_scheme'``      | The tagging scheme to use for generating per-token labels.                                                        |
  |                       |                                                                                                                   |
  |                       | Allowed values are:                                                                                               |
  |                       |                                                                                                                   |
  |                       | - ``'IOB'``: The `Inside-Outside-Beginning <https://en.wikipedia.org/wiki/Inside_Outside_Beginning>`_ tagging     |
  |                       |   format.                                                                                                         |
  |                       |                                                                                                                   |
  |                       | - ``'IOBES'``: An extension to the IOB format with an 'E' tag representing the ending token in an entity span,    |
  |                       |   and an 'S' tag representing single token entities.                                                              |
  +-----------------------+-------------------------------------------------------------------------------------------------------------------+

2. **Feature Extraction Settings** 

``'features'`` (:class:`dict`)
  |

  Is a dictionary where the keys are the names of the feature groups to be extracted. The corresponding values are dictionaries representing the feature extraction settings for each group. The table below enumerates the features that can be used for entity recognition.

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

  Is a dictionary containing the values to be used for different model hyperparameters during training. Examples include the ``'C'`` parameter (inverse of regularization strength), the ``'penalty'`` parameter (norm used in penalization) and so on. The hyperparameters for the MEMM model are the same as those for a `maximum entropy model (MaxEnt) <https://en.wikipedia.org/wiki/Multinomial_logistic_regression>`_. You can view the full list of allowed hyperparameters :sk_api:`here <sklearn.linear_model.LogisticRegression.html>`.

``'param_selection'`` (:class:`dict`)
  |

  Is a dictionary containing the settings for :sk_guide:`hyperparameter selection <grid_search>`. This is used as an alternative to the ``'params'`` dictionary above if the ideal hyperparameters for the model are not already known and need to be estimated.

  Workbench needs two pieces of information from the developer to do parameter estimation:

  #. The parameter space to search, captured by the value for the ``'grid'`` key
  #. The strategy for splitting the labeled data into training and validation sets, specified by the ``'type'`` key

  Depending on the splitting scheme selected, the :data:`param_selection` dictionary can contain other keys that define additional settings. The table below enumerates all the keys allowed in the dictionary.

  +-----------------------+-------------------------------------------------------------------------------------------------------------------+
  | Key                   | Value                                                                                                             |
  +=======================+===================================================================================================================+
  | ``'grid'``            | A dictionary mapping each hyperparameter to a list of potential values to be searched. Here is an example grid    |
  |                       | for a :sk_api:`logistic regression <sklearn.linear_model.LogisticRegression>` model:                              |
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

  The :meth:`fit` method does an :sk_guide:`exhaustive grid search <grid_search.html#exhaustive-grid-search>` over the parameter space, evaluating candidate models using the specified cross-validation strategy, to identify the parameters that give the highest accuracy. The optimal parameters can then be used in future calls to :meth:`fit` to skip the parameter selection process.

.. _build_entity_with_config:

Training with custom configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are two ways to override Workbench's default entity recognizer configuration with your custom settings.


1. Application configuration file
"""""""""""""""""""""""""""""""""

The first method, as described in the :ref:`NaturalLanguageProcessor <build_nlp_with_config>` chapter, is to define the classifier settings in your application configuration file, ``config.py``. Define a dictionary named :data:`ENTITY_MODEL_CONFIG` containing your custom settings. The :meth:`EntityRecognizer.fit` and :meth:`NaturalLanguageProcessor.build` methods will then use those settings instead of Workbench's defaults.

Here's an example of a ``config.py`` file where the preset configuration for the entity recognizer is being overridden by custom settings that have been optimized for the app.

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
           'scoring': 'logloss',
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

Since this method requires updating a file each time you want to modify a setting, it's less suitable for rapid prototyping than the second method described below. The recommended use for this functionality is to store your optimal classifier settings, once you have identified them via experimentation. This ensures that the classifier training methods will use the optimized configuration to rebuild the models in the future. A common use case is retraining models on newly acquired training data, without retuning the underlying model settings.


2. Arguments to the :meth:`fit` method
""""""""""""""""""""""""""""""""""""""

The recommended way to experiment with an entity recognizer is by using arguments to the :meth:`fit` method.


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

By default, the classifier only extracts n-grams within a context window of two tokens or less around the token of interest. It may be useful to have the classifier look at a larger context window since that could potentially provide more information than just the words in the immediate vicinity. To accomplish this, you need to change the ``'ngram_lengths_to_start_positions'`` settings to extract n-grams starting from tokens that are further away. Suppose you want to extract all the unigrams and bigrams in a window of three tokens around the current token, the :data:`my_features` dictionary should be updated as shown below.

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

To retrain the classifier with the updated feature set, pass in the :data:`my_features` dictionary as an argument to the :data:`features` parameter of the :meth:`fit` method. This trains the entity recognition model using the provided feature extraction settings, while continuing to use Workbench's defaults for model type (MEMM) and hyperparameter selection.

.. code-block:: python

   >>> er.fit(features=my_features)
   Fitting entity recognizer: domain='weather', intent='check_weather'
   No app configuration file found. Using default entity model configuration
   Selecting hyperparameters using k-fold cross-validation with 5 splits
   Best accuracy: 99.04%, params: {'C': 10000, 'penalty': 'l2'}

**Hyperparameter tuning**

Next, let's experiment with the model's hyperparameters. To get the hyperparameter selection settings for the current classifier, do:

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

Let's reduce the range of values to search for the ``'C'`` parameter (inverse of regularization strength). You could also allow the hyperparameter estimation process to choose whether or not to add an intercept term to the decision function (added by default). The updated settings can then be passed to :meth:`fit` as an argument to the :data:`param_selection` parameter.

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

The :meth:`fit` method now searches over the updated parameter grid and prints the hyperparameter values for the model with the highest cross-validation accuracy. By default, the entity recognizer uses k-fold cross-validation with 5 folds. To use a different cross-validation strategy, you can modify the value for the ``'type'`` key in the :data:`my_param_settings`. For instance, to use five randomized folds:

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

For a full list of configurable hyperparameters and available cross-validation methods, refer to the above section on defining :ref:`hyperparameter settings <entity_tuning>`.


**Model settings**

Lastly, let's try varying the model training settings. Start with the model settings of the current model:

.. code-block:: python

   >>> my_model_settings = er.config.model_settings
   >>> my_model_settings
   {'feature_scaler': 'max-abs', 'tag_scheme': 'IOB'}

The code below turns off feature scaling, changes the tagging scheme to IOBES and retrains the entity recognition model using these updated settings. It uses the Workbench defaults for feature extraction and hyperparameter selection settings.

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

A trained entity recognizer can be run on a test query using the :meth:`EntityRecognizer.predict` method.

.. code-block:: python

   >>> er.predict('Weather in San Francisco next week')
   (<QueryEntity 'San Francisco' ('city') char: [11-23], tok: [2-3]>,
    <QueryEntity 'next week' ('sys_time') char: [25-33], tok: [4-5]>)

The :meth:`predict` method returns a list of detected entities in the query. It gets called by the natural language processor's :meth:`process` method at runtime to recognize all the entities in an incoming query.

The :meth:`predict` method runs on one query at a time. To instead test a trained model on a batch of labeled test queries and evaluate classifier performance, see the next section.


Evaluate classifier performance
-------------------------------

To evaluate the accuracy of your trained entity recognizer, you first need to create labeled test data, as described in the :ref:`Natural Language Processor <evaluate_nlp>` chapter. Once you have the test data files in the right place in your Workbench project, you can measure your model's performance using the :meth:`EntityRecognizer.evaluate` method.

.. code-block:: python

   >>> er.evaluate()
   Loading queries from file weather/check_weather/test.txt
   <EntityModelEvaluation score: 89.19%, 33 of 37 examples correct>

The :meth:`evaluate` method strips away all ground truth annotations from the test queries and passes in the resulting unlabeled queries to the trained entity recognizer for prediction. The classifier's output predictions are then compared against the ground truth labels to compute the model's prediction accuracy. In the above example, the model got 33 out of 37 test queries correct, resulting in an accuracy of about 89%. Note that this is query-level accuracy. For a prediction on a query to be graded as "correct", all the entities detected by the entity recognizer need to match exactly with the annotated entities in the test query.

To debug the classifier performance, it can often be helpful to look at the token-level labeling accuracies as well. The :meth:`evaluate` method returns a rich object that contains a lot more information over and above the aggregate query-level accuracy score. The code below prints all the model performance statistics reported by the :meth:`evaluate` method.

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

The statistics are split into four sections.

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

  A `confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`_ with each row representing the number of instances in an actual class and each column representing the number of instances in a predicted class. It makes it easy to see if the classifier is frequently confusing two classes, i.e. commonly mislabelling one entity tag as another. For instance, in the above example, the entity recognizer has wrongly classified two instances of ``S|city|O|`` tokens as ``O||O|``.

**Sequence Statistics**
  |

  Sequence-level accuracy that tracks the fraction of queries for which the entity recognizer successfully identified **all** the expected entities.


While these detailed statistics provide a wealth of information about the classifier performance, you might additionally also want to inspect the classifier's prediction on individual queries to better understand error patterns.

To view the classifier predictions for the entire test set, you can use the :attr:`results` attribute of the returned :obj:`eval` object.

.. code-block:: python

   >>> eval.results
   [
     EvaluatedExample(example=<Query 'check temperature outside'>, expected=(), predicted=(), probas=None, label_type='entities'),
     EvaluatedExample(example=<Query 'check temperature in miami'>, expected=(<QueryEntity 'miami' ('city') char: [21-25], tok: [3-3]>,), predicted=(<QueryEntity 'miami' ('city') char: [21-25], tok: [3-3]>,), probas=None, label_type='entities'),
     ...
   ]

Each result is an instance of the :class:`EvaluatedExample` class which contains information about the original input query, the expected ground truth entities and the predicted entities. You can also selectively look at just the correct predictions or the incorrect predictions. The code below shows how to do that.

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

`List comprehensions <https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions>`_ can be used to easily slice and dice the results for error analysis. For instance, to easily inspect all incorrect predictions where the first entity in the query is supposed to be of a particular type, say ``city``, you could do:

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

In each of the above cases, the entity recognizer was unable to correctly detect the full ``city`` entity in the query. This is usually a sign that the training data lacks coverage for queries with language patterns or entities like those in the examples above. It could also mean that the gazetteer for this entity type is not comprehensive enough. 

On inspecting the :doc:`training data <../blueprints/home_assistant>`, you will find that the ``check_weather`` intent indeed lacks labeled training queries like the first two queries above. This issue could potentially be solved by adding more relevant queries annotated with the ``city`` entity to the ``check_weather`` intent's training data, so the recognition model can generalize better. The last two queries above are misclassified due to a lack of slang terms and nicknames in the :doc:`gazetteer data <../blueprints/home_assistant>` for the ``city`` entity. This can be mitigated by expanding the ``city`` gazetteer to contain entries like "San Fran", "Big Apple" and other popular synonyms for location names that are relevant to the ``weather`` domain.

Error analysis on the results of the :meth:`evaluate` method can thus inform your experimentation and help in building better models. In the example  above, adding more data to the training set or the gazetteers was proposed as a solution for improving accuracy. While data augmentation should be your first step, you could also explore other techniques such as experimenting with different model types, features and hyperparameters, as described :ref:`earlier <build_entity_with_config>` in this chapter.


Save model for future use
-------------------------

A trained entity recognizer can be saved for later use by calling the :meth:`EntityRecognizer.dump` method. The :meth:`dump` method serializes the trained model as a `pickle file <https://docs.python.org/3/library/pickle.html>`_ and saves it to the specified location on disk.

.. code:: python

   >>> er.dump(model_path='experiments/entity_recognizer.memm.20170701.pkl')
   Saving entity recognizer: domain='weather', intent='check_weather'

The saved model can then be loaded anytime using the :meth:`EntityRecognizer.load` method.

.. code:: python

   >>> er.load(model_path='experiments/entity_recognizer.memm.20170701.pkl')
   Loading entity recognizer: domain='weather', intent='check_weather'

