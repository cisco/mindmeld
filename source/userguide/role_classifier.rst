Working with the Role Classifier
================================

The :ref:`Role Classifier <arch_role_model>`

 - is run as the fourth step in the :ref:`natural language processing pipeline <instantiate_nlp>`
 - is a machine-learned `classification <https://en.wikipedia.org/wiki/Statistical_classification>`_ model that determines the target roles for entities in a given query
 - is trained per entity type, using all the labeled queries for a given intent, with labels derived from the role types annotated within the training queries

Every MindMeld app has one role classifier for every entity type with associated roles.

.. note::

    This is an in-depth tutorial to work through from start to finish. Before you begin, read the :ref:`Step-by-Step Guide <quickstart>`, paying special attention to the :ref:`Role Classification <role_classification>` section.

Access a role classifier
------------------------

Working with the natural language processor falls into two broad phases:

 - First, generate the training data for your app. App performance largely depends on having sufficient quantity and quality of training data. See :doc:`Step 6 <../quickstart/06_generate_representative_training_data>`.
 - Then, conduct experimentation in the Python shell.

When you are ready to begin experimenting, import the :class:`NaturalLanguageProcessor` (NLP) class from the MindMeld :mod:`nlp` module and :ref:`instantiate an object <instantiate_nlp>` with the path to your MindMeld project.

.. code-block:: python

   from mindmeld.components.nlp import NaturalLanguageProcessor
   nlp = NaturalLanguageProcessor(app_path='home_assistant')
   nlp

.. code-block:: console

   <NaturalLanguageProcessor 'home_assistant' ready: False, dirty: False>

Verify that the NLP has correctly identified all the domains and intents for your app.

.. code-block:: python

   nlp.domains

.. code-block:: console

   {
    'smart_home': <DomainProcessor 'smart_home' ready: False, dirty: False>,
    'times_and_dates': <DomainProcessor 'times_and_dates' ready: False, dirty: False>,
    'unknown': <DomainProcessor 'unknown' ready: False, dirty: False>,
    'weather': <DomainProcessor 'weather' ready: False, dirty: False>,
    'greeting': <DomainProcessor 'greeting' ready: False, dirty: False>
   }

.. code:: python

   nlp.domains['times_and_dates'].intents

.. code-block:: console

   {
    'change_alarm': <IntentProcessor 'change_alarm' ready: True, dirty: True>,
    'check_alarm': <IntentProcessor 'check_alarm' ready: False, dirty: False>,
    'remove_alarm': <IntentProcessor 'remove_alarm' ready: False, dirty: False>,
    'set_alarm': <IntentProcessor 'set_alarm' ready: True, dirty: True>,
    'start_timer': <IntentProcessor 'start_timer' ready: True, dirty: True>,
    'stop_timer': <IntentProcessor 'stop_timer' ready: False, dirty: False>,
    'specify_time': <IntentProcessor 'specify_time' ready: False, dirty: False>
   }
   ...

.. code:: python

   nlp.domains['weather'].intents

.. code-block:: console

   {
    'check_weather': <IntentProcessor 'check_weather' ready: False, dirty: False>
   }

.. note::

   Until the labeled training queries have been loaded, MindMeld is not aware of the different entity types for your app.

Use the :meth:`build` method to load the training queries for an intent of your choice. This can take several minutes for intents with a large number of training queries. Once the build is complete, inspect the entity types.

.. code-block:: python

   nlp.domains['times_and_dates'].intents['change_alarm'].build()
   nlp.domains['times_and_dates'].intents['change_alarm'].entities

.. code-block:: console

   {
    'sys_time': <EntityProcessor 'sys_time' ready: True, dirty: True>,
    'sys_interval': <EntityProcessor 'sys_interval' ready: True, dirty: True>
   }

Access the :class:`RoleClassifier` for an entity type of your choice, using the :attr:`role_classifier` attribute of the desired entity.

.. code-block:: python

   rc = nlp.domains['times_and_dates'].intents['change_alarm'].entities['sys_time'].role_classifier
   rc

.. code-block:: console

   <RoleClassifier ready: True, dirty: True>

.. _train_role_model:

Train a role classifier
-----------------------

Use the :meth:`RoleClassifier.fit` method to train a role classification model. Depending on the size of the training data, this can take anywhere from a few seconds to several minutes. With logging level set to ``INFO`` or below, you should see the build progress in the console along with cross-validation accuracy for the classifier.

.. _baseline_role_fit:

.. code-block:: python

   from mindmeld import configure_logs; configure_logs()
   from mindmeld.components.nlp import NaturalLanguageProcessor
   nlp = NaturalLanguageProcessor(app_path='home_assistant')
   nlp.domains['times_and_dates'].intents['change_alarm'].build()

   rc = nlp.domains['times_and_dates'].intents['change_alarm'].entities['sys_time'].role_classifier
   rc.fit()

.. code-block:: console

   Fitting role classifier: domain='times_and_dates', intent='change_alarm', entity_type='sys_time'
   No role model configuration set. Using default.

The :meth:`fit` method loads all necessary training queries and trains a role classification model. When called with no arguments (as in the example above), the method uses the settings from ``config.py``, the :ref:`app's configuration file <build_nlp_with_config>`. If ``config.py`` is not defined, the method uses the MindMeld preset :ref:`classifier configuration <config>`.

Using default settings is the recommended (and quickest) way to get started with any of the NLP classifiers. The resulting baseline classifier should provide a reasonable starting point from which to bootstrap your machine learning experimentation. You can then try alternate settings as you seek to identify the optimal classifier configuration for your app.


Classifier configuration
^^^^^^^^^^^^^^^^^^^^^^^^

Use the :attr:`config` attribute of a trained classifier to view the :ref:`configuration <config>` that the classifier is using. Here's an  example where we view the configuration of a role classifier trained using default settings:

.. code-block:: python

   rc.config.to_dict()

.. code-block:: console

   {
     'features': {
       'bag-of-words-after': {
         'ngram_lengths_to_start_positions': {1: [0, 1], 2: [0, 1]}
       },
       'bag-of-words-before': {
         'ngram_lengths_to_start_positions': {1: [-2, -1], 2: [-2, -1]}
       },
       'in-gaz': {},
       'other-entities': {}
     },
     'model_settings': {'classifier_type': 'logreg'},
     'model_type': 'text',
     'param_selection': None,
     'params': {'C': 100, 'penalty': 'l1'}
   }

Let's take a look at the allowed values for each setting in a role classifier configuration.

  .. _model_settings:

1. **Model Settings**

``'model_type'`` (:class:`str`)
  |

  Always ``'text'``, since role classification is a `text classification <https://en.wikipedia.org/wiki/Text_classification>`_ model.

``'model_settings'`` (:class:`dict`)
  |

  Always a dictionary with the single key ``'classifier_type'``, whose value specifies the machine learning model to use. Allowed values are shown in the table below.

  .. _sklearn_role_models:

  =============== ======================================================================= ==========================================
  Value           Classifier                                                              Reference for configurable hyperparameters
  =============== ======================================================================= ==========================================
  ``'logreg'``    :sk_guide:`Logistic regression <linear_model.html#logistic-regression>` :sk_api:`sklearn.linear_model.LogisticRegression <sklearn.linear_model.LogisticRegression>`
  ``'svm'``       :sk_guide:`Support vector machine <svm.html#svm-classification>`        :sk_api:`sklearn.svm.SVC <sklearn.svm.SVC>`
  ``'dtree'``     :sk_guide:`Decision tree <tree.html#tree>`                              :sk_api:`sklearn.tree.DecisionTreeClassifier <sklearn.tree.DecisionTreeClassifier>`
  ``'rforest'``   :sk_guide:`Random forest <ensemble.html#forest>`                        :sk_api:`sklearn.ensemble.RandomForestClassifier <sklearn.ensemble.RandomForestClassifier>`
  =============== ======================================================================= ==========================================

2. **Feature Extraction Settings**

``'features'`` (:class:`dict`)
  |

  A dictionary whose keys are names of feature groups to extract. The corresponding values are dictionaries representing the feature extraction settings for each group. The table below enumerates the features that can be used for role classification.


  .. _role_features:

  +---------------------------+------------------------------------------------------------------------------------------------------------+
  | Group Name                | Description                                                                                                |
  +===========================+============================================================================================================+
  | ``'bag-of-words-after'``  | Generates n-grams of specified lengths from the query text following the current entity.                   |
  |                           |                                                                                                            |
  |                           | Settings:                                                                                                  |
  |                           |                                                                                                            |
  |                           | A dictionary with n-gram lengths as keys and a list of different starting positions as values.             |
  |                           | Each starting position is a token index, relative to the the start of the current entity span.             |
  |                           |                                                                                                            |
  |                           | Examples:                                                                                                  |
  |                           |                                                                                                            |
  |                           | ``'ngram_lengths_to_start_positions': {1: [0], 2: [0]}``                                                   |
  |                           |  - extracts all words (unigrams) and bigrams starting with the first word of the current entity span       |
  |                           |                                                                                                            |
  |                           | ``'ngram_lengths_to_start_positions': {1: [0, 1], 2: [0, 1]}``                                             |
  |                           |  - additionally includes unigrams and bigrams starting from the word after the current entity's first token|
  |                           |                                                                                                            |
  |                           | Given the query "Change my {6 AM|sys_time|old_time} alarm to {7 AM|sys_time|new_time}"                     |
  |                           | and a classifier extracting features for the "6 AM" ``sys_time`` entity:                                   |
  |                           |                                                                                                            |
  |                           | ``{1: [0, 1]}``                                                                                            |
  |                           |  - extracts "6" and "AM"                                                                                   |
  |                           |                                                                                                            |
  |                           | ``{2: [0, 1]}``                                                                                            |
  |                           |  - extracts "6 AM" and "AM alarm"                                                                          |
  +---------------------------+------------------------------------------------------------------------------------------------------------+
  | ``'bag-of-words-before'`` | Generates n-grams of specified lengths from the query text preceding the current entity.                   |
  |                           |                                                                                                            |
  |                           | Settings:                                                                                                  |
  |                           |                                                                                                            |
  |                           | A dictionary with n-gram lengths as keys and a list of different starting positions as values, similar     |
  |                           | to the ``'bag-of-words-after'`` feature group.                                                             |
  |                           |                                                                                                            |
  |                           | Examples:                                                                                                  |
  |                           |                                                                                                            |
  |                           | Given the query "Change my {6 AM|sys_time|old_time} alarm to {7 AM|sys_time|new_time}"                     |
  |                           | and a classifier extracting features for the "6 AM" ``sys_time`` entity:                                   |
  |                           |                                                                                                            |
  |                           | ``{1: [-2, -1]}``                                                                                          |
  |                           |  - extracts "change" and "my"                                                                              |
  |                           |                                                                                                            |
  |                           | ``{2: [-2, -1]}``                                                                                          |
  |                           |  - extracts "change my" and "my 6"                                                                         |
  +---------------------------+------------------------------------------------------------------------------------------------------------+
  | ``'in-gaz'``              | Generates a set of features indicating the presence of query n-grams in different entity gazetteers,       |
  |                           | along with popularity information as defined in the gazetteer.                                             |
  +---------------------------+------------------------------------------------------------------------------------------------------------+
  | ``'numeric'``             | Generates a set of features indicating the presence of numeric entities in the query extracted by the      |
  |                           | numerical parser. These numeric entities include only time and interval entities and are labelled as       |
  |                           | ``sys_time`` and ``sys_interval``.                                                                         |
  +---------------------------+------------------------------------------------------------------------------------------------------------+
  | ``'other-entities'``      | Encodes information about the other entities present in the query than the current one.                    |
  +---------------------------+------------------------------------------------------------------------------------------------------------+

.. _role_tuning:

3. **Hyperparameter Settings**

``'params'`` (:class:`dict`)
  |

  A dictionary of values to be used for model hyperparameters during training. Examples include the ``'kernel'`` parameter for SVM, ``'penalty'`` for logistic regression, ``'max_depth'`` for decision tree, and so on. The list of allowable hyperparameters depends on the model selected. See the :ref:`reference links <sklearn_role_models>` above for parameter lists.

``'param_selection'`` (:class:`dict`)
  |

  Is a dictionary containing the settings for :sk_guide:`hyperparameter selection <grid_search>`. This is used as an alternative to the ``'params'`` dictionary above if the ideal hyperparameters for the model are not already known and need to be estimated.

  MindMeld needs two pieces of information from the developer to do parameter estimation:

  #. The parameter space to search, captured by the value for the ``'grid'`` key
  #. The strategy for splitting the labeled data into training and validation sets, specified by the ``'type'`` key

  Depending on the splitting scheme selected, the :data:`param_selection` dictionary can contain other keys that define additional settings. The table below enumerates all the keys allowed in the dictionary.

  +-----------------------+-------------------------------------------------------------------------------------------------------------------------+
  | Key                   | Value                                                                                                                   |
  +=======================+=========================================================================================================================+
  | ``'grid'``            | A dictionary mapping each hyperparameter to a list of potential values to be searched. Here is an example grid          |
  |                       | for a :sk_api:`logistic regression <sklearn.linear_model.LogisticRegression>` model:                                    |
  |                       |                                                                                                                         |
  |                       | .. code-block:: python                                                                                                  |
  |                       |                                                                                                                         |
  |                       |    {                                                                                                                    |
  |                       |      'penalty': ['l1', 'l2'],                                                                                           |
  |                       |      'C': [10, 100, 1000, 10000, 100000],                                                                               |
  |                       |       'fit_intercept': [True, False]                                                                                    |
  |                       |    }                                                                                                                    |
  |                       |                                                                                                                         |
  |                       | See the :ref:`reference links <sklearn_role_models>` above for details on the hyperparameters available for each model. |
  +-----------------------+-------------------------------------------------------------------------------------------------------------------------+
  | ``'type'``            | The :sk_guide:`cross-validation <cross_validation>` methodology to use. One of:                                         |
  |                       |                                                                                                                         |
  |                       | - ``'k-fold'``: :sk_api:`K-folds <sklearn.model_selection.KFold>`                                                       |
  |                       | - ``'shuffle'``: :sk_api:`Randomized folds <sklearn.model_selection.ShuffleSplit>`                                      |
  |                       | - ``'group-k-fold'``: :sk_api:`K-folds with non-overlapping groups <sklearn.model_selection.GroupKFold>`                |
  |                       | - ``'group-shuffle'``: :sk_api:`Group-aware randomized folds <sklearn.model_selection.GroupShuffleSplit>`               |
  |                       | - ``'stratified-k-fold'``: :sk_api:`Stratified k-folds <sklearn.model_selection.StratifiedKFold>`                       |
  |                       | - ``'stratified-shuffle'``: :sk_api:`Stratified randomized folds <sklearn.model_selection.StratifiedShuffleSplit>`      |
  |                       |                                                                                                                         |
  +-----------------------+-------------------------------------------------------------------------------------------------------------------------+
  | ``'k'``               | Number of folds (splits)                                                                                                |
  +-----------------------+-------------------------------------------------------------------------------------------------------------------------+

  To identify the parameters that give the highest accuracy, the :meth:`fit` method does an :sk_guide:`exhaustive grid search <grid_search.html#exhaustive-grid-search>` over the parameter space, evaluating candidate models using the specified cross-validation strategy. Subsequent calls to :meth:`fit` can use these optimal parameters and skip the parameter selection process

4. **Custom Train/Test Settings**

``'train_label_set'`` (:class:`str`)
  |

  A string representing a regex pattern that selects all training files for role model training with filenames that match the pattern. The default regex when this key is not specified is ``'train.*\.txt'``.

``'test_label_set'`` (:class:`str`)
  |

  A string representing a regex pattern that selects all evaluation files for role model testing with filenames that match the pattern. The default regex when this key is not specified is ``'test.*\.txt'``.

.. _build_role_with_config:

Training with custom configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To override MindMeld's default role classifier configuration with custom settings, you can either edit the app configuration file, or, you can call the :meth:`fit` method with appropriate arguments.


1. Application configuration file
"""""""""""""""""""""""""""""""""

When you define custom classifier settings in ``config.py``, the :meth:`RoleClassifier.fit` and :meth:`NaturalLanguageProcessor.build` methods use those settings instead of MindMeld's defaults. To do this, define a dictionary of your custom settings, named :data:`ROLE_CLASSIFIER_CONFIG`.

Here's an example of a ``config.py`` file where custom settings optimized for the app override the preset configuration for the role classifier.


.. code-block:: python

   ROLE_CLASSIFIER_CONFIG = {
       'model_type': 'text',
       'model_settings': {'classifier_type': 'logreg'}
       'params': {
           'C': 10,
           'penalty': 'l2'
       },
       'features': {
           'bag-of-words-before': {
               'ngram_lengths_to_start_positions': {
                   1: [-2, -1],
                   2: [-2, -1]
               }
           },
           'bag-of-words-after': {
               'ngram_lengths_to_start_positions': {
                   1: [0, 1],
                   2: [0, 1]
               }
           },
           'other-entities': {}
       }
   }

Settings defined in :data:`ROLE_CLASSIFIER_CONFIG` apply to role classifiers across all entity types in your application. For finer-grained control, you can implement the :meth:`get_role_classifier_config` function in ``config.py`` to specify suitable configurations for each entity. This gives you the flexibility to have customized configurations for different role classifiers based on the domain, intent, and entity type.

.. code-block:: python

   import copy

   def get_role_classifier_config(domain, intent, entity):
       SPECIAL_CONFIG = copy.deepcopy(ROLE_CLASSIFIER_CONFIG)
       if domain == 'times_and_dates' and intent == 'change_alarms' and entity == 'sys_time':
           SPECIAL_CONFIG['params']['penalty'] = 'l1'
       return SPECIAL_CONFIG

Using ``config.py`` is recommended for storing your optimal classifier settings once you have identified them through experimentation. Then the classifier training methods will use the optimized configuration to rebuild the models. A common use case is retraining models on newly-acquired training data, without retuning the underlying model settings.

Since this method requires updating a file each time you modify a setting, it's less suitable for rapid prototyping than the method described next.

2. Arguments to the :meth:`fit` method
""""""""""""""""""""""""""""""""""""""

For experimenting with the role classifier, the recommended method is to use arguments to the :meth:`fit` method. The main areas for exploration are feature extraction and hyperparameter tuning.

**Feature extraction**

View the default feature set, as seen in the baseline classifier that we trained :ref:`earlier <baseline_role_fit>`. Notice that the 'ngram_lengths_to_start_positions' settings tell the classifier to extract n-grams within a context window of two tokens or less around the token of interest — that is, to only look at words in the immediate vicinity.

.. code-block:: python

   my_features = rc.config.features
   my_features

.. code-block:: console

   {
     'bag-of-words-after': {'ngram_lengths_to_start_positions': {1: [0, 1], 2: [0, 1]}},
     'bag-of-words-before': {'ngram_lengths_to_start_positions': {1: [-2, -1], 2: [-2, -1]}},
     'other-entities': {}
   }

Next, have the classifier look at a larger context window, and extract n-grams starting from tokens that are further away. We'll see whether that provides better information than the smaller default window. Do this by changing the 'ngram_lengths_to_start_positions' settings to extract all the unigrams and bigrams in a window of three tokens around the current token, as shown below.

.. code-block:: python

   my_features['bag-of-words-after']['ngram_lengths_to_start_positions'] = {
       1: [0, 1, 2, 3],
       2: [0, 1, 2]
   }
   my_features['bag-of-words-before']['ngram_lengths_to_start_positions'] = {
       1: [-3, -2, -1],
       2: [-3, -2, -1]
   }
   my_features

.. code-block:: console

   {
     'bag-of-words-after': {'ngram_lengths_to_start_positions': {1: [0, 1, 2, 3], 2: [0, 1, 2]}},
     'bag-of-words-before': {'ngram_lengths_to_start_positions': {1: [-3, -2, -1], 2: [-3, -2, -1]}},
     'other-entities': {}
   }

Suppose w\ :sub:`i` represents the word at the *ith* index in the query, where the index is calculated relative to the start of the current entity span. Then, the above feature configuration should extract the following n-grams (w\ :sub:`0` is the first token of the current entity).

  - Unigrams: { w\ :sub:`-3`, w\ :sub:`-2`, w\ :sub:`-1`, w\ :sub:`0`, w\ :sub:`1`, w\ :sub:`2`, w\ :sub:`3` }

  - Bigrams: { w\ :sub:`-3`\ w\ :sub:`-2`, w\ :sub:`-2`\ w\ :sub:`-1`, w\ :sub:`-1`\ w\ :sub:`0`,  w\ :sub:`0`\ w\ :sub:`1`, w\ :sub:`1`\ w\ :sub:`2`, w\ :sub:`2`\ w\ :sub:`3` }

Retrain the classifier with the updated feature set by passing in the :data:`my_features` dictionary as an argument to the :data:`features` parameter of the :meth:`fit` method. This applies our new feature extraction settings, while retaining the MindMeld defaults for model and classifier types (logreg) and hyperparameter selection.

.. code-block:: python

   rc.fit(features=my_features)

.. code-block:: console

   Fitting role classifier: domain='times_and_dates', intent='change_alarm', entity_type='sys_time'
   No app configuration file found. Using default role model configuration

**Hyperparameter tuning**

View the model's hyperparameters, keeping in mind the :ref:`hyperparameters <model_settings>` for logistic regression, the default model for role classification in MindMeld. These include inverse of regularization strength as 'C', and the norm used in penalization as 'penalty'.

.. code-block:: python

   my_params = rc.config.params
   my_params

.. code-block:: console

   {'C': 100, 'penalty': 'l1'}

Instead of relying on the default preset values for ``'C'`` and ``'penalty'``, let's specify a parameter search grid to let MindMeld select ideal values for the dataset. We'll also specify a cross-validation strategy. Update the parameter selection settings such that the hyperparameter estimation process chooses the ideal ``'C'`` and ``'penalty'`` parameters using 10-fold cross-validation:

.. code-block:: python

   search_grid = {
     'C': [1, 10, 100, 1000],
     'penalty': ['l1', 'l2']
   }
   my_param_settings = {
     'grid': search_grid,
     'type': 'k-fold',
     'k': 10
   }

Pass the updated settings to :meth:`fit` as an argument to the :data:`param_selection` parameter. The :meth:`fit` method then searches over the updated parameter grid, and prints the hyperparameter values for the model whose 10-fold cross-validation accuracy is highest.

.. code-block:: python

   rc.fit(param_selection=my_param_settings)

.. code-block:: console

   Fitting role classifier: domain='times_and_dates', intent='change_alarm', entity_type='sys_time'
   No app configuration file found. Using default role model configuration
   Selecting hyperparameters using k-fold cross validation with 10 splits
   Best accuracy: 96.59%, params: {'C': 1, 'penalty': 'l2'}

Now we'll try a different cross-validation strategy: five randomized folds. Modify the values of the ``'k'`` and ``'type'`` keys in :data:`my_param_settings`, and call :meth:`fit` to see whether accuracy improves:

.. code-block:: python

   my_param_settings['k'] = 5
   my_param_settings['type'] = 'shuffle'
   my_param_settings

.. code-block:: console

   {
    'grid': {
              'C': [1, 10, 100, 1000],
              'penalty': ['l1', 'l2']
            },
    'k': 5,
    'type': 'shuffle'
   }

.. code:: python

   rc.fit(param_selection=my_param_settings)

.. code-block:: console

   Fitting role classifier: domain='times_and_dates', intent='change_alarm', entity_type='sys_time'
   No app configuration file found. Using default role model configuration
   Selecting hyperparameters using shuffle cross validation with 5 splits
   Best accuracy: 97.78%, params: {'C': 1, 'penalty': 'l2'}

For a list of configurable hyperparameters and cross-validation methods, see :ref:`hyperparameter settings <role_tuning>` above.

.. _predict_roles:

Run the role classifier
-----------------------

Before you run the trained role classifier on a test query, you must first detect all the entities in the query using a :ref:`trained entity recognizer <train_entity_model>`:

.. code-block:: python

   query = 'Change my 6 AM alarm to 7 AM'
   er = nlp.domains['times_and_dates'].intents['change_alarm'].entity_recognizer
   entities = er.predict(query)
   entities

.. code-block:: console

   (<QueryEntity '6 AM' ('sys_time') char: [10-13], tok: [2-3]>,
    <QueryEntity '7 AM' ('sys_time') char: [24-27], tok: [6-7]>)

Now you can choose an entity from among those detected, and call the role classifier's :meth:`RoleClassifier.predict` method to classify it. Although it classifies a single entity, the :meth:`RoleClassifier.predict` method uses the full query text, and information about all its entities, for :ref:`feature extraction <role_features>`.

Run the trained role classifier on the two entities from the example above, one by one. The :meth:`predict` method returns the label for the role whose predicted probability is highest.

.. code-block:: python

   rc.predict(query, entities, 0)

.. code-block:: console

   'old_time'

.. code:: python

   rc.predict(query, entities, 1)

.. code-block:: console

   'new_time'

.. note::

   At runtime, the natural language processor's :meth:`process` method calls :meth:`RoleClassifier.predict` to roles for all detected entities in the incoming query.

We want to know how confident our trained model is in its prediction. To view the predicted probability distribution over all possible role labels, use the :meth:`RoleClassifier.predict_proba` method. This is useful both for experimenting with classifier settings and for debugging classifier performance.

The result is a list of tuples whose first element is the role label and whose second element is the associated classification probability. These are ranked by roles, from most likely to least likely.

.. code-block:: python

   rc.predict_proba(query, entities, 0)

.. code-block:: console

   [('old_time', 0.9998281252873086), ('new_time', 0.00017187471269142218)]

.. code:: python

   rc.predict_proba(query, entities, 1)

.. code-block:: console

   [('new_time', 0.9999960507734881), ('old_time', 3.949226511944386e-06)]

An ideal classifier would assign a high probability to the expected (correct) class label for a test query, while assigning very low probabilities to incorrect labels.

The :meth:`predict` and :meth:`predict_proba` methods operate on one entity at a time. Next, we'll see how to test a trained model on a batch of labeled test queries.

Evaluate classifier performance
-------------------------------

To evaluate the accuracy of your trained role classifier, you first need to create labeled test data, as described in the :ref:`Natural Language Processor <evaluate_nlp>` chapter. Once you have the test data files in the right place in your MindMeld project, you can measure your model's performance using the :meth:`RoleClassifier.evaluate` method.

Before you can evaluate the accuracy of your trained role classifier, you must first create labeled test data and place it in your MindMeld project as described in the :ref:`Natural Language Processor <evaluate_nlp>` chapter.

Then, when you are ready, use the :meth:`RoleClassifier.evaluate` method, which

 - strips away all ground truth annotations from the test queries,
 - passes the resulting unlabeled queries to the trained role classifier for prediction, and
 - compares the classifier's output predictions against the ground truth labels to compute the model's prediction accuracy.

In the example below, the model gets 20 out of 21 test queries correct, resulting in an accuracy of about 95%.

.. code-block:: python

   rc.evaluate()

.. code-block:: console

   Loading queries from file times_and_dates/change_alarm/test.txt
   <StandardModelEvaluation score: 95.24%, 20 of 21 examples correct>

The aggregate accuracy score we see above is only the beginning, because the :meth:`evaluate` method returns a rich object containing overall statistics, statistics by class, and a confusion matrix.

Print all the model performance statistics reported by the :meth:`evaluate` method:

.. code-block:: python

   eval = rc.evaluate()
   eval.print_stats()

.. code-block:: console

   Overall statistics:

       accuracy f1_weighted          tp          tn          fp          fn    f1_macro    f1_micro
          0.952       0.952          20          20           1           1       0.952       0.952



   Statistics by class:

                  class      f_beta   precision      recall     support          tp          tn          fp          fn
                old_time       0.957       0.917       1.000          11          11           9           1           0
                new_time       0.947       1.000       0.900          10           9          11           0           1



   Confusion matrix:

                          old_time        new_time
           old_time             11              0
           new_time              1              9


The :meth:`eval.get_stats()` method returns all the above statistics in a structured dictionary without printing them to the console.

Let's decipher the statists output by the :meth:`evaluate` method.

**Overall Statistics**
  |

  Aggregate stats measured across the entire test set:

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

  - **Classes are balanced** — When the number of annotations for each role are comparable and each role is equally important, focusing on the accuracy metric is usually good enough.

  - **Classes are imbalanced** — In this case, it's important to take the f1 scores into account.

  - **All f1 and accuracy scores are low** — When role classification is performing poorly across all roles, either of the following may be the problem: 1) You do not have enough training data for the model to learn, or 2) you need to tune your model hyperparameters.

  - **f1 weighted is higher than f1 macro** — This means that roles with fewer evaluation examples are performing poorly. Try adding more data to these roles.

  - **f1 macro is higher than f1 weighted** — This means that roles with more evaluation examples are performing poorly. Verify that the number of evaluation examples reflects the class distribution of your training examples.

  - **f1 micro is higher than f1 macro** — This means that some roles are being misclassified more often than others. Identify the problematic roles by checking the class-wise statistics below. Some roles may be too similar to others, or you may need to add more training data to some roles.

  - **Some classes are more important than others** — If some roles are more important than others for your use case, it is best to focus especially on the class-wise statistics described below.

**Class-wise Statistics**
  |

  Stats computed at a per-class level:

  ===========  ===
  class        Role label
  f_beta       :sk_api:`F-beta score <sklearn.metrics.fbeta_score>`
  precision    `Precision <https://en.wikipedia.org/wiki/Precision_and_recall#Precision>`_
  recall       `Recall <https://en.wikipedia.org/wiki/Precision_and_recall#Recall>`_
  support      Number of test entities with this role (based on ground truth)
  tp           Number of `true positives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  tn           Number of `true negatives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  fp           Number of `false positives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  fn           Number of `false negatives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  ===========  ===

**Confusion Matrix**
  |

  A `confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`_ where each row represents the number of instances in an actual class and each column represents the number of instances in a predicted class. This reveals whether the classifier tends to confuse two classes, i.e., mislabel one class as another. In the above example, the role classifier wrongly classified one instance of a ``new_time`` entity as ``old_time``.

Now we have a wealth of information about the performance of our classifier. Let's go further and inspect the classifier's predictions at the level of individual queries, to better understand error patterns.

View the classifier predictions for the entire test set using the :attr:`results` attribute of the returned :obj:`eval` object. Each result is an instance of the :class:`EvaluatedExample` class, which contains information about the original input query, the expected ground truth label, the predicted label, and the predicted probability distribution over all the class labels.

.. code-block:: python

   eval.results

.. code-block:: console

   [
     EvaluatedExample(example=(<Query 'change my 6 am alarm'>, (<QueryEntity '6 am' ('sys_time') char: [10-13], tok: [2-3]>,), 0), expected='old_time', predicted='old_time', probas={'sys_time': 0.10062246873286373, 'old_time': 0.89937753126713627}, label_type='class'),
     EvaluatedExample(example=(<Query 'change my 6 am alarm to 7 am'>, (<QueryEntity '6 am' ('sys_time') char: [10-13], tok: [2-3]>, <QueryEntity '7 am' ('sys_time') char: [24-27], tok: [6-7]>), 0), expected='old_time', predicted='old_time', probas={'sys_time': 0.028607105880949835, 'old_time': 0.97139289411905017}, label_type='class'),
    ...
   ]

Next, we look selectively at just the correct or incorrect predictions.

.. code-block:: python

   list(eval.correct_results())

.. code-block:: console

   [
     EvaluatedExample(example=(<Query 'change my 6 am alarm'>, (<QueryEntity '6 am' ('sys_time') char: [10-13], tok: [2-3]>,), 0), expected='old_time', predicted='old_time', probas={'new_time': 0.10062246873286373, 'old_time': 0.89937753126713627}, label_type='class'),
     EvaluatedExample(example=(<Query 'change my 6 am alarm to 7 am'>, (<QueryEntity '6 am' ('sys_time') char: [10-13], tok: [2-3]>, <QueryEntity '7 am' ('sys_time') char: [24-27], tok: [6-7]>), 0), expected='old_time', predicted='old_time', probas={'new_time': 0.028607105880949835, 'old_time': 0.97139289411905017}, label_type='class'),
    ...
   ]

.. code:: python

   list(eval.incorrect_results())

.. code-block:: console

   [
     EvaluatedExample(example=(<Query 'replace the 8 am alarm with a 10 am alarm'>, (<QueryEntity '8 am' ('sys_time') char: [12-15], tok: [2-3]>, <QueryEntity '10 am' ('sys_time') char: [30-34], tok: [7-8]>), 1), expected='new_time', predicted='old_time', probas={'new_time': 0.48770513415754235, 'old_time': 0.51229486584245765}, label_type='class')
   ]

Slicing and dicing these results for error analysis is easily done with `list comprehensions <https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions>`_.

Our example dataset is fairly small, and we get just one case of misclassification. But for a real-world app with a large test set, we'd need to be able inspect incorrect predictions for a particular role. Try this using the ``new_time`` role from our example:

.. code-block:: python

   [(r.example, r.probas) for r in eval.incorrect_results() if r.expected == 'new_time']

.. code-block:: console

   [
     (
       (
         <Query 'replace the 8 am alarm with a 10 am alarm'>,
         (<QueryEntity '8 am' ('sys_time') char: [12-15], tok: [2-3]>, <QueryEntity '10 am' ('sys_time') char: [30-34], tok: [7-8]>),
         1
       ),
       {
         'new_time': 0.48770513415754235,
         'old_time': 0.51229486584245765
       }
     )
   ]

Next, we use a list comprehension to identify the kind of queries that the current training data might lack. To do this, we list all queries with a given role where the classifier's confidence for the true label was relatively low. We'll demonstrate this with the ``new_time`` role and a confidence of <60%.

.. code-block:: python

   [(r.example, r.probas) for r in eval.results if r.expected == 'new_time' and r.probas['new_time'] < .6]

.. code-block:: console

   [
     (
       (
         <Query 'replace the 8 am alarm with a 10 am alarm'>,
         (<QueryEntity '8 am' ('sys_time') char: [12-15], tok: [2-3]>, <QueryEntity '10 am' ('sys_time') char: [30-34], tok: [7-8]>),
         1
       ),
       {
         'new_time': 0.48770513415754235,
         'old_time': 0.51229486584245765
       }
     ),
     (
       (
         <Query 'cancel my 6 am and replace it with a 6:30 am alarm'>,
         (<QueryEntity '6 am' ('sys_time') char: [10-13], tok: [2-3]>, <QueryEntity '6:30 am' ('sys_time') char: [37-43], tok: [9-10]>),
         1
       ),
       {
         'new_time': 0.5872536946800766,
         'old_time': 0.41274630531992335
       }
     )
   ]

For both of these results, the classifier's prediction probability for the ``'new_time'`` role was fairly low. The classifier got one of them wrong, and barely got the other one right with a confidence of about 59%.

Try looking at the :doc:`training data <../blueprints/home_assistant>`. You should discover that the ``new_time`` role does indeed lack labeled training queries like the ones above.

One potential solution is to add more training queries for the ``new_time`` role, so the classification model can generalize better.

Error analysis on the results of the :meth:`evaluate` method can inform your experimentation and help in building better models. Augmenting training data should be the first step, as in the above example. Beyond that, you can experiment with different model types, features, and hyperparameters, as described :ref:`earlier <build_role_with_config>` in this chapter.

View features extracted for classification
------------------------------------------

While training a new model or investigating a misclassification by the classifier, it is sometimes useful to view the extracted features to make sure they are as expected. For example, there may be non-ASCII characters in the query that are treated differently by the feature extractors. Or the value assigned to a particular feature may be computed differently than you expected. Not extracting the right features could lead to misclassifications. In the example below, we view the features extracted for the query 'set alarm for 7 am' using :meth:`RoleClassifier.view_extracted_features` method.

.. code:: python

   rc.view_extracted_features("set alarm for 7 am", entities, 0)

.. code-block:: console

   {'bag_of_words|ngram_before|length:1|pos:-2': 'alarm',
    'bag_of_words|ngram_before|length:1|pos:-1': 'for',
    'bag_of_words|ngram_before|length:2|pos:-2': 'alarm for',
    'bag_of_words|ngram_before|length:2|pos:-1': 'for 7',
    'bag_of_words|ngram_after|length:1|pos:0': 'am',
    'bag_of_words|ngram_after|length:1|pos:1': '<$>',
    'bag_of_words|ngram_after|length:2|pos:0': 'am <$>',
    'bag_of_words|ngram_after|length:2|pos:1': '<$> <$>'}

This is especially useful when you are writing :doc:`custom feature extractors <./custom_features>` to inspect whether the right features are being extracted.

Save model for future use
-------------------------

Save the trained role classifier for later use by calling the :meth:`RoleClassifier.dump` method. The :meth:`dump` method serializes the trained model as a `pickle file <https://docs.python.org/3/library/pickle.html>`_ and saves it to the specified location on disk.

.. code:: python

   rc.dump(model_path='experiments/role_classifier.maxent.20170701.pkl')

.. code-block:: console

   Saving role classifier: domain='times_and_dates', intent='change_alarm', entity_type='sys_time'

You can load the saved model anytime using the :meth:`RoleClassifier.load` method.

.. code:: python

   rc.load(model_path='experiments/role_classifier.maxent.20170701.pkl')

.. code-block:: console

   Loading role classifier: domain='times_and_dates', intent='change_alarm', entity_type='sys_time'

