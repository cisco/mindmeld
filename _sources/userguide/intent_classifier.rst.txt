Working with the Intent Classifier
==================================

The :ref:`Intent Classifier <arch_intent_model>`

 - is run as the second step in the :ref:`natural language processing pipeline <arch_nlp>`
 - is a `text classification <https://en.wikipedia.org/wiki/Text_classification>`_ model that determines the target intent for a given query
 - is trained using all of the labeled queries across all the intents in a given domain

Every MindMeld app has one intent classifier for every domain with multiple intents. The name of each intent folder serves as the label for the training queries contained within that folder.

See :doc:`Step 6 <../quickstart/06_generate_representative_training_data>` for more details on training data preparation.

.. note::

   - This is an in-depth tutorial to work through from start to finish. Before you begin, read the :ref:`Step-by-Step Guide <quickstart>`, paying special attention to the :ref:`Intent Classification <intent_classification>` section.
   - This section requires the :doc:`Home Assistant <../blueprints/home_assistant>` blueprint application. To get the app, open a terminal and run ``mindmeld blueprint home_assistant``.


Access an intent classifier
---------------------------

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

Verify that the NLP has correctly identified all the domains for your app.

.. code-block:: python

   nlp.domains

.. code-block:: console

   {
    'greeting': <DomainProcessor 'greeting' ready: False, dirty: False>,
    'smart_home': <DomainProcessor 'smart_home' ready: False, dirty: False>,
    'times_and_dates': <DomainProcessor 'times_and_dates' ready: False, dirty: False>,
    'unknown': <DomainProcessor 'unknown' ready: False, dirty: False>,
    'weather': <DomainProcessor 'weather' ready: False, dirty: False>
   }

Access the :class:`IntentClassifier` for a domain of your choice, using the :attr:`intent_classifier` attribute of the desired entity.

.. code-block:: python

   # Intent classifier for the 'smart_home' domain:
   ic = nlp.domains['smart_home'].intent_classifier
   ic

.. code-block:: console

   <IntentClassifier ready: False, dirty: False>
   ...

.. code-block:: python

   # Intent classifier for the 'weather' domain:
   ic = nlp.domains['weather'].intent_classifier
   ic

.. code-block:: console

   <IntentClassifier ready: False, dirty: False>


Train an intent classifier
--------------------------

Use the :meth:`IntentClassifier.fit` method to train an intent classification model for a domain of your choice. Depending on the size of the training data, this can take anywhere from a few seconds to several minutes to finish. With logging level set to ``INFO`` or below, you should see the build progress in the console and the cross-validation accuracy of the trained model.

.. _baseline_intent_fit:

.. code-block:: python

   from mindmeld import configure_logs; configure_logs()
   ic = nlp.domains['times_and_dates'].intent_classifier
   ic.fit()

.. code-block:: console

   Fitting intent classifier: domain='times_and_dates'
   Loading queries from file times_and_dates/change_alarm/train.txt
   Loading queries from file times_and_dates/check_alarm/train.txt
   Loading queries from file times_and_dates/remove_alarm/train.txt
   Loading queries from file times_and_dates/set_alarm/train.txt
   Loading queries from file times_and_dates/start_timer/train.txt
   Loading queries from file times_and_dates/stop_timer/train.txt
   Selecting hyperparameters using k-fold cross-validation with 10 splits
   Best accuracy: 97.68%, params: {'C': 100, 'class_weight': {0: 2.3033333333333332, 1: 1.066358024691358, 2: 0.68145956607495073, 3: 0.54068857589984354, 4:    0.98433048433048431, 5: 3.3872549019607843}, 'fit_intercept': True}


The :meth:`fit` method loads all the necessary training queries and trains an intent classification model. When called with no arguments (as in the example above), the method uses the settings from ``config.py``, the :ref:`app's configuration file <build_nlp_with_config>`. If ``config.py`` is not defined, the method uses the MindMeld preset :ref:`classifier configuration <config>`.

Using default settings is the recommended (and quickest) way to get started with any of the NLP classifiers. The resulting baseline classifier should provide a reasonable starting point from which to bootstrap your machine learning experimentation. You can then try alternate settings as you seek to identify the optimal classifier configuration for your app.


Classifier configuration
^^^^^^^^^^^^^^^^^^^^^^^^

Use the :attr:`config` attribute of a trained classifier to view the :ref:`configuration <config>` that the classifier is using. Here’s an example where we view the configuration of a baseline intent classifier trained using default settings:

.. code-block:: python

   ic.config.to_dict()

.. code-block:: console

   {
    'features': {
      'bag-of-words': {'lengths': [1, 2]},
      'edge-ngrams': {'lengths': [1, 2]},
      'exact': {'scaling': 10},
      'freq': {'bins': 5},
      'in-gaz': {},
      'length': {}
    },
    'model_settings': {'classifier_type': 'logreg'},
    'model_type': 'text',
    'param_selection': {
      'grid': {
        'C': [0.01, 1, 100, 10000, 1000000],
        'class_weight': [
          ...
        ],
        'fit_intercept': [True, False]
      },
      'k': 10,
      'type': 'k-fold'
    },
    'params': None,
    'train_label_set': 'train.*\.txt',
    'test_label_set': 'test.*\.txt'
   }

Let's take a look at the allowed values for each setting in an intent classifier configuration.

1. **Model Settings**

``'model_type'`` (:class:`str`)
  |

  Always ``'text'``, since an intent classifier is a `text classification <https://en.wikipedia.org/wiki/Text_classification>`_ model.

``'model_settings'`` (:class:`dict`)
  |

  Always a dictionary with the single key ``'classifier_type'`` whose value specifies the machine learning model to use. Allowed values are shown in the table below.


  .. _sklearn_intent_models:

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

  A dictionary whose keys are the names of the feature groups to extract. The corresponding values are dictionaries representing the feature extraction settings for each group. The table below enumerates the features that can be used for intent classification.

  .. _intent_features:

  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | Group Name            | Description                                                                                                |
  +=======================+============================================================================================================+
  | ``'bag-of-words'``    | Generates n-grams of the specified lengths from the query text.                                            |
  |                       |                                                                                                            |
  |                       | Settings:                                                                                                  |
  |                       |                                                                                                            |
  |                       | A list of n-gram lengths to extract.                                                                       |
  |                       |                                                                                                            |
  |                       | Examples:                                                                                                  |
  |                       |                                                                                                            |
  |                       | ``{'lengths': [1]}``                                                                                       |
  |                       |  - only extracts words (unigrams)                                                                          |
  |                       |                                                                                                            |
  |                       | ``{'lengths': [1, 2, 3]}``                                                                                 |
  |                       |  - extracts unigrams, bigrams and trigrams                                                                 |
  |                       |                                                                                                            |
  |                       | Given the query "how are you":                                                                             |
  |                       |                                                                                                            |
  |                       | ``{'lengths': [1]}``                                                                                       |
  |                       |  - extracts "how", "are", and "you"                                                                        |
  |                       |                                                                                                            |
  |                       | ``{'lengths': [1, 2]}``                                                                                    |
  |                       |  - extracts "how", "are", "you", "how are", and "are you"                                                  |
  |                       |                                                                                                            |
  |                       | Additionally, you can also limit the n-grams considered while extracting the feature by setting a          |
  |                       | threshold on their frequency. These frequencies are computed over the entire training set. This prevents   |
  |                       | infrequent n-grams from being used as features. By default, this frequency is set to 0.                    |
  |                       |                                                                                                            |
  |                       | Examples:                                                                                                  |
  |                       |                                                                                                            |
  |                       |  .. code-block:: python                                                                                    |
  |                       |                                                                                                            |
  |                       |    {                                                                                                       |
  |                       |      'lengths':[2, 3],                                                                                     |
  |                       |      'thresholds': [5, 8]                                                                                  |
  |                       |    }                                                                                                       |
  |                       |                                                                                                            |
  |                       |  - extracts all bigrams whose frequency in the training set is 5 or greater and all trigrams whose         |
  |                       |    frequency is 8 or greater.                                                                              |
  |                       |                                                                                                            |
  |                       |  .. code-block:: python                                                                                    |
  |                       |                                                                                                            |
  |                       |    {                                                                                                       |
  |                       |      'lengths':[1, 3],                                                                                     |
  |                       |      'thresholds': [8]                                                                                     |
  |                       |    }                                                                                                       |
  |                       |                                                                                                            |
  |                       |  - extracts all unigrams whose frequency in the training set is 8 or greater and all trigrams.             |
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | ``'freq'``            | Generates a log-scaled count for each frequency bin, where the count represents the number of query tokens |
  |                       | whose frequency falls into that bin. Frequency is measured by number of occurrences in the training data.  |
  |                       |                                                                                                            |
  |                       | Settings:                                                                                                  |
  |                       |                                                                                                            |
  |                       | Number of bins.                                                                                            |
  |                       |                                                                                                            |
  |                       | Example:                                                                                                   |
  |                       |                                                                                                            |
  |                       | ``{'bins': 5}``                                                                                            |
  |                       |  - quantizes the vocabulary frequency into 5 bins                                                          |
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | ``'enable-stemming'`` | Stemming is the process of reducing inflected words to their word stem or base form. For example, word stem|
  |                       | of "eating" is "eat", word stem of "backwards" is "backward". MindMeld extracts word stems using a variant |
  |                       | of the `Porter stemming algorithm <https://tartarus.org/martin/PorterStemmer/>`_ that only removes         |
  |                       | inflectional suffixes.                                                                                     |
  |                       |                                                                                                            |
  |                       | This feature extends the ``'bag-of-words'`` and ``'freq'`` features described above.                       |
  |                       |                                                                                                            |
  |                       | If this flag is set to ``True``:                                                                           |
  |                       |                                                                                                            |
  |                       | - The stemmed versions of the n-grams are extracted from the query in addition to regular n-grams when     |
  |                       |   using the ``'bag-of-words'`` feature                                                                     |
  |                       |                                                                                                            |
  |                       | - Frequency counts for both unstemmed as well as stemmed versions of the query tokens are computed when    |
  |                       |   using the ``'freq'`` feature                                                                             |
  |                       |                                                                                                            |
  |                       | Example:                                                                                                   |
  |                       |                                                                                                            |
  |                       |  .. code-block:: python                                                                                    |
  |                       |                                                                                                            |
  |                       |    'features': {                                                                                           |
  |                       |        'bag-of-words': {'lengths': [1]},                                                                   |
  |                       |        'enable-stemming': True                                                                             |
  |                       |     }                                                                                                      |
  |                       |                                                                                                            |
  |                       |  - extracts ["two", “orders", "of", "breadsticks", **"order"**, **"breadstick"**] from the query “two      |
  |                       |    orders of breadsticks”.                                                                                 |
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | ``'word-shape'``      | Generates word shapes of n-grams of the specified lengths from the query text. Word shapes are simplified  |
  |                       | representations which encode attributes such as capitalization, numerals, punctuation etc.                 |
  |                       | Currently, we only encode whether a character is a digit or not.                                           |
  |                       |                                                                                                            |
  |                       | Settings:                                                                                                  |
  |                       |                                                                                                            |
  |                       | A list of n-gram lengths to extract.                                                                       |
  |                       |                                                                                                            |
  |                       | Examples:                                                                                                  |
  |                       |                                                                                                            |
  |                       | ``{'lengths': [1]}``                                                                                       |
  |                       |  - only extracts word shapes of individual tokens (unigrams)                                               |
  |                       |                                                                                                            |
  |                       | ``{'lengths': [1, 2, 3]}``                                                                                 |
  |                       |  - extracts word shapes of unigrams, bigrams and trigrams                                                  |
  |                       |                                                                                                            |
  |                       | Given the query "i want 12":                                                                               |
  |                       |                                                                                                            |
  |                       | ``{'lengths': [1]}``                                                                                       |
  |                       |  - extracts "x", "xxxx", and "dd"                                                                          |
  |                       |                                                                                                            |
  |                       | ``{'lengths': [1, 2]}``                                                                                    |
  |                       |  - extracts "x", "xxxx", "dd", "x xxxx", and "xxxx dd"                                                     |
  |                       |                                                                                                            |
  |                       | Note:                                                                                                      |
  |                       |                                                                                                            |
  |                       | - Shapes of words which are all digits or non-digits and have more than 5 characters are collapsed to      |
  |                       |   `ddddd+` and `xxxxx+` respectively.                                                                      |
  |                       | - Feature value for each shape is its log-scaled count.                                                    |
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | ``'edge-ngrams'``     | Generates n-grams of the specified lengths from the edges (that is, the start and the end) of the query.   |
  |                       |                                                                                                            |
  |                       | Settings:                                                                                                  |
  |                       |                                                                                                            |
  |                       | A list of n-gram lengths to extract.                                                                       |
  |                       |                                                                                                            |
  |                       | Examples:                                                                                                  |
  |                       |                                                                                                            |
  |                       | ``{'lengths': [1]}``                                                                                       |
  |                       |  - only extracts the first and last word                                                                   |
  |                       |                                                                                                            |
  |                       | ``{'lengths': [1, 2, 3]}``                                                                                 |
  |                       |  - extracts all leading and trailing n-grams up to size 3                                                  |
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | ``'char-ngrams'``     | Generates character n-grams of specified lengths from the query text.                                      |
  |                       |                                                                                                            |
  |                       | Examples:                                                                                                  |
  |                       |                                                                                                            |
  |                       | ``{'lengths': [1]}``                                                                                       |
  |                       |  - extracts each character in the query (unigrams)                                                         |
  |                       |                                                                                                            |
  |                       | ``{'lengths': [1, 2, 3]}``                                                                                 |
  |                       |  - extracts character unigrams, bigrams and trigrams                                                       |
  |                       |                                                                                                            |
  |                       | Given the query "hi there":                                                                                |
  |                       |                                                                                                            |
  |                       | ``{'lengths': [1]}``                                                                                       |
  |                       |  - extracts 'h', 'i', ' ', t', 'h', 'e', 'r', and 'e'                                                      |
  |                       |                                                                                                            |
  |                       | ``{'lengths': [1, 2]}``                                                                                    |
  |                       |  - extracts  'h', 'i', ' ', 't', 'h', 'e', 'r', 'e', 'hi', 'i ', ' t', 'th', 'he', 'er', and 're'          |
  |                       |                                                                                                            |
  |                       | Additionally, you can also limit the character n-grams considered while extracting the feature by setting  |
  |                       | a threshold on their frequency. These frequencies are computed over the entire training set. This prevents |
  |                       | infrequent n-grams from being used as features. By default, this frequency is set to 0.                    |
  |                       |                                                                                                            |
  |                       | Examples:                                                                                                  |
  |                       |                                                                                                            |
  |                       |  .. code-block:: python                                                                                    |
  |                       |                                                                                                            |
  |                       |    {                                                                                                       |
  |                       |      'lengths':[2, 3],                                                                                     |
  |                       |      'thresholds': [5, 8]                                                                                  |
  |                       |    }                                                                                                       |
  |                       |                                                                                                            |
  |                       |  - extracts all character bigrams whose frequency in the training set is 5 or greater and all character    |
  |                       |    trigrams whose frequency is 8 or greater.                                                               |
  |                       |                                                                                                            |
  |                       |  .. code-block:: python                                                                                    |
  |                       |                                                                                                            |
  |                       |    {                                                                                                       |
  |                       |      'lengths':[1, 3],                                                                                     |
  |                       |      'thresholds': [8]                                                                                     |
  |                       |    }                                                                                                       |
  |                       |                                                                                                            |
  |                       |  - extracts all character unigrams whose frequency in the training set is 8 or greater and all character   |
  |                       |    trigrams.                                                                                               |
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | ``'sys-candidates'``  | Generates a set of features indicating the presence of system entities in the query.                       |
  |                       |                                                                                                            |
  |                       | Settings:                                                                                                  |
  |                       |                                                                                                            |
  |                       | The types of system entities to extract. If unspecified, all system entities will be considered by default.|
  |                       |                                                                                                            |
  |                       | Example:                                                                                                   |
  |                       |                                                                                                            |
  |                       | ``{'entities': ['sys_number', 'sys_time', 'sys_phone-number']}``                                           |
  |                       |  - extracts features indicating the presence of the above system entities                                  |
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | ``'in-gaz'``          | Generates a set of features indicating the presence of query n-grams in different entity gazetteers,       |
  |                       | along with popularity information as defined in the gazetteer.                                             |
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | ``'length'``          | Generates a set of features that capture query length information.                                         |
  |                       | Computes the number of tokens and characters in the query, on both linear and log scales.                  |
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | ``'exact'``           | Returns the entire query text as a feature.                                                                |
  +-----------------------+------------------------------------------------------------------------------------------------------------+

.. _intent_tuning:

3. **Hyperparameter Settings**

``'params'`` (:class:`dict`)
  |

  A dictionary of values to be used for model hyperparameters during training. Examples include the ``'kernel'`` parameter for SVM, the ``'penalty'`` parameter for logistic regression, ``'max_depth'`` for decision tree, and so on. The list of allowable hyperparameters depends on the model selected. See the :ref:`reference links <sklearn_intent_models>` above for parameter lists.

``'param_selection'`` (:class:`dict`)
  |

  A dictionary of settings for :sk_guide:`hyperparameter selection <grid_search>`. Provides an alternative to the ``'params'`` dictionary above if the ideal hyperparameters for the model are not already known and need to be estimated.

  To estimate parameters, MindMeld needs two pieces of information from the developer:

  #. The parameter space to search, as the value for the ``'grid'`` key
  #. The strategy for splitting the labeled data into training and validation sets, as the value for the ``'type'`` key

  Depending on the splitting scheme selected, the :data:`param_selection` dictionary can contain other keys that define additional settings. The table below enumerates the allowable keys.

  +-----------------------+---------------------------------------------------------------------------------------------------------------------------+
  | Key                   | Value                                                                                                                     |
  +=======================+===========================================================================================================================+
  | ``'grid'``            | A dictionary which maps each hyperparameter to a list of potential values to search.                                      |
  |                       | Here is an example for a :sk_api:`logistic regression <sklearn.linear_model.LogisticRegression>` model:                   |
  |                       |                                                                                                                           |
  |                       | .. code-block:: python                                                                                                    |
  |                       |                                                                                                                           |
  |                       |    {                                                                                                                      |
  |                       |      'penalty': ['l1', 'l2'],                                                                                             |
  |                       |      'C': [10, 100, 1000, 10000, 100000],                                                                                 |
  |                       |       'fit_intercept': [True, False]                                                                                      |
  |                       |    }                                                                                                                      |
  |                       |                                                                                                                           |
  |                       | See the :ref:`reference links <sklearn_intent_models>` above for details on the hyperparameters available for each model. |
  +-----------------------+---------------------------------------------------------------------------------------------------------------------------+
  | ``'type'``            | The :sk_guide:`cross-validation <cross_validation>` methodology to use. One of:                                           |
  |                       |                                                                                                                           |
  |                       | - ``'k-fold'``: :sk_api:`K-folds <sklearn.model_selection.KFold>`                                                         |
  |                       | - ``'shuffle'``: :sk_api:`Randomized folds <sklearn.model_selection.ShuffleSplit>`                                        |
  |                       | - ``'group-k-fold'``: :sk_api:`K-folds with non-overlapping groups <sklearn.model_selection.GroupKFold>`                  |
  |                       | - ``'group-shuffle'``: :sk_api:`Group-aware randomized folds <sklearn.model_selection.GroupShuffleSplit>`                 |
  |                       | - ``'stratified-k-fold'``: :sk_api:`Stratified k-folds <sklearn.model_selection.StratifiedKFold>`                         |
  |                       | - ``'stratified-shuffle'``: :sk_api:`Stratified randomized folds <sklearn.model_selection.StratifiedShuffleSplit>`        |
  |                       |                                                                                                                           |
  +-----------------------+---------------------------------------------------------------------------------------------------------------------------+
  | ``'k'``               | Number of folds (splits)                                                                                                  |
  +-----------------------+---------------------------------------------------------------------------------------------------------------------------+

  To identify the parameters that give the highest accuracy, the :meth:`fit` method does an :sk_guide:`exhaustive grid search <grid_search.html#exhaustive-grid-search>` over the parameter space, evaluating candidate models using the specified cross-validation strategy. Subsequent calls to :meth:`fit` can use these optimal parameters and skip the parameter selection process.

4. **Custom Train/Test Settings**

``'train_label_set'`` (:class:`str`)
  |

  A string representing a regex pattern that selects all training files for intent model training with filenames that match the pattern. The default regex when this key is not specified is ``'train.*\.txt'``.

``'test_label_set'`` (:class:`str`)
  |

  A string representing a regex pattern that selects all evaluation files for intent model testing with filenames that match the pattern. The default regex when this key is not specified is ``'test.*\.txt'``.


.. _build_intent_with_config:

Training with custom configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To override MindMeld’s default intent classifier configuration with custom settings, you can either edit the app configuration file, or, you can call the :meth:`fit` method with appropriate arguments.


1. Application configuration file
"""""""""""""""""""""""""""""""""

When you define custom classifier settings in  ``config.py``, the :meth:`IntentClassifier.fit` and :meth:`NaturalLanguageProcessor.build` methods use those settings instead of MindMeld’s defaults. To do this, define a dictionary of your custom settings, named :data:`INTENT_CLASSIFIER_CONFIG`.

Here's an example of a ``config.py`` file where custom settings optimized for the app override the preset configuration for the intent classifier.

.. code-block:: python

   INTENT_CLASSIFIER_CONFIG = {
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

Settings defined in :data:`INTENT_CLASSIFIER_CONFIG` apply to intent classifiers across all domains in your application. For finer-grained control, you can implement the :meth:`get_intent_classifier_config` function in ``config.py`` to specify suitable configurations for each domain. This gives you the flexibility to modify models and features based on the domain.

.. code-block:: python

   import copy

   def get_intent_classifier_config(domain):
       SPECIAL_CONFIG = copy.deepcopy(INTENT_CLASSIFIER_CONFIG)
       if domain == 'smart_home':
           SPECIAL_CONFIG['features']['bag-of-words']['lengths'] = [2, 3]
       elif domain == 'greeting':
           SPECIAL_CONFIG['params']['C'] = 100
       return SPECIAL_CONFIG

Using ``config.py`` is recommended for storing your optimal classifier settings once you have identified them through experimentation. Then the classifier training methods will use the optimized configuration to rebuild the models. A common use case is retraining models on newly-acquired training data, without retuning the underlying model settings.

Since this method requires updating a file each time you modify a setting, it’s less suitable for rapid prototyping than the method described next.


2. Arguments to the :meth:`fit` method
""""""""""""""""""""""""""""""""""""""

For experimenting with an intent classifier, the recommended method is to use arguments to the :meth:`fit` method. The main areas for exploration are feature extraction, hyperparameter tuning, and model selection.


**Feature extraction**

Let’s start with the baseline classifier we trained :ref:`earlier <baseline_intent_fit>`. Viewing the feature set reveals that, by default, the classifier just uses a bag of words (unigrams) for features.

.. code-block:: python

   my_features = ic.config.features
   my_features

.. code-block:: console

   {
    'bag-of-words': {'lengths': [1, 2]},
    'edge-ngrams': {'lengths': [1, 2]},
    'exact': {'scaling': 10},
    'freq': {'bins': 5},
    'in-gaz': {},
    'length': {}
   }

Now we want the classifier to look at longer phrases, which carry more context than unigrams. Change the ``'lengths'`` setting of the ``'bag-of-words'`` feature to extract longer n-grams. For this example, to extract single words (unigrams), bigrams, and trigrams, we’ll edit the :data:`my_features` dictionary as shown below.

.. code-block:: python

   my_features['bag-of-words']['lengths'] = [1, 2, 3]

We can also add more :ref:`supported features <intent_features>`. Suppose that our intents are such that the natural language patterns at the start or the end of a query can be highly indicative of one intent or another. To capture this, we extract the leading and trailing phrases of different lengths — known as *edge n-grams* — from the query. The code below adds the new ``'edge-ngrams'`` feature to the existing :data:`my_features` dictionary.

If ``'edge-ngrams'`` feature already exists in :data:`my_features` dictionary this will update the feature value.

.. code-block:: python

   my_features['edge-ngrams'] = { 'lengths': [1, 2, 3] }
   my_features

.. code-block:: console

   {
    'bag-of-words': {'lengths': [1, 2, 3]},
    'edge-ngrams': {'lengths': [1, 2, 3]},
    'freq': {'bins': 5},
    'in-gaz': {},
    'length': {}
   }

To retrain the classifier with the updated feature set, pass in the :data:`my_features` dictionary as an argument to the :data:`features` parameter of the :meth:`fit` method.  This trains the intent classification model with our new feature extraction settings, while continuing to use MindMeld defaults for model type (logistic regression) and hyperparameter selection.

.. code-block:: python

   ic.fit(features=my_features)

.. code-block:: console

   Fitting intent classifier: domain='times_and_dates'
   Selecting hyperparameters using k-fold cross-validation with 10 splits
   Best accuracy: 97.83%, params: {'C': 100, 'class_weight': {0: 1.9123333333333332, 1: 1.0464506172839507, 2: 0.77702169625246553, 3: 0.67848200312989049, 4: 0.989031339031339, 5: 2.6710784313725489}, 'fit_intercept': False}

The exact accuracy number and the selected params might be different each time we run hyperparameter tuning, which we will explore in detail in the next section.

**Hyperparameter tuning**

View the model’s :ref:`hyperparameters <intent_tuning>`, keeping in mind the hyperparameters for logistic regression, the default model in MindMeld. These include: ``'C'``, the inverse of regularization strength; and, penalization, which is not shown in the response but defaults to ``'l2'``.

.. code-block:: python

   my_param_settings = ic.config.param_selection
   my_param_settings

.. code-block:: console

   {
    'grid': {
              'C': [0.01, 1, 100, 10000, 1000000],
              'class_weight': [ ... ],
              'fit_intercept': [True, False]
            },
    'k': 5,
    'type': 'k-fold'
   }

Instead of relying on default preset values, let’s reduce the range of values to search for ``'C'``, and allow the hyperparameter estimation process to choose the ideal norm (``'l1'`` or ``'l2'``) for penalization. Pass the updated settings to :meth:`fit` as arguments to the :data:`param_selection` parameter. The :meth:`fit` method then searches over the updated parameter grid, and prints the hyperparameter values for the model whose cross-validation accuracy is highest.

.. code-block:: python

   my_param_settings['grid']['C'] = [0.01, 1, 100]
   my_param_settings['grid']['penalty'] = ['l1', 'l2']
   my_param_settings

.. code-block:: console

   {
    'grid': {
              'C': [10, 100, 1000],
              'class_weight': [ ... ],
              'fit_intercept': [True, False],
              'penalty': ['l1', 'l2']
            },
    'k': 5,
    'type': 'k-fold'
   }

.. code-block:: python

   ic.fit(param_selection=my_param_settings)


.. code-block:: console

   Fitting intent classifier: domain='times_and_dates'
   Selecting hyperparameters using k-fold cross-validation with 5 splits
   Best accuracy: 97.97%, params: {'C': 100, 'class_weight': {0: 2.3033333333333332, 1: 1.066358024691358, 2: 0.68145956607495073, 3: 0.54068857589984354, 4: 0.98433048433048431, 5: 3.3872549019607843}, 'fit_intercept': False, 'penalty': 'l1'}

Finally, we’ll try a new cross-validation strategy of randomized folds instead of the 5-fold cross-validation currently specified in the config. To do this, we modify the value of the ``'type'`` key in :data:`my_param_settings`:

.. code-block:: python

   my_param_settings['type'] = 'shuffle'
   my_param_settings


.. code-block:: console

   {
    'grid': {
              'C': [10, 100, 1000],
              'class_weight': [ ... ],
              'fit_intercept': [True, False],
              'penalty': ['l1', 'l2']
            },
    'k': 5,
    'type': 'shuffle'
   }

.. code-block:: python

   ic.fit(param_selection=my_param_settings)

.. code-block:: console

   Fitting intent classifier: domain='times_and_dates'
   Selecting hyperparameters using shuffle cross-validation with 5 splits
   Best accuracy: 97.70%, params: {'C': 100, 'class_weight': {0: 2.3033333333333332, 1: 1.066358024691358, 2: 0.68145956607495073, 3: 0.54068857589984354, 4: 0.98433048433048431, 5: 3.3872549019607843}, 'fit_intercept': False, 'penalty': 'l2'}

For a list of configurable hyperparameters for each model, along with available cross-validation methods, see :ref:`hyperparameter settings <intent_tuning>`.

**Model selection**

To try :ref:`machine learning models <sklearn_intent_models>` other than the default of logistic regression, we specify the new model as the argument to ``model_settings``, then update the hyperparameter grid accordingly.

For example, a :sk_guide:`support vector machine (SVM) <svm>` with the same features as before, and parameter selection settings updated to search over the :sk_api:`SVM hyperparameters <sklearn.svm.SVC.html#sklearn.svm.SVC>`, looks like this:

.. code-block:: python

   my_param_settings['grid'] = {
    'C': [0.1, 0.5, 1, 5, 10, 50, 100, 1000, 5000],
    'kernel': ['linear', 'rbf', 'poly']
   }
   my_param_settings


.. code-block:: console

   {
    'grid': {
              'C': [0.1, 0.5, 1, 5, 10, 50, 100, 1000, 5000],
              'kernel': ['linear', 'rbf', 'poly']
            },
    'k': 5,
    'type': 'shuffle'
   }

.. code-block:: python

   ic.fit(model_settings={'classifier_type': 'svm'}, param_selection=my_param_settings)


.. code-block:: console

   Fitting intent classifier: domain='times_and_dates'
   Selecting hyperparameters using shuffle cross-validation with 5 splits
   Best accuracy: 97.41%, params: {'C': 1, 'kernel': 'linear'}

Meanwhile, a :sk_api:`random forest <sklearn.ensemble.RandomForestClassifier>` :sk_guide:`ensemble <ensemble>` classifier would look like this:

.. code-block:: python

   my_param_settings['grid'] = {
    'n_estimators': [5, 10, 15, 20],
    'criterion': ['gini', 'entropy'],
    'warm_start': [True, False]
   }
   ic.fit(model_settings={'classifier_type': 'rforest'}, param_selection=my_param_settings)

.. code-block:: console

   Fitting intent classifier: domain='times_and_dates'
   Selecting hyperparameters using shuffle cross-validation with 5 splits
   Best accuracy: 90.50%, params: {'criterion': 'gini', 'n_estimators': 15, 'warm_start': False}


Run the intent classifier
-------------------------

Run the trained intent classifier on a test query using the :meth:`IntentClassifier.predict` method. The :meth:`IntentClassifier.predict` method returns the label for the intent whose predicted probability is highest.

.. code-block:: python

   ic.predict('cancel my morning alarm')

.. code-block:: console

   'remove_alarm'

.. note::

   At runtime, the natural language processor's :meth:`process` method calls :meth:`IntentClassifier.predict` to classify the domain for an incoming query.

We want to know how confident our trained model is in its prediction. To view the predicted probability distribution over all possible intent labels, use the :meth:`IntentClassifier.predict_proba` method. This is useful both for experimenting with classifier settings and for debugging classifier performance.

The result is a list of tuples whose first element is the intent label and whose second element is the associated classification probability. These are ranked by intent, from most likely to least likely.

.. code-block:: python

   ic.predict_proba('cancel my alarm')

.. code-block:: console

   [
    ('remove_alarm', 0.80000000000000004),
    ('set_alarm', 0.20000000000000001),
    ('change_alarm', 0.0),
    ('check_alarm', 0.0),
    ('start_timer', 0.0),
    ('stop_timer', 0.0)]
   ]

An ideal classifier would assign a high probability to the expected (correct) class label for a test query, while assigning very low probabilities to incorrect labels.

The :meth:`predict` and :meth:`predict_proba` methods take one query at a time. Next, we’ll see how to test a trained model on a batch of labeled test queries.


Evaluate classifier performance
-------------------------------

Before you can evaluate the accuracy of your trained domain classifier, you must first create labeled test data and place it in your MindMeld project as described in the :ref:`Natural Language Processor <evaluate_nlp>` chapter.

Then, when you are ready, use the :meth:`IntentClassifier.evaluate` method, which

 - strips away all ground truth annotations from the test queries,
 - passes the resulting unlabeled queries to the trained intent classifier for prediction, and
 - compares the classifier’s output predictions against the ground truth labels to compute the model’s prediction accuracy.

In the example below, the model gets 339 out of 345 test queries correct, resulting in an accuracy of about 98.3%.

.. code-block:: python

   ic.evaluate()

.. code-block:: console

   Loading queries from file times_and_dates/change_alarm/test.txt
   Loading queries from file times_and_dates/check_alarm/test.txt
   Loading queries from file times_and_dates/remove_alarm/test.txt
   Loading queries from file times_and_dates/set_alarm/test.txt
   Loading queries from file times_and_dates/start_timer/test.txt
   Loading queries from file times_and_dates/stop_timer/test.txt
   <StandardModelEvaluation score: 98.26%, 339 of 345 examples correct>

The aggregate accuracy score we see above is only the beginning, because the :meth:`evaluate` method returns a rich object containing overall statistics, statistics by class, and a confusion matrix.

Print all the model performance statistics reported by the :meth:`evaluate` method:

.. code-block:: python

   eval = ic.evaluate()
   eval.print_stats()

.. code-block:: console

   Overall statistics:

      accuracy f1_weighted          tp          tn          fp          fn    f1_macro    f1_micro
         0.983       0.982         339        2064           6           6       0.942       0.983



   Statistics by class:

                 class      f_beta   precision      recall     support          tp          tn          fp          fn
          change_alarm       0.952       1.000       0.909          11          10         334           0           1
          remove_alarm       0.947       0.964       0.931          29          27         315           1           2
           check_alarm       0.974       1.000       0.950          20          19         325           0           1
             set_alarm       0.889       0.800       1.000           8           8         335           2           0
          specify_time       0.994       0.989       1.000         264         264          78           3           0
           start_timer       0.833       1.000       0.714           7           5         338           0           2
            stop_timer       1.000       1.000       1.000           6           6         339           0           0



   Confusion matrix:

                    change_ala..   remove_ala..   check_alar..      set_alarm   specify_ti..   start_time..     stop_timer
     change_ala..             10              1              0              0              0              0              0
     remove_ala..              0             27              0              0              2              0              0
     check_alar..              0              0             19              1              0              0              0
        set_alarm              0              0              0              8              0              0              0
     specify_ti..              0              0              0              0            264              0              0
     start_time..              0              0              0              1              1              5              0
       stop_timer              0              0              0              0              0              0              6


The :meth:`eval.get_stats()` method returns all the above statistics in a structured dictionary without printing them to the console.

Let’s decipher the statistics output by the :meth:`evaluate` method.

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

  - **Classes are balanced** — When the number of training examples in your intents are comparable and each intent is equally important, focusing on the accuracy metric is usually good enough.

  - **Classes are imbalanced** — In this case, it's important to take the f1 scores into account.

  - **All f1 and accuracy scores are low** — When intent classification is performing poorly across all intents, any of the following may be the problem: 1) You do not have enough training data for the model to learn; 2) you need to tune your model hyperparameters; 3) you need to reconsider your intent structure to ensure that queries in different intents have different natural language patterns — this may involve either combining or separating intents so that the resulting classes are easier for the classifier to distinguish.

  - **f1 weighted is higher than f1 macro** — This means that intents with fewer evaluation examples are performing poorly. Try adding more data to these intents or adding class weights to your hyperparameters.

  - **f1 macro is higher than f1 weighted** — This means that intents with more evaluation examples are performing poorly. Verify that the number of evaluation examples reflects the class distribution of your training examples.

  - **f1 micro is higher than f1 macro** — This means that some intents are being misclassified more often than others. Identify the problematic intents by checking the class-wise statistics below. Some intents may be too similar to others, or you may need to add more training data to some intents.

  - **Some classes are more important than others** — If some intents are more important than others for your use case, it is best to focus especially on the class-wise statistics described below.

**Class-wise Statistics**
  |

  Stats computed at a per-class level:

  ===========  ===
  class        Intent label
  f_beta       :sk_api:`F-beta score <sklearn.metrics.fbeta_score>`
  precision    `Precision <https://en.wikipedia.org/wiki/Precision_and_recall#Precision>`_
  recall       `Recall <https://en.wikipedia.org/wiki/Precision_and_recall#Recall>`_
  support      Number of test queries in this intent (based on ground truth)
  tp           Number of `true positives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  tn           Number of `true negatives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  fp           Number of `false positives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  fn           Number of `false negatives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  ===========  ===

**Confusion Matrix**
  |

  A `confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`_ where each row represents the number of instances in an actual class and each column represents the number of instances in a predicted class. This reveals whether the classifier tends to confuse two classes, i.e., mislabel one class as another. In the above example, the domain classifier wrongly classified four instances of ``check_alarm`` queries as ``set_alarm``, and another four as ``remove_alarm``.

Now we have a wealth of information about the performance of our classifier. Let’s go further and inspect the classifier’s predictions at the level of individual queries, to better understand error patterns.

View the classifier predictions for the entire test set using the :attr:`results` attribute of the returned :obj:`eval` object. Each result is an instance of the :class:`EvaluatedExample` class which contains information about the original input query, the expected ground truth label, the predicted label, and the predicted probability distribution over all the class labels.

.. code-block:: python

   eval.results

.. code-block:: console

   [
    EvaluatedExample(example=<Query 'change my 6 am alarm'>, expected='change_alarm', predicted='change_alarm', probas={'change_alarm': 0.40000000000000002, 'check_alarm': 0.0, 'remove_alarm': 0.26666666666666666, 'set_alarm': 0.33333333333333331, 'start_timer': 0.0, 'stop_timer': 0.0}, label_type='class'),
    EvaluatedExample(example=<Query 'change my 6 am alarm to 7 am'>, expected='change_alarm', predicted='change_alarm', probas={'change_alarm': 1.0, 'check_alarm': 0.0, 'remove_alarm': 0.0, 'set_alarm': 0.0, 'start_timer': 0.0, 'stop_timer': 0.0}, label_type='class'),
    ...
   ]

Next, we look selectively at just the correct or incorrect predictions.

.. code-block:: python

   list(eval.correct_results())

.. code-block:: console

   [
    EvaluatedExample(example=<Query 'change my 6 am alarm'>, expected='change_alarm', predicted='change_alarm', probas={'change_alarm': 0.40000000000000002, 'check_alarm': 0.0, 'remove_alarm': 0.26666666666666666, 'set_alarm': 0.33333333333333331, 'start_timer': 0.0, 'stop_timer': 0.0}, label_type='class'),
    EvaluatedExample(example=<Query 'change my 6 am alarm to 7 am'>, expected='change_alarm', predicted='change_alarm', probas={'change_alarm': 1.0, 'check_alarm': 0.0, 'remove_alarm': 0.0, 'set_alarm': 0.0, 'start_timer': 0.0, 'stop_timer': 0.0}, label_type='class'),
    ...
   ]

.. code-block:: python

   list(eval.incorrect_results())

.. code-block:: console

   [
    EvaluatedExample(example=<Query 'reschedule my 6 am alarm to tomorrow morning at 10'>, expected='change_alarm', predicted='set_alarm', probas={'change_alarm': 0.26666666666666666, 'check_alarm': 0.0, 'remove_alarm': 0.26666666666666666, 'set_alarm': 0.46666666666666667, 'start_timer': 0.0, 'stop_timer': 0.0}, label_type='class'),
    EvaluatedExample(example=<Query 'move my 6 am alarm to 3pm in the afternoon'>, expected='change_alarm', predicted='remove_alarm', probas={'change_alarm': 0.20000000000000001, 'check_alarm': 0.20000000000000001, 'remove_alarm': 0.33333333333333331, 'set_alarm': 0.066666666666666666, 'start_timer': 0.20000000000000001, 'stop_timer': 0.0}, label_type='class'),
    ...
   ]

Slicing and dicing these results for error analysis is easily done with `list comprehensions <https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions>`_.

A simple example of this is inspecting incorrect predictions for a particular intent. For the ``start_timer`` intent, we get:

.. code-block:: python

   [(r.example, r.probas) for r in eval.incorrect_results() if r.expected == 'start_timer']

.. code-block:: console

   [
    (<Query 'remind me in 1 hour'>,
     {
      'change_alarm': 0.0,
      'check_alarm': 0.066666666666666666,
      'remove_alarm': 0.066666666666666666,
      'set_alarm': 0.53333333333333333,
      'start_timer': 0.33333333333333331,
      'stop_timer': 0.0
     }
    )
   ]

In this case, only one test query from the ``start_timer`` intent got misclassified as ``set_alarm``. The correct label came in second, but lost by a significant margin in classification probability.

Next, we use a list comprehension to identify the kind of queries that the current training data might lack. To do this, we list all misclassified queries from a given intent, where the classifier’s confidence for the true label is very low. We’ll demonstrate this with the ``check_alarm`` intent and a confidence of <25%.

.. code-block:: python

   [(r.example, r.probas) for r in eval.incorrect_results() if r.expected == 'check_alarm' and r.probas['check_alarm'] < .25]

.. code-block:: console

   [
    ...
    (<Query 'did you set an alarm for 6 am'>,
     {
      'change_alarm': 0.0,
      'check_alarm': 0.066666666666666666,
      'remove_alarm': 0.0,
      'set_alarm': 0.80000000000000004,
      'start_timer': 0.13333333333333333,
      'stop_timer': 0.0
     }
  ),
    (<Query 'did you set an alarm to wake me up at 6 am'>,
     {
      'change_alarm': 0.0,
      'check_alarm': 0.066666666666666666,
      'remove_alarm': 0.0,
      'set_alarm': 0.80000000000000004,
      'start_timer': 0.13333333333333333,
      'stop_timer': 0.0
     }
    ),
    ...
   ]

The result reveals queries where the intent was misclassified as ``set_alarm``, and where the language pattern was some words followed the phrase "set an alarm" followed by more words. We'll call this the "... set an alarm ..." pattern.

Try looking for similar queries in the :doc:`training data <../blueprints/home_assistant>`. You should discover that the ``check_alarm`` intent does indeed lack labeled training queries that match the pattern. But the ``set_alarm`` intent has plenty of queries that fit. This explains why the model chose ``set_alarm`` over ``check_alarm`` when classifying such queries.

One potential solution is to add more training queries that match the "... set an alarm ..." pattern to the ``check_alarm`` intent. Then the classification model should more effectively learn to distinguish the two intents that it confused.

Error analysis on the results of the :meth:`evaluate` method can inform your experimentation and help in building better models. Augmenting training data should be the first step, as in the above example. Beyond that, you can experiment with different model types, features, and hyperparameters, as described :ref:`earlier <build_intent_with_config>` in this chapter.

Viewing features extracted for classification
---------------------------------------------

While training a new model or investigating a misclassification by the classifier, it is sometimes useful to view the extracted features to make sure they are as expected. For example, there may be non-ASCII characters in the query that are treated differently by the feature extractors. Or the value assigned to a particular feature may be computed differently than you expected. Not extracting the right features could lead to misclassifications. In the example below, we view the features extracted for the query 'set alarm for 7 am' using :meth:`IntentClassifier.view_extracted_features` method.

.. code:: python

   ic.view_extracted_features("set alarm for 7 am")

.. code-block:: console

   {'bag_of_words|edge:left|length:1|ngram:set': 1,
    'bag_of_words|edge:left|length:2|ngram:set alarm': 1,
    'bag_of_words|edge:right|length:1|ngram:am': 1,
    'bag_of_words|edge:right|length:2|ngram:#NUM am': 1,
    'bag_of_words|length:1|ngram:#NUM': 1,
    'bag_of_words|length:1|ngram:alarm': 1,
    'bag_of_words|length:1|ngram:am': 1,
    'bag_of_words|length:1|ngram:for': 1,
    'bag_of_words|length:1|ngram:set': 1,
    'bag_of_words|length:2|ngram:#NUM am': 1,
    'bag_of_words|length:2|ngram:alarm for': 1,
    'bag_of_words|length:2|ngram:for #NUM': 1,
    'bag_of_words|length:2|ngram:set alarm': 1,
    'exact|query:<OOV>': 10,
    'in_gaz|type:city|gaz_freq_bin:2': 0.2,
    'in_gaz|type:city|gaz_freq_bin:4': 0.2,
    'in_vocab:IV|freq_bin:0': 0.31699250014423125,
    'in_vocab:IV|freq_bin:1': 0.4,
    'in_vocab:IV|in_gaz|type:city|gaz_freq_bin:4': 0.2,
    'in_vocab:OOV|in_gaz|type:city|gaz_freq_bin:2': 0.2}

This is especially useful when you are writing :doc:`custom feature extractors <./custom_features>` to inspect whether the right features are being extracted.

Inspect features and their importance
-------------------------------------

Examining the learned feature weights of a machine-learned model can offer insights into its behavior. To analyze the prediction of the intent classifier on any query, you can inspect its features and their weights using :meth:`NaturalLanguageProcessor.inspect` method. In particular, it is useful to compare the computed feature values for the query for the predicted class and the expected ground truth (also called **gold**) class. Looking at the feature values closely can help in identifying the features that are useful, those that aren't, and even those that may be misleading or confusing for the model.

Here is an example of the results returned by :meth:`NaturalLanguageProcessor.inspect` method on the query "have i set an alarm to awaken me" with the expected gold intent ``check_alarm``. Focus on the 'Feature' and 'Diff' columns. The high negative value in the 'Diff' column  for the ngram 'set' indicates that its presence biases the decision of the classifier towards ``set_alarm`` intent over ``check_alarm``.  A possible solution is to add more training queries (like the example query) to the ``check_alarm`` intent, making the classifier rely on tokens like 'have' as well.

.. note::

    This section requires trained domain and intent models for the Home Assistant app. If you have not built them yet, run ``nlp.build()``. If you have already built and saved the models, do ``nlp.load()``.

.. code-block:: python

   nlp.inspect("have i set an alarm to awaken me", intent="check_alarm")

.. code-block:: console
   :emphasize-lines: 10

   Inspecting intent classification
                                                                                    Feature   Value Pred_W(set_alarm)     Pred_P Gold_W(check_alarm)     Gold_P       Diff
   bag_of_words|edge:left|length:1|ngram:have    bag_of_words|edge:left|length:1|ngram:have       1          [0.6906]   [0.6906]           [-0.4421]  [-0.4421]  [-1.1328]
   bag_of_words|edge:right|length:1|ngram:me      bag_of_words|edge:right|length:1|ngram:me       1         [-0.1648]  [-0.1648]           [-0.3431]  [-0.3431]  [-0.1782]
   bag_of_words|length:1|ngram:alarm                      bag_of_words|length:1|ngram:alarm       1          [1.6087]   [1.6087]            [1.5089]   [1.5089]  [-0.0997]
   bag_of_words|length:1|ngram:an                            bag_of_words|length:1|ngram:an       1          [1.6324]   [1.6324]            [0.2536]   [0.2536]  [-1.3788]
   bag_of_words|length:1|ngram:have                        bag_of_words|length:1|ngram:have       1         [-1.0182]  [-1.0182]            [1.3052]   [1.3052]   [2.3234]
   bag_of_words|length:1|ngram:i                              bag_of_words|length:1|ngram:i       1          [0.4271]   [0.4271]            [1.6761]   [1.6761]    [1.249]
   bag_of_words|length:1|ngram:me                            bag_of_words|length:1|ngram:me       1          [2.1782]   [2.1782]            [0.4724]   [0.4724]  [-1.7058]
   bag_of_words|length:1|ngram:set                          bag_of_words|length:1|ngram:set       1           [3.682]    [3.682]            [1.0064]   [1.0064]  [-2.6756]
   bag_of_words|length:1|ngram:to                            bag_of_words|length:1|ngram:to       1          [0.0281]   [0.0281]           [-0.8413]  [-0.8413]  [-0.8694]
   bag_of_words|length:2|ngram:alarm to                bag_of_words|length:2|ngram:alarm to       1         [-0.4646]  [-0.4646]           [-0.1883]  [-0.1883]   [0.2763]
   bag_of_words|length:2|ngram:an alarm                bag_of_words|length:2|ngram:an alarm       1          [1.1225]   [1.1225]            [0.3721]   [0.3721]  [-0.7504]
   bag_of_words|length:2|ngram:set an                    bag_of_words|length:2|ngram:set an       1         [-1.8094]  [-1.8094]            [0.0306]   [0.0306]     [1.84]
   exact|query:<OOV>                                                      exact|query:<OOV>      10         [-0.5906]  [-5.9056]           [-0.6247]  [-6.2467]  [-0.3411]
   in_gaz|type:city|gaz_freq_bin:1                          in_gaz|type:city|gaz_freq_bin:1  0.1981         [-0.6438]  [-0.1275]            [1.2285]   [0.2434]   [0.3709]
   in_gaz|type:city|gaz_freq_bin:3                          in_gaz|type:city|gaz_freq_bin:3   0.125         [-0.8062]  [-0.1008]           [-0.0586]  [-0.0073]   [0.0934]
   in_gaz|type:city|gaz_freq_bin:4                          in_gaz|type:city|gaz_freq_bin:4   0.125         [-0.1004]  [-0.0125]           [-0.6153]  [-0.0769]  [-0.0644]
   in_vocab:IV|freq_bin:0                                            in_vocab:IV|freq_bin:0   0.125         [-0.9523]   [-0.119]           [-0.5941]  [-0.0743]   [0.0448]
   in_vocab:IV|freq_bin:1                                            in_vocab:IV|freq_bin:1   0.125          [0.1404]   [0.0176]           [-0.4717]   [-0.059]  [-0.0765]
   in_vocab:IV|freq_bin:2                                            in_vocab:IV|freq_bin:2   0.125          [0.3538]   [0.0442]           [-0.7243]  [-0.0905]  [-0.1348]
   in_vocab:IV|freq_bin:3                                            in_vocab:IV|freq_bin:3  0.1981         [-0.4922]  [-0.0975]           [-0.5453]   [-0.108]  [-0.0105]
   in_vocab:IV|freq_bin:4                                            in_vocab:IV|freq_bin:4  0.1981         [-0.2612]  [-0.0517]           [-0.7934]  [-0.1572]  [-0.1055]
   in_vocab:IV|in_gaz|type:city|gaz_freq_bin:1  in_vocab:IV|in_gaz|type:city|gaz_freq_bin:1  0.1981         [-0.9942]   [-0.197]            [1.4016]   [0.2777]   [0.4746]
   in_vocab:IV|in_gaz|type:city|gaz_freq_bin:3  in_vocab:IV|in_gaz|type:city|gaz_freq_bin:3   0.125         [-0.8062]  [-0.1008]           [-0.0586]  [-0.0073]   [0.0934]
   in_vocab:IV|in_gaz|type:city|gaz_freq_bin:4  in_vocab:IV|in_gaz|type:city|gaz_freq_bin:4   0.125         [-0.1004]  [-0.0125]           [-0.6153]  [-0.0769]  [-0.0644]
   in_vocab:OOV                                                                in_vocab:OOV   0.125          [0.0209]   [0.0026]           [-0.2293]  [-0.0287]  [-0.0313]

You can combine both domain and intent inspection by passing both parameters into the function.

.. code-block:: python

   nlp.inspect("have i set an alarm to awaken me", domain="times_and_dates", intent="check_alarm")


The columns returned by the method are explained below:

========  ===
Feature   Name of the feature extracted from the query
Value     Value of the extracted feature
Pred_W    Feature weight from the co-efficient matrix for the predicted label
Pred_P    Product of the co-efficient and the feature value for the predicted label
Gold_W    Feature weight from the co-efficient matrix for the gold label
Gold_P    Product of the co-efficient and the feature value for the gold label
Diff      Difference between Gold_P and Pred_P
========  ===

Currently, feature inspection is only available for logistic regression models.

Save model for future use
-------------------------

Save the trained intent classifier for later use by calling the :meth:`IntentClassifier.dump` method. The :meth:`dump` method serializes the trained model as a `pickle file <https://docs.python.org/3/library/pickle.html>`_ and saves it to the specified location on disk.

.. code:: python

   ic.dump(model_path='experiments/intent_classifier.pkl')

.. code-block:: console

   Saving intent classifier: domain='times_and_dates'

You can load the saved model anytime using the :meth:`IntentClassifier.load` method.

.. code:: python

   ic.load(model_path='experiments/intent_classifier.pkl')

.. code-block:: console

   Loading intent classifier: domain='times_and_dates'
