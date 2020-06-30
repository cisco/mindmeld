Working with the Domain Classifier
==================================

The Domain Classifier

 - is run as the first step in the :ref:`natural language processing pipeline <arch_nlp>`
 - is a `text classification <https://en.wikipedia.org/wiki/Text_classification>`_ model that determines the target domain for a given query
 - is trained using all of the labeled queries across all the domains in an application
 - can be trained only when the labeled data contains more than one domain

Every MindMeld app has exactly one domain classifier. The name of each domain folder serves as the label for the training queries contained within that folder.

.. note::

   - This is an in-depth tutorial to work through from start to finish. Before you begin, read the :ref:`Step-by-Step Guide <quickstart>`, paying special attention to the :ref:`Domain Classification <domain_classification>` section.
   - This section requires the :doc:`Home Assistant <../blueprints/home_assistant>` blueprint application. To get the app, open a terminal and run ``mindmeld blueprint home_assistant``.

Access the domain classifier
----------------------------

Working with any natural language processor component falls into two broad phases:

 - First, generate the training data for your app. App performance largely depends on having sufficient quantity and quality of training data. See :doc:`Step 6 <../quickstart/06_generate_representative_training_data>`.
 - Then, conduct experimentation in the Python shell.

When you are ready to begin experimenting, import the :class:`NaturalLanguageProcessor` class from the MindMeld :mod:`nlp` module and instantiate an object with the path to your MindMeld project.

.. code-block:: python

   from mindmeld.components.nlp import NaturalLanguageProcessor
   nlp = NaturalLanguageProcessor(app_path='home_assistant')
   nlp

.. code-block:: console

   <NaturalLanguageProcessor 'home_assistant' ready: False, dirty: False>

Access the :class:`DomainClassifier` using the :attr:`domain_classifier` attribute of the :class:`NaturalLanguageProcessor` class.

.. code-block:: python

   dc = nlp.domain_classifier
   dc

.. code-block:: console

  <DomainClassifier ready: False, dirty: False>


Train the domain classifier
---------------------------

Use the :meth:`DomainClassifier.fit` method to train a domain classification model. Depending on the size of the training data, this can take anywhere from a few seconds to several minutes. With logging level set to ``INFO`` or below, you should see the build progress in the console along with cross-validation accuracy for the classifier.

.. _baseline_domain_fit:

.. code-block:: python

   from mindmeld import configure_logs; configure_logs()
   dc.fit()

.. code-block:: console

   Fitting domain classifier
   No domain model configuration set. Using default.
   Loading queries from file smart_home/check_thermostat/train.txt
   Loading queries from file smart_home/close_door/train.txt
   Loading queries from file smart_home/lock_door/train.txt
   Loading queries from file smart_home/open_door/train.txt
   Loading queries from file smart_home/set_thermostat/train.txt
   Loading queries from file smart_home/turn_appliance_off/train.txt
   Loading queries from file smart_home/turn_appliance_on/train.txt
   Loading queries from file smart_home/turn_down_thermostat/train.txt
   Loading queries from file smart_home/turn_lights_off/train.txt
   Loading queries from file smart_home/turn_lights_on/train.txt
   Loading queries from file smart_home/turn_off_thermostat/train.txt
   Loading queries from file smart_home/turn_on_thermostat/train.txt
   Loading queries from file smart_home/turn_up_thermostat/train.txt
   Loading queries from file smart_home/unlock_door/train.txt
   Loading queries from file weather/check-weather/train.txt
   Loading queries from file times_and_dates/change_alarm/train.txt
   Loading queries from file times_and_dates/check_alarm/train.txt
   Loading queries from file times_and_dates/remove_alarm/train.txt
   Loading queries from file times_and_dates/set_alarm/train.txt
   Loading queries from file times_and_dates/start_timer/train.txt
   Loading queries from file times_and_dates/stop_timer/train.txt
   Loading queries from file unknown/unknown/training.txt
   Selecting hyperparameters using k-fold cross-validation with 10 splits
   Best accuracy: 99.50%, params: {'C': 10, 'fit_intercept': True}

The :meth:`fit` method loads all necessary training queries and trains a domain classification model. When called with no arguments (as in the example above), the method uses the settings from ``config.py``, the :ref:`app's configuration file <build_nlp_with_config>`. If no custom settings for domain classification are defined in ``config.py``, the method uses the MindMeld preset :ref:`classifier configuration <config>`.

Using default settings is the recommended (and quickest) way to get started with any of the NLP classifiers. The resulting baseline classifier should provide a reasonable starting point from which to bootstrap your machine learning experimentation. You can then try alternate settings as you seek to identify the optimal classifier configuration for your app.

Classifier configuration
^^^^^^^^^^^^^^^^^^^^^^^^

Use the :attr:`config` attribute of a trained classifier to view the :ref:`configuration <config>` that the classifier is using. Here's an example where we view the configuration of a baseline domain classifier trained using default settings:

.. code-block:: python

   dc.config.to_dict()

.. code-block:: console

   {
    'features': {
        'bag-of-words': {'lengths': [1]},
        'freq': {'bins': 5},
        'in-gaz': {}
    },
    'model_settings': {'classifier_type': 'logreg'},
    'model_type': 'text',
    'param_selection': {
        'grid': {
          'C': [10, 100, 1000, 10000, 100000],
          'fit_intercept': [True, False]
        },
        'k': 10,
        'type': 'k-fold'
    },
    'params': None,
    'train_label_set': 'train.*\.txt',
    'test_label_set': 'test.*\.txt'
   }

Let's take a look at the allowed values for each setting in a domain classifier configuration.

1. **Model Settings**

``'model_type'`` (:class:`str`)
  |

  Always ``'text'``, since the domain classifier is a `text classification <https://en.wikipedia.org/wiki/Text_classification>`_ model.

``'model_settings'`` (:class:`dict`)
  |

  Always a dictionary with the single key ``'classifier_type'``, whose value specifies the machine learning model to use. Allowed values are shown in the table below.

  .. _sklearn_domain_models:

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

  A dictionary whose keys are names of feature groups to extract. The corresponding values are dictionaries representing the feature extraction settings for each group. The table below enumerates the features that can be used for domain classification.

  .. _domain_features:

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

.. _domain_tuning:

3. **Hyperparameter Settings**

``'params'`` (:class:`dict`)
  |

  A dictionary of values to be used for model hyperparameters during training. Examples include the ``'kernel'`` parameter for SVM, ``'penalty'`` for logistic regression, ``'max_depth'`` for decision tree, and so on. The list of allowable hyperparameters depends on the model selected. See the :ref:`reference links <sklearn_domain_models>` above for parameter lists.

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
  |                       | See the :ref:`reference links <sklearn_domain_models>` above for details on the hyperparameters available for each model. |
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

  A string representing a regex pattern that selects all training files for domain model training with filenames that match the pattern. The default regex when this key is not specified is ``'train.*\.txt'``.

``'test_label_set'`` (:class:`str`)
  |

  A string representing a regex pattern that selects all evaluation files for domain model testing with filenames that match the pattern. The default regex when this key is not specified is ``'test.*\.txt'``.


.. _build_domain_with_config:

Training with custom configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To override MindMeld's default domain classifier configuration with custom settings, you can either edit the app configuration file, or, you can call the :meth:`fit` method with appropriate arguments.


1. Application configuration file
"""""""""""""""""""""""""""""""""

When you define custom classifier settings in ``config.py``, the :meth:`DomainClassifier.fit` and :meth:`NaturalLanguageProcessor.build` methods use those settings instead of MindMeld's defaults. To do this, define a dictionary of your custom settings, named :data:`DOMAIN_CLASSIFIER_CONFIG`.

Here's an example of a ``config.py`` file where custom settings optimized for the app override the preset configuration for the domain classifier.

.. code-block:: python

   DOMAIN_CLASSIFIER_CONFIG = {
       'model_type': 'text',
       'model_settings': {
           'classifier_type': 'logreg'
       },
       'param_selection': {
           'type': 'k-fold',
           'k': 10,
           'grid': {
               'fit_intercept': [True, False],
               'C': [10, 100, 1000, 10000, 100000]
           },
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

This method is recommended for storing your optimal classifier settings once you have identified them through experimentation. Then the classifier training methods will use the optimized configuration to rebuild the models. A common use case is retraining models on newly-acquired training data, without retuning the underlying model settings.

Since this method requires updating a file each time you modify a setting, it's less suitable for rapid prototyping than the method described next.

2. Arguments to the :meth:`fit` method
""""""""""""""""""""""""""""""""""""""

For experimenting with the domain classifier, the recommended method is to use arguments to the :meth:`fit` method. The main areas for exploration are feature extraction, hyperparameter tuning, and model selection.


**Feature extraction**

Let's start with the baseline classifier we trained :ref:`earlier <baseline_domain_fit>`. Viewing the feature set reveals that, by default, the classifier uses unigrams for its bag of words features.

.. code-block:: python

   my_features = dc.config.features
   my_features

.. code-block:: console

   {
    'bag-of-words': {'lengths': [1, 2]},
    'edge-ngrams': {'lengths': [1, 2]},
    'exact': {'scaling': 10},
    'freq': {'bins': 5},
    'gaz-freq': {},
    'in-gaz': {}}
   }

Now we want the classifier to look at longer phrases, which carry more context than unigrams. Change the ``'lengths'`` setting of the ``'bag-of-words'`` feature to extract longer n-grams. For this example, to extract single words (unigrams), bigrams, and trigrams, we'll edit the :data:`my_features` dictionary as shown below.

.. code-block:: python

   my_features['bag-of-words']['lengths'] = [1, 2, 3]

We can also add more :ref:`supported features <domain_features>`. Suppose that our domains are such that the natural language patterns at the start or the end of a query are highly indicative of one domain or another. To capture this, we extract the leading and trailing phrases of different lengths — known as *edge n-grams* — from the query. The code below adds the new ``'edge-ngrams'`` feature to the existing :data:`my_features` dictionary.

.. code-block:: python

   my_features['edge-ngrams'] = { 'lengths': [1, 2] }
   my_features

.. code-block:: console

   {
    'bag-of-words': {'lengths': [1, 2, 3]},
    'edge-ngrams': {'lengths': [1, 2]},
    'exact': {'scaling': 10},
    'freq': {'bins': 5},
    'gaz-freq': {},
    'in-gaz': {}
   }

To retrain the classifier with the updated feature set, pass in the :data:`my_features` dictionary as an argument to the :data:`features` parameter of the :meth:`fit` method. This trains the domain classification model with our new feature extraction settings, while continuing to use MindMeld defaults for model type (logistic regression) and hyperparameter selection.

.. code-block:: python

   dc.fit(features=my_features)

.. code-block:: console

   Fitting domain classifier
   Selecting hyperparameters using k-fold cross-validation with 10 splits
   Best accuracy: 99.60%, params: {'C': 10, 'fit_intercept': True}

   The exact accuracy number and the selected params might be different each time we run hyperparameter tuning, which we will explore in detail in the next section.

**Hyperparameter tuning**

View the model's hyperparameters, keeping in mind the hyperparameters for logistic regression, the default model for domain classification in MindMeld. These include: ``'C'``, the inverse of regularization strength; and, penalization, which is not shown in the response but defaults to ``'l2'``.

.. code-block:: python

   my_param_settings = dc.config.param_selection
   my_param_settings

.. code-block:: console

   {
    'grid': {
              'C': [10, 100, 1000, 10000, 100000],
              'fit_intercept': [True, False]
            },
    'k': 10,
    'type': 'k-fold'
   }

For our first experiment, let's reduce the range of values to search for ``'C'``, and allow the hyperparameter estimation process to choose the ideal norm (``'l1'`` or ``'l2'``) for penalization. Pass the updated settings to :meth:`fit` as arguments to the :data:`param_selection` parameter. The :meth:`fit` method then searches over the updated parameter grid, and prints the hyperparameter values for the model whose cross-validation accuracy is highest.

.. code-block:: python

   my_param_settings['grid']['C'] = [10, 100, 1000]
   my_param_settings['grid']['penalty'] = ['l1', 'l2']
   my_param_settings

.. code-block:: console

   {
    'grid': {
              'C': [10, 100, 1000],
              'fit_intercept': [True, False],
              'penalty': ['l1', 'l2']
            },
    'k': 10,
    'type': 'k-fold'
   }

.. code-block:: python

   dc.fit(param_selection=my_param_settings)

.. code-block:: console

   Fitting domain classifier
   Selecting hyperparameters using k-fold cross-validation with 10 splits
   Best accuracy: 99.59%, params: {'C': 1000, 'penalty': 'l2', 'fit_intercept': True}

Again, the exact accuracy number and the selected params might be different for a particular run.

Finally, we'll try a new cross-validation strategy of randomized folds, replacing the default of k-fold. We'll also specify five folds instead of the default of ten folds. To so this, we modify the values of the   ``'type'`` and ``'k'`` keys in :data:`my_param_settings`:

.. code-block:: python

   my_param_settings['k'] = 5
   my_param_settings['type'] = 'shuffle'
   my_param_settings

.. code-block:: console

   {
    'grid': {
              'C': [10, 100, 1000],
              'fit_intercept': [True, False],
              'penalty': ['l1', 'l2']
            },
    'k': 5,
    'type': 'shuffle'
   }

.. code-block:: python

   dc.fit(param_selection=my_param_settings)

.. code-block:: console

   Fitting domain classifier
   Selecting hyperparameters using shuffle cross-validation with 5 splits
   Best accuracy: 99.50%, params: {'C': 100, 'fit_intercept': False, 'penalty': 'l2'}

For a list of configurable hyperparameters for each model, along with available cross-validation methods, see :ref:`hyperparameter settings <domain_tuning>`.


**Model selection**

To try :ref:`machine learning models <sklearn_domain_models>` other than the default of logistic regression, we specify the new model as the argument to ``model_settings``, then update the hyperparameter grid accordingly.

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

   dc.fit(model_settings={'classifier_type': 'svm'}, param_selection=my_param_settings)

.. code-block:: console

   Fitting domain classifier
   Selecting hyperparameters using shuffle cross-validation with 5 splits
   Best accuracy: 99.56%, params: {'C': 1000, 'kernel': 'rbf'}

Meanwhile, a :sk_api:`random forest <sklearn.ensemble.RandomForestClassifier>` :sk_guide:`ensemble <ensemble>` classifier would look like this:

.. code-block:: python

   my_param_settings['grid'] = {
    'n_estimators': [5, 10, 15, 20],
    'criterion': ['gini', 'entropy'],
    'warm_start': [True, False]
   }
   dc.fit(model_settings={'classifier_type': 'rforest'}, param_selection=my_param_settings)

.. code-block:: console

  Fitting domain classifier
  Selecting hyperparameters using shuffle cross-validation with 5 splits
  Best accuracy: 98.37%, params: {'criterion': 'gini', 'n_estimators': 15, 'warm_start': False}


Run the domain classifier
-------------------------

Run the trained domain classifier on a test query using the :meth:`DomainClassifier.predict` method, which returns the label for the domain whose predicted probability is highest.

.. code-block:: python

   dc.predict('weather in san francisco?')

.. code-block:: console

   'weather'

.. note::

   At runtime, the natural language processor's :meth:`process` method calls :meth:`DomainClassifier.predict` to classify the domain for an incoming query.

We want to know how confident our trained model is in its prediction. To view the predicted probability distribution over all possible domain labels, use the :meth:`DomainClassifier.predict_proba` method. This is useful both for experimenting with classifier settings and for debugging classifier performance.

The result is a list of tuples whose first element is the domain label and whose second element is the associated classification probability. These are ranked by domain, from most likely to least likely.

.. code-block:: python

   dc.predict_proba('weather in san francisco?')

.. code-block:: console

   [
    ('weather', 0.6),
    ('smart_home', 0.05),
    ('unknown', 0.25),
    ('times_and_dates', 0.1),
    ('greeting', 0.1),
   ]

An ideal classifier would assign a high probability to the expected (correct) class label for a test query, while assigning very low probabilities to incorrect labels.

The :meth:`predict` and :meth:`predict_proba` methods take one query at a time. Next, we'll see how to test a trained model on a batch of labeled test queries.

Evaluate classifier performance
-------------------------------

Before you can evaluate the accuracy of your trained domain classifier, you must first create labeled test data and place it in your MindMeld project as described in the :ref:`Natural Language Processor <evaluate_nlp>` chapter.

Then, when you are ready, use the :meth:`DomainClassifier.evaluate` method, which

 - strips away all ground truth annotations from the test queries,
 - passes the resulting unlabeled queries to the trained domain classifier for prediction, and
 - compares the classifier's output predictions against the ground truth labels to compute the model's prediction accuracy.

In the example below, the model gets 2,550 out of 2,563 test queries correct, resulting in an accuracy of 99.5%.

.. code-block:: python

   dc.evaluate()

.. code-block:: console

   Loading queries from file times_and_dates/change_alarm/test.txt
   Loading queries from file times_and_dates/check_alarm/test.txt
   Loading queries from file times_and_dates/remove_alarm/test.txt
   Loading queries from file times_and_dates/set_alarm/test.txt
   Loading queries from file times_and_dates/start_timer/test.txt
   Loading queries from file times_and_dates/stop_timer/test.txt
   Loading queries from file unknown/unknown/test.txt
   Loading queries from file smart_home/check_thermostat/test.txt
   Loading queries from file smart_home/close_door/test.txt
   Loading queries from file smart_home/lock_door/test.txt
   Loading queries from file smart_home/open_door/test.txt
   Loading queries from file smart_home/set_thermostat/test.txt
   Loading queries from file smart_home/turn_appliance_off/test.txt
   Loading queries from file smart_home/turn_appliance_on/test.txt
   Loading queries from file smart_home/turn_down_thermostat/test.txt
   Loading queries from file smart_home/turn_lights_off/test.txt
   Loading queries from file smart_home/turn_lights_on/test.txt
   Loading queries from file smart_home/turn_off_thermostat/test.txt
   Loading queries from file smart_home/turn_on_thermostat/test.txt
   Loading queries from file smart_home/turn_up_thermostat/test.txt
   Loading queries from file smart_home/unlock_door/test.txt
   Loading queries from file weather/check-weather/test.txt
   <StandardModelEvaluation score: 99.49%, 2550 of 2563 examples correct>

The aggregate accuracy score we see above is only the beginning, because the :meth:`evaluate` method returns a rich object containing overall statistics, statistics by class, and a confusion matrix.

Print all the model performance statistics reported by the :meth:`evaluate` method:

.. code-block:: python

   eval = dc.evaluate()
   eval.print_stats()

.. code-block:: console

   Overall statistics:

      accuracy f1_weighted          tp          tn          fp          fn    f1_macro    f1_micro
         0.995       0.995        2550        7676          13          13       0.954       0.995



   Statistics by class:

                 class      f_beta   precision      recall     support          tp          tn          fp          fn
            smart_home       0.994       0.990       0.998        1074        1072        1478          11           2
               weather       0.825       1.000       0.703          37          26        2526           0          11
               unknown       1.000       1.000       1.000        1107        1107        1456           0           0
       times_and_dates       0.997       0.994       1.000         345         345        2216           2           0



   Confusion matrix:

                      smart_home        weather        unknown   times_and_..
       smart_home           1072              0              0              2
          weather             11             26              0              0
          unknown              0              0           1107              0
     times_and_..              0              0              0            345


The :meth:`eval.get_stats()` method returns all the above statistics in a structured dictionary without printing them to the console.

Let's decipher the statistics output by the :meth:`evaluate` method.

**Overall Statistics**
  |

  Aggregate stats measured across the entire test set:

  ===========  ===
  accuracy     :sk_guide:`Classification accuracy score <model_evaluation.html#accuracy-score>`
  f1_weighted  :sk_api:`Class-weighted average f1 score <sklearn.metrics.f1_score.html>` — to take class imbalance into account, weights the f1 scores by class support
  tp           Number of `true positives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  tn           Number of `true negatives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  fp           Number of `false positives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  fn           Number of `false negatives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  f1_macro     :sk_api:`Macro-averaged f1 score <sklearn.metrics.f1_score.html>` — the mean of f1 scores calculated by class
  f1_micro     :sk_api:`Micro-averaged f1 score <sklearn.metrics.f1_score.html>` — calculated with global precision and recall metrics
  ===========  ===

  When interpreting these statistics, consider whether your app and evaluation results fall into one of the cases below, and if so, apply the accompanying guideline. This list is basic, not exhaustive, but should get you started.

  - **Classes are balanced** — When the number of training examples in your domains are comparable and each domain is equally important, focusing on the accuracy metric is usually good enough.

  - **Classes are imbalanced** — In this case, it's important to take the f1 scores into account.

  - **All f1 and accuracy scores are low** — When domain classification is performing poorly across all domains, any of the following may be the problem: 1) You do not have enough training data for the model to learn, 2) you need to tune your model hyperparameters, or 3) you need to reconsider your domain structure to ensure that queries in different domain have different vocabularies — this may involve either combining or separating domains so that the resulting classes are easier for the classifier to distinguish.

  - **f1 weighted is higher than f1 macro** — This means that domains with fewer evaluation examples are performing poorly. Try adding more data to these domains or adding class weights to your hyperparameters.

  - **f1 macro is higher than f1 weighted** — This means that domains with more evaluation examples are performing poorly. Verify that the number of evaluation examples reflects the class distribution of your training examples.

  - **f1 micro is higher than f1 macro** — This means that some domains are being misclassified more often than others. Identify the problematic domains by checking the class-wise statistics below. It is possible that some domains are too similar to others, or that you need to add more training data to some domains.

  - **Some classes are more important than others** — If some domains are more important than others for your use case, it is best to focus especially on the class-wise statistics described below.

**Class-wise Statistics**
  |

  Stats computed at a per-class level:

  ===========  ===
  class        Domain label
  f_beta       :sk_api:`F-beta score <sklearn.metrics.fbeta_score>`
  precision    `Precision <https://en.wikipedia.org/wiki/Precision_and_recall#Precision>`_
  recall       `Recall <https://en.wikipedia.org/wiki/Precision_and_recall#Recall>`_
  support      Number of test queries in this domain (based on ground truth)
  tp           Number of `true positives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  tn           Number of `true negatives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  fp           Number of `false positives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  fn           Number of `false negatives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  ===========  ===

**Confusion Matrix**
  |

  A `confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`_ where each row represents the number of instances in an actual class and each column represents the number of instances in a predicted class. This reveals whether the classifier tends to confuse two classes, i.e., mislabel one class as another. In the above example, the domain classifier wrongly classified 32 instances of ``weather`` queries as ``smart_home``.

Now we have a wealth of information about the performance of our classifier. Let's go further and inspect the classifier's predictions at the level of individual queries, to better understand error patterns.

View the classifier predictions for the entire test set using the :attr:`results` attribute of the returned :obj:`eval` object. Each result is an instance of the :class:`EvaluatedExample` class which contains information about the original input query, the expected ground truth label, the predicted label, and the predicted probability distribution over all the class labels.

.. code-block:: python

   eval.results

.. code-block:: console

   [
    EvaluatedExample(example=<Query 'change my 6 am alarm'>, expected='times_and_dates', predicted='times_and_dates', probas={'smart_home': 0.050000000000000003, 'times_and_dates': 0.94999999999999996, 'unknown': 0.0, 'weather': 0.0}, label_type='class'),
    EvaluatedExample(example=<Query 'change my 6 am alarm to 7 am'>, expected='times_and_dates', predicted='times_and_dates', probas={'smart_home': 0.050000000000000003, 'times_and_dates': 0.94999999999999996, 'unknown': 0.0, 'weather': 0.0}, label_type='class')],
    ...
   ]

Next, we look selectively at just the correct or incorrect predictions.

.. code-block:: python

   list(eval.correct_results())

.. code-block:: console

   [
    EvaluatedExample(example=<Query 'change my 6 am alarm'>, expected='times_and_dates', predicted='times_and_dates', probas={'smart_home': 0.050000000000000003, 'times_and_dates': 0.94999999999999996, 'unknown': 0.0, 'weather': 0.0}, label_type='class'),
    EvaluatedExample(example=<Query 'change my 6 am alarm to 7 am'>, expected='times_and_dates', predicted='times_and_dates', probas={'smart_home': 0.050000000000000003, 'times_and_dates': 0.94999999999999996, 'unknown': 0.0, 'weather': 0.0}, label_type='class'),
    ...
   ]

.. code-block:: python

   list(eval.incorrect_results())

.. code-block:: console

   [
    EvaluatedExample(example=<Query 'stop my timers'>, expected='times_and_dates', predicted='smart_home', probas={'smart_home': 0.65000000000000002, 'times_and_dates': 0.29999999999999999, 'unknown': 0.050000000000000003, 'weather': 0.0}, label_type='class'),
    EvaluatedExample(example=<Query 'what is happening in germany right now?'>, expected='unknown', predicted='weather', probas={'smart_home': 0.14999999999999999, 'times_and_dates': 0.0, 'unknown': 0.40000000000000002, 'weather': 0.45000000000000001}, label_type='class'),
    ...
   ]

Slicing and dicing these results for error analysis is easily done with `list comprehensions <https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions>`_.

A simple example of this is inspecting incorrect predictions for a particular domain. For the ``times_and_dates`` domain, we get:

.. code-block:: python

   [(r.example, r.probas) for r in eval.incorrect_results() if r.expected == 'times_and_dates']


.. code-block:: console

   [
    (<Query 'stop my timers'>,
     {
       'smart_home': 0.65000000000000002,
       'times_and_dates': 0.29999999999999999,
       'unknown': 0.050000000000000003,
       'weather': 0.0
     }
    )
   ]

In this case, only one test query from the ``times_and_dates`` domain got misclassified as ``smart_home``. The correct label came in second, but lost by a significant margin in classification probability.

Next, we use a list comprehension to identify the kind of queries that the current training data might lack. To do this, we list all misclassified queries from a given domain, where the classifier's confidence for the true label is very low. We'll demonstrate this with the ``weather`` domain and a confidence of <25%.

.. code-block:: python

   [(r.example, r.probas) for r in eval.incorrect_results() if r.expected == 'weather' and r.probas['weather'] < .25]

.. code-block:: console

   [
    (<Query 'check temperature outside'>,
     {
      'smart_home': 0.84999999999999998,
      'times_and_dates': 0.0,
      'unknown': 0.0,
      'weather': 0.14999999999999999
     }
    ),
    (<Query 'check current temperature in chicago'>,
     {
      'smart_home': 0.84999999999999998,
      'times_and_dates': 0.050000000000000003,
      'unknown': 0.050000000000000003,
      'weather': 0.050000000000000003
     }
    ),
    ...
   ]

The result reveals queries where the domain was misclassified as ``smart_home``, and where the language pattern was the word "check" followed some words, then the word "temperature" and some more words. We'll call this the "check ... temperature ..." pattern.

Try looking for similar queries in the :doc:`training data <../blueprints/home_assistant>`. You should discover that the ``weather`` domain does indeed lack labeled training queries that fit the pattern. But the ``smart_home`` domain, and the ``check_thermostat`` intent in particular, has plenty of queries that fit. This explains why the model chose ``smart_home`` over ``weather`` when classifying such queries.

One potential solution is to add more training queries that fit the "check ... temperature ..." pattern to the ``weather`` domain. Then the classification model should more effectively learn to distinguish the two domains that it confused.

Error analysis on the results of the :meth:`evaluate` method can inform your experimentation and help in building better models. Augmenting training data based on what you find should be the first step, as in the above example. Beyond that, you can experiment with different model types, features, and hyperparameters, as described :ref:`earlier <build_domain_with_config>` in this chapter.

Viewing features extracted for classification
---------------------------------------------

While training a new model or investigating a misclassification by the classifier, it is sometimes useful to view the extracted features to make sure they are as expected. For example, there may be non-ASCII characters in the query that are treated differently by the feature extractors, or the value assigned to a particular feature may be computed differently than you expected. Not extracting the right features could lead to misclassifications. In the example below, we view the features extracted for the query 'set alarm for 7 am' using :meth:`DomainClassifier.view_extracted_features` method.

.. code:: python

   dc.view_extracted_features("set alarm for 7 am")

.. code-block:: console

   {'bag_of_words|length:1|ngram:set': 1,
    'bag_of_words|length:1|ngram:alarm': 1,
    'bag_of_words|length:1|ngram:for': 1,
    'bag_of_words|length:1|ngram:#NUM': 1,
    'bag_of_words|length:1|ngram:am': 1,
    'bag_of_words|length:2|ngram:set alarm': 1,
    'bag_of_words|length:2|ngram:alarm for': 1,
    'bag_of_words|length:2|ngram:for #NUM': 1,
    'bag_of_words|length:2|ngram:#NUM am': 1,
    'bag_of_words|edge:left|length:1|ngram:set': 1,
    'bag_of_words|edge:right|length:1|ngram:am': 1,
    'bag_of_words|edge:left|length:2|ngram:set alarm': 1,
    'bag_of_words|edge:right|length:2|ngram:#NUM am': 1,
    'exact|query:<OOV>': 10,
    'in_gaz|type:city|gaz_freq_bin:2': 0.2,
    'in_vocab:OOV|in_gaz|type:city|gaz_freq_bin:2': 0.2,
    'in_gaz|type:city|gaz_freq_bin:4': 0.2,
    'in_vocab:IV|in_gaz|type:city|gaz_freq_bin:4': 0.2,
    'in_vocab:IV|freq_bin:3': 0.4,
    'in_vocab:IV|freq_bin:2': 0.2,
    'in_vocab:IV|freq_bin:1': 0.2}

This is especially useful when you are writing :doc:`custom feature extractors <./custom_features>` to inspect whether the right features are being extracted.

Inspect features and their importance
-------------------------------------

Examining the learned feature weights of a machine-learned model can offer insights into its behavior. To analyze the prediction of the domain classifier on any query, you can inspect its features and their weights using :meth:`NaturalLanguageProcessor.inspect` method. In particular, it is useful to compare the computed feature values for the query for the predicted class and the expected ground truth (also called **gold**) class. Looking at the feature values closely can help in identifying the features that are useful, those that aren't, and even those that may be misleading or confusing for the model.

Let us examine the results of :meth:`NaturalLanguageProcessor.inspect` on the query "check temperature outside" with the gold domain ``weather``. Focus on the 'Feature' and 'Diff' columns. A high negative value in the 'Diff' column for the features representing the presence of a ``location`` entity and the word 'temperature' imply that they are major indicators to the classifier that the query belongs to the ``smart_home`` domain over the ``weather`` domain. This reinforces our hypothesis from the previous section that we lack labeled training queries that fit the "check ... temperature ..." pattern in the ``weather`` domain.

.. note::

   Model inspection is currently only available for logistic models, so the example below demonstrates the functionality using a domain classifier trained with a logistic regression model.

.. code-block:: python

   dc.fit(model_settings={'classifier_type':'logreg'})
   nlp.inspect("check temperature outside", domain="weather")

.. code-block:: console
   :emphasize-lines: 8,12

   Inspecting domain classification
                                                                                                 Feature   Value Pred_W(smart_home)     Pred_P Gold_W(weather)     Gold_P       Diff
   bag_of_words|edge:left|length:1|ngram:check               bag_of_words|edge:left|length:1|ngram:check       1          [-0.0664]  [-0.0664]        [0.2985]   [0.2985]   [0.3649]
   bag_of_words|edge:left|length:2|ngram:check tem...  bag_of_words|edge:left|length:2|ngram:check te...       1          [-0.0067]  [-0.0067]        [0.0212]   [0.0212]   [0.0279]
   bag_of_words|edge:right|length:1|ngram:outside         bag_of_words|edge:right|length:1|ngram:outside       1          [-0.9582]  [-0.9582]         [0.867]    [0.867]   [1.8252]
   bag_of_words|length:1|ngram:check                                   bag_of_words|length:1|ngram:check       1          [-0.0609]  [-0.0609]        [0.2225]   [0.2225]   [0.2833]
   bag_of_words|length:1|ngram:outside                               bag_of_words|length:1|ngram:outside       1          [-2.2307]  [-2.2307]        [2.1483]   [2.1483]   [4.3791]
   bag_of_words|length:1|ngram:temperature                       bag_of_words|length:1|ngram:temperature       1           [2.9699]   [2.9699]        [1.7425]   [1.7425]  [-1.2274]
   bag_of_words|length:2|ngram:check temperature           bag_of_words|length:2|ngram:check temperature       1          [-0.0067]  [-0.0067]        [0.0212]   [0.0212]   [0.0279]
   bag_of_words|length:2|ngram:temperature outside       bag_of_words|length:2|ngram:temperature outside       1          [-1.3214]  [-1.3214]        [1.0047]   [1.0047]   [2.3261]
   exact|query:<OOV>                                                                   exact|query:<OOV>      10           [0.1527]   [1.5268]       [-0.0955]  [-0.9554]  [-2.4821]
   in_gaz|type:location                                                             in_gaz|type:location       1           [2.4073]   [2.4073]       [-0.7694]  [-0.7694]  [-3.1766]
   in_gaz|type:location|gaz_freq_bin:0                               in_gaz|type:location|gaz_freq_bin:0  0.3333              [1.7]   [0.5667]       [-0.2367]  [-0.0789]  [-0.6456]
   in_gaz|type:location|pop                                                     in_gaz|type:location|pop       1            [1.817]    [1.817]       [-0.9516]  [-0.9516]  [-2.7686]
   in_gaz|type:location|ratio                                                 in_gaz|type:location|ratio    0.28            [1.657]    [0.464]       [-0.3005]  [-0.0841]  [-0.5481]
   in_gaz|type:location|ratio_pop                                         in_gaz|type:location|ratio_pop    0.28            [1.657]    [0.464]       [-0.3005]  [-0.0841]  [-0.5481]
   in_vocab:IV|freq_bin:3                                                         in_vocab:IV|freq_bin:3  0.3333            [1.164]    [0.388]       [-0.9314]  [-0.3105]  [-0.6985]
   in_vocab:IV|freq_bin:5                                                         in_vocab:IV|freq_bin:5  0.5283          [-1.2485]  [-0.6596]        [-1.434]  [-0.7576]   [-0.098]
   in_vocab:IV|in_gaz|type:location|gaz_freq_bin:0       in_vocab:IV|in_gaz|type:location|gaz_freq_bin:0  0.3333              [1.7]   [0.5667]       [-0.2367]  [-0.0789]  [-0.6456]

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

Save the trained domain classifier for later use by calling the :meth:`DomainClassifier.dump` method. The :meth:`dump` method serializes the trained model as a `pickle file <https://docs.python.org/3/library/pickle.html>`_ and saves it to the specified location on disk.

.. code:: python

   dc.dump(model_path='experiments/domain_classifier.pkl')

.. code-block:: console

   Saving domain classifier

You can load the saved model anytime using the :meth:`DomainClassifier.load` method.

.. code:: python

   dc.load(model_path='experiments/domain_classifier.pkl')

.. code-block:: console

   Loading domain classifier
