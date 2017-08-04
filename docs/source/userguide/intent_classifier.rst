Working with the Intent Classifier
==================================

The :ref:`Intent Classifier <arch_intent_model>`

 - is run as the second step in the :ref:`natural language processing pipeline <arch_nlp>`
 - is a `text classification <https://en.wikipedia.org/wiki/Text_classification>`_ model that determines the target intent for a given query
 - is trained using all of the labeled queries across all the intents in a given domain

Every Workbench app has one intent classifier for every domain with multiple intents. The name of each intent folder serves as the label for the training queries contained within that folder.

See :doc:`Step 6 <../quickstart/06_generate_representative_training_data>` for more details on training data preparation.

.. note::

    This is an in-depth tutorial to work through from start to finish. Before you begin, read the :ref:`Step-by-Step Guide <quickstart>`, paying special attention to the :ref:`Intent Classification <intent_classification>` section.

Access an intent classifier
---------------------------

Before using any of the NLP components, you need to generate the necessary training data for your app. See :doc:`Step 6 <../quickstart/06_generate_representative_training_data>`. Once you have training data, import the :class:`NaturalLanguageProcessor` (NLP) class from the Workbench :mod:`nlp` module and :ref:`instantiate an object <instantiate_nlp>` with the path to your Workbench project.

.. code-block:: python

   >>> from mmworkbench.components.nlp import NaturalLanguageProcessor
   >>> nlp = NaturalLanguageProcessor(app_path='home_assistant')
   >>> nlp
   <NaturalLanguageProcessor 'home_assistant' ready: False, dirty: False>

Verify that the NLP has correctly identified all the domains for your app.

.. code-block:: python

   >>> nlp.domains
   {
    'smart_home': <DomainProcessor 'smart_home' ready: False, dirty: False>,
    'times_and_dates': <DomainProcessor 'times_and_dates' ready: False, dirty: False>,
    'unknown': <DomainProcessor 'unknown' ready: False, dirty: False>,
    'weather': <DomainProcessor 'weather' ready: False, dirty: False>
   }

Access the :class:`IntentClassifier` for a domain of your choice, using the :attr:`intent_classifier` attribute of the desired entity.

.. code-block:: python

   >>> # Intent classifier for the 'smart_home' domain:
   >>> ic = nlp.domains['smart_home'].intent_classifier
   >>> ic
   <IntentClassifier ready: False, dirty: False>
   ...
   >>> # Intent classifier for the 'weather' domain:
   >>> ic = nlp.domains['weather'].intent_classifier
   >>> ic
   <IntentClassifier ready: False, dirty: False>


Train an intent classifier
--------------------------

Use the :meth:`IntentClassifier.fit` method to train an intent classification model for a domain of your choice. Depending on the size of the training data, this can take anywhere from a few seconds to several minutes to finish. With logging level set to ``INFO`` or below, you should see the build progress in the console and the cross-validation accuracy of the trained model.

.. _baseline_intent_fit:

.. code-block:: python

   >>> from mmworkbench import configure_logs; configure_logs()
   >>> ic = nlp.domains['times_and_dates'].intent_classifier
   >>> ic.fit()
   Fitting intent classifier: domain='times_and_dates'
   No app configuration file found. Using default intent model configuration
   Loading queries from file times_and_dates/change_alarm/train.txt
   Loading queries from file times_and_dates/check_alarm/train.txt
   Loading queries from file times_and_dates/remove_alarm/train.txt
   Loading queries from file times_and_dates/set_alarm/train.txt
   Loading queries from file times_and_dates/start_timer/train.txt
   Loading queries from file times_and_dates/stop_timer/train.txt
   Selecting hyperparameters using k-fold cross-validation with 10 splits
   Best accuracy: 97.68%, params: {'C': 100, 'class_weight': {0: 2.3033333333333332, 1: 1.066358024691358, 2: 0.68145956607495073, 3: 0.54068857589984354, 4:    0.98433048433048431, 5: 3.3872549019607843}, 'fit_intercept': True}


The :meth:`fit` method loads all the necessary training queries and trains an intent classification model. When called with no arguments (as in the example above), the method uses the settings from ``config.py``, the :ref:`app's configuration file <build_nlp_with_config>`. If ``config.py`` is not defined, the method uses the Workbench preset :ref:`classifier configuration <config>`.

Using default settings is the recommended (and quickest) way to get started with any of the NLP classifiers. The resulting baseline classifier should provide a reasonable starting point from which to bootstrap your machine learning experimentation. You can then try alternate settings as you seek to identify the optimal classifier configuration for your app.


Classifier configuration
^^^^^^^^^^^^^^^^^^^^^^^^

Use the :attr:`config` attribute of a trained classifier to view the :ref:`configuration <config>` that the classifier is using. Here’s an example where we view the configuration of a baseline intent classifier trained using default settings:

.. code-block:: python

   >>> ic.config.to_dict()
   {
    'features': {
      'bag-of-words': {'lengths': [1]},
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
    'params': None
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

  =============== =======================================================
  Classifier Type Description (with list of configurable hyperparameters)
  =============== =======================================================
  ``'logreg'``    :sk_guide:`Logistic regression <linear_model.html#logistic-regression>` (see :sk_api:`parameter list <sklearn.linear_model.LogisticRegression>`)
  ``'svm'``       :sk_guide:`Support vector machine <svm.html#svm-classification>` (see :sk_api:`parameter list <sklearn.svm.SVC>`)
  ``'dtree'``     :sk_guide:`Decision tree <tree.html#tree>` (see :sk_api:`parameter list <sklearn.tree.DecisionTreeClassifier>`)
  ``'rforest'``   :sk_guide:`Random forest <ensemble.html#forest>` (see :sk_api:`parameter list <sklearn.ensemble.RandomForestClassifier>`)
  =============== =======================================================


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
  |                       | A list of n-gram lengths to extract.                                                                       |
  |                       | For instance, ``{'lengths': [1]}`` only extracts words (unigrams), whereas ``{'lengths': [1, 2, 3]}``      |
  |                       | extracts unigrams, bigrams and trigrams.                                                                   |
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | ``'edge-ngrams'``     | Generates n-grams of the specified lengths from the edges (i.e., start and end) of the query.              |
  |                       |                                                                                                            |
  |                       | Settings:                                                                                                  |
  |                       | A list of n-gram lengths to extract.                                                                       |
  |                       | For instance, ``{'lengths': [1]}`` only extracts the first and last word,                                  |
  |                       | whereas ``{'lengths': [1, 2, 3]}`` extracts all leading and trailing n-grams up to size 3.                 |
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | ``'freq'``            | Generates a log-scaled count for each frequency bin, where the count represents the number of query tokens |
  |                       | whose frequency falls into that bin. Frequency is measured by number of occurrences in the training data.  |
  |                       |                                                                                                            |
  |                       | Settings:                                                                                                  |
  |                       | Number of bins.                                                                                            |
  |                       | For instance, ``{'bins': 5}`` quantizes the vocabulary frequency into 5 bins.                              |
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

  A dictionary of values to be used for model hyperparameters during training. Examples include the ``'kernel'`` parameter for SVM, the ``'penalty'`` parameter for logistic regression, ``'max_depth'`` for decision tree, and so on. The list of allowable hyperparameters depends on the model selected. See the parameter list in :ref:`the model table <sklearn_intent_models>` above.

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
  |                       | :ref:`The model table <sklearn_intent_models>` above lists hyperparameters available for each supported model.    |
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

  To identify the parameters that give the highest accuracy, the :meth:`fit` method does an :sk_guide:`exhaustive grid search <grid_search.html#exhaustive-grid-search>` over the parameter space, evaluating candidate models using the specified cross-validation strategy. Subsequent calls to :meth:`fit` can use these optimal parameters and skip the parameter selection process.

.. _build_intent_with_config:

Training with custom configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To override Workbench’s default intent classifier configuration with custom settings, you can either edit the app configuration file, or, you can call the :meth:`fit` method with appropriate arguments.


1. Application configuration file
"""""""""""""""""""""""""""""""""

When you define custom classifier settings in  ``config.py``, the :meth:`IntentClassifier.fit` and :meth:`NaturalLanguageProcessor.build` methods use those settings instead of Workbench’s defaults. To do this, define a dictionary of your custom settings, named :data:`INTENT_MODEL_CONFIG`.

Here's an example of a ``config.py`` file where custom settings optimized for the app override the preset configuration for the intent classifier.

.. code-block:: python

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

This method is recommended for storing your optimal classifier settings once you have identified them through experimentation. Then the classifier training methods will use the optimized configuration to rebuild the models. A common use case is retraining models on newly-acquired training data, without retuning the underlying model settings.

Since this method requires updating a file each time you modify a setting, it’s less suitable for rapid prototyping than the method described next.


2. Arguments to the :meth:`fit` method
""""""""""""""""""""""""""""""""""""""

For experimenting with an intent classifier, the recommended method is to use arguments to the :meth:`fit` method. The main areas for exploration are feature extraction, hyperparameter tuning, and model selection.


**Feature extraction**

Let’s start with the baseline classifier we trained :ref:`earlier <baseline_intent_fit>`. Viewing the feature set reveals that, by default, the classifier just uses a bag of words (unigrams) for features.

.. code-block:: python

   >>> my_features = ic.config.features
   >>> my_features
   {
    'bag-of-words': {'lengths': [1]},
    'freq': {'bins': 5},
    'in-gaz': {},
    'length': {}
   }

Now we want the classifier to look at longer phrases, which carry more context than unigrams. Change the ``'lengths'`` setting of the ``'bag-of-words'`` feature to extract longer n-grams. For this example, to extract single words (unigrams), bigrams, and trigrams, we’ll edit the :data:`my_features` dictionary as shown below.

.. code-block:: python

   >>> my_features['bag-of-words']['lengths'] = [1, 2, 3]

We can also add more :ref:`supported features <intent_features>`. Suppose that our intents are such that the natural language patterns at the start or the end of a query can be highly indicative of one intent or another. To capture this, we extract the leading and trailing phrases of different lengths — known as *edge n-grams* — from the query. The code below adds the new ``'edge-ngrams'`` feature to the existing :data:`my_features` dictionary.

.. code-block:: python

   >>> my_features['edge-ngrams'] = { 'lengths': [1, 2] }
   >>> my_features
   {
    'bag-of-words': {'lengths': [1, 2, 3]},
    'edge-ngrams': {'lengths': [1, 2]},
    'freq': {'bins': 5},
    'in-gaz': {},
    'length': {}
   }

To retrain the classifier with the updated feature set, pass in the :data:`my_features` dictionary as an argument to the :data:`features` parameter of the :meth:`fit` method.  This trains the intent classification model with our new feature extraction settings, while continuing to use Workbench defaults for model type (logistic regression) and hyperparameter selection.

.. code-block:: python

   >>> ic.fit(features=my_features)
   Fitting intent classifier: domain='times_and_dates'
   No app configuration file found. Using default intent model configuration
   Selecting hyperparameters using k-fold cross-validation with 10 splits
   Best accuracy: 97.83%, params: {'C': 100, 'class_weight': {0: 1.9123333333333332, 1: 1.0464506172839507, 2: 0.77702169625246553, 3: 0.67848200312989049, 4: 0.989031339031339, 5: 2.6710784313725489}, 'fit_intercept': False}


**Hyperparameter tuning**

View the model’s :ref:`hyperparameters <intent_tuning>`, keeping in mind the hyperparameters for logistic regression, the default model in Workbench. These include: ``'C'``, the inverse of regularization strength; and, penalization, which is not shown in the response but defaults to ``'l2'``.

.. code-block:: python

   >>> my_param_settings = ic.config.param_selection
   >>> my_param_settings
   {
    'grid': {
              'C': [0.01, 1, 100, 10000, 1000000],
              'class_weight': [ ... ],
              'fit_intercept': [True, False]
            },
    'k': 10,
    'type': 'k-fold'
   }

Instead of relying on default preset values, let’s reduce the range of values to search for ``'C'``, and allow the hyperparameter estimation process to choose the ideal norm (``'l1'`` or ``'l2'``) for penalization. Pass the updated settings to :meth:`fit` as arguments to the :data:`param_selection` parameter. The :meth:`fit` method then searches over the updated parameter grid, and prints the hyperparameter values for the model whose cross-validation accuracy is highest.

.. code-block:: python

   >>> my_param_settings['grid']['C'] = [0.01, 1, 100]
   >>> my_param_settings['grid']['penalty'] = ['l1', 'l2']
   >>> my_param_settings
   {
    'grid': {
              'C': [10, 100, 1000],
              'class_weight': [ ... ],
              'fit_intercept': [True, False],
              'penalty': ['l1', 'l2']
            },
    'k': 10,
    'type': 'k-fold'
   }
   >>> ic.fit(param_selection=my_param_settings)
   Fitting intent classifier: domain='times_and_dates'
   No app configuration file found. Using default intent model configuration
   Selecting hyperparameters using k-fold cross-validation with 10 splits
   Best accuracy: 97.97%, params: {'C': 100, 'class_weight': {0: 2.3033333333333332, 1: 1.066358024691358, 2: 0.68145956607495073, 3: 0.54068857589984354, 4: 0.98433048433048431, 5: 3.3872549019607843}, 'fit_intercept': False, 'penalty': 'l1'}

Finally, we’ll override the default k-fold cross-validation, which is 10 folds, and specify five randomized folds instead. To so this, we modify the values of the ``'k'`` and ``'type'`` keys in :data:`my_param_settings`:

.. code-block:: python

   >>> my_param_settings['k'] = 5
   >>> my_param_settings['type'] = 'shuffle'
   >>> my_param_settings
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
   >>> ic.fit(param_selection=my_param_settings)
   Fitting intent classifier: domain='times_and_dates'
   No app configuration file found. Using default intent model configuration
   Selecting hyperparameters using shuffle cross-validation with 5 splits
   Best accuracy: 97.70%, params: {'C': 100, 'class_weight': {0: 2.3033333333333332, 1: 1.066358024691358, 2: 0.68145956607495073, 3: 0.54068857589984354, 4: 0.98433048433048431, 5: 3.3872549019607843}, 'fit_intercept': False, 'penalty': 'l2'}

For a list of configurable hyperparameters for each model, along with available cross-validation methods, see :ref:`hyperparameter settings <intent_tuning>`.

**Model selection**

To try :ref:`machine learning models <sklearn_intent_models>` other than the default of logistic regression, we specify the new model as the argument to ``model_settings``, then update the hyperparameter grid accordingly.

For example, a :sk_guide:`support vector machine (SVM) <svm>` with the same features as before, and parameter selection settings updated to search over the :sk_api:`SVM hyperparameters <sklearn.svm.SVC.html#sklearn.svm.SVC>`, looks like this:

.. code-block:: python

   >>> my_param_settings['grid'] = {
   ...  'C': [0.1, 0.5, 1, 5, 10, 50, 100, 1000, 5000],
   ...  'kernel': ['linear', 'rbf', 'poly'],
   ... }
   >>> my_param_settings
   {
    'grid': {
              'C': [0.1, 0.5, 1, 5, 10, 50, 100, 1000, 5000],
              'kernel': ['linear', 'rbf', 'poly']
            },
    'k': 5,
    'type': 'shuffle'
   }
   >>> ic.fit(model_settings={'classifier_type': 'svm'}, param_selection=my_param_settings)
   Fitting intent classifier: domain='times_and_dates'
   No app configuration file found. Using default intent model configuration
   Selecting hyperparameters using shuffle cross-validation with 5 splits
   Best accuracy: 97.41%, params: {'C': 1, 'kernel': 'linear'}

Meanwhile, a :sk_api:`random forest <sklearn.ensemble.RandomForestClassifier>` :sk_guide:`ensemble <ensemble>` classifier would look like this:

.. code-block:: python

   >>> my_param_settings['grid'] = {
   ...  'n_estimators': [5, 10, 15, 20],
   ...  'criterion': ['gini', 'entropy'],
   ...  'warm_start': [True, False]
   ... }
   >>> ic.fit(model_settings={'classifier_type': 'rforest'}, param_selection=my_param_settings)
   Fitting intent classifier: domain='times_and_dates'
   No app configuration file found. Using default intent model configuration
   Selecting hyperparameters using shuffle cross-validation with 5 splits
   Best accuracy: 90.50%, params: {'criterion': 'gini', 'n_estimators': 15, 'warm_start': False}


Run the intent classifier
-------------------------

Run the trained intent classifier on a test query using the :meth:`IntentClassifier.predict` method. The :meth:`IntentClassifier.predict` method returns the label for the intent whose predicted probability is highest.

.. code-block:: python

   >>> ic.predict('cancel my morning alarm')
   'remove_alarm'

.. note::

   At runtime, the natural language processor's :meth:`process` method calls :meth:`IntentClassifier.predict` to classify the domain for an incoming query.

We want to know how confident our trained model is in its prediction. To view the predicted probability distribution over all possible intent labels, use the :meth:`IntentClassifier.predict_proba` method. This is useful both for experimenting with classifier settings and for debugging classifier performance.

The result is a list of tuples whose first element is the intent label and whose second element is the associated classification probability. These are ranked by intent, from most likely to least likely.

.. code-block:: python

   >>> ic.predict_proba('cancel my alarm')
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

Before you can evaluate the accuracy of your trained domain classifier, you must first create labeled test data and place it in your Workbench project as described in the :ref:`Natural Language Processor <evaluate_nlp>` chapter.

Then, when you are ready, use the :meth:`IntentClassifier.evaluate` method, which

 - strips away all ground truth annotations from the test queries,
 - passes the resulting unlabeled queries to the trained intent classifier for prediction, and
 - compares the classifier’s output predictions against the ground truth labels to compute the model’s prediction accuracy.

In the example below, the model gets 63 out of 78 test queries correct, resulting in an accuracy of about 81%.

.. code-block:: python

   >>> ic.evaluate()
   Loading queries from file times_and_dates/change_alarm/test.txt
   Loading queries from file times_and_dates/check_alarm/test.txt
   Loading queries from file times_and_dates/remove_alarm/test.txt
   Loading queries from file times_and_dates/set_alarm/test.txt
   Loading queries from file times_and_dates/start_timer/test.txt
   Loading queries from file times_and_dates/stop_timer/test.txt
   <StandardModelEvaluation score: 80.77%, 63 of 78 examples correct>

The aggregate accuracy score we see above is only the beginning, because the :meth:`evaluate` method returns a rich object containing overall statistics, statistics by class, and a confusion matrix.

Print all the model performance statistics reported by the :meth:`evaluate` method:

.. code-block:: python

   >>> eval = ic.evaluate()
   >>> eval.print_stats()
   Overall Statistics:

       accuracy f1_weighted          TP          TN          FP          FN    f1_macro    f1_micro
          0.808       0.811          63         375          15          15       0.800       0.808



   Statistics by Class:

                  class      f_beta   precision      recall     support          TP          TN          FP          FN
           change_alarm       0.857       1.000       0.750           8           6          70           0           2
              set_alarm       0.667       0.500       1.000           8           8          62           8           0
           remove_alarm       0.871       0.818       0.931          29          27          43           6           2
            check_alarm       0.750       1.000       0.600          20          12          58           0           8
            start_timer       0.857       0.857       0.857           7           6          70           1           1
             stop_timer       0.800       1.000       0.667           6           4          72           0           2



   Confusion Matrix:

                     change_ala..      set_alarm   remove_ala..   check_alar..   start_time..
      change_ala..              6              1              1              0              0
         set_alarm              0              8              0              0              0
      remove_ala..              0              2             27              0              0
      check_alar..              0              4              4             12              0
      start_time..              0              1              0              0              6
        stop_timer              0              0              1              0              1

Let’s decipher the statistics output by the :meth:`evaluate` method.

**Overall Statistics**
  |

  Aggregate stats measured across the entire test set:

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

  - **Classes are balanced**: When the number of training examples in your intents are comparable and each intent is equally important, focusing on the accuracy metric is usually good enough.

  - **Classes are imbalanced**: When classes are imbalanced it is important to take the F1 scores into account.

  - **All F1 and accuracy scores are low**: Intent classification is performing poorly across all intents. You may not have enough training data for the model to learn or you may need to tune your model hyperparameters. You may also need to reconsider your intent structure and make sure queries in different intents have distinct natural language patterns. You may need to combine intents or separate them, so that the resulting classes are easier for the classifier to distinguish.

  - **F1 weighted is higher than F1 macro**: Your intents with fewer evaluation examples are performing poorly. You may need to add more data to intents that have fewer examples. You could also try adding class weights to your hyperparameters.

  - **F1 macro is higher than F1 weighted**: Your intents with more evaluation examples are performing poorly. Verify that the number of evaluation examples reflects the class distribution of your training examples.

  - **F1 micro is higher than F1 macro**: Certain intents are being misclassified more often than others. Check the class-wise below statistics to identify these intents. Some intents may be too similar to another intent or you may need to add more training data to some intents.

  - **Some classes are more important than others**: If some intents are more important than others for your use case, it is good to focus more on the class-wise statistics described below.

**Class-wise Statistics**
  |

  Stats computed at a per-class level:

  ===========  ===
  class        Intent label
  f_beta       :sk_api:`F-beta score <sklearn.metrics.fbeta_score>`
  precision    `Precision <https://en.wikipedia.org/wiki/Precision_and_recall#Precision>`_
  recall       `Recall <https://en.wikipedia.org/wiki/Precision_and_recall#Recall>`_
  support      Number of test queries in this intent (based on ground truth)
  TP           Number of `true positives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  TN           Number of `true negatives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  FP           Number of `false positives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  FN           Number of `false negatives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  ===========  ===

**Confusion Matrix**
  |

  A `confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`_ where each row represents the number of instances in an actual class and each column represents the number of instances in a predicted class. This reveals whether the classifier tends to confuse two classes, i.e., mislabel one class as another. In the above example, the domain classifier wrongly classified four instances of ``check_alarm`` queries as ``set_alarm``, and another four as ``remove_alarm``.

Now we have a wealth of information about the performance of our classifier. Let’s go further and inspect the classifier’s predictions at the level of individual queries, to better understand error patterns.

View the classifier predictions for the entire test set using the :attr:`results` attribute of the returned :obj:`eval` object. Each result is an instance of the :class:`EvaluatedExample` class which contains information about the original input query, the expected ground truth label, the predicted label, and the predicted probability distribution over all the class labels.

.. code-block:: python

   >>> eval.results
   [
    EvaluatedExample(example=<Query 'change my 6 am alarm'>, expected='change_alarm', predicted='change_alarm', probas={'change_alarm': 0.40000000000000002, 'check_alarm': 0.0, 'remove_alarm': 0.26666666666666666, 'set_alarm': 0.33333333333333331, 'start_timer': 0.0, 'stop_timer': 0.0}, label_type='class'),
    EvaluatedExample(example=<Query 'change my 6 am alarm to 7 am'>, expected='change_alarm', predicted='change_alarm', probas={'change_alarm': 1.0, 'check_alarm': 0.0, 'remove_alarm': 0.0, 'set_alarm': 0.0, 'start_timer': 0.0, 'stop_timer': 0.0}, label_type='class'),
    ...
   ]

Next, we look selectively at just the correct or incorrect predictions.

.. code-block:: python

   >>> list(eval.correct_results())
   [
    EvaluatedExample(example=<Query 'change my 6 am alarm'>, expected='change_alarm', predicted='change_alarm', probas={'change_alarm': 0.40000000000000002, 'check_alarm': 0.0, 'remove_alarm': 0.26666666666666666, 'set_alarm': 0.33333333333333331, 'start_timer': 0.0, 'stop_timer': 0.0}, label_type='class'),
    EvaluatedExample(example=<Query 'change my 6 am alarm to 7 am'>, expected='change_alarm', predicted='change_alarm', probas={'change_alarm': 1.0, 'check_alarm': 0.0, 'remove_alarm': 0.0, 'set_alarm': 0.0, 'start_timer': 0.0, 'stop_timer': 0.0}, label_type='class'),
    ...
   ]
   >>> list(eval.incorrect_results())
   [
    EvaluatedExample(example=<Query 'reschedule my 6 am alarm to tomorrow morning at 10'>, expected='change_alarm', predicted='set_alarm', probas={'change_alarm': 0.26666666666666666, 'check_alarm': 0.0, 'remove_alarm': 0.26666666666666666, 'set_alarm': 0.46666666666666667, 'start_timer': 0.0, 'stop_timer': 0.0}, label_type='class'),
    EvaluatedExample(example=<Query 'move my 6 am alarm to 3pm in the afternoon'>, expected='change_alarm', predicted='remove_alarm', probas={'change_alarm': 0.20000000000000001, 'check_alarm': 0.20000000000000001, 'remove_alarm': 0.33333333333333331, 'set_alarm': 0.066666666666666666, 'start_timer': 0.20000000000000001, 'stop_timer': 0.0}, label_type='class'),
    ...
   ]

Slicing and dicing these results for error analysis is easily done with `list comprehensions <https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions>`_.

A simple example of this is inspecting incorrect predictions for a particular intent. For the ``start_timer`` intent, we get:

.. code-block:: python

   >>> [(r.example, r.probas) for r in eval.incorrect_results() if r.expected == 'start_timer']
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

   >>> [(r.example, r.probas) for r in eval.incorrect_results()
   ... if r.expected == 'check_alarm' and r.probas['check_alarm'] < .25]
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


Save model for future use
-------------------------

Save the trained intent classifier for later use by calling the :meth:`IntentClassifier.dump` method. The :meth:`dump` method serializes the trained model as a `pickle file <https://docs.python.org/3/library/pickle.html>`_ and saves it to the specified location on disk.

.. code:: python

   >>> ic.dump(model_path='experiments/intent_classifier.rforest.20170701.pkl')
   Saving intent classifier: domain='times_and_dates'

You can load the saved model anytime using the :meth:`IntentClassifier.load` method.

.. code:: python

   >>> ic.load(model_path='experiments/intent_classifier.rforest.20170701.pkl')
   Loading intent classifier: domain='times_and_dates'
