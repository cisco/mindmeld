.. meta::
    :scope: private

Intent Classifier
=================

The Intent Classifier is run as the second step in the natural language processing pipeline to determine the target intent for a given query. It is a `text classification <https://en.wikipedia.org/wiki/Text_classification>`_ model that is trained using all of the labeled queries across all the intents in a given domain. The name of each intent folder serves as the label for the training queries contained within that folder. See :doc:`Step 6 <../quickstart/06_generate_representative_training_data>` for more details on training data preparation. Intent classification models are trained per domain. A Workbench app hence has one intent classifier for every domain with multiple intents.

.. note::

   For a quick introduction, refer to :ref:`Step 7 <intent_classification>` of the Step-By-Step Guide.
   
   Recommended prior reading: :doc:`Natural Language Processor <nlp>` chapter of the User Guide.


Access an intent classifier
---------------------------

Before using any of the NLP componenets, you need to generate the necessary training data for your app by following the guidelines in :doc:`Step 6 <../quickstart/06_generate_representative_training_data>`. You can then start by :ref:`instantiating an object <instantiate_nlp>` of the :class:`NaturalLanguageProcessor` (NLP) class.

.. code-block:: python

   >>> from mmworkbench.components.nlp import NaturalLanguageProcessor
   >>> nlp = NaturalLanguageProcessor(app_path='home_assistant')
   >>> nlp
   <NaturalLanguageProcessor 'home_assistant' ready: False, dirty: False>

Next, verify that the NLP has correctly identified all the domains for your app.

.. code-block:: python

   >>> nlp.domains
   {
    'smart_home': <DomainProcessor 'smart_home' ready: False, dirty: False>,
    'times_and_dates': <DomainProcessor 'times_and_dates' ready: False, dirty: False>,
    'unknown': <DomainProcessor 'unknown' ready: False, dirty: False>,
    'weather': <DomainProcessor 'weather' ready: False, dirty: False>
   }

Each domain has its own :class:`IntentClassifier` which can be accessed using the :attr:`intent_classifier` attribute of the corresponding domain.

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

To train an intent classification model for a specific domain, use the :meth:`IntentClassifier.fit` method. Depending on the size of the training data, this can take anywhere from a few seconds to several minutes to finish. If the logging level is set to ``INFO`` or below, you should see the build progress in the console and the cross-validation accuracy of the trained model.

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


The :meth:`fit` method loads all the necessary training queries and trains an intent classification model using the provided machine learning settings. When the method is called without any parameters (as in the example above), it uses the settings from the :ref:`app's configuration file <build_nlp_with_config>` (``config.py``), if defined, or Workbench's preset :ref:`classifier configuration <config>`.

The quickest and recommended way to get started with any of the NLP classifiers is by using Workbench's default settings. The resulting baseline classifier should provide a reasonable starting point to bootstrap your machine learning experimentation from. You can then experiment with alternate settings to identify the optimal classifier configuration for your app.


Classifier configuration
^^^^^^^^^^^^^^^^^^^^^^^^

To view the current :ref:`configuration <config>` being used by a trained classifier, use its :attr:`config` attribute. For example, here is the configuration being used by a baseline intent classifier trained using Workbench's default settings.

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

  Is always ``'text'``, since an intent classifier is a `text classification <https://en.wikipedia.org/wiki/Text_classification>`_ model.

``'model_settings'`` (:class:`dict`)
  |

  Is always a dictionary with a single key called ``'classifier_type'``. The value of the key specifies the machine learning model to use. Allowed values are 

.. _sklearn_models:

  =============== =======================================================
  Classifier Type Description (with list of configurable hyperparameters)
  =============== =======================================================
  ``'logreg'``    :sk_guide:`Logistic regression <linear_model.html#logistic-regression>` (See :sk_api:`parameter list <sklearn.linear_model.LogisticRegression>`)
  ``'svm'``       :sk_guide:`Support vector machine <svm.html#svm-classification>` (See :sk_api:`parameter list <sklearn.svm.SVC>`)
  ``'dtree'``     :sk_guide:`Decision tree <tree.html#tree>` (See :sk_api:`parameter list <sklearn.tree.DecisionTreeClassifier>`)
  ``'rforest'``   :sk_guide:`Random forest <ensemble.html#forest>` (See :sk_api:`parameter list <sklearn.ensemble.RandomForestClassifier>`)
  =============== =======================================================


2. **Feature Extraction Settings** 

``'features'`` (:class:`dict`)
  |

  Is a dictionary where the keys are the names of the feature groups to be extracted. The corresponding values are dictionaries representing the feature extraction settings for each group. The table below enumerates the features that can be used for intent classification.

.. _intent_features:

  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | Group Name            | Description                                                                                                |
  +=======================+============================================================================================================+
  | ``'bag-of-words'``    | Generates n-grams of the specified lengths from the query text.                                            |
  |                       |                                                                                                            |
  |                       | Supported settings:                                                                                        |
  |                       | A list containing the different n-gram lengths to extract.                                                 |
  |                       | E.g., ``{'lengths': [1]}`` only extracts words (unigrams), whereas ``{'lengths': [1, 2, 3]}`` extracts     |
  |                       | unigrams, bigrams and trigrams.                                                                            |
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | ``'edge-ngrams'``     | Generates n-grams of the specified lengths from the edges (i.e. the start and the end) of the query.       |
  |                       |                                                                                                            |
  |                       | Supported settings:                                                                                        |
  |                       | A list containing the different n-gram lengths to extract.                                                 |
  |                       | E.g., ``{'lengths': [1]}`` only extracts the first and last word, whereas ``{'lengths': [1, 2, 3]}``       |
  |                       | extracts all leading and trailing n-grams up to size 3.                                                    |
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | ``'freq'``            | Generates a log-scaled count for each frequency bin, where the count represents the number of query tokens |
  |                       | whose frequency (as measured by number of occurrences in the training data) falls into that bin.           |
  |                       |                                                                                                            |
  |                       | Supported settings:                                                                                        |
  |                       | Number of bins to quantize the vocabulary frequency into.                                                  |
  |                       | E.g., ``{'bins': 5}`` quantizes the vocabulary frequency into 5 bins.                                      |
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | ``'in-gaz'``          | Generates a set of features indicating the presence of query n-grams in different entity gazetteers,       |
  |                       | along with popularity information (as defined in the gazetteer).                                           |
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | ``'length'``          | Generates a set of features that capture query length information. Computes the number of tokens and       |
  |                       | characters in the query, on both linear and log scales.                                                    | 
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | ``'exact'``           | Returns the entire query text as a feature.                                                                |
  +-----------------------+------------------------------------------------------------------------------------------------------------+

.. _tuning:

3. **Hyperparameter Settings**

``'params'`` (:class:`dict`)
  |

  Is a dictionary containing the values to be used for different model hyperparameters during training. Examples include the ``'kernel'`` parameter for SVM, the ``'penalty'`` parameter for logistic regression, the ``'max_depth'`` parameter for decision tree, and so on. The list of allowable hyperparameters depends on the selected model. Refer to the parameter list in :ref:`the model table <sklearn_models>` above.

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
  |                       | :ref:`The model table <sklearn_models>` above lists the hyperparameters available for each supported model.       |
  +-----------------------+-------------------------------------------------------------------------------------------------------------------+
  | ``'type'``            | The :sk_guide:`cross-validation <cross_validation>` methodology to use. One of:                                   |
  |                       |                                                                                                                   |
  |                       | ========================= ===                                                                                     |
  |                       | ``'k-fold'``              :sk_api:`K-folds <sklearn.model_selection.KFold>`                                       |
  |                       | ``'shuffle'``             :sk_api:`Randomized folds <sklearn.model_selection.ShuffleSplit>`                       |
  |                       | ``'group-k-fold'``        :sk_api:`K-folds with non-overlapping groups <sklearn.model_selection.GroupKFold>`      |
  |                       | ``'group-shuffle'``       :sk_api:`Group-aware randomized folds <sklearn.model_selection.GroupShuffleSplit>`      |
  |                       | ``'stratified-k-fold'``   :sk_api:`Stratified k-folds <sklearn.model_selection.StratifiedKFold>`                  |
  |                       | ``'stratified-shuffle'``  :sk_api:`Stratified randomized folds <sklearn.model_selection.StratifiedShuffleSplit>`  |
  |                       | ========================= ===                                                                                     |
  +-----------------------+-------------------------------------------------------------------------------------------------------------------+
  | ``'k'``               | Number of folds (splits)                                                                                          |
  +-----------------------+-------------------------------------------------------------------------------------------------------------------+

  The :meth:`fit` method does an :sk_guide:`exhaustive grid search <grid_search.html#exhaustive-grid-search>` over the parameter space, evaluating candidate models using the specified cross-validation strategy, to identify the parameters that give the highest accuracy. The optimal parameters can then be used in future calls to :meth:`fit` to skip the parameter selection process.

.. _build_intent_with_config:

Training with custom configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are two ways to override Workbench's default intent classifier configuration with your custom settings.


1. Application configuration file
"""""""""""""""""""""""""""""""""

The first method, as described in the :ref:`NaturalLanguageProcessor <build_nlp_with_config>` chapter, is to define the classifier settings in your application configuration file, ``config.py``. Define a dictionary named :data:`INTENT_MODEL_CONFIG` containing your custom settings. The :meth:`IntentClassifier.fit` and :meth:`NaturalLanguageProcessor.build` methods will then use those settings instead of Workbench's defaults.

Here's an example of a ``config.py`` file where the preset configuration for the intent classifier is being overridden by custom settings that have been optimized for the app.

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

Since this method requires updating a file each time you want to modify a setting, it's less suitable for rapid prototyping than the second method described below. The recommended use for this functionality is to store your optimal classifier settings, once you have identified them via experimentation. This ensures that the classifier training methods will use the optimized configuration to rebuild the models in the future. A common use case is retraining models on newly acquired training data, without retuning the underlying model settings.


2. Arguments to the :meth:`fit` method
""""""""""""""""""""""""""""""""""""""

The recommended way to experiment with an intent classifier is by using arguments to the :meth:`fit` method.


**Feature extraction**

Let's start with the baseline classifier that was trained :ref:`above <baseline_intent_fit>`. Here's how you get the default feature set used by the classifer.

.. code-block:: python

   >>> my_features = ic.config.features
   >>> my_features
   {
    'bag-of-words': {'lengths': [1]},
    'freq': {'bins': 5},
    'in-gaz': {},
    'length': {}
   }

By default, the classifier only uses a bag of words (unigrams) as features. It may be useful to have the classifier look at longer phrases since they carry more context. To accomplish this, you need to change the ``'lengths'`` setting of the ``'bag-of-words'`` feature to extract longer n-grams. Suppose you want to extract single words (unigrams), bigrams and trigrams, the :data:`my_features` dictionary should be updated as shown below.

.. code-block:: python

   >>> my_features['bag-of-words']['lengths'] = [1, 2, 3]

You could also add other :ref:`supported features <intent_features>`. In some cases, the natural language patterns at the start or the end of a query can be highly indicative of of a certain intent. To capture this information, you can extract the leading and trailing phrases of different lengths, also called edge n-grams, from the query. The code below adds the new ``'edge-ngrams'`` feature to the existing :data:`my_features` dictionary.

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

To retrain the classifier with the updated feature set, pass in the :data:`my_features` dictionary as an argument to the :data:`features` parameter of the :meth:`fit` method. This trains the intent classification model using the provided feature extraction settings, while continuing to use Workbench's defaults for model type (logistic regression) and hyperparameter selection.

.. code-block:: python

   >>> ic.fit(features=my_features)
   Fitting intent classifier: domain='times_and_dates'
   No app configuration file found. Using default intent model configuration
   Selecting hyperparameters using k-fold cross-validation with 10 splits
   Best accuracy: 97.83%, params: {'C': 100, 'class_weight': {0: 1.9123333333333332, 1: 1.0464506172839507, 2: 0.77702169625246553, 3: 0.67848200312989049, 4: 0.989031339031339, 5: 2.6710784313725489}, 'fit_intercept': False}


**Hyperparameter tuning**

Next, let's experiment with the model's hyperparameters. To get the hyperparameter selection settings for the current classifier, do:

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

Let's reduce the range of values to search for the ``'C'`` parameter (inverse of regularization strength). Also, instead of always choosing an ``'l2'`` penalty by default, let's allow the hyperparameter estimation process to choose the ideal norm (``'l1'`` or ``'l2'``) for penalization. The updated settings can then be passed to :meth:`fit` as an argument to the :data:`param_selection` parameter.

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

The :meth:`fit` method now searches over the updated parameter grid and prints the hyperparameter values for the model with the highest cross-validation accuracy. By default, the intent classifier uses k-fold cross-validation with 10 folds. To use a different cross-validation strategy, you can modify the value for the ``'type'`` key in the :data:`my_param_settings`. For instance, to use five randomized folds:

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

For a full list of configurable hyperparameters for each model and available cross-validation methods, refer to the above section on defining :ref:`hyperparameter settings <tuning>`.


**Model selection**

Lastly, let's try other :ref:`machine learning models <sklearn_models>` in place of the default logistic regression. The hyperparameter grid needs to updated accordingly to be compatible with the selected model. Here's an example using a :sk_guide:`support vector machine (SVM) <svm>` with the same features as before, and the parameter selection settings updated to search over the :sk_api:`SVM hyperparameters <sklearn.svm.SVC.html#sklearn.svm.SVC>`.

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

Here's another example that trains a :sk_api:`random forest <sklearn.ensemble.RandomForestClassifier>` :sk_guide:`ensemble <ensemble>` classifier:

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

A trained intent classifier can be run on a test query using the :meth:`IntentClassifier.predict` method.

.. code-block:: python

   >>> ic.predict('cancel my morning alarm')
   'remove_alarm'   

The :meth:`predict` method returns the label for the intent with highest predicted probability. This method gets called by the natural language processor's :meth:`process` method at runtime to classify the intent for an incoming query.

When experimenting with different classifier settings or debugging classifier performance, it is often useful to inspect how confident a trained model is at predicting the right label. To view the predicted probability distribution over all the possible intent labels, use the :meth:`IntentClassifier.predict_proba` method.

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

The result of :meth:`predict_proba` is a list of tuples ranked from the most likely intent to the least. The first element of each tuple is the intent label and the second element is the associated classification probability. Ideally, you want a classifier that assigns a high probability to the expected (correct) class label for a test query, while having very low prediction probabilities for the incorrect labels.

The :meth:`predict` and :meth:`predict_proba` methods run on one query at a time. See the next section for details on evaluating classifier performance by testing a trained model on a batch of labeled test queries.


Evaluate classifier performance
-------------------------------

To evaluate the accuracy of your trained intent classifier, you first need to create labeled test data, as described in the :ref:`Natural Language Processor <evaluate_nlp>` chapter. Once you have the test data files in the right place in your Workbench project, you can measure your model's performance using the :meth:`IntentClassifier.evaluate` method.

.. code-block:: python

   >>> ic.evaluate()
   Loading queries from file times_and_dates/change_alarm/test.txt
   Loading queries from file times_and_dates/check_alarm/test.txt
   Loading queries from file times_and_dates/remove_alarm/test.txt
   Loading queries from file times_and_dates/set_alarm/test.txt
   Loading queries from file times_and_dates/start_timer/test.txt
   Loading queries from file times_and_dates/stop_timer/test.txt
   <StandardModelEvaluation score: 80.77%, 63 of 78 examples correct>

The :meth:`evaluate` method strips away all ground truth annotations from the test queries and passes in the resulting unlabeled queries to the trained intent classifier for prediction. The classifier's output predictions are then compared against the ground truth labels to compute the model's prediction accuracy. In the above example, the model got 63 out of 78 test queries correct, resulting in an accuracy of about 81%

The :meth:`evaluate` method returns a rich object that contains a lot more information over and above the aggregate accuracy score. The code below prints all the model performance statistics reported by the :meth:`evaluate` method.

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

The statistics are split into three sections.

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

  A `confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`_ with each row representing the number of instances in an actual class and each column representing the number of instances in a predicted class. It makes it easy to see if the classifier is frequently confusing two classes, i.e. commonly mislabelling one class as another. For instance, in the above example, the intent classifier has wrongly classified four instances of ``check_alarm`` queries as ``set_alarm``, and another four as ``remove_alarm``.

While these detailed statistics provide a wealth of information about the classifier performance, you might additionally also want to inspect the classifier's prediction on individual queries to better understand error patterns.

To view the classifier predictions for the entire test set, you can use the :attr:`results` attribute of the returned :obj:`eval` object.

.. code-block:: python

   >>> eval.results
   [
    EvaluatedExample(example=<Query 'change my 6 am alarm'>, expected='change_alarm', predicted='change_alarm', probas={'change_alarm': 0.40000000000000002, 'check_alarm': 0.0, 'remove_alarm': 0.26666666666666666, 'set_alarm': 0.33333333333333331, 'start_timer': 0.0, 'stop_timer': 0.0}, label_type='class'),
    EvaluatedExample(example=<Query 'change my 6 am alarm to 7 am'>, expected='change_alarm', predicted='change_alarm', probas={'change_alarm': 1.0, 'check_alarm': 0.0, 'remove_alarm': 0.0, 'set_alarm': 0.0, 'start_timer': 0.0, 'stop_timer': 0.0}, label_type='class'),
    ...
   ]

Each result is an instance of the :class:`EvaluatedExample` class which contains information about the original input query, the expected ground truth label, the predicted label, and the predicted probability distribution over all the class labels. You can also selectively look at just the correct predictions or the incorrect predictions. The code below shows how to do that.

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

`List comprehensions <https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions>`_ can be used to easily slice and dice the results for error analysis. For instance, to easily inspect all the incorrect predictions for a particular intent, say ``start_timer``, you could do:

.. code-block:: python

   >>>  [(r.example, r.probas) for r in eval.incorrect_results() if r.expected == 'start_timer']
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
 
In this case, there was just one test query from the ``start_timer`` intent that got misclassified as ``set_alarm``. You can also see that the correct label did come in second, but was still beaten by a decent margin in classification probability.

Here's an example listing all the misclassified queries from the ``check_alarm`` intent where the classifier's confidence for the true label was very low (<25%). These could often be indicative of the kind of queries that are lacking in the current training data.

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


In both of the cases above, the intent was misclassified as ``set_alarm``. On inspecting the :doc:`training data <../blueprints/home_assistant>`, you will find that the ``check_alarm`` intent indeed lacks labeled training queries like the ones above. On the other hand, these queries are very similar in language patterns ("... set an alarm ...") to the training data for the ``set_alarm`` intent. The model hence chose ``set_alarm`` over ``check_alarm`` when classifying them. This issue could potentially be solved by adding more relevant training queries to the ``check_alarm`` intent, so the classification model can better learn to distinguish between these two confusable intents.

Error analysis on the results of the :meth:`evaluate` method can thus inform your experimentation and help in building better models. In the example  above, adding more training data was proposed as a solution for improving accuracy. While training data augmentation should be your first step, you could also explore other techniques such as experimenting with different model types, features and hyperparameters, as described :ref:`earlier <build_intent_with_config>` in this chapter.


Save model for future use
-------------------------

A trained intent classifier can be saved for later use by calling the :meth:`IntentClassifier.dump` method. The :meth:`dump` method serializes the trained model as a `pickle file <https://docs.python.org/3/library/pickle.html>`_ and saves it to the specified location on disk.

.. code:: python

   >>> ic.dump(model_path='experiments/intent_classifier.rforest.20170701.pkl')
   Saving intent classifier: domain='times_and_dates'

The saved model can then be loaded anytime using the :meth:`IntentClassifier.load` method.

.. code:: python

   >>> ic.load(model_path='experiments/intent_classifier.rforest.20170701.pkl')
   Loading intent classifier: domain='times_and_dates'

