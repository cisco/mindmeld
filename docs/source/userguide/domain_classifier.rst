.. meta::
    :scope: private

Domain Classifier
=================

The Domain Classifier is run as the first step in the natural language processing pipeline to determine the target domain for a given query. It is a `text classification <https://en.wikipedia.org/wiki/Text_classification>`_ model that is trained using all of the labeled queries across all the domains in an application. The name of each domain folder serves as the label for the training queries contained within that folder. See :doc:`Step 6 <../quickstart/06_generate_representative_training_data>` for more details on training data preparation. A Workbench app has exactly one domain classifier which gets trained only when the labeled data contains more than one domain.

.. note::

   For a quick introduction, refer to :ref:`Step 7 <domain_classification>` of the Step-By-Step Guide.

   Recommended prior reading: :doc:`Natural Language Processor <nlp>` chapter of the User Guide.


Access the domain classifier
----------------------------

Before using any of the NLP componenets, you need to generate the necessary training data for your app by following the guidelines in :doc:`Step 6 <../quickstart/06_generate_representative_training_data>`. You can then start by :ref:`instantiating an object <instantiate_nlp>` of the :class:`NaturalLanguageProcessor` (NLP) class. 

.. code-block:: python

   >>> from mmworkbench.components.nlp import NaturalLanguageProcessor
   >>> nlp = NaturalLanguageProcessor(app_path='home_assistant')
   >>> nlp
   <NaturalLanguageProcessor 'home_assistant' ready: False, dirty: False>

The :class:`DomainClassifier` for the app can then be accessed using the :attr:`domain_classifier` attribute of the :class:`NaturalLanguageProcessor` class.

.. code-block:: python

  >>> dc = nlp.domain_classifier
  >>> dc
  <DomainClassifier ready: False, dirty: False>


Train the domain classifier
---------------------------

To train a domain classification model, use the :meth:`DomainClassifier.fit` method. Depending on the size of the training data, this can take anywhere from a few seconds to several minutes to finish. If the logging level is set to ``INFO`` or below, you should see the build progress in the console and the cross-validation accuracy of the trained model.

.. _baseline_domain_fit:

.. code-block:: python

   >>> from mmworkbench import configure_logs; configure_logs()
   >>> dc.fit()
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

The :meth:`fit` method loads all the necessary training queries and trains a domain classification model using the provided machine learning settings. When the method is called without any parameters (as in the example above), it uses the settings from the :ref:`app's configuration file <build_nlp_with_config>` (``config.py``), if defined, or Workbench's preset :ref:`classifier configuration <config>`.

The quickest and recommended way to get started with any of the NLP classifiers is by using Workbench's default settings. The resulting baseline classifier should provide a reasonable starting point to bootstrap your machine learning experimentation from. You can then experiment with alternate settings to identify the optimal classifier configuration for your app.


Classifier configuration
^^^^^^^^^^^^^^^^^^^^^^^^

To view the current :ref:`configuration <config>` being used by a trained classifier, use its :attr:`config` attribute. For example, here is the configuration being used by a baseline domain classifier trained using Workbench's default settings.

.. code-block:: python

   >>> dc.config.to_dict()
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
    'params': None
  }

Let's take a look at the allowed values for each setting in a domain classifier configuration.

1. **Model Settings** 

``'model_type'`` (:class:`str`)
  |

  Is always ``'text'``, since the domain classifier is a `text classification <https://en.wikipedia.org/wiki/Text_classification>`_ model.

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

  Is a dictionary where the keys are the names of the feature groups to be extracted. The corresponding values are dictionaries representing the feature extraction settings for each group. The table below enumerates the features that can be used for domain classification.

.. _domain_features:

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

.. _build_domain_with_config:

Training with custom configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are two ways to override Workbench's default domain classifier configuration with your custom settings.


1. Application configuration file
"""""""""""""""""""""""""""""""""

The first method, as described in the :ref:`NaturalLanguageProcessor <build_nlp_with_config>` chapter, is to define the classifier settings in your application configuration file, ``config.py``. Define a dictionary named :data:`DOMAIN_MODEL_CONFIG` containing your custom settings. The :meth:`DomainClassifier.fit` and :meth:`NaturalLanguageProcessor.build` methods will then use those settings instead of Workbench's defaults.

Here's an example of a ``config.py`` file where the preset configuration for the domain classifier is being overridden by custom settings that have been optimized for the app.

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

Since this method requires updating a file each time you want to modify a setting, it's less suitable for rapid prototyping than the second method described below. The recommended use for this functionality is to store your optimal classifier settings, once you have identified them via experimentation. This ensures that the classifier training methods will use the optimized configuration to rebuild the models in the future. A common use case is retraining models on newly acquired training data, without retuning the underlying model settings.


2. Arguments to the :meth:`fit` method
""""""""""""""""""""""""""""""""""""""

The recommended way to experiment with the domain classifier is by using arguments to the :meth:`fit` method.


**Feature extraction**

Let's start with the baseline classifier that was trained :ref:`above <baseline_domain_fit>`. Here's how you get the default feature set used by the classifer.

.. code-block:: python

   >>> my_features = dc.config.features
   >>> my_features
   {
    'bag-of-words': {'lengths': [1]},
    'freq': {'bins': 5},
    'in-gaz': {}
   }

By default, the classifier only uses a bag of words (unigrams) as features. It may be useful to have the classifier look at longer phrases since they carry more context. To accomplish this, you need to change the ``'lengths'`` setting of the ``'bag-of-words'`` feature to extract longer n-grams. Suppose you want to extract single words (unigrams), bigrams and trigrams, the :data:`my_features` dictionary should be updated as shown below.

.. code-block:: python

   >>> my_features['bag-of-words']['lengths'] = [1, 2, 3]

You could also add other :ref:`supported features <domain_features>`. In some cases, the natural language patterns at the start or the end of a query can be highly indicative of of a certain domain. To capture this information, you can extract the leading and trailing phrases of different lengths, also called edge n-grams, from the query. The code below adds the new ``'edge-ngrams'`` feature to the existing :data:`my_features` dictionary.

.. code-block:: python

   >>> my_features['edge-ngrams'] = { 'lengths': [1, 2] } 
   >>> my_features
   {
    'bag-of-words': {'lengths': [1, 2, 3]},
    'edge-ngrams': {'lengths': [1, 2]},
    'freq': {'bins': 5},
    'in-gaz': {}
   }

To retrain the classifier with the updated feature set, pass in the :data:`my_features` dictionary as an argument to the :data:`features` parameter of the :meth:`fit` method. This trains the domain classification model using the provided feature extraction settings, while continuing to use Workbench's defaults for model type (logistic regression) and hyperparameter selection.

.. code-block:: python

   >>> dc.fit(features=my_features)
   Fitting domain classifier
   No app configuration file found. Using default domain model configuration
   Selecting hyperparameters using k-fold cross-validation with 10 splits
   Best accuracy: 99.60%, params: {'C': 100, 'fit_intercept': False}


**Hyperparameter tuning**

Next, let's experiment with the model's hyperparameters. To get the hyperparameter selection settings for the current classifier, do:

.. code-block:: python

   >>> my_param_settings = dc.config.param_selection
   >>> my_param_settings
   {
    'grid': {
              'C': [10, 100, 1000, 10000, 100000],
              'fit_intercept': [True, False]
            },
    'k': 10,
    'type': 'k-fold'
   }

Let's reduce the range of values to search for the ``'C'`` parameter (inverse of regularization strength). Also, instead of always choosing an ``'l2'`` penalty by default, let's allow the hyperparameter estimation process to choose the ideal norm (``'l1'`` or ``'l2'``) for penalization. The updated settings can then be passed to :meth:`fit` as an argument to the :data:`param_selection` parameter.

.. code-block:: python

   >>> my_param_settings['grid']['C'] = [10, 100, 1000]
   >>> my_param_settings['grid']['penalty'] = ['l1', 'l2']
   >>> my_param_settings
   {
    'grid': {
              'C': [10, 100, 1000],
              'fit_intercept': [True, False],
              'penalty': ['l1', 'l2']
            },
    'k': 10,
    'type': 'k-fold'
   }
   >>> dc.fit(param_selection=my_param_settings)
   Fitting domain classifier
   No app configuration file found. Using default domain model configuration
   Selecting hyperparameters using k-fold cross-validation with 10 splits
   Best accuracy: 99.56%, params: {'C': 10, 'fit_intercept': False, 'penalty': 'l2'}

The :meth:`fit` method now searches over the updated parameter grid and prints the hyperparameter values for the model with the highest cross-validation accuracy. By default, the domain classifier uses k-fold cross-validation with 10 folds. To use a different cross-validation strategy, you can modify the value for the ``'type'`` key in the :data:`my_param_settings`. For instance, to use five randomized folds:

.. code-block:: python

   >>> my_param_settings['k'] = 5
   >>> my_param_settings['type'] = 'shuffle'
   >>> my_param_settings
   {
    'grid': {
              'C': [10, 100, 1000],
              'fit_intercept': [True, False],
              'penalty': ['l1', 'l2']
            },
    'k': 5,
    'type': 'shuffle'
   }
   >>> dc.fit(param_selection=my_param_settings)
   Fitting domain classifier
   No app configuration file found. Using default domain model configuration
   Selecting hyperparameters using shuffle cross-validation with 5 splits
   Best accuracy: 99.50%, params: {'C': 100, 'fit_intercept': False, 'penalty': 'l2'}

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
   >>> dc.fit(model_settings={'classifier_type': 'svm'}, param_selection=my_param_settings)
   Fitting domain classifier
   No app configuration file found. Using default domain model configuration
   Selecting hyperparameters using shuffle cross-validation with 5 splits
   Best accuracy: 99.56%, params: {'C': 1000, 'kernel': 'rbf'}

Here's another example that trains a :sk_api:`random forest <sklearn.ensemble.RandomForestClassifier>` :sk_guide:`ensemble <ensemble>` classifier:

.. code-block:: python

   >>> my_param_settings['grid'] = {
   ...  'n_estimators': [5, 10, 15, 20],
   ...  'criterion': ['gini', 'entropy'],
   ...  'warm_start': [True, False]
   ... }
   >>> dc.fit(model_settings={'classifier_type': 'rforest'}, param_selection=my_param_settings)
  Fitting domain classifier
  No app configuration file found. Using default domain model configuration
  Selecting hyperparameters using shuffle cross-validation with 5 splits
  Best accuracy: 98.37%, params: {'criterion': 'gini', 'n_estimators': 15, 'warm_start': False}


Run the domain classifier
-------------------------

A trained domain classifier can be run on a test query using the :meth:`DomainClassifier.predict` method.

.. code-block:: python

   >>> dc.predict('weather in san francisco?')
   'weather'

The :meth:`predict` method returns the label for the domain with highest predicted probability. This method gets called by the natural language processor's :meth:`process` method at runtime to classify the domain for an incoming query.

When experimenting with different classifier settings or debugging classifier performance, it is often useful to inspect how confident a trained model is at predicting the right label. To view the predicted probability distribution over all the possible domain labels, use the :meth:`DomainClassifier.predict_proba` method.

.. code-block:: python

   >>> dc.predict_proba('weather in san francisco?')
   [
    ('weather', 0.66666666666666663),
    ('smart_home', 0.13333333333333333),
    ('unknown', 0.13333333333333333),
    ('times_and_dates', 0.066666666666666666)
   ]

The result of :meth:`predict_proba` is a list of tuples ranked from the most likely domain to the least. The first element of each tuple is the domain label and the second element is the associated classification probability. Ideally, you want a classifier that assigns a high probability to the expected (correct) class label for a test query, while having very low prediction probabilities for the incorrect labels.

The :meth:`predict` and :meth:`predict_proba` methods run on one query at a time. See the next section for details on evaluating classifier performance by testing a trained model on a batch of labeled test queries.


Evaluate classifier performance
-------------------------------

To evaluate the accuracy of your trained domain classifier, you first need to create labeled test data, as described in the :ref:`Natural Language Processor <evaluate_nlp>` chapter. Once you have the test data files in the right place in your Workbench project, you can measure your model's performance using the :meth:`DomainClassifier.evaluate` method.

.. code-block:: python

   >>> dc.evaluate()
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
   <StandardModelEvaluation score: 98.05%, 1811 of 1847 examples correct>

The :meth:`evaluate` method strips away all ground truth annotations from the test queries and passes in the resulting unlabeled queries to the trained domain classifier for prediction. The classifier's output predictions are then compared against the ground truth labels to compute the model's prediction accuracy. In the above example, the model got 1,811 out of 1,847 test queries correct, resulting in an accuracy of 98%

The :meth:`evaluate` method returns a rich object that contains a lot more information over and above the aggregate accuracy score. The code below prints all the model performance statistics reported by the :meth:`evaluate` method.

.. code-block:: python

   >>> eval = dc.evaluate()
   >>> eval.print_stats()
   Overall Statistics: 

       accuracy f1_weighted          TP          TN          FP          FN    f1_macro    f1_micro
          0.981       0.972        1811        5505          36          36       0.741       0.981



   Statistics by Class: 

                  class      f_beta   precision      recall     support          TP          TN          FP          FN
        times_and_dates       0.994       1.000       0.987          78          77        1769           0           1
             smart_home       0.974       0.949       1.000         629         629        1184          34           0
                unknown       0.999       0.999       0.998        1107        1105         739           1           2
                weather       0.000       0.000       0.000          33           0        1813           1          33



   Confusion Matrix: 

                     times_and_..     smart_home        unknown
      times_and_..             77              1              0
        smart_home              0            629              0
           unknown              0              1           1105
           weather              0             32              1


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
  class        Domain label
  f_beta       :sk_api:`F-beta score <sklearn.metrics.fbeta_score>`
  precision    `Precision <https://en.wikipedia.org/wiki/Precision_and_recall#Precision>`_
  recall       `Recall <https://en.wikipedia.org/wiki/Precision_and_recall#Recall>`_
  support      Number of test queries in this domain (based on ground truth)
  TP           Number of `true positives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  TN           Number of `true negatives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  FP           Number of `false positives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  FN           Number of `false negatives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  ===========  ===

**Confusion Matrix**
  |

  A `confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`_ with each row representing the number of instances in an actual class and each column representing the number of instances in a predicted class. It makes it easy to see if the classifier is frequently confusing two classes, i.e. commonly mislabelling one class as another. For instance, in the above example, the domain classifier has wrongly classified 32 instances of ``weather`` queries as ``smart_home``.

While these detailed statistics provide a wealth of information about the classifier performance, you might additionally also want to inspect the classifier's prediction on individual queries to better understand error patterns.

To view the classifier predictions for the entire test set, you can use the :attr:`results` attribute of the returned :obj:`eval` object.

.. code-block:: python

   >>> eval.results
   [
    EvaluatedExample(example=<Query 'change my 6 am alarm'>, expected='times_and_dates', predicted='times_and_dates', probas={'smart_home': 0.050000000000000003, 'times_and_dates': 0.94999999999999996, 'unknown': 0.0, 'weather': 0.0}, label_type='class'),
    EvaluatedExample(example=<Query 'change my 6 am alarm to 7 am'>, expected='times_and_dates', predicted='times_and_dates', probas={'smart_home': 0.050000000000000003, 'times_and_dates': 0.94999999999999996, 'unknown': 0.0, 'weather': 0.0}, label_type='class')],
    ...
   ]

Each result is an instance of the :class:`EvaluatedExample` class which contains information about the original input query, the expected ground truth label, the predicted label, and the predicted probability distribution over all the class labels. You can also selectively look at just the correct predictions or the incorrect predictions. The code below shows how to do that.

.. code-block:: python

   >>> list(eval.correct_results())
   [
    EvaluatedExample(example=<Query 'change my 6 am alarm'>, expected='times_and_dates', predicted='times_and_dates', probas={'smart_home': 0.050000000000000003, 'times_and_dates': 0.94999999999999996, 'unknown': 0.0, 'weather': 0.0}, label_type='class'),
    EvaluatedExample(example=<Query 'change my 6 am alarm to 7 am'>, expected='times_and_dates', predicted='times_and_dates', probas={'smart_home': 0.050000000000000003, 'times_and_dates': 0.94999999999999996, 'unknown': 0.0, 'weather': 0.0}, label_type='class'),
    ...
   ]
   >>> list(eval.incorrect_results())
   [
    EvaluatedExample(example=<Query 'stop my timers'>, expected='times_and_dates', predicted='smart_home', probas={'smart_home': 0.65000000000000002, 'times_and_dates': 0.29999999999999999, 'unknown': 0.050000000000000003, 'weather': 0.0}, label_type='class'),
    EvaluatedExample(example=<Query 'what is happening in germany right now?'>, expected='unknown', predicted='weather', probas={'smart_home': 0.14999999999999999, 'times_and_dates': 0.0, 'unknown': 0.40000000000000002, 'weather': 0.45000000000000001}, label_type='class'),
    ...
   ]

`List comprehensions <https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions>`_ can be used to easily slice and dice the results for error analysis. For instance, to easily inspect all the incorrect predictions for a particular domain, say ``times_and_dates``, you could do:

.. code-block:: python

   >>>  [(r.example, r.probas) for r in eval.incorrect_results() if r.expected == 'times_and_dates'] 
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

In this case, there was just one test query from the ``times_and_dates`` domain that got misclassified as ``smart_home``. You can also see that the correct label did come in second, but was still beaten by a significant margin in classification probability.

Here's an example listing all the misclassified queries from the ``weather`` domain where the classifier's confidence for the true label was very low (<25%). These could often be indicative of the kind of queries that are lacking in the current training data.

.. code-block:: python

   >>> [(r.example, r.probas) for r in eval.incorrect_results()
   ... if r.expected == 'weather' and r.probas['weather'] < .25]
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

In both of the cases above, the domain was misclassified as ``smart_home``. On inspecting the :doc:`training data <../blueprints/home_assistant>`, you will find that the ``weather`` domain indeed lacks labeled training queries like the ones above. On the other hand, these queries are very similar in language patterns ("check temperature ...", "check current temperature ...") to the training data for the ``check_thermostat`` intent in the ``smart_home`` domain. The model hence chose ``smart_home`` over ``weather`` when classifying them. This issue could potentially be solved by adding more relevant training queries to the ``weather`` domain, so the classification model can better learn to distinguish between these two confusable domains.

Error analysis on the results of the :meth:`evaluate` method can thus inform your experimentation and help in building better models. In the example  above, adding more training data was proposed as a solution for improving accuracy. While training data augmentation should be your first step, you could also explore other techniques such as experimenting with different model types, features and hyperparameters, as described :ref:`earlier <build_domain_with_config>` in this chapter.


Save model for future use
-------------------------

A trained domain classifier can be saved for later use by calling the :meth:`DomainClassifier.dump` method. The :meth:`dump` method serializes the trained model as a `pickle file <https://docs.python.org/3/library/pickle.html>`_ and saves it to the specified location on disk.

.. code:: python

   >>> dc.dump(model_path='experiments/domain_classifier.rforest.20170701.pkl')
   Saving domain classifier

The saved model can then be loaded anytime using the :meth:`DomainClassifier.load` method.

.. code:: python

   >>> dc.load(model_path='experiments/domain_classifier.rforest.20170701.pkl')
   Loading domain classifier
