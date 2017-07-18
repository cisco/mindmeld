.. meta::
    :scope: private

Role Classifier
===============

The Role Classifier is run as the fourth step in the natural language processing pipeline to determine the target roles for entities in a given query. It is a machine-learned `classification <https://en.wikipedia.org/wiki/Statistical_classification>`_ model that is trained using all the labeled queries for a given intent. Labels are derived from the role types annotated within the training queries. See :doc:`Step 6 <../quickstart/06_generate_representative_training_data>` for more details on training data preparation. Role classification models are trained per entity type. A Workbench app hence has one role classifier for every entity type with associated roles.

.. note::

   For a quick introduction, refer to :ref:`Step 7 <role_classification>` of the Step-By-Step Guide.

   Recommended prior reading: :doc:`Natural Language Processor <nlp>` chapter of the User Guide.


Access a role classifier
------------------------

Before using any of the NLP componenets, you need to generate the necessary training data for your app by following the guidelines in :doc:`Step 6 <../quickstart/06_generate_representative_training_data>`. You can then start by :ref:`instantiating an object <instantiate_nlp>` of the :class:`NaturalLanguageProcessor` (NLP) class.

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
    'check-weather': <IntentProcessor 'check-weather' ready: False, dirty: False>
   }

Workbench isn't aware of the different entity types for your app till it has loaded the labeled training queries. Load the relevant queries by calling the :meth:`build` method for the intent you are interested in. The :meth:`build` operation can take several minutes if the number of training queries for the chosen intent is large. Once the build is complete, you can inspect the identified entity types.

.. code-block:: python

   >>> nlp.domains['times_and_dates'].intents['change_alarm'].build()
   >>> nlp.domains['times_and_dates'].intents['change_alarm'].entities
   {
    'time': <EntityProcessor 'time' ready: True, dirty: True>
   }

The :class:`RoleClassifier` for each entity type can then be accessed using the :attr:`role_classifier` attribute of the corresponding entity.

.. code-block:: python

   >>> rc = nlp.domains['times_and_dates'].intents['change_alarm'].entities['time'].role_classifier
   >>> rc
   <RoleClassifier ready: True, dirty: True>


Train a role classifier
-----------------------

To train a role classification model for a specific entity, use the :meth:`RoleClassifier.fit` method. Depending on the size of the training data, this can take anywhere from a few seconds to several minutes to finish. If the logging level is set to ``INFO`` or below, you should see the build progress in the console.

.. _baseline_role_fit:

.. code-block:: python

   >>> from mmworkbench import configure_logs; configure_logs()
   >>> rc = nlp.domains['times_and_dates'].intents['change_alarm'].entities['time'].role_classifier
   >>> rc.fit()
   Fitting role classifier: domain='times_and_dates', intent='change_alarm', entity_type='time'
   No app configuration file found. Using default role model configuration

The :meth:`fit` method loads all the necessary training queries and trains a role classification model using the provided machine learning settings. When the method is called without any parameters (as in the example above), it uses the settings from the :ref:`app's configuration file <build_nlp_with_config>` (``config.py``), if defined, or Workbench's preset :ref:`classifier configuration <config>`.

The quickest and recommended way to get started with any of the NLP classifiers is by using Workbench's default settings. The resulting baseline classifier should provide a reasonable starting point to bootstrap your machine learning experimentation from. You can then experiment with alternate settings to identify the optimal classifier configuration for your app.


Classifier configuration
^^^^^^^^^^^^^^^^^^^^^^^^

To view the current :ref:`configuration <config>` being used by a trained classifier, use its :attr:`config` attribute. For example, here is the configuration being used by a baseline role classifier trained using Workbench's default settings.

.. code-block:: python

   >>> rc.config.to_dict()
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
     'model_settings': None,
     'model_type': 'maxent',
     'param_selection': None,
     'params': {'C': 100, 'penalty': 'l1'}
   }

Let's take a look at the allowed values for each setting in a role classifier configuration.

1. **Model Settings** 

``'model_type'`` (:class:`str`)
  |

  Is always ``'maxent'``, since Workbench currently only supports training a `maximum entropy model (MaxEnt) <https://en.wikipedia.org/wiki/Multinomial_logistic_regression>`_ for role classification.

``'model_settings'`` (:class:`dict`)
  |

  Is always ``None``.

2. **Feature Extraction Settings** 

``'features'`` (:class:`dict`)
  |

  Is a dictionary where the keys are the names of the feature groups to be extracted. The corresponding values are dictionaries representing the feature extraction settings for each group. The table below enumerates the features that can be used for role classification.

.. _role_features:

  +---------------------------+------------------------------------------------------------------------------------------------------------+
  | Group Name                | Description                                                                                                |
  +===========================+============================================================================================================+
  | ``'bag-of-words-after'``  | Generates n-grams of specified lengths from the query text following the current entity.                   |
  |                           |                                                                                                            |
  |                           | Supported settings:                                                                                        |
  |                           | A dictionary with n-gram lengths as keys and a list of different starting positions as values.             |
  |                           | Each starting position is a token index, relative to the the start of the current entity span.             |
  |                           |                                                                                                            |
  |                           | E.g.,``'ngram_lengths_to_start_positions': {1: [0], 2: [0]}`` will extract all words (unigrams) and bigrams|
  |                           | starting with the first word of the current entity span. To additionally include unigrams and bigrams      |
  |                           | starting from the word after the current entity's first token, the settings can be modified to             |
  |                           | ``'ngram_lengths_to_start_positions': {1: [0, 1], 2: [0, 1]}``.                                            |
  |                           |                                                                                                            |
  |                           | Suppose the query is "Change my {6 AM|time|oldtime} alarm to {7 AM|time|newtime}" and the classifier is    |
  |                           | extracting features for the "6 AM" ``time`` entity. Then,                                                  |
  |                           |                                                                                                            |
  |                           | - ``{1: [0, 1]}`` would extract "6" and "AM"                                                               |
  |                           | - ``{2: [0, 1]}`` would extract "6 AM" and "AM alarm"                                                      |
  +---------------------------+------------------------------------------------------------------------------------------------------------+
  | ``'bag-of-words-before'`` | Generates n-grams of specified lengths from the query text preceding the current entity.                   |
  |                           |                                                                                                            |
  |                           | Supported settings:                                                                                        |
  |                           | A dictionary with n-gram lengths as keys and a list of different starting positions as values, similar     |
  |                           | to the ``'bag-of-words-after'`` feature group.                                                             |
  |                           |                                                                                                            |
  |                           | If the query is "Change my {6 AM|time|oldtime} alarm to {7 AM|time|newtime}" and the classifier is         |
  |                           | extracting features for the "6 AM" ``time`` entity,                                                        |
  |                           |                                                                                                            |
  |                           | - ``{1: [-2, -1]}`` would extract "change" and "my"                                                        |
  |                           | - ``{2: [-2, -1]}`` would extract "change my" and "my 6"                                                   | 
  +---------------------------+------------------------------------------------------------------------------------------------------------+
  | ``'other-entities'``      | Encodes information about the other entities present in the query.                                         |
  +---------------------------+------------------------------------------------------------------------------------------------------------+

.. _role_tuning:

3. **Hyperparameter Settings**

``'params'`` (:class:`dict`)
  |

  Is a dictionary containing the values to be used for different model hyperparameters during training. Examples include the ``'C'`` parameter (inverse of regularization strength), the ``'penalty'`` parameter (norm used in penalization) and so on. You can view the full list of allowed hyperparameters :sk_api:`here <sklearn.linear_model.LogisticRegression.html>`.

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

  The :meth:`fit` method does an :sk_guide:`exhaustive grid search <grid_search.html#exhaustive-grid-search>` over the parameter space, evaluating candidate models using the specified cross-validation strategy, to identify the parameters that give the highest accuracy. The optimal parameters can then be used in future calls to :meth:`fit` to skip the parameter selection process.

.. _build_role_with_config:

Training with custom configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are two ways to override Workbench's default role classifier configuration with your custom settings.


1. Application configuration file
"""""""""""""""""""""""""""""""""

The first method, as described in the :ref:`NaturalLanguageProcessor <build_nlp_with_config>` chapter, is to define the classifier settings in your application configuration file, ``config.py``. Define a dictionary named :data:`ROLE_MODEL_CONFIG` containing your custom settings. The :meth:`RoleClassifier.fit` and :meth:`NaturalLanguageProcessor.build` methods will then use those settings instead of Workbench's defaults.

Here's an example of a ``config.py`` file where the preset configuration for the role classifier is being overridden by custom settings that have been optimized for the app.

.. code-block:: python

   ROLE_MODEL_CONFIG = {
       'model_type': 'maxent',
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

Since this method requires updating a file each time you want to modify a setting, it's less suitable for rapid prototyping than the second method described below. The recommended use for this functionality is to store your optimal classifier settings, once you have identified them via experimentation. This ensures that the classifier training methods will use the optimized configuration to rebuild the models in the future. A common use case is retraining models on newly acquired training data, without retuning the underlying model settings.


2. Arguments to the :meth:`fit` method
""""""""""""""""""""""""""""""""""""""

The recommended way to experiment with a role classifier is by using arguments to the :meth:`fit` method.


**Feature extraction**

Let's start with the baseline classifier that was trained :ref:`above <baseline_role_fit>`. Here's how you get the default feature set used by the classifer.

.. code-block:: python

   >>> my_features = rc.config.features
   >>> my_features
   {
     'bag-of-words-after': {'ngram_lengths_to_start_positions': {1: [0, 1], 2: [0, 1]}},
     'bag-of-words-before': {'ngram_lengths_to_start_positions': {1: [-2, -1], 2: [-2, -1]}},
     'other-entities': {}
   }

By default, the classifier only extracts n-grams within a context window of two tokens around the entity of interest. It may be useful to have the classifier look at a larger context window since that could potentially provide more information than just the words in the immediate vicinity. To accomplish this, you need to change the ``'ngram_lengths_to_start_positions'`` settings to extract n-grams starting from tokens that are further away. Suppose you want to extract all the unigrams and bigrams in a window of three tokens around the current entity, the :data:`my_features` dictionary should be updated as shown below.

.. code-block:: python

   >>> my_features['bag-of-words-after']['ngram_lengths_to_start_positions'] = {
   ...     1: [0, 1, 2, 3],
   ...     2: [0, 1, 2]
   ... }
   >>> my_features['bag-of-words-before']['ngram_lengths_to_start_positions'] = {
   ...     1: [-3, -2, -1],
   ...     2: [-3, -2, -1]
   ... }
   >>> my_features
   {
     'bag-of-words-after': {'ngram_lengths_to_start_positions': {1: [0, 1, 2, 3], 2: [0, 1, 2]}},
     'bag-of-words-before': {'ngram_lengths_to_start_positions': {1: [-3, -2, -1], 2: [-3, -2, -1]}},
     'other-entities': {}
   }

Suppose w\ :sub:`i` represents the word at the *ith* index in the query, where the index is calculated relative to the start of the current entity span. Then, the above feature configuration should extract the following n-grams (w\ :sub:`0` is the first token of the current entity).

  - Unigrams: { w\ :sub:`-3`, w\ :sub:`-2`, w\ :sub:`-1`, w\ :sub:`0`, w\ :sub:`1`, w\ :sub:`2`, w\ :sub:`3` } 

  - Bigrams: { w\ :sub:`-3`\ w\ :sub:`-2`, w\ :sub:`-2`\ w\ :sub:`-1`, w\ :sub:`-1`\ w\ :sub:`0`,  w\ :sub:`0`\ w\ :sub:`1`, w\ :sub:`1`\ w\ :sub:`2`, w\ :sub:`2`\ w\ :sub:`3` }

To retrain the classifier with the updated feature set, pass in the :data:`my_features` dictionary as an argument to the :data:`features` parameter of the :meth:`fit` method. This trains the role classification model using the provided feature extraction settings, while continuing to use Workbench's defaults for model type (MaxEnt) and hyperparameter selection.

.. code-block:: python

   >>> rc.fit(features=my_features)
   Fitting role classifier: domain='times_and_dates', intent='change_alarm', entity_type='time'
   No app configuration file found. Using default role model configuration

**Hyperparameter tuning**

Next, let's experiment with the model's hyperparameters. To view the hyperparameters for the current classifier, do:

.. code-block:: python

   >>> my_params = rc.config.params
   >>> my_params
   {'C': 100, 'penalty': 'l1'}

The default role classifier comes with preset values for the ``'C'`` parameter (inverse of regularization strength) and the ``'penalty'`` parameter (norm used in penalization). However, you could also let Workbench select the ideal hyperparameters for your dataset by specifying a parameter search grid and a cross-validation strategy. Suppose you want the hyperparameter estimation process to choose the ideal ``'C'`` and ``'penalty'`` parameters using 10-fold cross-validation. Here's how you define your parameter selection settings:

.. code-block:: python

   >>> search_grid = {
   ...   'C': [1, 10, 100, 1000],
   ...   'penalty': ['l1', 'l2']
   ... }
   >>> my_param_settings = {
   ...   'grid': search_grid,
   ...   'type': 'k-fold',
   ...   'k': 10
   ... }

These settings can then be passed to :meth:`fit` as an argument to the :data:`param_selection` parameter.

.. code-block:: python

   >>>  
   Fitting role classifier: domain='times_and_dates', intent='change_alarm', entity_type='time'
   No app configuration file found. Using default role model configuration
   Selecting hyperparameters using k-fold cross validation with 10 splits
   Best accuracy: 96.59%, params: {'C': 1, 'penalty': 'l2'}

The :meth:`fit` method now searches over the provided parameter grid and prints the hyperparameter values for the model with the highest 10-fold cross-validation accuracy. To try a different cross-validation strategy, you can modify the value for the ``'type'`` key in the :data:`my_param_settings`. For instance, to use five randomized folds:

.. code-block:: python

   >>> my_param_settings['k'] = 5
   >>> my_param_settings['type'] = 'shuffle'
   >>> my_param_settings
   {
    'grid': {
              'C': [1, 10, 100, 1000],
              'penalty': ['l1', 'l2']
            },
    'k': 5,
    'type': 'shuffle'
   }
   >>> rc.fit(param_selection=my_param_settings)
   Fitting role classifier: domain='times_and_dates', intent='change_alarm', entity_type='time'
   No app configuration file found. Using default role model configuration
   Selecting hyperparameters using shuffle cross validation with 5 splits
   Best accuracy: 97.78%, params: {'C': 1, 'penalty': 'l2'}

For a full list of configurable hyperparameters and available cross-validation methods, refer to the above section on defining :ref:`hyperparameter settings <role_tuning>`.


Run the role classifier
-----------------------

A trained role classifier can be run on a test query using the :meth:`RoleClassifier.predict` method. First, detect all the entities in the query using a :ref:`trained entity recognizer <train_entity_model>`:

.. code-block:: python

   >>> query = 'Change my 6 AM alarm to 7 AM'
   >>> entities = er.predict(query)
   >>> entities
   (<QueryEntity '6 AM' ('time') char: [10-13], tok: [2-3]>,
    <QueryEntity '7 AM' ('time') char: [24-27], tok: [6-7]>)

Once the entities have been detected, you can call the role classifier's :meth:`predict` method on the entity of interest. The :meth:`predict` method classifies a single entity, but uses the full query text and information about all the entities in the query for :ref:`feature extraction <role_features>`. Here's how you run role classification on the above two entities, one by one:

.. code-block:: python

   >>> rc.predict(query, entities, 0)
   'oldtime'   
   >>> rc.predict(query, entities, 1)
   'newtime'

The :meth:`predict` method returns the label for the role with highest predicted probability. It gets called by the natural language processor's :meth:`process` method at runtime to classify the roles for all detected entities. 

The :meth:`predict` method runs on one entity at a time. To instead test a trained model on a batch of labeled test queries and evaluate classifier performance, see the next section.


Evaluate classifier performance
-------------------------------

To evaluate the accuracy of your trained role classifier, you first need to create labeled test data, as described in the :ref:`Natural Language Processor <evaluate_nlp>` chapter. Once you have the test data files in the right place in your Workbench project, you can measure your model's performance using the :meth:`RoleClassifier.evaluate` method.

.. code-block:: python

   >>> rc.evaluate()
   Loading queries from file times_and_dates/change_alarm/test.txt
   <StandardModelEvaluation score: 95.24%, 20 of 21 examples correct>

The :meth:`evaluate` method strips away all ground truth annotations from the test queries and passes in the resulting unlabeled queries to the trained role classifier for prediction. The classifier's output predictions are then compared against the ground truth labels to compute the model's prediction accuracy. In the above example, the model got 20 out of 21 test queries correct, resulting in an accuracy of about 95%.

The :meth:`evaluate` method returns a rich object that contains a lot more information over and above the aggregate accuracy score. The code below prints all the model performance statistics reported by the :meth:`evaluate` method.

.. code-block:: python

   >>> eval = rc.evaluate()
   >>> eval.print_stats()
   Overall Statistics: 

       accuracy f1_weighted          TP          TN          FP          FN    f1_macro    f1_micro
          0.952       0.952          20          20           1           1       0.952       0.952



   Statistics by Class: 

                  class      f_beta   precision      recall     support          TP          TN          FP          FN
                oldtime       0.957       0.917       1.000          11          11           9           1           0
                newtime       0.947       1.000       0.900          10           9          11           0           1



   Confusion Matrix: 

                          oldtime        newtime
           oldtime             11              0
           newtime              1              9


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
  class        Role label
  f_beta       :sk_api:`F-beta score <sklearn.metrics.fbeta_score>`
  precision    `Precision <https://en.wikipedia.org/wiki/Precision_and_recall#Precision>`_
  recall       `Recall <https://en.wikipedia.org/wiki/Precision_and_recall#Recall>`_
  support      Number of test entities with this role (based on ground truth)
  TP           Number of `true positives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  TN           Number of `true negatives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  FP           Number of `false positives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  FN           Number of `false negatives <https://en.wikipedia.org/wiki/Precision_and_recall>`_
  ===========  ===

**Confusion Matrix**
  |

  A `confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`_ with each row representing the number of instances in an actual class and each column representing the number of instances in a predicted class. It makes it easy to see if the classifier is frequently confusing two classes, i.e. commonly mislabelling one class as another. For instance, in the above example, the role classifier has wrongly classified one instance of a ``newtime`` entity as ``oldtime``.

While these detailed statistics provide a wealth of information about the classifier performance, you might additionally also want to inspect the classifier's prediction on individual queries to better understand error patterns.

To view the classifier predictions for the entire test set, you can use the :attr:`results` attribute of the returned :obj:`eval` object.

.. code-block:: python

   >>> eval.results
   [
     EvaluatedExample(example=(<Query 'change my 6 am alarm'>, (<QueryEntity '6 am' ('time') char: [10-13], tok: [2-3]>,), 0), expected='oldtime', predicted='oldtime', probas={'newtime': 0.10062246873286373, 'oldtime': 0.89937753126713627}, label_type='class'),
     EvaluatedExample(example=(<Query 'change my 6 am alarm to 7 am'>, (<QueryEntity '6 am' ('time') char: [10-13], tok: [2-3]>, <QueryEntity '7 am' ('time') char: [24-27], tok: [6-7]>), 0), expected='oldtime', predicted='oldtime', probas={'newtime': 0.028607105880949835, 'oldtime': 0.97139289411905017}, label_type='class'),
    ...
   ]

Each result is an instance of the :class:`EvaluatedExample` class which contains information about the original input query, the expected ground truth label, the predicted label, and the predicted probability distribution over all the class labels. You can also selectively look at just the correct predictions or the incorrect predictions. The code below shows how to do that.

.. code-block:: python

   >>> list(eval.correct_results())
   [
     EvaluatedExample(example=(<Query 'change my 6 am alarm'>, (<QueryEntity '6 am' ('time') char: [10-13], tok: [2-3]>,), 0), expected='oldtime', predicted='oldtime', probas={'newtime': 0.10062246873286373, 'oldtime': 0.89937753126713627}, label_type='class'),
     EvaluatedExample(example=(<Query 'change my 6 am alarm to 7 am'>, (<QueryEntity '6 am' ('time') char: [10-13], tok: [2-3]>, <QueryEntity '7 am' ('time') char: [24-27], tok: [6-7]>), 0), expected='oldtime', predicted='oldtime', probas={'newtime': 0.028607105880949835, 'oldtime': 0.97139289411905017}, label_type='class'),
    ...
   ]
   >>> list(eval.incorrect_results())
   [
     EvaluatedExample(example=(<Query 'replace the 8 am alarm with a 10 am alarm'>, (<QueryEntity '8 am' ('time') char: [12-15], tok: [2-3]>, <QueryEntity '10 am' ('time') char: [30-34], tok: [7-8]>), 1), expected='newtime', predicted='oldtime', probas={'newtime': 0.48770513415754235, 'oldtime': 0.51229486584245765}, label_type='class')
   ]

`List comprehensions <https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions>`_ can be used to easily slice and dice the results for error analysis. In the above case, given the fairly small dataset size, there is just one case of misclassification. But in a real-world app with a large test set, you can still easily inspect all the incorrect predictions for a particular role, say ``newtime``, as shown below:

.. code-block:: python

   >>>  [(r.example, r.probas) for r in eval.incorrect_results() if r.expected == 'newtime']
   [
     (
       (
         <Query 'replace the 8 am alarm with a 10 am alarm'>,
         (<QueryEntity '8 am' ('time') char: [12-15], tok: [2-3]>, <QueryEntity '10 am' ('time') char: [30-34], tok: [7-8]>),
         1
       ),
       {
         'newtime': 0.48770513415754235,
         'oldtime': 0.51229486584245765
       }
     )
   ]

Here's another example listing all queries with a ``newtime`` role where the classifier's confidence for the true label was relatively low (<60%). These could often be indicative of the kind of queries that are lacking in the current training data.

.. code-block:: python

   >>> [(r.example, r.probas) for r in eval.results
   ... if r.expected == 'newtime' and r.probas['newtime'] < .6]
   [
     (
       (
         <Query 'replace the 8 am alarm with a 10 am alarm'>,
         (<QueryEntity '8 am' ('time') char: [12-15], tok: [2-3]>, <QueryEntity '10 am' ('time') char: [30-34], tok: [7-8]>),
         1
       ),
       {
         'newtime': 0.48770513415754235,
         'oldtime': 0.51229486584245765
       }
     ),
     (
       (
         <Query 'cancel my 6 am and replace it with a 6:30 am alarm'>,
         (<QueryEntity '6 am' ('time') char: [10-13], tok: [2-3]>, <QueryEntity '6:30 am' ('time') char: [37-43], tok: [9-10]>),
         1
       ),
       {
         'newtime': 0.5872536946800766,
         'oldtime': 0.41274630531992335
       }
     )
   ]

In both of the above cases, the classifier's prediction probability for the ``'newtime'`` role was fairly low. The classifier got one of them wrong, and barely got the other one right with a confidence of about 59%. On inspecting the :doc:`training data <../blueprints/home_assistant>`, you will find that the ``newtime`` role indeed lacks labeled training queries like the ones above. This issue could potentially be solved by adding more relevant training queries for the ``newtime`` role, so the classification model can generalize better.

Error analysis on the results of the :meth:`evaluate` method can thus inform your experimentation and help in building better models. In the example  above, adding more training data was proposed as a solution for improving accuracy. While training data augmentation should be your first step, you could also explore other techniques such as experimenting with different model types, features and hyperparameters, as described :ref:`earlier <build_role_with_config>` in this chapter.


Save model for future use
-------------------------

A trained role classifier can be saved for later use by calling the :meth:`RoleClassifier.dump` method. The :meth:`dump` method serializes the trained model as a `pickle file <https://docs.python.org/3/library/pickle.html>`_ and saves it to the specified location on disk.

.. code:: python

   >>> rc.dump(model_path='experiments/role_classifier.maxent.20170701.pkl')
   Saving role classifier: domain='times_and_dates', intent='change_alarm', entity_type='time'

The saved model can then be loaded anytime using the :meth:`RoleClassifier.load` method.

.. code:: python

   >>> rc.load(model_path='experiments/role_classifier.maxent.20170701.pkl')
   Loading role classifier: domain='times_and_dates', intent='change_alarm', entity_type='time'

