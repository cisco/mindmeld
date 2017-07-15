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
   Selecting hyperparameters using k-fold cross validation with 10 splits
   Best accuracy: 99.50%, params: {'C': 10, 'fit_intercept': True}

The :meth:`fit` method loads all the necessary training queries and trains a domain classification model using the provided machine learning settings. When the method is called without any parameters (as in the example above), it uses the settings from the :ref:`app's configuration file <build_nlp_with_config>` (``config.py``), if defined, or Workbench's preset :ref:`classifier configuration <config>`.

The quickest and recommended way to get started with any of the NLP classifiers is by using Workbench's default settings. The resulting baseline classifier should provide a reasonable starting point to bootstrap your machine learning experimentation from. You can then experiment with alternate settings to identify the optimal classifier configuration for your app.


Training with custom configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To view the current :ref:`configuration <config>` being used by a trained classifier, use its :attr:`config` attribute. For example, here is the configuration being used by a baseline domain classifier trained using Workbench's default settings.

.. code-block:: python

   >>> dc.config.to_dict()
   {
    'features': 
      {
        'bag-of-words': {'lengths': [1]},
        'freq': {'bins': 5},
        'in-gaz': {}
      },
    'model_settings': {'classifier_type': 'logreg'},
    'model_type': 'text',
    'param_selection': 
      {
        'grid': {'C': [10, 100, 1000, 10000, 100000],
        'fit_intercept': [True, False]},
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
  Is always a dictionary with a single key called ``'classifier_type'``. The value of the key denotes the machine learning model to use. Allowed values are 

.. _sklearn_models:

  =============== =======================================================
  Classifier Type Description (with list of configurable hyperparameters)
  =============== =======================================================
  ``'logreg'``    `Logistic regression <http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression>`_ (See `parameter list <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_ )
  ``'svm'``       `Support vector machine <http://scikit-learn.org/stable/modules/svm.html#svm-classification>`_ (See `parameter list <http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_)
  ``'dtree'``     `Decision tree <http://scikit-learn.org/stable/modules/tree.html#tree>`_ (See `parameter list <http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>`_)
  ``'rforest'``   `Random forest <http://scikit-learn.org/stable/modules/ensemble.html#forest>`_ (See `parameter list <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_)
  =============== =======================================================


2. **Feature Extraction Settings** 

``'features'`` (:class:`dict`)
  |
  Is a dictionary where the keys are the names of the feature groups to be extracted. The corresponding values are dictionaries representing the feature extraction settings for each group. The table below enumerates the features that can be used for domain classification.

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
  |                       | E.g., ``{'bins': 5}`` quantizes the vocabulary frequency into 5 bins, and returns counts (log-scaled)      |
  |                       | representing the number of query tokens in each bin.                                                       |
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | ``'in-gaz'``          | Generates a set of features indicating the presence of query n-grams in different entity gazetteers,       |
  |                       | along with popularity information (as defined in the gazetteer).                                           |
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | ``'length'``          | Generates a set of features that capture query length information. Computes the number of tokens and       |
  |                       | characters in the query, on both linear and log scales.                                                    | 
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | ``'exact'``           | Returns the entire query text as a feature.                                                                |
  +-----------------------+------------------------------------------------------------------------------------------------------------+


3. **Hyperparameter Settings**

``'params'`` (:class:`dict`)
  |
  Is a dictionary containing the values to be used for different model hyperparameters during training. Examples include the ``'kernel'`` parameter for SVM, the ``'penalty'`` parameter for logistic regression, the ``'max_depth'`` parameter for decision tree, and so on. The list of allowable hyperparameters depends on the selected model. Refer to the parameter list in :ref:`the model table <sklearn_models>` above.

``'param_selection'`` (:class:`dict`)
  |
  Is a dictionary containing the settings for `hyperparameter selection <http://scikit-learn.org/stable/modules/grid_search.html>`_. This is used as an alternative to the ``'params'`` dictionary above if the ideal hyperparameters for the model are not already known and need to be estimated.

  Workbench mainly needs two pieces of information from the developer to do parameter estimation:

  #. The parameter space to search, captured by the value for the ``'grid'`` key
  #. The strategy for splitting the labeled data into training and validation sets, specified by the ``'type'`` key

  Depending on the splitting scheme selected, the ``param_selection`` dictionary can contain other keys that define additional settings. The table below enumerates all the keys allowed in the dictionary.

  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | Key                   | Value                                                                                                      |
  +=======================+============================================================================================================+
  | ``'grid'``            | A dictionary mapping each hyperparameter to a list of potential values to be searched. Here is an example  |
  |                       | grid for a :sk_model:`logistic regression <LogisticRegression>` model:                                     |
  |                       |.. code-block:: python                                                                                      |
  |                       |                                                                                                            |
  |                       |   {                                                                                                        | 
  |                       |     'penalty': ['l1', 'l2'],                                                                               |
  |                       |     'C': [10, 100, 1000, 10000, 100000],                                                                   |
  |                       |      'fit_intercept': [True, False]                                                                        |
  |                       |   }                                                                                                        | 
  |                       |                                                                                                            |
  +-----------------------+------------------------------------------------------------------------------------------------------------+



  Generates n-grams of the specified lengths from the query text.                                            |
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
  |                       | E.g., ``{'bins': 5}`` quantizes the vocabulary frequency into 5 bins, and returns counts (log-scaled)      |
  |                       | representing the number of query tokens in each bin.                                                       |
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | ``'in-gaz'``          | Generates a set of features indicating the presence of query n-grams in different entity gazetteers,       |
  |                       | along with popularity information (as defined in the gazetteer).                                           |
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | ``'length'``          | Generates a set of features that capture query length information. Computes the number of tokens and       |
  |                       | characters in the query, on both linear and log scales.                                                    | 
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | ``'exact'``           | Returns the entire query text as a feature.                                                                |
  +-----------------------+------------------------------------------------------------------------------------------------------------+



from sklearn.model_selection import (KFold, GridSearchCV, GroupKFold, GroupShuffleSplit,
                                     ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit)


# K-fold
 'k': 10,
 'type': 'k-fold'}


        try:
            cv_iterator = {"k-fold": self._k_fold_iterator,
            http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold

                           "shuffle": self._shuffle_iterator,
            http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit

                           "group-k-fold": self._groups_k_fold_iterator,
                           http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html#sklearn.model_selection.GroupKFold

                           "group-shuffle": self._groups_shuffle_iterator, 
                           http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html#sklearn.model_selection.GroupShuffleSplit

                           "stratified-k-fold": self._stratified_k_fold_iterator, http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold

                           "stratified-shuffle": self._stratified_shuffle_iterator, http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit















Which one takes precedence?


  The dictionary must contain a key called ``grid`` whose value is a dictionary mapping hyperparameters to a lists of possible 




    'param_selection': 
      {
        'grid': {'C': [10, 100, 1000, 10000, 100000],
        'fit_intercept': [True, False]},
        'k': 10,
        'type': 'k-fold'
      },




Evaluation
----------


Optimization
------------











Introduce the general ML techniques and methodology common to all NLP classifiers:
Getting the right kind of training data using in-house data generation and crowdsourcing, QAing and analyzing the data
Training a Workbench classifier, using k-fold cross-validation for hyperparameter selection
Training with default settings
Training with different classifier configurations (varying the model type, features or hyperparameter selection settings)
Testing a Workbench classifier on a held-out validation set
Doing error analysis on the validation set, retraining based on observations from error analysis by adding more training examples or feature tweaks
Getting final evaluation numbers on an unseen “blind” test set
Saving models for production use 
