Intent Classifier
=================

The intent classifier is trained using all of the labeled queries across all intents for all intents. It determines the target intent for a given query. The labels for the training data are the intent names associated with each query. Similar to the Domain Classifier, the intent classifier uses the "marked down" form of each query i.e all query annotations are removed and raw text is sent to a text classifier.

Training The Model
------------------

.. code-block:: python

  from mindmeld.intent_classification import IntentClassifier

  # Load training data to a Data Frame
  training_data = mm.load_data('/path/to/domain/training_data.txt')

  # Select the feature settings
  features = {
    "bag-of-words": { "lengths": [1, 2] },
    "edge-ngrams": { "lengths": [1, 2] },
    "in-gaz": { "scaling": 10 },
    "length": {},
    "gaz-freq": {},
    "freq": { "bins": 5 }
  }

  # Train the classifier
  intent_classifier = IntentClassifier(model_type='logreg', features=features)
  intent_classifier.fit(data=training_data, model='logreg')

  # Evaluate the model
  eval_set = mm.load_data('/path/to/eval_set.txt')
  intent_classifier.evaluate(data=eval_set)

For a grid sweep over model hyperparameters, you can specify a param_grid dict object in the fit method. For example, for a SVM model you can define the **kernel** and the regularization parameter **C**. Additionally, if you want to do Cross Validation, you can define a CV iterator by specifying the number of splits.

.. code-block:: python

  from mindmeld.cross_validation import StratifiedKFold

  # Specify grid search params
  params = {
    "C": [1, 10, 100, 1000, 5000],
    "class_bias": [0.5],
    "kernel": ["linear"],
    "probability": [true]
  }

  # Define CV iterator
  skfold_cv = StratifiedKFold(num_splits=10)

  # Train classifier with grid search + CV
  intent_classifier.fit(data=training_data, model='svm', params_grid=params, cv=skfold_cv)

A confusion matrix will be generated in the printed stats.

.. code-block:: javascript

    +---------------------+------------------------+
    |Error analysis       |        predicted       |
    +---------------------+----+---+---+---+---+---+
    |                     |gold| 0 | 1 | 2 | 3 | 4 |
    +=====================+====+===+===+===+===+===+
    |search               |  0 | . | 4 | . | 2 | . |
    +---------------------+----+---+---+---+---+---+
    |check-store          |  1 | 9 | . | . | . | 1 |
    +---------------------+----+---+---+---+---+---+
    |help                 |  2 | . | . | . | . | . |
    +---------------------+----+---+---+---+---+---+
    |greeting             |  3 | 1 | . | . | . | . |
    +---------------------+----+---+---+---+---+---+
    |exit                 |  4 | . | 1 | . | . | . |
    +---------------------+----+---+---+---+---+---+


Training Accuracy Statistics::

  Average CV accuracy: 98.34% ± 0.26%
  Best accuracy: 98.56%, settings: {u'kernel': u'linear', u'C': 5000, u'probability': True, 'class_weight': {0: 0.8454625164401579, 1: 1.404707233065442}}


Feature Specification
---------------------

The features to be used in the Machine Learning model are specified in the **features** field of your model specification. The following feature-specifications are available to use.

+--------------+----------------------------------------------------------------------------------------------------------------+
|Feature Group | Description                                                                                                    |
+==============+================================================================================================================+
| bag-of-words | Takes a query and generates N-grams of the specified "lengths"                                                 |
+--------------+----------------------------------------------------------------------------------------------------------------+
| edge-ngrams  | N-grams of the specified lengths at the start and end of query                                                 |
+--------------+----------------------------------------------------------------------------------------------------------------+
| freq         | Counts of query tokens within each frequency bin (log-scaled)                                                  |
+--------------+----------------------------------------------------------------------------------------------------------------+
| in-gaz       | A set of features indicating presence of N-grams in Gazetteers                                                 |
+--------------+----------------------------------------------------------------------------------------------------------------+
| gaz-freq     | Extracts frequency bin features for each gazetteer (log-scaled)                                                |
+--------------+----------------------------------------------------------------------------------------------------------------+
| length       | Extracts length measures (linear & log scale) on whole query                                                   |
+--------------+----------------------------------------------------------------------------------------------------------------+
| exact        | Extracts whole query string as a feature - useful for high accuracy on command & control applications          |
+--------------+----------------------------------------------------------------------------------------------------------------+

Evaluation
----------

Next, see how the trained model performs against the test data set. Run the **evaluate** method on the classifier.

.. code-block:: python

  ev = intent_classifier.evaluate(data='test_set.txt')

You can then print out the accuracy and error analysis of the classification:

.. code-block:: python

  accuracy = ev.accuracy_score()
  print("Accuracy: {0:f}".format(accuracy))

  # Error Analysis
  errors = ev.prediction_errors()
  for e in errors:
    print("{0} \t {1} \t {2}".format(e.data, e.gold_label, e.predicted_label))

Prediction
----------

Finally, use the model to predict the intent for any new query input:

.. code-block:: python

  q = "My new query for classification"
  pred_intent = intent_classifier.predict(query=q)

Detailed Inspection
-------------------

You can use the **verbose=True** flag for deeper analysis on the feature values used for classifying that query.

.. code-block:: python

  q = "I'm looking for a pair of jeans"
  pred_intent = intent_classifier.predict(query=q, verbose=True)

This outputs a detailed dump of the top feature values used for classifying that query. This provides valuable insights into model behavior towards specific queries and guides you to making alternate modeling choices.

.. code-block:: text

  Predicted intent:

  FEATURE                            VALUE          PRED_W          PRED_P          GOLD_W          GOLD_P            DIFF

  IV&category|freq|0                 0.226          -0.101          -0.023          -0.101          -0.023          +0.000
  IV&popularitysort|freq|0           0.143           0.114           0.016           0.114           0.016          +0.000
  IV&sale|freq|0                     0.226           0.102           0.023           0.102           0.023          +0.000
  IV&size|freq|1                     0.143           0.039           0.006           0.039           0.006          +0.000
  IV&special|freq|0                  0.143          -0.077          -0.011          -0.077          -0.011          +0.000
  IV&unsupported-emoji|freq|0        0.143          -0.007          -0.001          -0.007          -0.001          +0.000
  category|freq|0                    0.226          -0.101          -0.023          -0.101          -0.023          +0.000
  chars_log                          3.466           0.561           1.946           0.561           1.946          +0.000
  clothing_category_exists           1.000          -0.828          -0.828          -0.828          -0.828          +0.000
  clothing_category_pop              0.219          -0.095          -0.021          -0.095          -0.021          +0.000
  clothing_category_ratio_pop        0.354           0.068           0.024           0.068           0.024          +0.000
  collar|freq|0                      0.143           0.279           0.040           0.279           0.040          +0.000
  design|freq|0                      0.143           0.087           0.012           0.087           0.012          +0.000
  faq|freq|0                         0.143          -0.325          -0.046          -0.325          -0.046          +0.000
  freq|0                             0.226           0.003           0.001           0.003           0.001          +0.000
  freq|2                             0.286          -0.569          -0.162          -0.569          -0.162          +0.000
  freq|3                             0.143          -0.461          -0.066          -0.461          -0.066          +0.000
  freq|4                             0.143          -0.508          -0.073          -0.508          -0.073          +0.000
  left-edge|1:i\'m                   1.000           0.178           0.178           0.178           0.178          +0.000
  left-edge|2:i\'m|looking           1.000           0.088           0.088           0.088           0.088          +0.000
  ngram:a|pair                       1.000          -0.039          -0.039          -0.039          -0.039          +0.000
  ngram:jeans                        1.000          -0.088          -0.088          -0.088          -0.088          +0.000
  ...



