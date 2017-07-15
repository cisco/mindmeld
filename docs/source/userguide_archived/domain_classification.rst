

  # Evaluate the model
  eval_set = mm.load_data('/path/to/eval_set.txt')
  domain_classifier.evaluate(data=eval_set)

For a grid sweep over model hyperparameters, you can specify a param_grid dict object in the fit method. For example, for a Logistic Regression model, you can specify the regularization penalty function (l1/l2) and the strength parameter **C**. Additionally, if you want to do Cross Validation, you can define a CV iterator by specifying the number of splits.

.. code-block:: python

  from mindmeld.cross_validation import KFold

  # Specify grid search params
  params = {
    "C": [1, 10, 100, 1000, 5000],
    "class_bias": [0.5],
    "penalty": ["l2"]
  }

  # Define CV iterator
  kfold_cv = KFold(num_splits=10)

  # Train classifier with grid search + CV
  domain_classifier.fit(data=training_data, params_grid=params, cv=kfold_cv)

If you set **cv=KFold** or **cv=StratifiedKFold**, a confusion matrix will be generated in the printed stats. If **cv=None** (default), the entire data will be used for training and no confusion matrix is generated.

.. code-block:: javascript

    +---------------------+--------------------+---+
    |Error analysis       |        predicted   |   |
    +---------------------+----+---+---+---+---+---+
    |                     |gold| 0 | 1 | 2 | 3 | 4 |
    +=====================+====+===+===+===+===+===+
    |smart-home           |  0 | . | . | 4 | . | 2 |
    +---------------------+----+---+---+---+---+---+
    |tv-and-movies        |  1 | . | . | . | 12| . |
    +---------------------+----+---+---+---+---+---+
    |times-and-dates      |  2 | 7 | . | . | . | . |
    +---------------------+----+---+---+---+---+---+
    |weather              |  4 | 1 | . | 4 | . | . |
    +---------------------+----+---+---+---+---+---+
    |unknown              |  3 | . | 9 | . | . | 2 |
    +---------------------+----+---+---+---+---+---+


Training Accuracy Statistics::

  Average CV accuracy: 99.21% ± 0.36%
  Best accuracy: 99.60%, settings: {u'penalty': u'l2', u'C': 100, u'probability': True, 'class_weight': {0: 0.8454625164401579, 1: 1.404707233065442}}


+--------------+----------------------------------------------------------------------------------------------------------------+

Evaluation
----------

Next, see how the trained model performs against the test data set. Run the **evaluate** method on the classifier.

.. code-block:: python

  ev = domain_classifier.evaluate(data='test_set')

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

Finally, you can use the model to predict the domain for any new query input:

.. code-block:: python

  q = "Set a timer for 25 minutes"
  pred_domain = domain_classifier.predict(query=q)
  print pred_domain

.. code-block:: text

  "times-and-dates"