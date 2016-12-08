Entity Recognizer
=================

The Named Entity Recognizer detects specific words or phrases in a query which might be useful when fulfilling its intent.

.. raw:: html

    <style> .aqua {background-color:#00FFFF} </style>

.. role:: aqua

* show me tom hanks movies => show me :aqua:`tom hanks` movies

.. _Maximum Entropy Markov Model: https://en.wikipedia.org/wiki/Maximum-entropy_Markov_model
.. _Conditional Random Field: https://en.wikipedia.org/wiki/Conditional_random_field

In practice, sequence models such as the `Maximum Entropy Markov Model`_ or `Conditional Random Field`_ work very effectively for bootstrapping your application.

.. _Stanford lecture notes: https://web.stanford.edu/class/cs124/lec/Information_Extraction_and_Named_Entity_Recognition.pdf
.. _IOB tagging: https://en.wikipedia.org/wiki/Inside_Outside_Beginning

A formal introduction to sequence NER models can be found in the `Stanford lecture notes`_ slides. MindMeld Workbench will prepare the data to an "IOBES" tagging scheme (an extension to the well known `IOB tagging`_ scheme for NER sequence models).

Train The Model
---------------

.. code-block:: python

  from mindmeld.entity_recognition import EntityRecognizer

  # Load the training data
  training_data = mm.load_data('/path/to/domain/intent/training_data.txt')

  # Load the gazetteers
  gazetteers = mm.load_gaz('/path/to/domain/intent/gazetteers')

  # Define the feature settings
  features = {
    "bag-of-words": { "lengths": [1, 2] },
    "in-gaz": { "scaling": 10 },
    "length": {}
  }

  # Train the classifier
  entity_model = EntityRecognizer(model_type='memm', features=features, gaz=gazetteers)
  entity_model.fit(data=training_data)

  # Evaluate the model
  eval_set = mm.load_data('/path/to/eval_set.txt')
  entity_model.evaluate(data=eval_set)

For determining the Cross Validation accuracy, you can define a CV iterator and train with the **cv** argument.

.. code-block:: python

  from mindmeld.cross_validation import KFold

  # Define CV iterator
  kfold_cv = KFold(num_splits=10)

  # Train classifier with grid search + CV
  entity_model.fit(data=training_data, model='svm', params_grid=params, cv=kfold_cv, scoring='precision')

Feature Specification
---------------------

The features to be used in the Machine Learning model are specified in the **features** field of your model specification. The following feature-specifications are available to use.

+----------------+----------------------------------------------------------------------------------------------------------------+
|Feature Group   | Description                                                                                                    |
+================+================================================================================================================+
| bag-of-words   | Takes a query and generates N-grams of the specified "lengths"                                                 |
+----------------+----------------------------------------------------------------------------------------------------------------+
| freq           | Counts of query tokens within each frequency bin (log-scaled)                                                  |
+----------------+----------------------------------------------------------------------------------------------------------------+
| in-gaz         | A set of features indicating presence of N-grams in Gazetteers                                                 |
+----------------+----------------------------------------------------------------------------------------------------------------+
| in-gaz-span    | Extracts various properties of gazetteer spans                                                                 |
+----------------+----------------------------------------------------------------------------------------------------------------+
| num-candidates | Heuristically extracted numeric entities                                                                       |
+----------------+----------------------------------------------------------------------------------------------------------------+


Prediction
----------

For predicting the sequence tags on a new query, simply use the **predict** method on the model. This returns a structured JSON with start/end information at the character-level.

.. code-block:: python

  q = "Show me tom hanks movies featuring meg ryan"
  entities = entity_model.predict(query=q)

Output::

  # Character indices are zero-indexed.
  [
    {
      chstart: 8,
      chend: 16,
      entity: "tom hanks",
      raw entity: "tom hanks",
      tstart: 2,
      tend: 3,
      type: "cast"
    },
    {
      chstart: 35,
      chend: 42,
      entity: "meg ryan",
      raw entity: "meg ryan",
      tstart: 6,
      tend: 7,
      type: "cast"
    }
  ]

Detailed Inspection
-------------------

You can use the **verbose=true** flag for detailed inspection on the predicted tags with their log probabilities.

.. code-block:: python

  q = "what are stanley kubrick's best rated movies"
  entities = entity_model.predict(query=q, verbose=True)

This outputs a detailed dump of the top feature values used for classifying that query. This provides valuable insights into model behavior towards specific queries and guides you to making alternate modeling choices. For illustration, let us assume that some configuration of the model wrongly predicted an entity tag.

.. code-block:: text

  Token                   Pred Tag                (Gold Tag)              (Log Prob)
  ------------------      ------------------      ------------------      ------------------
  what                    O||O|                        "\"
  are                     O||O|                        "\"
  stanley                 B|directors|O|           B|cast|O|               [-16.80866592]
  kubrick                 I|directors|O|           I|cast|O|               [-16.67216257]
  s                       O||O|                        "\"
  best                    B|sort|O|                    "\"
  rated                   I|sort|O|                    "\"
  movies                  B|type|O|                    "\"


In the above case, the model was unable to successfully distinguish "stanley kubrick" between cast and director (He appears as both in the training data). For further investigation, detailed feature values are printed along with the the feature names. This provides valuable insights into model and feature engineering for training the system better.

.. code-block:: javascript

  --------                                               --------        --------        --------        --------        --------        --------
  name                                                   feat_val         pred_w          gold_w          pred_p          gold_p           diff
  --------                                               --------        --------        --------        --------        --------        --------
  bag-of-words|length:1|pos:-1=are                         1.000          -0.183          -0.427          -0.183          -0.427          -0.244
  bag-of-words|length:1|pos:-2=what                        1.000          -0.536          -0.090          -0.536          -0.090           0.446
  bag-of-words|length:1|pos:0=stanley                      1.000           0.079          -0.000           0.079          -0.000          -0.079
  bag-of-words|length:1|pos:1=kubrick                      1.000           0.079          -0.000           0.079          -0.000          -0.079
  bag-of-words|length:2|pos:0=stanley kubrick              1.000           0.079          -0.000           0.079          -0.000          -0.079
  in-gaz|conflict|exact|type1:directors|type2:producers    1.000          -0.277          -0.496          -0.277          -0.496          -0.219
  in-gaz|conflict|exact|type1:directors|type2:writers      1.000          -0.074          -0.853          -0.074          -0.853          -0.779
  in-gaz|type:directors                                    1.000           0.976          -0.025           0.976          -0.025          -1.001
  in-gaz|type:directors|log-char-len                       0.876           0.834          -0.239           0.731          -0.210          -0.940
  in-gaz|type:directors|ngram-first|length:1=stanley       1.000           0.079          -0.000           0.079          -0.000          -0.079
  in-gaz|type:directors|ngram-last|length:1=kubrick        1.000           0.066          -0.000           0.066          -0.000          -0.066
  in-gaz|type:directors|p_ef                              -1.000          -0.750           0.118           0.750          -0.118          -0.868
    ...
    ...

