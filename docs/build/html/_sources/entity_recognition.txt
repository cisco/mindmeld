Building the Entity Recognizer
==============================

The Named Entity Recognizers detect specific words or phrases in a query which might be useful when fulfilling its intent.

.. code-block:: javascript

  show me tom hanks movies --> show me {tom hanks|cast} movies

The named entity recognizers are also called the facet models or facet classifiers. In practice, sequence models such as the Maximum-Entropy Markov Model or Conditional Random Fields work very effectively for bootstrapping your application.

.. _notes: https://web.stanford.edu/class/cs124/lec/Information_Extraction_and_Named_Entity_Recognition.pdf
.. _IOB: https://en.wikipedia.org/wiki/Inside_Outside_Beginning

A formal introduction to sequence NER models can be found in the Stanford lecture notes_. MindMeld Workbench will prepare the data to an IOBES tagging scheme (an extension to the well known IOB_ tagging for NER sequence models).

Define The Config
*****************

Similar to the Domain & Intent Classifiers, we can define a config file for specifying the model and feature settings to use. In the following example, we use the `memm` model with the respective lexical and syntactic features specified -

.. code-block:: javascript

  {
      "models": {
        "memm": {
          "model-type": "memm",
          "features": {
            "bag-of-words": {
              "lengths": [1, 2] 
            },
            "edge-ngrams": {
              "lengths": [1, 2]
            },
            "in-gaz": { "scaling": 10 },
            "length": {},
            "gaz-freq": {},
            "freq": { "bins": 5 }
          }
        },
        "ngram": {
          "model-type": "logreg",
          "features": {
            "bag-of-words": { "lengths": [1, 2] },
            "in-gaz": { "scaling": 10 },
            "length": {}
          }
        }
      }
    }

Train The Model
***************

Feature Specification
*********************

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

