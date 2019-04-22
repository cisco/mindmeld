Working with User-Defined Features
==================================

In addition to the features available for each NLP classifier in MindMeld,
you can also define your own custom feature extractors that are relevant to your application.
User-defined features must follow the same format as MindMeld's in-built features.
In this section, we will examine the components of a feature extractor function and
explain how to write your own custom features.


Custom Features File
--------------------

Start by creating a new Python file, say ``custom_features.py``, that will contain the definitions
of all your custom feature extractors. If your MindMeld project was created using the "template"
blueprint or adapted from an existing blueprint application, you should already have this file at
the root level of your project directory. If you created your
MindMeld project from scratch, you can refer to any of the blueprints for an example of the
custom features file.

In order to use your custom features, the custom features file must be imported in the
``__init__.py`` file. For example, in the Home Assistant blueprint app you can import
a custom features file named ``custom_features.py`` by adding the following line to the
``__init__.py`` file.

.. code-block:: python

   import home_assistant.custom_features

You can then reference your newly defined features in the classifier
configurations you specify in the application configuration file, ``config.py``.

The Natural Language Processor uses two kinds of features. **Query features** can be used in
domain, intent, and entity model configs, and are extracted by feature extractors that operate on
the entire input query. **Entity Features**, on the other hand, can only be used in the role
classifier config, and are extracted by feature extractors that operate on a single extracted
entity. An example for each kind of feature extractor is provided in the following sections.

To summarize, in order to implement and use your own custom features, you must do the following:

  • Define your feature extractors in a ``.py`` file (referred to as the *custom features file*)

  • Import the custom features file in ``__init__.py``.

  • Add your newly defined feature names to the ``'features'`` dictionary within a classifier
    configuration.


Example of a Query Feature Extractor
------------------------------------

Each feature extractor is defined as a Python function that returns an inner ``_extractor``
function. This ``_extractor`` function performs the actual feature extraction. The following code
block shows an example of a query feature extractor that computes the average token length of
an input query.

.. code-block:: python

    @register_query_feature(feature_name='average-token-length')
    def extract_average_token_length(**args):
        """
        Example query feature that gets the average length of normalized tokens in the query

        Returns:
            (function) A feature extraction function that takes a query and
                returns the average normalized token length
        """
        def _extractor(query, resources):
            tokens = query.normalized_tokens
            average_token_length = sum([len(t) for t in tokens]) / len(tokens)
            return {'average_token_length': average_token_length}

        return _extractor

Let's take a closer look at the salient parts of a feature extractor.

1. The ``@register_query_feature`` decorator at the top registers the feature with MindMeld.

.. code-block:: python

    @register_query_feature(feature_name='average-token-length') 

The ``feature_name`` parameter specifies the name by which the extractor will be referenced in the
app's configuration file, ``config.py``. The feature name must be added as a key within the
'features' dictionary of the classifier config, as shown below. If the feature extractor function
has parameters, the corresponding value in the key-value pair must specify these parameters. If
there are no parameters, as in this case, an empty dictionary is sufficient.

.. code-block:: python
   :emphasize-lines: 15

    DOMAIN_CLASSIFIER_CONFIG = {
        ...
        ...
        ...

        'features': {
            "bag-of-words": {
                "lengths": [1, 2]
            },
            "edge-ngrams": {"lengths": [1, 2]},
            "in-gaz": {},
            "exact": {"scaling": 10},
            "gaz-freq": {},
            "freq": {"bins": 5},
            "average-token-length": {},
        }
    }

2. The arguments passed to the feature extractor can be accessed by the inner ``_extractor``
function.

.. code-block:: python

    def extract_average_token_length(**args):

The values of the parameters must be specified in the 'features' dictionary of the classifier
config as values corresponding to the appropriate feature keys.

3. The feature extractor returns an ``_extractor`` function which encapsulates the actual feature
extraction logic.

.. code-block:: python

    def _extractor(query, resources):

Query feature extractors have access to the ``query`` object, which contains the query text,
normalized query tokens, and system entity candidates.

4. The ``_extractor`` function must return a dictionary mapping feature names to their corresponding values.

.. code-block:: python

    return {'average_token_length': average_token_length}


Example of an Entity Feature Extractor
--------------------------------------

Entity features are similar to the query features described above with a few key differences. The
most important distinction is that entity features can only be used by the role classifier.
Specifying an entity feature in the domain classifier, intent classifier, or entity recognizer
config specifications will raise an error.

There are two other differences.

  1. Entity features are registered using a different decorator, ``@register_entity_feature``.

  2. The inner ``_extractor`` function of an entity feature extractor receives an ``example``
     object that contains information about the query and the extracted entities.

.. code-block:: python

    def _extractor(example, resources):
        query, entities, entity_index = example

The ``query`` object is the same as above, ``entities`` is a list of all the entities detected in
the query, and the ``entity_index`` specifies which of the ``entities`` the extractor function is
currently operating on.

Here's an example of an entity feature extractor that computes the starting character index for a
given entity.

.. code-block:: python

    @register_entity_feature(feature_name='entity-span-start')
    def extract_entity_span_start(**args):
        """
        Example entity feature that gets the start span for the given entity

        Returns:
            (function) A feature extraction function that returns the span start of the entity
        """
        def _extractor(example, resources):
            query, entities, entity_index = example
            features = {}

            current_entity = entities[entity_index]
            current_entity_token_start = current_entity.token_span.start

            features['entity_span_start'] = current_entity_token_start
            return features

        return _extractor
