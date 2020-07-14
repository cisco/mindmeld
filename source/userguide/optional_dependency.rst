Managing Optional Dependencies
==============================

MindMeld depends on Duckling for numerical parsing and Elasticsearch for Entity Resolution and Question Answering. If your application does not need these components or the fully optimized versions of these components, you can configure your MindMeld application to not rely on Duckling or Elasticsearch.


Turning off Duckling
--------------------

To turn off Duckling, specify an empty dictionary for the ``'system_entity_recognizer'`` key:

.. code-block:: python

   NLP_CONFIG = {
       'system_entity_recognizer': {}
   }


Turning off ElasticSearch for Entity Resolution
-----------------------------------------------

If you choose not to use Elasticsearch, MindMeld provides a simple baseline version of entity resolution. This Exact Match Model only resolves to an object when the text exactly matches a canonical name or synonym. To use the Exact Match Model, add the following to your app config (``config.py``) located in the top level of your app folder:

.. code-block:: python

    ENTITY_RESOLVER_CONFIG = {
        'model_type': 'exact_match'
    }

This Exact Match model only resolves to an object when the text exactly matches a canonical name or synonym. Alternatively, when your ``mapping.json`` is empty (i.e. when your application does not include any information for entity resolution), MindMeld will not make a call to Elasticsearch.
