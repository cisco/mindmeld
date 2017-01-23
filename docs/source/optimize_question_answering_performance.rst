Step 9: Optimize Question Answering Performance
===============================================

The Question Answering module is responsible for ranking results retrieved from the Knowledge Base, based on some notion of relevance. The relevance of each document is represented by a positive floating point number - the ``score``. The higher the score, the more relevant the document. MindMeld Workbench offers a robust, built-in "ranking formula" for defining a general-purpose scoring function. However, in cases where the default ranking formula is not sufficient in ensuring good performance across a large number of test queries, MindMeld Workbench provides a facility for defining custom ranking formulae. The concept of "performance" is explained in Section 1.10.4 on "Evaluation Metrics".

Sorting
~~~~~~~

Among the various signals used in computing the relevance score, sorting is an important operation offered by MindMeld Workbench. Sorting is applicable on any real-valued field in the Knowledge Base (either ascending or descending order). The Question Answering module gets its cue to invoke the sorting function based on the presence of ``sort`` entities. If one or more sort entities are detected, the documents with resolved numerical field values corresponding to those entities will get a boost in the score function. Additionally, a decay is applied to this sorting boost to ensure a balance between the applied sort and other relevance signals.

For example, consider the following query:

* What are the cheapest doughnuts available in Kwik-E-Mart?

Let's say we have the following documents in the Knowledge Base:

.. code-block:: javascript

  {
    "item_id": 1,
    "item_name": "Pink Doughnut",
    "price": 20
  },
  {
    "item_id": 2,
    "item_name": "Green Doughnut",
    "price": 12
  },
  {
    "item_id": 3,
    "item_name": "Yellow Doughnut",
    "price": 15
  }
  ...

The Natural Language Processor would detect ``cheapest`` as a sort entity and populates the context object accordingly:

.. code-block:: python

  query = "What are the cheapest doughnuts available in Kwik-E-Mart?"
  context = {
    'domain': 'item_information',
    'intent': 'order_item',
    'entities': [
      {
        'type': 'item_name'
        'mode': 'search',
        'text': 'doughnut'
        'value': 'item_name:doughnut',
        'chstart': 22,
        'chend': 30
      },
      {
        'type': 'pricesort',
        'mode': 'sort',
        'text': 'cheapest',
        'value': 'price:asc',
        'chstart': 13,
        'chend': 20
      }
    ]
  }

  results = qa.get(index='items', query, context)
  print results

The final ranking of doughnut names that MindMeld Workbench returns would be the following:

.. code-block:: javascript

  "Green Doughnut",
  "Pink Doughnut",
  "Yellow Doughnut"

Text Relevance
~~~~~~~~~~~~~~

In general, "Text Relevance" refers to the algorithm used to calculate how *similar* the contents of a full-text field are to a full-text query string. The Question Answerer offered by MindMeld Workbench uses a standard similarity algorithm called the `TF_IDF <https://en.wikipedia.org/wiki/Tf-idf>`_ algorithm. TF-IDF is a widely used text similarity statistic intended to reflect how important a word is to a document collection. This is usually normalized by the field-length values so that greater simiality emphasis is placed on shorter field values.

Consider the following example documents on three different products:

.. code-block:: javascript

  {
    "item_id": 1,
    "item_name": "Pink Frosty Doughnuts"
  },
  { 
    "item_id": 2,
    "item_name": "Pink Sprinklicious Doughnuts"
  },
  {
    "item_id": 3,
    "item_name": "Frosty Yellow Doughnuts With Frosty Sprinkles"
  }

For an incoming query like -

* "I want some frosty doughnuts"

The returned list of documents as per text relevance would be:

.. code-block:: javascript

  {
    "item_id": 1,
    "item_name": "Pink Frosty Doughnuts"
  },
  {
    "item_id": 3,
    "item_name": "Frosty Yellow Doughnuts With Frosty Sprinkles"
  },
  {
    "item_id": 2,
    "item_name": "Pink Sprinklicious Doughnuts"
  }

* item_id 1 is more relevant because it's ``item_name`` is short
* item_id 3 comes next because "frosty" appears twice and "doughnut" appears once
* item_id 2 is the last - only "doughnut" matched

The Question Answerer's **get** method offers a versatile set of arguments for controlling text relevance. The ``minimum_should_match`` parameter specifies what percentage of query terms should match with the field value (at least). For the above example, if we wanted to specify more stringent match criteria - e.g both "frosty" and "doughnut" must appear in the returned documents - the ``minimum_should_match`` argument can be used as follows:

.. code-block:: python

  # All query terms must match the terms in the field value
  qa.get(index='items', query, context, minimum_should_match=100)

The default value of the ``minimum_should_match`` parameter is set to 75%.

While the above example gives a glimpse of the text-matching strategies available in MindMeld Workbench, much more complex functionality (such as "Exact Matching" and "Boosting Query Clauses") is available in the User Guide chapter on Knowledge Base.

Advanced Settings
~~~~~~~~~~~~~~~~~

When creating the Knowledge Base index, all fields in the data go through a process called "Analysis". Analyzers can be defined per field to define the following:

* Tokenizing a block of text into individual terms before adding to inverted index
* Normalizing these terms into a standard form to improve searchability

When searching on a full-text field, the query string is passed through the same analysis process, to ensure that we are searching for terms in the same form as those that exist in the index.

In MindMeld Workbench, you can optionally define custom analyzers per field by specifying an **es_mapping.json** file at the application root level. While the default MindMeld Workbench Analyzer uses a robust set of character filtering operations for tokenizing, custom analyzers can be handy for special character/token handling.

For example, lets say we have a store named *"Springfield™ store"*. We want the indexer to ignore characters like "™" and "®" since users never specify these in their queries. We need to define special ``char_filter`` and ``analyzers`` mappings as follows:

File **es_mapping.json** -

.. code-block:: javascript

  {
    "field_mappings": {
      "store_name": {
        "type": "string",
        "analyzer": "my_custom_analyzer"
      }
    },
    "settings": {
      "char_filter": {
        "remove_tm_and_r": {
            "pattern":"™|®",
            "type":"pattern_replace",
            "replacement":""
        }
      },
      "analyzers": {
        "my_custom_analyzer": {
          "type": "custom",
          "tokenizer": "whitespace",
          "char_filter": [
            "remove_tm_and_r"
          ]
        }
      }
    }
  }

More information on custom analyzers and the **es_mapping.json** file is available in the :ref:`User Manual <userguide>`. Example mapping files for a variety of use-cases and content types are also provided.

Evaluation Metrics
~~~~~~~~~~~~~~~~~~

In Information Retrieval, Top 1 accuracy, Top K accuracy, Precision, Recall and F1 scores are all great evaluation metrics to get started. To optimize Precision and Recall, you will need to create a "relevant set" of documents for each query in your test set. This relevant set is typically generated by a human expert, or by repeated error analysis.

.. code-block:: text

  Query                                   Relevant Set

  "get me a doughnut"                     1, 3, 28, 67, 253, 798
  "i want a lemon Squishee"               4, 363, 692
  "can I get a buzz cola"                 291
  "pink frosty sprinklicous doughnut"     67
  ...

For a thorough evaluation, it is advisable to create relevant sets for thousands of test queries for the initial pass. This bank of queries and their expected results should grow over time into hundreds of thousands, or even millions of query examples. This then becomes the golden set of data on which future models can be trained and evaluated.

Custom Ranking Functions
~~~~~~~~~~~~~~~~~~~~~~~~

In general, you should not have to worry about writing your own scoring function for ranking. MindMeld Workbench provides numerous knobs and dials for detailed, granular control over the built-in scoring function. However, in cases where the existing scoring function simply does not fit the needs of your application, you can specify your own custom scoring function for ranking. Define your custom ranking function in the **my_app.py** file as follows:

File **my_app.py** -

.. code-block:: python

  @app.qa.handle(domain='items')
  def items_ranking_fn(query, context, document):
    # Custom scoring logic goes here.
    score = compute_doc_score(query, context, document)
    return score

The custom ranking function can then be used in the **get** method of the QuestionAnswerer object.

.. code-block:: python

 # Assume KnowledgeBase object has been created and
 # the data is loaded into the 'items' index.

 # Get ranked results from KB
 ranked_results = qa.get(index='stores', query,
        context, ranking_fn=items_ranking_fn)

The function gets applied to each document in the retrieved set to compute their final scores, and the ranked set is then returned.

.. note::

  A note on system latency - In applications where hundreds or thousands of documents are retrieved on each query, applying a custom scoring function on each document can make the requests terribly slow, depending on how well the function is engineered. Please be mindful of request latencies and overall system performance when designing custom ranking functions.

Learning To Rank
~~~~~~~~~~~~~~~~

Given the right kind of training data (and lots of it), Machine Learning methods can be applied for ranking in a variety of ways. To learn how to develop a Machine Learning approach to ranking, i.e. `Learning To Rank <https://en.wikipedia.org/wiki/Learning_to_rank>`_, please refer to the guidelines on assembling the right kind of training data and building models in the :ref:`User Manual <userguide>`.

