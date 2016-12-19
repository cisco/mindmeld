Question Answerer
=================

.. raw:: html

    <style> .red {color:red} </style>

.. raw:: html

    <style> .green {color:green} </style>

.. raw:: html

    <style> .blue {color:blue} </style>

.. raw:: html

    <style> .orange {color:orange} </style>

.. role:: red
.. role:: green
.. role:: blue
.. role:: orange

The Question Answering module is responsible for retrieving relevant documents from the Knowledge Base. It takes as input the Natural Language Parser parse result, and outputs a ranked list of relevant documents from the content catalog. 

.. image:: images/question_answerer.png
   :target: _images/question_answerer.png

The goal is to map the parsed entities and associated slot values for a given query. In the following example, we want to convert the following query -

* Do you have some :orange:`orange` :blue:`chino` :red:`pants` on :green:`sale`?

to some sort of structured query (or logical form), which would like like this for Elasticsearch -

.. code-block:: javascript

	{ES query example goes here}

Query Clauses
-------------

Each entity mode has it's own format of mapping to the KB query language. 

Filter Entity Clauses
~~~~~~~~~~~~~~~~~~~~~

For each "filter" mode entity, a "must" clause is generated to apply to the Elasticsearch Knowledge Base.

.. code-block:: javascript

	{Filter clause example goes here}

Note that the entity map supports both conjunctive (i.e. 'and') clauses as well as disjunctive (i.e. 'or') clauses. These get translated into the appropriate boolean logic in the query language for querying the Knowledge Base. In the following examples, if we had SQL-like boolean clause, we would have -

.. code-block:: text

	"raincoat": "mindmeld_category:coats AND function:water-repellant" => 'category=coats AND function=water-repellant'
	"3-4 year olds": "size:3y,4y,3-4y" => 'size=3y OR size=4y OR size=3-4y'

Search Entity Clauses
~~~~~~~~~~~~~~~~~~~~~

Search entity clauses take the following form -

.. code-block:: javascript

	{Search clause example goes here}

To control the text-relevance match score in ranking, a "method" field can be specified in the ranking configuration. To exercise fine-grained control over the effect of the Lucene Practical Scoring Function in Elasticsearch, more knobs are available in the **es_mapping** file. More information is available in the sections on Controlling Text Relevance and Creating the Ranking Config below.

Search Terms Clauses
~~~~~~~~~~~~~~~~~~~~

An additional search clause is created containing any statistically significant words or phrases contained in the query. The 'query-docfreq' index, stored in the intent classifier model, is used to assess which terms and ngrams are 'statistically significant' by ignoring any terms which appear in more than a specified number of queries (called the 'query term frequency cutoff'). In addition, a small stopword list is also used to filter any very common words which might appear in the query.

.. code-block:: text

	movies about somali pirates --> 'somali OR pirates OR "somali pirates"'

Similar to the search entity clauses, configurations can be specified to alter the effect of text-relevance in the scoring functions used for search ranking. We will see how to tune these configurations below.

Creating The Question Answerer
------------------------------

To generate the final ranking of the retrieved candidate results, we want to control the impact each of the entity modes have on the final ranking. Once an ES query is generated from the various entity clauses, an ES Function Score query is generated. This implements a ranking formula to blend text relevance and popularity with any "sort" entities.  The ranking function is the linear sum of ranking components for text relevance, popularity and the sort entities.

Define a ranking_config.json file as follows -

.. code-block:: python

	from mindmeld.question_answering import QuestionAnswerer

	# Define the ranking configs
	ranking_coeff = {
	    "sort_popularity_coeff": 0.015404286207392436,
	    "sort_coeff": 10.0,
	    "common_term_cutoff_freq": 0.001,
	    "popularity_coeff": 1
	}

	# Create the QuestionAnswerer object
	qa = QuestionAnswerer(ranking_coefficients=ranking_coeff)

	# Generate ranked results using the QA object
	results = qa.answer(query, entities)

	print results

Ranking Coefficients -

+-------------------------+-----------------------------------------------------------------------+
| Parameter               | Definition                                                            |
+=========================+=======================================================================+
| popularity_coef         | weight given to the normalized popularity factor                      |
+-------------------------+-----------------------------------------------------------------------+
| sort_coeff              | weight given to the normalized sort entity factor                     |
+-------------------------+-----------------------------------------------------------------------+
| common_term_cutoff_freq | maximum frequency of terms that should not be treated as common terms |
+-------------------------+-----------------------------------------------------------------------+
| sort_popularity_coeff   | popularity weight when a sort entity is detected                      |
+-------------------------+-----------------------------------------------------------------------+

Controlling Text Relevance
--------------------------

Search entities are applied as a text match query against their specified field.  It is recommended that the ElasticSearch analyzer used for the specified field employ a shingle filter so that the text relevance score takes into account word proximity in addition to the presence of individual words.

A JSON entry for "text_relevance" can be added to the ranking config file -

.. code-block:: python

	text_relevance_params = {
		"match_boost_method": "match_backoff",
		"important_terms_field": "name",
		"search_term_method": "match_backoff",
		"search_term_boost": 0.5,
		"search_entity_method": "match_backoff",
		"minimum_should_match": "75%"
	}

	qa = QuestionAnswerer(ranking_coefficients=ranking_coeffs, text_relevance_params=text_relevance_params)
	results = qa.answer(query, entities)

	print results

Standalone search terms are applied as a text match query against the text search field. If the **es_mapping** defines an 'all_terms' field, that field will be used for text match. If 'all_terms' does not exist, the text search field will be the '_all' field, which exists in ElasticSearch by default and indexes all of the other text fields in the document. The text match will be performed using whatever normalizer and tokenizer is specified for that field in ElasticSearch.

The "match_boost_method" parameter is used to give a ranking boost to any documents where all search entities match exactly with their target fields. This boost is necessary to filter out irrelevant documents when applying sorts.  It is also helpful for displaying only exact matching results on the front end. The method type can be configured in the following ways -

  +---------------+---------------------------------------------------------------------------------------+
  | Method Type   | Description                                                                           |
  +===============+=======================================================================================+
  | exact         | exact whole-field matching                                                            |
  +---------------+---------------------------------------------------------------------------------------+
  | match         | basic matching, "tropical island" scored as "tropical" + "island" + "tropical island" |
  +---------------+---------------------------------------------------------------------------------------+
  | match_and     | like basic matching, but requiring a match for each token                             |
  +---------------+---------------------------------------------------------------------------------------+
  | match_backoff | match_and + exact, plus match_and on name_search_field                                |
  +---------------+---------------------------------------------------------------------------------------+

If the "important_terms_field" is specified, an additional clause is added to boost matches on that field. The "search_term_method" parameter is used to determine the matching method for search terms (with the same abstractions as "match_boost_method"). 

Tuning The Ranking Algorithm
----------------------------

For boostrapping applications where no prior search logs are available, the ranking will need to be hand tuned. The process is normally as follows -

#. Collect a set of few hundred (or a thousand) diverse, representative queries
#. Run the queries through the parse + QA system with an initial set of configurations
#. Analyze the results for Top 1 or Top K accuracy (depending on the use case)
#. Modify the configs to improve accuracy results for bulk of the misses (without compromising the correct ones)
#. Repeat from Step 2

Once you have launched the app and have collect a large amount of click data (or if you have prior search logs that can be used as a proxy), the above parameters can be learned automatically (instead of repeatedly tuning by hand). There is a vast amount of literature around search and recommendations using Machine Learning. Using the clicks as positive example and sampling on negatives, you can use F1/F2/F0.5 scores or other evaluation metrics such as NDCG, MRR and MAP to tune your models.