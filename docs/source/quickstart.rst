Building a conversational interface in 11 steps
===============================================

This section outlines the step-by-step approach to building conversational interfaces using MindMeld workbench. This section provides four real-world examples (tutorials) to illustrate each of the steps in practice.


This section provides a high-level overview introducing the steps.

Select the right use case
-------------------------
Selecting the right use case is critical. Selecting an unrealistic or incorrect use case will render even the smartest voice or chat assistant dead on arrival. 
The best user cases today typically mimic an existing, familiar real-world interaction. This is the best way to ensure that users will know what they can ask.
Any use case must offer a practical path for collecting the training data required for the app.
The best use cases provide a clear value or utility where voice can be a faster way to find information or accomplish a task.

Define the dialog state flows
-----------------------------
In my view, this is where we define the natural language response templates which should be returned at each dialogue state in an interaction. We should illustrate a simple flow in a flow chart and then in a snippet of python code which illustrates how the logic is implemented in the dialogue manager.

Define the domain, intent, entity and role hierarchy
----------------------------------------------------
Show and describe a simple diagram which illustrates the domain, intent, entity and role hierarchy.  Show the directory structure which captures this hierarchy for a simple example.

Create the Knowledge Base
-------------------------
A Knowledge Base is a repository for storing complex, structured and unstructured information relevant to a content catalog. In Workbench, we support the use of Elasticsearch - a powerful, distributed, Full Text Search-featured search engine built on top of Lucene. The following section assumes that you have Elasticsearch setup on your cloud or on-premise infrastructure.

To import your catalog data into Elasticsearch, 3 steps are required -

#. Define a KB Configuration (operational specifications for your KB cluster)
#. Define a Schema (structural definition for your data)
#. Import your data

Define a **kb_conf.json** file at the top-level as follows -

.. code-block:: javascript

  {
    "knowledgebase-type": "elasticsearch",
    "elasticsearch-host": "search.prod", // URL alias to your ES cluster
    "elasticsearch-port": 9200,
    "elasticsearch-index-name": "kwik-e-mart"
  }

A "schema" defines the structure of your Knowledge Base. Define a **schema.json** file at the top-level as follows -

.. code-block:: javascript

  {
    "object-type": "stores", // name for the table of documents
    "popularity-field": "default", // name of the field (of type INTEGER or REAL) to be used for default popularity ranking
    "fields": [
      {
        "old-name": "store_id", // name of field in the data source to be imported
        "new-name": "id", // name of corresponding field in KB
        "type": "ID" // data type for this field (ID, TEXT, LIST, INTEGER, REAL, DATE, TIME, JSON)
      },
      {
        "old-name": "store_name",
        "new-name": "name",
        "type": "TEXT",
        "detect-entities": true // flag which indicates if this field should be used for extracting entity data files
      },
      {
        "old-name": "full_address",
        "new-name": "address",
        "type": "TEXT"
      },
      {
        "old-name": "street",
        "new-name": "street",
        "type": "TEXT",
        "detect-entities": true
      },
      {
        "old-name": "intersection",
        "new-name": "intersection",
        "type": "TEXT",
        "detect-entities": true
      },
      {
        "old-name": "open_time",
        "new-name": "open_time",
        "type": "TIME"
      },
      {
        "old-name": "close_time",
        "new-name": "close_time",
        "type": "TIME"
      },
    ]
  }

We are now ready to import the data into the Knowledge Base. The following example assumes the data is stored as JSON flat files locally -

.. code-block:: python

  from mmworkbench.knowledge_base import KnowledgeBase

  # Initialize the KB
  kb = KnowledgeBase(app_name, app_path, domain_name)

  # Read the data
  with open('store_data.json') as json_data:
    data = json.load(json_data)

  # Import Data to KB
  kb.import_data(data, format='json')

Running **import_data** will setup a new Elasticsearch index with the latest imported data.


Generate representative training data
-------------------------------------
Components in Mindmeld Workbench utilize Supervised Learning models to analyze a user's query and derive meaning out of it. To train each of these components, we typically require thousands to millions of *labeled* queries to build powerful models. **It is critical that you obtain high-quality, representative training data** to ensure high accuracy. The training data serves as the ground truth for the models, so it is imperative that the ground truth data is clean and represents the exact use-case that you are training the model for.

Some strategies for collecting training data are -

#. Human Data Entry
#. Mining The Web
#. Crowdsourcing
#. Operational Logs (Customer Service, Search etc.)

For the **store_information** domain, here are snippets of training examples for Intent Classification -

* **greet**

.. code-block:: text

  Hi
  Hello
  Good morning
  ...

* **get_close_time**

.. code-block:: text

  when does the elm street store close?
  what's the shut down time for pine & market store?
  ...

.. _Amazon Mechanical Turk: https://www.mturk.com

To collect data at scale, platforms such as `Amazon Mechanical Turk`_ are popular and relatively inexpensive to get an initial dataset. The spec to send out to "Turkers" should be **highly precise** but should also encourage **language diversity** (formal and informal variants, slang, common abbreviations etc.) within the task. Lack of clarity or specificity can lead to noisy data, which hurts training accuracy.

.. raw:: html

    <style> .green {color:green} </style>

.. role:: green

Task Spec - :green:`"Information on whether a particular Kwik-E-Mart store is open."`

.. code-block:: text

  Scenario: You are interested in knowing if a particular Kwik-E-Mart store is open.

  Task: Ask a Conversational Agent if a specific Kwik-E-Mart store is open. You may specify a 
  time and/or location to enquire if the store is open. Please try to vary your phrasing on each
  query.

  Examples:

  Is the Central Plaza Kwik-E-Mart open now?
  The store near Pine & Market - is it open?
  Is the Rockerfeller Kwik-E-Mart on 30th Street open for business?
  Can you check if the Market St store is open at 6 pm tomorrow?

Tasks can also be micro-targeted at specific sections of the content catalog. For example, if the initial collection tasks did not yield many queries with **intersections** examples, we can create a task specific to that section -

Task Spec - :green:`"Queries covering popular intersections in the US"`

.. code-block:: text

  Scenario: You are interested in the closing times of Kwik-E-Mart stores at
  specific intersections.

  Task: Ask a Conversational Agent about the closing times of Kwik-E-Mart stores based on
  it's nearest intersection location. Please try to vary your phrasing on each query.

  Examples:

  When does the Bush & Kearny store close?
  What is the closing time of Kwik-E-Mart on 24th and Mission?
  Can you tell me when the 5th & Market one closes?

Annotating Data
~~~~~~~~~~~~~~~

To train the MindMeld Entity Recognizer, we need to add annotations to our training data to identify all the entities within our collected queries. Mark up the parts of the query that correspond to an entity in the following syntax -

* Enclose the entity in curly braces
* Follow the entity with its type
* Use the pipe character as separator

Examples -

.. code-block:: text

  Is the {Central Plaza|name} Kwik-E-Mart open {now|time}?
  The store near {Pine & Market|intersection} - is it open?
  Is the {Rockerfeller|name} Kwik-E-Mart on {30th Street|street} open for business?
  Can you check if the {Market St|street} store is open at {6 pm tomorrow|time}?

.. note::

  Pro tip - Academic datasets (though instrumental in researching advanced algorithms), are not always reflective of real-world conversational data. Therefore, datasets from popular conferences such as TREC and ACM-SIGDIAL might not be the best choice for developing production applications.


Train the domain and intent models
----------------------------------
Introduce the topic of loading training data, training text classification models, measuring CV and held-out performance.

Train the entity and role recognizers
-------------------------------------
Introduce the topic of loading training data, training entity and role classification models, measuring CV and held-out performance.

Train the entity resolvers
--------------------------
Introduce the topic of loading training data, training entity resolution models, measuring CV and held-out performance, performing disambiguation.

Implement the semantic parser
-----------------------------
Introduce the topic of semantic and dependency parsing. Illustrate a simple example of a rule-based or grammar-based parser which groups entities into a tree data structure.

Optimize Question Answering
---------------------------
The Question Answering module is responsible for retrieving relevant documents from the Knowledge Base. It first maps the resolved entities to a structured logical query form, executes the structured query on Elasticsearch, and then ranks the retrieved candidates based on some learned or specified relevance parameters.

To generate the final ranking of the retrieved candidate results, we want to control the impact each of the entity modes have on the final ranking. The ranking formula is a blend of text relevance, popularity and any “sort” entities (if present). Define your ranking coefficients and instantiate a QuestionAnswerer object as follows -

.. code-block:: python

  from mmworkbench.question_answering import QuestionAnswerer

  # Define the ranking configs
  ranking_coeff = {
      "sort_coeff": 0.01, # weight given to the normalized sort entity factor
      "common_term_cutoff_freq": 0.001, # document frequency threshold to prevent scoring high-frequency terms (absolute or relative)
      "popularity_coeff": 1.0 # weight given to the normalized popularity factor
  }

  # Create the QuestionAnswerer object
  qa = QuestionAnswerer(ranking_coefficients=ranking_coeff)

  # Generate ranked results using the QA object
  results = qa.answer(query, entities)

  print results

.. _Question Answering: question_answering.html

Detailed explanations on all ranking coefficients are available in the User Guide chapter on `Question Answering`_. You can also use find additional configurations for finer-grained control on Text Relevance. Check out "Tuning The Ranking Algorithm" section in that chapter for a step-by-step guide on optimizing the parameters by hand-tuning or Machine Learning.


Deploy trained models to production
-----------------------------------
Show a simple example of the steps required to deploy to production