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

Create the knowledge base
-------------------------
Describe what a knowledge base is and what it is used for. Describe options for storing the knowledge base. Show a simple example of creating a knowledge base by uploading a JSON file to ES via Workbench.

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

Optimize question answering
---------------------------
Introduce the topic of ranking for answer recommendations.

Deploy trained models to production
-----------------------------------
Show a simple example of the steps required to deploy to production