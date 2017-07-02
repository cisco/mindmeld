.. meta::
    :scope: private

Platform Architecture
=====================

The MindMeld Deep-Domain Conversational AI Platform provides a robust end-to-end pipeline for building and deploying intelligent data-driven conversational apps. The high level architecture of the platform is illustrated in the figure below.

.. image:: /images/architecture.png
    :align: center
    :name: architecture_diagram

We will now take a look at each of the components in the MindMeld platform.


Natural Language Processor
--------------------------

The Natural Language Processor (NLP) is tasked with understanding the user's natural language input. This involves processing the user query using a combination of techniques such as `pattern matching <https://en.wikipedia.org/wiki/Pattern_matching#Pattern_matching_and_strings>`_, `text classification <https://en.wikipedia.org/wiki/Text_classification>`_, `information extraction <https://en.wikipedia.org/wiki/Information_extraction>`_, and `parsing <https://en.wikipedia.org/wiki/Parsing>`_. The end goal is to produce a representation that captures all the salient pieces of information in the query. This summarized representation is then used by the app to decide on a suitable action or response to satisfy the user's goals.

The figure below shows the NLP output on a sample user input.

.. image:: /images/nlp.png
    :align: center


The Natural Language Processor analyzes the input using a hierarchy of machine-learned classification models, as introduced in Steps :doc:`3 <../quickstart/03_define_the_hierarchy>`, :doc:`6 <../quickstart/06_generate_representative_training_data>` and :doc:`7 <../quickstart/07_train_the_natural_language_processing_classifiers>` of the Step-by-Step Guide. In addition to the four layers of classifiers, the NLP also has modules for entity resolution and language parsing. The user query is processed sequentially by each of these six subcomponents in the left-to-right order shown in the :ref:`architecture diagram <architecture_diagram>` above. The role of each step in the NLP pipeline is described below.

Domain Classifier
~~~~~~~~~~~~~~~~~

The Domain Classifier performs the first level of categorization on a user query by classifying it into one of a pre-defined set of domains that can be handled by the app. Each domain is a unique area of knowledge with its own vocabulary and specialized terminology.

For instance, a conversational app serving as a "Smart Home Assistant" would be expected to handle several distinct tasks such as:

* Setting the temperature on the thermostat
* Toggling the light fixtures in different rooms
* Locking and unlocking different doors
* Controlling multimedia devices around the home
* Answering informational queries about time, weather, etc.

The vocabulary used for instructing the app to change the settings on a thermostat is very different from interacting with the television. These could therefore be modeled as separate domains - a ``thermostat`` domain for handling all interactions related to the thermostat and a ``multimedia`` domain for talking to media devices in the house. Personal assistants like Siri, Cortana, Google Assistant and Alexa are trained to handle more than a dozen different domains like ``weather``, ``navigation``, ``sports``, ``music``, ``calendar``, etc.

On the opposite end of the spectrum are apps with just one de facto domain. This is usually the case if all the functions that the app provides are conceptually related and span a single realm of knowledge. For instance, a "Food Ordering" app could potentially handle multiple tasks like searching for restaurants, getting more information about a particular restaurant, placing an order, etc. But the vocabulary used for accomplishing all of these tasks are highly shared, and hence could be modeled as one single domain called ``food``.

The number of domains thus depends on the scope of the conversational application. For apps with multiple domains, the :doc:`Domain Classifier User Guide <>` describes how Workbench can be used to train a machine-learned domain classification model.




  4.3.1.2 Intent Classifier
  4.3.1.3 Entity Recognizer
  4.3.1.4 Role Classifier
  4.3.1.5 Entity Recognizer
  4.3.1.6 Language Parser
4.3.2 Question Answerer
4.3.3 Dialogue Manager
4.3.4 Application Manager
4.3.5 Gateway

[ARCHIVED CONTENT BELOW]




.. raw:: html

    <style> .red {color:red} </style>

.. raw:: html

    <style> .green {color:green} </style>

.. raw:: html

    <style> .orange {color:orange} </style>

.. raw:: html

    <style> .pink {color:#DB7093} </style>

.. raw:: html

   <style> .indigo {color:#4B0082} </style>

.. role:: red
.. role:: green
.. role:: pink
.. role:: indigo
.. role:: orange


We next take a look at each of the classifiers within the MindMeld Parser one by one.



Intent Classifier
~~~~~~~~~~~~~~~~~

Once the domain for the user input has been determined, the next level of categorization is provided by the Intent Classifier. An **intent** refers to a very specific kind of informational or transactional user need. The user may want to book a flight, search for movies from a catalog, know about the weather conditions somewhere or set the temperature on their home thermostat. Each of these is an example of a user intent.

A domain can, and usually has multiple intents. For instance, the de facto "food" domain in a Food Ordering app would at least contain intents such as:

  +--------------------+------------------------------------------------------------------------------------------------+
  |    Intent          |  Description                                                                                   |
  +====================+================================================================================================+
  |search_restaurant   | Searching for restaurants matching a particular set of criteria                                |
  +--------------------+------------------------------------------------------------------------------------------------+
  |get_restaurant_info | Get general information about a selected restaurant like hours, cuisine, price range, etc.     |
  +--------------------+------------------------------------------------------------------------------------------------+
  |list_dishes         | List all the dishes available at a selected restaurant, optionally filtered by certain criteria|
  +--------------------+------------------------------------------------------------------------------------------------+
  |place_order         | Place an order for pick up or delivery                                                         |
  +--------------------+------------------------------------------------------------------------------------------------+

By convention, we use verbs to name our intents as they inherently refer to an action that needs to be taken.

The Intent Classifier, similar to the Domain Classifier uses a Machine-Learned text classification model that is trained using labelled training data. We train one intent classification model per domain and the system chooses the appropriate classifier model at runtime, based on the predicted domain for the input query. The output of the Intent Classifier is an intent label which allows us to identify the exact task that the user is trying to solve.

We describe how to build intent classification models in :doc:`Intent Classifier </intent_classification>`.


Entity Recognizer
~~~~~~~~~~~~~~~~~

After the user intent has been established by the Intent Classifier, the next step is to identify all the entities relevant to satisfying the user intent. An **entity** is any important word or phrase that provides further information about the user's end goal. For instance, if the user intent was to search for a movie, the relevant entities would be things like movie titles, genre, cast names, etc. If the intent was to update the thermostat, the entity would be the numerical value of the temperature to set the thermostat to.

For programmers, a good analogy is to think of intents as functions and entities as the arguments you pass into the function call. E.g:

* Set_thermostat (:red:`temperature` = 70)
* Get_weather_info (:green:`city` = 'San Francisco')
* Find_movies (:indigo:`release_year` = '2016', :pink:`actor` = 'Tom Hanks', :orange:`genre` = 'Drama').

The Entity Recognizer's job is to analyze the user input and extract all the entities relevant to the current intent. In NLP literature, this problem is commonly referred to as `Named Entity Recognition <https://en.wikipedia.org/wiki/Named-entity_recognition>`_.

The problem essentially consists of two parts:

1. Detect which spans of words within the input text correspond to entities of interest
2. Classify those detected text spans into a pre-determined set of entity types

The Entity Recognizer uses a Machine-Learned Sequence Labeling model to look at each word in the input query sequentially and assign a label to it. It is trained using labeled training data where queries are annotated to mark entity spans along with their corresponding entity types. We train a separate entity recognition model for each user intent since the types of entities required to satisfy the end goal vary from intent to intent. We will get into the details of build entity recognition models in :doc:`Entity Recognizer </entity_recognition>`.

At runtime, the Entity Recognizer loads the appropriate model, based on the predicted intent for the query. Once this step is done and we have extracted the relevant entities, we will finally have all the raw ingredients required to make sense out of the user input. The next step would be to put those together to build a semantic representation that encapsulates all the information necessary to execute the user's intended action.


Entity Resolver
~~~~~~~~~~~~~~~

The Entity Resolver transforms the entity spans extracted by the Entity Recognizer into canonical forms that can be looked up in a catalog or a Knowledge Base. For instance, the extracted entity :red:`"lemon bread"` may get resolved to :red:`"Iced Lemon Pound Cake"` and :green:`"SF"` may get resolved to :green:`"San Francisco"`. This problem of entity resolution is also referred to as `Entity Linking <https://en.wikipedia.org/wiki/Entity_linking>`_ in NLP literature.

The MindMeld Entity Resolver uses a resource called an **Entity Map** to transform extracted entities into their desired normalized forms. The chapters on :doc:`Entity Map </entity_map>` and :doc:`Entity Resolver </entity_resolution>` provide more details on the entity resolution step.


Role Classifier
~~~~~~~~~~~~~~~

Role Classification is the task of identifying predicates and predicate arguments. A **semantic role** in language is the relationship that a syntactic constituent has with a predicate. In Conversational NLU, a **role** represents the semantic theme a given entity can take. It can also be used to define how a named entity should be used for fulfilling a query intent. For example, in the query :red:`"Play Black Sabbath by Black Sabbath"`, the **title** entity :green:`"Black Sabbath"` has different semantic themes - **song** and **artist** respectively.

Treating Named Entity Recognition (NER) and Semantic Role Labeling (SRL) as separate tasks has a few advantages -

* NER models are hurt by splitting examples across fairly similar categories. Grouping entities with significantly overlapping entities and similar surrounding natural language will lead to better parsing and let us use more powerful models.
* Joint NER & SRL needs global dependencies, but fast & good NER models only do local. NER models (MEMM, CRF) quickly become intractable with long-distance dependencies. Separating NER from SRL let us use local dependencies for NER and long-distance dependencies in SRL.
* Role labeling might be a multi-label problem. With multi-label roles, we can use the same entity to query multiple fields.


Language Parser
~~~~~~~~~~~~~~~

The Semantic Parser is the last subcomponent within the MindMeld Natural Language Parser. It takes all the resolved entities and groups them into semantically related items. Each item represents a single real-world entity or concept along with all its describing attributes.

We provide more details in :doc:`Language Parser </language_parsing>`.


Dialogue Manager
----------------

The Dialogue Manager is responsible for directing the flow of the conversation. In contrast to other parts of the system that are stateless, the Dialogue Manager is stateful and maintains information about each state or step in the dialogue flow. It is therefore able to use historical context from previous conversation turns to move the dialogue along towards the end goal of satisfying the user's intent.

The Natural Language Generator (NLG) component frames the natural language response to be output to the user. It receives information about how the user's intent has been processed and uses that in conjunction with a set of pre-defined templates to construct a fluent natural language text response. We will go into further details in Natural Language Generator chapter.

Question Answerer
-----------------

In the context of Deep-Domain Conversational AI, Question Answering is the task of retrieving relevant documents from a large content catalog in response to a natural language question. A large-vocabulary content catalog is first imported into a **Knowledge Base**. The Question Answerer uses the structured output of the Language Parser to first construct a database query. The query is then executed on the Knowledge Base to retrieve a wide net of candidate answers to the query. Finally, these candidate answers are scored and ranked, and the top ranked results are returned as the most relevant documents to the natural language query.

The parameters and weights assigned to the various entity types determine the effect of those entities on the final ranking. More context is provided in the chapter on :doc:`Question Answerer </question_answering>`.
