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

The number of domains thus depends on the scope of the application. For apps with multiple domains, the :doc:`Domain Classifier User Guide <domain_intent_classifiers>` describes how Workbench can be used to train a machine-learned domain classification model.


Intent Classifier
~~~~~~~~~~~~~~~~~

Once the domain has been determined, the Intent Classifier provides the next level of categorization by assigning each query to one of the intents defined for the app. The intent reflects what the user is trying to accomplish. For instance, the user may want to book a flight, search for movies from a catalog, know about the weather conditions or set the temperature on their home thermostat. Each of these is an example of a user intent. The intent also prescribes a specific action or answer type which defines the desired outcome for the query.

Each domain in a conversational app usually has multiple intents. By convention, intent names are verbs that describe what the user is trying accomplish. Here are some example intents from the ``food`` domain in a "Food Ordering" app.

+---------------------+-------------------------------------------------------------------------------------------+
| Intent              | Description                                                                               |
+=====================+===========================================================================================+
| search_restaurant   | Searching for restaurants matching a particular set of criteria                           |
+---------------------+-------------------------------------------------------------------------------------------+
| get_restaurant_info | Get information about a selected restaurant like hours, cuisine, price range, etc         |
+---------------------+-------------------------------------------------------------------------------------------+
| browse_dish         | List dishes available at a selected restaurant, filtered by given criteria                |
+---------------------+-------------------------------------------------------------------------------------------+
| place_order         | Place an order for pick up or delivery                                                    |
+---------------------+-------------------------------------------------------------------------------------------+

Every domain has its own separate intent classifier for categorizing the query into one of the intent defined within that domain. The app chooses the appropriate intent model at runtime, based on the predicted domain for the input query. Refer to the :doc:`Intent Classifier User Guide <domain_intent_classifiers>` for details on training intent classification models using Workbench.


Entity Recognizer
~~~~~~~~~~~~~~~~~

The next step in the NLP pipeline, the Entity Recognizer, identifies all the entities that are relevant to a given intent. An entity is any important word or phrase that provides the necessary information to understand and fulfill the user's end goal. For instance, if the user intent is to search for a movie, the relevant entities would be movie titles, genre, cast names, etc. If the intent is to update the thermostat, the entity would be the numerical value of the temperature to set the thermostat to.

Each intent within a domain usually has multiple entities. By convention, entity names are nouns that describe the entity type. Here are some examples of entity types that might be required for different conversational intents.

+---------+-------------------+-----------------------------------------------------------------------+
| Domain  | Intent            | Entity Types                                                          |
+=========+===================+=======================================================================+
| weather | check_weather     | location, day                                                         |
+---------+-------------------+-----------------------------------------------------------------------+
| movies  | find_movie        | movie_title, genre, cast, director, release_date, rating              |
+---------+-------------------+-----------------------------------------------------------------------+
| food    | search_restaurant | restaurant_name, cuisine, dish_name, location, price_range, rating    |
+---------+-------------------+-----------------------------------------------------------------------+
| food    | browse_dish       | dish_name, category, ingredient, spice_level, price_range             |
+---------+-------------------+-----------------------------------------------------------------------+

Since the set of relevant entity types might differ for each intent (even within the same domain), every intent has its own separate entity recognizer. Once the domain and intent have been established at runtime, the app uses the appropriate entity model to detect entities in the query that are specific to the predicted intent. We will get into the details of building machine-learned entity recognition models using Workbench in the :doc:`Entity Recognizer User Guide <entity_recognizer>`.


Role Classifier
~~~~~~~~~~~~~~~

Role Classification is the task of identifying predicates and predicate arguments. A **semantic role** in language is the relationship that a syntactic constituent has with a predicate. In Conversational NLU, a **role** represents the semantic theme a given entity can take. It can also be used to define how a named entity should be used for fulfilling a query intent. For example, in the query :red:`"Play Black Sabbath by Black Sabbath"`, the **title** entity :green:`"Black Sabbath"` has different semantic themes - **song** and **artist** respectively.

Treating Named Entity Recognition (NER) and Semantic Role Labeling (SRL) as separate tasks has a few advantages -

* NER models are hurt by splitting examples across fairly similar categories. Grouping entities with significantly overlapping entities and similar surrounding natural language will lead to better parsing and let us use more powerful models.
* Joint NER & SRL needs global dependencies, but fast & good NER models only do local. NER models (MEMM, CRF) quickly become intractable with long-distance dependencies. Separating NER from SRL let us use local dependencies for NER and long-distance dependencies in SRL.
* Role labeling might be a multi-label problem. With multi-label roles, we can use the same entity to query multiple fields.



We describe how to build role classification models with Workbench in :doc:`Role Classifier </role_classifier>`.





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








Entity Resolver
~~~~~~~~~~~~~~~

The Entity Resolver transforms the entity spans extracted by the Entity Recognizer into canonical forms that can be looked up in a catalog or a Knowledge Base. For instance, the extracted entity :red:`"lemon bread"` may get resolved to :red:`"Iced Lemon Pound Cake"` and :green:`"SF"` may get resolved to :green:`"San Francisco"`. This problem of entity resolution is also referred to as `Entity Linking <https://en.wikipedia.org/wiki/Entity_linking>`_ in NLP literature.

The MindMeld Entity Resolver uses a resource called an **Entity Map** to transform extracted entities into their desired normalized forms. The chapters on :doc:`Entity Map </entity_map>` and :doc:`Entity Resolver </entity_resolution>` provide more details on the entity resolution step.




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
