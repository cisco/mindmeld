Platform Architecture
=====================

The MindMeld Deep-Domain Conversational AI Platform provides a robust end-to-end pipeline for building and deploying intelligent data-driven conversational apps. The high level architecture of the platform is illustrated in the figure below.

.. image:: /images/architecture.png
    :align: center
    :name: architecture_diagram

We will now take a look at each of the components in the MindMeld platform.

.. _arch_nlp:

Natural Language Processor
--------------------------

The Natural Language Processor (NLP) is tasked with understanding the user's natural language input. This involves processing the user query using a combination of techniques such as `pattern matching <https://en.wikipedia.org/wiki/Pattern_matching#Pattern_matching_and_strings>`_, `text classification <https://en.wikipedia.org/wiki/Text_classification>`_, `information extraction <https://en.wikipedia.org/wiki/Information_extraction>`_, and `parsing <https://en.wikipedia.org/wiki/Parsing>`_. The end goal is to produce a representation that captures all the salient pieces of information in the query. This summarized representation is then used by the app to decide on a suitable action or response to satisfy the user's goals.

The figure below shows the NLP output on a sample user input.

.. image:: /images/nlp.png
    :align: center
    :name: nlp_output


The Natural Language Processor analyzes the input using a hierarchy of machine-learned classification models, as introduced in Steps :doc:`3 <../quickstart/03_define_the_hierarchy>`, :doc:`6 <../quickstart/06_generate_representative_training_data>` and :doc:`7 <../quickstart/07_train_the_natural_language_processing_classifiers>` of the Step-by-Step Guide. In addition to the four layers of classifiers, the NLP also has modules for entity resolution and language parsing. The user query is processed sequentially by each of these six subcomponents in the left-to-right order shown in the :ref:`architecture diagram <architecture_diagram>` above. The role of each step in the NLP pipeline is described below.


.. _arch_domain_model:

Domain Classifier
~~~~~~~~~~~~~~~~~

The Domain Classifier performs the first level of categorization on a user query by classifying it into one of a pre-defined set of domains that can be handled by the app. Each domain is a unique area of knowledge with its own vocabulary and specialized terminology.

For instance, a conversational app serving as a "Smart Home Assistant" would be expected to handle several distinct tasks such as:

* Setting the temperature on the thermostat
* Toggling the light fixtures in different rooms
* Locking and unlocking different doors
* Controlling multimedia devices around the home
* Answering informational queries about time, weather, etc.

The vocabulary used for instructing the app to change the settings on a thermostat is very different from interacting with the television. These could therefore be modeled as separate domains - a ``thermostat`` domain for handling all interactions related to the thermostat and a ``multimedia`` domain for talking to media devices in the house. Personal assistants like Siri, Cortana, Google Assistant and Alexa are trained to handle dozens of different domains like ``weather``, ``navigation``, ``sports``, ``music``, ``calendar``, etc.

On the opposite end of the spectrum are apps with just one de facto domain. This is usually the case if all the functions that the app provides are conceptually related and span a single realm of knowledge. For instance, a "Food Ordering" app could potentially handle multiple tasks like searching for restaurants, getting more information about a particular restaurant, placing an order, etc. But the vocabulary used for accomplishing all of these tasks are highly shared, and hence could be modeled as one single domain called ``food``.

The number of domains thus depends on the scope of the application. For apps with multiple domains, the :doc:`Domain Classifier User Guide <domain_classifier>` describes how Workbench can be used to train a machine-learned domain classification model.


.. _arch_intent_model:

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

Every domain has its own separate intent classifier for categorizing the query into one of the intent defined within that domain. The app chooses the appropriate intent model at runtime, based on the predicted domain for the input query. Refer to the :doc:`Intent Classifier User Guide <intent_classifier>` for details on training intent classification models using Workbench.


.. _arch_entity_model:

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


.. _arch_role_model:

Role Classifier
~~~~~~~~~~~~~~~

The Role Classifier is the last level in the 4-layer NLP classification hierarchy. It assigns a differentiating label, called a role, to the entities extracted by the entity recognizer. Sub-categorizing entities in this manner is only necessary where an entity of a particular type can have multiple meanings depending on the context. For example, “9 AM” and “5 PM” could both be classified as time entities, but one might need to be interpreted as playing the role of an opening time and the other as playing the role of a closing time. The role classifiers label such entities with the appropriate roles.

Here are examples of some entity types that might require role classification when dealing with certain intents.

+---------+------------------+-------------+----------------------+
| Domain  | Intent           | Entity Type | Role Types           |
+=========+==================+=============+======================+
| meeting | schedule_meeting | time        | start_time, end_time |
+---------+------------------+-------------+----------------------+
| travel  | book_flight      | location    | origin, destination  |
+---------+------------------+-------------+----------------------+
| retail  | search_product   | price       | min_price, max_price |
+---------+------------------+-------------+----------------------+
| banking | transfer_funds   | account_num | sender, recipient    |
+---------+------------------+-------------+----------------------+

Role classifiers are trained separately for each entity that requires the additional categorization. We describe how to build role classification models with Workbench in the :doc:`Role Classifier User Guide <role_classifier>`.

After the domain, intent, entities and roles have been determined by the 4-level classifier hierarchy discussed above, the processed query is sent to the Entity Resolver and the Language Parser modules to complete the natural language understanding of the user input.


.. _arch_resolver:

Entity Resolver
~~~~~~~~~~~~~~~

The Entity Resolver was introduced in Steps :ref:`6 <entity-mapping-files>` and :ref:`7 <entity_resolution>` of the Step-By-Step Guide. Entity resolution entails mapping each identified entity to a canonical value that can be looked up in an official catalog or database. For instance, the extracted entity "lemon bread" may get resolved to "Iced Lemon Pound Cake (Product ID: 470)" and "SF" might be resolved to "San Francisco, CA". 

In conversational interactions, users generally refer to entities in informal terms, using abbreviations, nicknames, and other aliases, rather than their official standardized names. Robust entity resolution is hence key to a seamless conversational experience. The MindMeld Entity Resolver leverages advanced text relevance algorithms, similar to the ones used in state-of-the-art information retrieval systems to ensure high resolution accuracies.

Each entity has its own resolver that is trained to capture all the name variations specific to that entity. We will learn more about how to build about entity resolvers using Workbench in the :doc:`Entity Resolver User Guide <entity_resolver>`.


.. _arch_parser:

Language Parser
~~~~~~~~~~~~~~~

As described in the :doc:`Step-By-Step Guide <../quickstart/08_configure_the_language_parser>`, the Language Parser is the final module in the NLP pipeline. The parser finds relationships between the extracted entities and clusters them into meaningful entity groups. Each entity group has an inherent hierarchy, representing a real-world organizational structure.

In the :ref:`example <nlp_output>` above, the resolved entities have been arranged into three separate entity groups, with each group describing a distinct real-world concept:

.. image:: /images/entity_groups.png
    :align: center

The first two groups represent the products to be ordered, whereas the last group contains the store information. The main entity at the top in each group is called the parent or the `head <https://en.wikipedia.org/wiki/Head_(linguistics)>`_, whereas the other entities are called its children or `dependents <https://en.wikipedia.org/wiki/Dependent_(grammar)>`_. This structured representation of the user's natural language input can then be interpreted by the app to decide on the next action and/or response. E.g. submitting the order to a point-of-sale system to complete the user's order.

Most natural language parsers used in NLP academic research need to be trained using expensive `treebank <https://en.wikipedia.org/wiki/Treebank>`_ data, which is hard to find and annotate for custom conversational domains. The MindMeld Language Parser, on the other hand, is a config-driven rule-based parser which works out-of-the-box without the need for training. Refer to the :doc:`User Guide <parser>` for details on how Workbench can be used to configure the parser for optimum performance for a specific app.

The Natural Language Processor gets half of the job done, namely, understanding what the user wants. The next two components in the MindMeld pipeline address the other half by responding appropriately to the user and advancing the conversation.


.. _arch_qa:

Question Answerer
-----------------

Most of the modern conversational apps today rely on a Knowledge Base to understand user requests and answer questions. The knowledge base is a comprehensive repository of all the important world knowledge for a given application use case. The component responsible for interfacing with the knowledge base is called the Question Answerer. See Steps :doc:`5 <../quickstart/05_create_the_knowledge_base>` and :doc:`9 <../quickstart/09_optimize_question_answering_performance>` of the Step-By-Step Guide for an introduction to the topics of Knowledge Base and Question Answering.

The question answerer uses information retrieval techniques to identify the best answer candidates from the knowledge base that satisfy a given set of constraints. For example, the question answerer for a restaurant app might rely on a knowledge base containing a detailed menu of all the available items, in order to identify the user requested dishes and answer questions about them. Similarly, the question answerer for a voice-activated multimedia device might have a knowledge base containing detailed information about every song or album in a music library.

The MindMeld Question Answerer provides a flexible mechanism for retrieving and ranking relevant results from the knowledge base, with convenient interfaces for both simple and highly advanced searches. Refer to the :doc:`Question Answerer User Guide <kb>` for detailed documentation along with examples.


.. _arch_dm:

Dialogue Manager
----------------

The Dialogue Manager is a stateful component responsible for directing the flow of the conversation. It analyzes each incoming request and assigns it to a dialogue state handler which then executes the required logic and returns a response to the user.

Architecting the dialogue manager correctly is often one of the most challenging software engineering tasks when building a conversational app for a non-trivial use case. Workbench offers a simple solution by abstracting away many of the underlying complexities of dialogue management and offering developers a simple but powerful mechanism for defining their application logic. Workbench provides advanced capabilities for dialogue state tracking, beginning with a flexible syntax for defining rules and patterns for mapping requests to dialogue states. It also allows dialogue state handlers to invoke any arbitrary code for taking a specific action, completing a transaction or getting the information necessary for formulating a response.

Refer to Step :doc:`4 <../quickstart/04_define_the_dialogue_handlers>` of the Step-By-Step guide for a practical introduction to dialogue state tracking using Workbench. We will see more examples in the :doc:`Dialogue Manager User Guide <dm>`. 



.. _arch_gateway:

Gateway
-------

The Gateway is the component responsible for processing external requests via various endpoints, and for persisting user state. Supported endpoints include messaging platforms such as Cisco Spark or Facebook Messenger, intelligent assistants such as Google Assistant or Amazon Alexa, and custom endpoints on the web, in mobile apps, or on custom hardware.

The gateway is able to identify users from various endpoints, load their context, and convert requests into a format the Workbench-trained components can consume. After a request has been processed, it converts responses to the appropriate client format, and sends the response back to the endpoint.


.. _arch_app_manager:

Application Manager
-------------------

The Application Manager is the core orchestrator of the MindMeld platform. It performs the following functions:

    - Receives the client request from the gateway
    - Processes the request by passing it through all the Workbench-trained components of the MindMeld platform
    - Returns the final response back to the gateway once the processing is complete

The application manager is hidden from the Workbench developer, and accomplishes its tasks behind the scenes.


That concludes our quick tour of the MindMeld Conversational AI platform. Now that we are familiar with all its components, the rest of this user guide will focus on hands-on tutorials using Workbench to build modern data-driven conversational apps that run on the MindMeld platform.
