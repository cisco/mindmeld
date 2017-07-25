Key Concepts
============

Every Workbench developer should know the terms defined in this section. It's best to read the entire section through, since later definitions build on earlier ones.

.. glossary::

    app
      A conversational application built using the MindMeld Workbench framework. Examples include voice and chat assistants deployed on a messaging platform (e.g. `Spark <https://depot.ciscospark.com/bots>`_, `Slack <https://slack.com/apps/category/At0MQP5BEF-bots>`_, `Messenger <https://messenger.fb.com>`_, `Skype <https://bots.botframework.com>`_, etc.) or a voice-activated device (e.g. `Alexa <https://developer.amazon.com/alexa-skills-kit>`_, `Google Home <https://developers.google.com/actions/>`_).


    project
      A collection of all the resources (data and code) necessary for building a conversational app, stored in the Workbench-specified directory structure.


    blueprint
      A pre-configured project for a specific conversational use case, distributed with Workbench as an example app. Available blueprints include :doc:`Food Ordering <../blueprints/food_ordering>`, :doc:`Video Discovery <../blueprints/video_discovery>` and :doc:`Home Assistant <../blueprints/home_assistant>`.


    query / request
      A user's spoken or typed natural language input to the conversational app. A Workbench app uses statistical `natural language processing <https://en.wikipedia.org/wiki/Natural_language_processing>`_ models to understand the query and determine what the user wants.


    domain
      A unique area of knowledge with its own vocabulary. E.g. ``weather``, ``sports``, ``traffic``, ``movies``, ``travel``, etc.


    intent
      The user intention expressed in the query, reflecting what the user is trying to accomplish. Each intent prescribes a specific action or answer type which defines the desired outcome for the query. E.g. ``set_alarm``, ``get_artist``, ``book_flight``, etc.


    entity
      An important word or phrase that provides information necessary to fulfill a particular intent. Each entity has a type associated with it which identifies the category it belongs to.

      For instance, a ``book_flight`` intent could have a ``location`` type entity (e.g. 'Miami', 'San Francisco', etc.), an ``airline`` type entity (e.g. 'United', 'Southwest', 'American', etc.) and a ``date`` entity (e.g. 'tomorrow', 'July 4th', 'Thanskgiving', etc.).


    role
      An additional label assigned to an entity to subcategorize entities of the same type. Roles are used when identically typed entities need to be interpreted differently in different contexts.

      For instance, in the query 'Book a flight from SFO to JFK', the location entity 'SFO' would be assigned the role label of ``origin``, whereas 'JFK' would be labeled as ``destination``.


    knowledge base (KB)
      A comprehensive repository containing the universe of helpful information needed by an app to understand user queries and answer questions successfully. The relevant knowledge is stored as a set of collections, each containing objects of a specified type.

      For instance, the knowledge base for a restaurant chain's food ordering app might require an extensive directory of all its branch locations, and detailed information about the menu served at each store.


    knowledge base index
      A collection of objects of the same type, stored in a manner that aids efficient retrieval and ranking based on some desired search criteria. The knowledge base for an app is comprised of one or more indexes.

      For instance, a food ordering app might have a ``restaurant`` index for storing the details of all its branch locations, including metadata such as name, address, phone number, cuisine type and ratings for each entry. It might also have a separate ``menu_items`` index that stores all the dishes offered at its restaurant location with details like name, price, description and add-ons.


    canonical name
      The standardized or official name under which an entity is stored in the knowledge base. For instance, the beverage commonly known as 'vanilla frappe' or 'vanilla frappuccino' might have the canonical name of 'Caffè Vanilla Frappuccino® Blended Coffee' in the app's `official catalog <https://www.starbucks.com/menu/drinks/frappuccino-blended-beverages/caffe-vanilla-frappuccino-blended-beverage>`_.


    entity synonym
      An alternate way of referring to an entity, apart from its canonical name. Synonyms can be abbreviated forms, nicknames, slang terms, translations, semantically equivalent expressions or just different names for the same entity.

      For instance, 'vanilla frappuccino', 'vanilla frap' and 'iced vanilla frappe' are all synonyms for the canonical name, 'Caffè Vanilla Frappuccino® Blended Coffee'.


    entity group
      A group of entities that are related to each other and together, form a meaningful real-world concept. Each entity group has a main entity and optionally, additional entities that are attributes of that main entity.

      For instance, there are two entity groups in the query 'Order one large squishee and a dozen donuts'. The first group is {'one', 'large', 'squishee'}, which fully describes the order details for the product named 'squishee', including the quantity and the size. The second entity group is {'a dozen', 'donuts'}, which describes the order for the 'donuts'.


    head / parent
      The principal entity being described in an entity group. For instance, the entity 'squishee' is the head of the group {'one', 'large', 'squishee'}. Similarly, 'donuts' heads the entity group {'a dozen', 'donuts'}. See also, `head (linguistics) <https://en.wikipedia.org/wiki/Head_(linguistics)>`_.


    dependent / child
      An entity that describes or `modifies <https://en.wikipedia.org/wiki/Grammatical_modifier>`_ the principal entity (head) in an entity group. For instance, the entities 'one' and 'large' are the dependents of the head entity 'squishee' in the group {'one', 'large', 'squishee'}. Similarly, 'a dozen' depends on the head 'donuts' in the entity group {'a dozen', 'donuts'}. See also, `dependent (linguistics) <https://en.wikipedia.org/wiki/Dependent_(grammar)>`_.


    natural language response (NLR)
      The app's natural language reply to the user. It could be in the form of a text-based response, a spoken voice response, or both, depending on the platform capabilities. Fully automated `natural language response generation <https://en.wikipedia.org/wiki/Natural_language_generation>`_ is still an area of active academic research. Real world applications today, including all the popular personal assistants, instead rely on canned responses (**templates**) with placeholders (**slots**) that are filled in by the app at runtime.

      E.g.

      +-------------------------------+-----------------------------------------------------------------------------+
      | Template with unfilled slots: | '``{flight}`` will depart from ``{gate}`` of ``{terminal}`` at ``{time}``.' |
      +-------------------------------+-----------------------------------------------------------------------------+
      | NLR with filled slots:        | 'AA 20 will depart from gate 56A of terminal 2 at 3:30 PM'.                 |
      +-------------------------------+-----------------------------------------------------------------------------+


    conversational turn
      A single instance of either dialogue participant (the user or the app) communicating with the other. The dialogue between a user and the app proceeds as a series of back and forth communication with each party `taking turns <https://en.wikipedia.org/wiki/Turn-taking>`_ to advance the conversation.


    dialogue state
      The state that the application is in at each step of the dialogue. The app transitions from one dialogue state to another with every turn in the conversation.


    dialogue state handler
      The code to be executed when the app is in a particular dialogue state. It determines the appropriate form of response for that state, and invokes any logic necessary to determine the content of the response.


    dialogue frame
      A container for any information that needs to be persisted across turns over the course of a single conversational interaction with a user. The dialogue frame serves as the app's short-term memory and allows it to hold a coherent conversation with the user.