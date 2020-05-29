Key Concepts
============

Every MindMeld developer should know the terms defined in this section. It's best to read the entire section through, since later definitions build on earlier ones.

.. glossary::

    app
      |
      | A conversational application built using the MindMeld framework. Examples include voice and chat assistants deployed on messaging platforms like `Webex Teams <https://apphub.webex.com/bots>`_, `Slack <https://slack.com/apps/category/At0MQP5BEF-bots>`_, `Messenger <https://messenger.fb.com>`_, or `Skype <https://bots.botframework.com>`_; or, on voice-activated devices like `Alexa <https://developer.amazon.com/alexa-skills-kit>`_ or `Google Home <https://developers.google.com/actions/>`_.

    project
      |
      | A collection of all the data and code necessary to build a conversational app for a particular use case, stored in the MindMeld-specified directory structure.

    blueprint
      |
      | A pre-configured project, distributed with MindMeld as an example app. Available blueprints include :doc:`Food Ordering <../blueprints/food_ordering>`, :doc:`Video Discovery <../blueprints/video_discovery>`, and :doc:`Home Assistant <../blueprints/home_assistant>`.

    query / request
      |
      | A user's spoken or typed natural language input to the conversational app. MindMeld apps use statistical `natural language processing <https://en.wikipedia.org/wiki/Natural_language_processing>`_ models to understand the query and determine what the user wants.

    domain
      |
      | A distinct area of knowledge with its own vocabulary, such as ``weather``, ``sports``, ``traffic``, ``movies``, or ``travel``.

    intent
      |
      | Something that the user wants to accomplish, expressed as a short phrase, and associated with a domain. For example, the ``book_flight`` intent could be defined for the ``travel`` domain, and could be detected in query like 'Get me on the redeye to New York tonight.' Most domains have multiple intents.
      |
      | The intent also defines the desired outcome for the query, by prescribing that the app take a specific action and/or respond with a particular type of answer.

    entity
      |
      | A word or phrase that provides information necessary to fulfill a particular intent. Each entity belongs to a category specified by the entity's associated *type*.
      |
      | For instance, a ``book_flight`` intent could have a ``location`` type for entities like 'Miami' and 'Chicago O'Hare', an ``airline`` type for entities like 'Air India' and 'Southwest', and a ``date`` type for entities like 'July 4th' and 'New Years Day'.

    system entity
      |
      | An application-agnostic entity that is automatically detected by MindMeld. Examples include numbers, time expressions, email addresses, URLs and measured quantities like distance, volume, currency and temperature. See :ref:`system-entities`.

    role
      |
      | A label that specifies a sub-category in entities of the same type. Roles are used when identically-typed entities need to be interpreted differently in different contexts.
      |
      | For instance, in the query 'Book a flight from SFO to JFK', both 'SFO' and 'JFK' are location entities, but they would be assigned different roles: SFO would have the ``origin`` role, while 'JFK' would have the ``destination`` role.

    knowledge base (KB)
      |
      | A comprehensive repository containing the universe of information an app needs to understand user queries and answer questions successfully.
      |
      | For instance, the knowledge base for a restaurant chain's food ordering app might require a directory of its branch locations and details of the menu served at each store.

    knowledge base index
      |
      | A collection of objects of the same type within the knowledge base. MindMeld organizes the knowledge base as a set of indexes for efficient retrieval and ranking based on developer-supplied search criteria.
      |
      | For instance, a food ordering app might have a ``restaurant`` index for storing the details of its branch locations, including metadata like name, address, phone number, cuisine type, and ratings for each entry. It might also have a separate ``menu_items`` index that stores all the dishes offered at its restaurant location with details like name, price, description and add-ons.

    canonical name
       |
       | The standardized or official name under which an entity is stored in the knowledge base. For instance, the beverage commonly known as 'vanilla frappe' or 'vanilla frappuccino' might have the canonical name of 'Caffè Vanilla Frappuccino® Blended Coffee' in the app's `official catalog <https://www.starbucks.com/menu/drinks/frappuccino-blended-beverages/caffe-vanilla-frappuccino-blended-beverage>`_.

    entity synonym
       |
       | An alternate way of referring to an entity, apart from its canonical name. Synonyms can be abbreviated forms, nicknames, slang terms, translations, semantically equivalent expressions or just different names for the same entity.
       |
       | For instance, 'vanilla frappuccino', 'vanilla frap' and 'iced vanilla frappe' are all synonyms for the canonical name, 'Caffè Vanilla Frappuccino® Blended Coffee'.

    entity group
       |
       | A group of entities that together form a meaningful real-world concept. Each entity group has one main entity. Other entities in the group (if any) are considered attributes of the main entity.
       |
       | For instance, the query 'Order one large squishee and a dozen donuts' contains two entity groups. The order details for the 'squishee' product, including quantity and size, make up the entity group {'one', 'large', 'squishee'}. The order details for the 'donuts' make up the entity group {'a dozen', 'donuts'}.

    head / parent
        |
        | The principal entity being described in an entity group. In the entity group {'one', 'large', 'squishee'}, 'squishee' is the head entity. Similarly, 'donuts' heads the entity group {'a dozen', 'donuts'}. See `head (linguistics) <https://en.wikipedia.org/wiki/Head_(linguistics)>`_.

    dependent / child
        |
        | An entity that describes or `modifies <https://en.wikipedia.org/wiki/Grammatical_modifier>`_ the principal entity (head) in an entity group.
        |
        | For instance, in the group {'one', 'large', 'squishee'} the entities 'one' and 'large' are the dependents of the head entity 'squishee'. Similarly, 'a dozen' depends on the head 'donuts' in the entity group {'a dozen', 'donuts'}. See `dependent (linguistics) <https://en.wikipedia.org/wiki/Dependent_(grammar)>`_.

    natural language response (NLR)
        |
        | The app's natural language reply to the user. It could be in the form of a text-based response, a spoken voice response, or both, depending on the platform capabilities. Fully automated `natural language response generation <https://en.wikipedia.org/wiki/Natural_language_generation>`_ is still an area of active academic research. Real-world applications today, including all the popular personal assistants, instead rely on canned responses (**templates**) with placeholders (**slots**) that are filled in by the app at runtime.
        |

        +-------------------------------+-----------------------------------------------------------------------------+
        | Template with unfilled slots  |'``{flight}`` will depart from ``{gate}`` of ``{terminal}`` at ``{time}``.'  |
        +-------------------------------+-----------------------------------------------------------------------------+
        | NLR with filled slots         | 'AA 20 will depart from gate 56A of terminal 2 at 3:30 PM'.                 |
        +-------------------------------+-----------------------------------------------------------------------------+

    conversational turn
        |
        | A single instance of either dialogue participant (the user or the app) communicating with the other. The dialogue between a user and the app proceeds as a series of back and forth communication with each party `taking turns <https://en.wikipedia.org/wiki/Turn-taking>`_ to advance the conversation.

    dialogue state
        |
        | The state of the application at a given moment in the dialogue. The app transitions from one dialogue state to another with every turn in the conversation.

    dialogue state handler
        |
        | The code to be executed when the app is in a particular dialogue state. It determines the appropriate form of response for that state, and invokes any logic necessary to determine the content of the response.

    dialogue frame
        |
        | A container for any information that the app needs to persist across turns over the course of a single conversational interaction with a user. The dialogue frame serves as the app's short-term memory and allows it to hold a coherent conversation with the user.
