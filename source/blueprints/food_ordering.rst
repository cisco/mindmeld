Food Ordering
=============

In this step-by-step walkthrough, you'll build a conversational application that allows users to order food for delivery from nearby restaurants, using the MindMeld blueprint for this purpose.

.. note::

   Please make sure to install and run all of the :ref:`pre-requisites <getting_started_virtualenv_setup>` for MindMeld before continuing on with this blueprint tutorial.


1. The Use Case
^^^^^^^^^^^^^^^

In a hands-free manner, from the convenience of their homes, users should have the same food-ordering experience as when conversing with a restaurant waiter or a drive-through attendant. The food is ordered for delivery from a service similar to Amazon Restaurants, Grubhub or DoorDash.

2. Example Dialogue Interactions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The conversational flows for food ordering can be highly complex, depending on the desired app functionality and the amount of user guidance required at each step. Enumerating and finalizing all anticipated user interactions requires multiple iterations.

Here are some examples of scripted dialogue interactions for conversational flows.

.. image:: /images/food_ordering_interactions.png
    :width: 700px
    :align: center

.. admonition:: Exercise

   Pick a convenient textual or graphical representation. Try to design as many user flows as you can. Always capture the entire dialogue from start to finish. Think of scenarios that differ from the examples above, such as: asking to order from a specific restaurant without choosing a dish; requesting a dish that is not available at the selected restaurant; asking for a restaurant location that doesn't exist; choosing a customization option that is not applicable for the chosen dish, and so on.


3. Domain-Intent-Entity Hierarchy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is the NLP model hierarchy for our food ordering application.

.. image:: /images/food_ordering_hierarchy.png
    :width: 700px
    :align: center

The single domain, ``ordering``, encompasses all of the functionality required to order food for delivery from nearby restaurants. The ``ordering`` domain supports the following intents:

   - ``greet`` — User wants to start a conversation
   - ``build_order`` — User wants to make selections for food delivery
   - ``place_order`` — User wants to confirm and place an order
   - ``start_over`` — User wants to abandon current selections and restart the ordering process
   - ``exit`` — User wants to end the current conversation
   - ``help`` — User is confused or needs instructions on how to proceed
   - ``unsupported`` — User is talking about something other than food ordering

For this app, only the ``build_order`` intent requires entity recognition. This intent supports the following entity types:

   - ``restaurant`` — The name of a restaurant location
   - ``cuisine`` — The name of a cuisine
   - ``category`` — The name of a food category on a restaurant menu
   - ``dish`` — The name of a dish on a restaurant menu
   - ``option`` — The name of an available option (customization, add-on, etc.) for a dish
   - ``sys_number`` — The quantity of a given dish, captured by the :doc:`number system entity <../userguide/entity_recognizer>`

The ``sys_number`` entity can play two roles (``num_orders`` and ``item_size``) so queries containing ``sys_number`` entities
must be labeled accordingly. The ``num_orders`` role represents the number of desired orders and the
``item_size`` role represents the size of a desired item. The role annotation follows the entity annotation
in the following manner: ``can I have a {12|sys_number|item_size} ounce burger`` and ``{one|sys_number|num_orders} {parfait|dish} please.``

.. admonition:: Exercise

   While the blueprint provides a good starting point, you may need additional intents and entities to support the desired scope of your app. Enumerate some other intents (e.g., ``check_order_status``, ``get_calories``, and so on) and entities (e.g., ``location``, ``price_level``, and so on) that make sense for a food ordering use case.

To train the different machine learning models in the NLP pipeline for this app, we need labeled training data that covers all our intents and entities. To download the data and code required to run this blueprint, run the command below in a directory of your choice. (If you have already completed the Quick Start for this blueprint, you should skip this step.)

.. warning::

   This application requires Elasticsearch for the QuestionAnswerer. Please make sure that Elasticsearch is running in another shell before proceeding to setup this blueprint.

.. code-block:: console

   python -c "import mindmeld as mm; mm.blueprint('food_ordering');"

This should create a MindMeld project folder called ``food_ordering`` in your current directory with the following structure:

.. image:: /images/food_ordering_directory.png
    :width: 250px
    :align: center


4. Dialogue States
^^^^^^^^^^^^^^^^^^

To support the functionality we envision, our app needs one dialogue state for each intent, as shown in the table below.

+------------------+---------------------+-----------------------------------------------+
| | Intent         | |  Dialogue State   | | Dialogue State                              |
| |                | |  name             | | function                                    |
+==================+=====================+===============================================+
| | ``greet``      | | ``welcome``       | | Begin an interaction and welcome the user   |
+------------------+---------------------+-----------------------------------------------+
| | ``build_order``| | ``build_order``   | | Guide the user through making selections    |
| |                | |                   | | and build up the delivery order             |
+------------------+---------------------+-----------------------------------------------+
| | ``place_order``| | ``place_order``   | | Place the order to complete the transaction |
+------------------+---------------------+-----------------------------------------------+
| | ``start_over`` | | ``start_over``    | | Cancel the ongoing transaction              |
| |                | |                   | | and prompt the user for a new request       |
+------------------+---------------------+-----------------------------------------------+
| | ``exit``       | | ``say_goodbye``   | | End the current interaction                 |
+------------------+---------------------+-----------------------------------------------+
| | ``help``       | | ``provide_help``  | | Provide help information                    |
| |                | |                   | | in case the user gets stuck                 |
+------------------+---------------------+-----------------------------------------------+
| | ``unsupported``| | ``default``       | | Prompt a user who has gone off-topic        |
| |                | |                   | | to get back to food ordering                |
+------------------+---------------------+-----------------------------------------------+


All dialogue states and their associated handlers are defined in the ``__init__.py`` application container file at the top level of the blueprint folder.

Handler logic can be simple, complex, or in between. At one end of this spectrum, the handler simply returns a canned response, sometimes choosing randomly from a set of responses. A more sophisticated handler could execute knowledge base queries to fill in the slots of a partially-templatized response. And a handler that applies more complex business logic could call an external API, process what the API returns, and incorporate the result into the response template.

The handler logic is fairly straightforward for most of our dialogue states. The main actions are choosing from a set of pre-scripted natural language responses, and replying to the user. These simple states include ``welcome``, ``start_over``, ``say_goodbye``, ``provide_help`` and ``default``.

For example, here's the ``say_goodbye`` state handler, where we clear the :doc:`dialogue frame <../userguide/dm>` and use the :doc:`responder <../userguide/dm>` object to reply with one of our scripted "goodbye" responses:

.. code:: python

    @app.handle(intent='exit')
    def say_goodbye(request, responder):
        """
        When the user ends a conversation, clear the dialogue frame and say goodbye.
        """
        # Clear the dialogue frame to start afresh for the next user request.
        responder.frame = {}

        # Respond with a random selection from one of the canned "goodbye" responses.
        responder.reply(['Bye!', 'Goodbye!', 'Have a nice day.', 'See you later.'])

By contrast, the handler logic for the ``build_order`` and ``place_order`` dialogue states is more substantial, because they contain the core business logic for our application. In these dialogue state handlers, we use the :doc:`Question Answerer <../userguide/kb>` and external API calls to process the transaction.

We can illustrate this with a simplistic implementation of the ``build_order`` handler:

.. code:: python

    @app.handle(intent='build_order')
    def build_order(request, responder):
        """
        When the user expresses an intent to make food selections, build up the order by
        adding the requested dishes to their "check-out" basket.
        """
        # Get the first recognized restaurant entity in the user query.
        rest_entity = next(e for e in request.entities if e['type'] == 'restaurant')

        # Resolve the restaurant entity to a specific entry in the knowledge base (KB).
        selected_restaurant = _get_restaurant_from_kb(rest_entity['value'][0]['id'])

        # Next, get all the recognized dish entities in the user query.
        dish_entities = [e for e in request.entities if e['type'] == 'dish']

        # Add dishes one by one to the "check-out" shopping basket.
        selected_dishes = list()
        for entity in dish_entities:
            # Resolve the dish entity to a KB entry using restaurant information.
            selected_dishes.append(_resolve_dish(entity, selected_restaurant))

        # Store dish and restaurant selections in the dialogue frame.
        responder.frame['restaurant'] = selected_restaurant
        responder.frame['dishes'] = selected_dishes

        # Respond with a preview of the current basket and prompt for order confirmation.
        responder.slots['restaurant_name'] = selected_restaurant['name']
        responder.slots['dish_names'] = ', '.join([dish['name'] for dish in selected_dishes])
        responder.slots['price'] = sum([dish['price'] for dish in selected_dishes])
        responder.reply('Sure, I got {dish_names} from {restaurant_name} for a total '
                        'price of ${price:.2f}. Would you like to place the order?')
        responder.listen()

This code assumes that every user query contains a ``restaurant`` entity and at least one ``dish`` entity. Using the Question Answerer (within the :func:`_get_restaurant_from_kb()` and :func:`_resolve_dish()` functions not shown above), it selects the most likely restaurant and dishes requested by the user. Next, it saves this information in the dialogue frame for use in future conversational turns, and presents it to the user via the responder object.

For a more realistic implementation of ``build_order`` that deals with varied user flows and the full code behind all the dialogue state handlers, see the ``__init__.py`` file in the blueprint folder.

.. admonition:: Exercise

   Extend the ``build_order`` dialogue state handler in ``__init__.py`` to handle more user flows or handle the existing ones in a smarter way. See the suggestions for improvements in the comments in the ``__init__.py`` code. Here are a few more:

   - Make it possible to select restaurants by ``cuisine`` or to search for dishes by ``category``. These are already modeled as entities in the blueprint and available as part of the restaurant and dish metadata stored in the knowledge base. But ``build_order`` needs some additional code to handle queries containing these entities.

   - After providing restaurant suggestions to a user based on a requested dish, do not ask the user to repeat the dish selection from scratch in the next turn. Instead, keep track of the dish the user was originally interested in. Then, when the user makes a restaurant selection, add the dish to the check-out basket.

   - Update the natural language response templates to include resolved options that are stored in the dialogue frame (``responder.frame['dishes']``) for each dish. Try to construct your responses such that if someone orders the same dish with different options, instead of saying '*I have 3 orders of steak from ...*' your app could respond with something like '*I have 1 order of steak medium rare and 2 orders of steak well done from ...*'.

5. Knowledge Base
^^^^^^^^^^^^^^^^^

The knowledge base for our food ordering app leverages publicly available information about San Francisco restaurants from `Amazon Restaurants <https://primenow.amazon.com/restaurants>`_. The knowledge base comprises two indexes in `Elasticsearch <https://www.elastic.co/products/elasticsearch>`_:

   - ``restaurants`` — information about restaurant locations
   - ``menu_items`` — information about dishes on different restaurants' menus

For example, here's the knowledge base entry in the ``restaurants`` index for "Basa Seafood Express," a seafood restaurant in San Francisco:

.. code:: javascript

    {
        "cuisine_types": ["Seafood", "Sushi"],
        "rating": 3.2,
        "name": "Basa Seafood Express",
        "num_reviews": 8,
        "menus": [
            {
                "option_groups": [
                    {
                        "max_selected": 1,
                        "options": [
                            {
                                "price": 0.0,
                                "description": null,
                                "name": "Coke",
                                "id": "B01N1ME52H"
                            },
                            {
                                "price": 0.0,
                                "description": null,
                                "name": "Water",
                                "id": "B01MTUONJU"
                            }
                        ],
                        "min_selected": 1,
                        "id": "Coke or Water",
                        "name": "Choose One"
                    }
                ],
                "id": "78eb0100-029d-4efc-8b8c-77f97dc875b5",
                "size_groups": [
                    {
                        "sizes": [
                            {
                                "alt_name": "Small",
                                "name": "Small"
                            },
                            {
                                "alt_name": "Medium",
                                "name": "Medium"
                            }
                        ],
                        "description": null,
                        "name": "Choose Size",
                        "id": "Size"
                    }
                ]
            }
        ],
        "image_url": "https://images-na.ssl-images-amazon.com/images/G/01/ember/restaurants/SanFrancisco/BasaSeafoodExpress/logo_232x174._CB523176793_SX600_QL70_.png",
        "price_range": 1.0,
        "id": "B01N97KQNJ",
        "categories": ["Hawaiian Style Poke (HP)", "Ceviche (C)", "Nigiri Sushi (2 Pcs)", "Popular Dishes",
            "Makimono-Sushi Rolls (6 Pcs)", "Clam Chowder (CC)", "Side Order (SO)", "Sashimi (5 Pcs)",
            "Fish & Chips (FC)", "Salads (SL)", "Rice Bowl (RB)", "Sandwiches (SW)", "Special Rolls",
            "Special Combo (PC)"]
    }

Here's a knowledge base entry in the ``menu_items`` index for a dish from the above restaurant.

.. code:: javascript

    {
        "category": "Nigiri Sushi (2 Pcs)",
        "menu_id": "78eb0100-029d-4efc-8b8c-77f97dc875b5",
        "description": "Nigiri Sushi",
        "price": 4.5,
        "option_groups": [],
        "restaurant_id": "B01N97KQNJ",
        "size_prices": [],
        "size_group": null,
        "popular": false,
        "img_url": null,
        "id": "B01MTUOW2R",
        "name": "Masago (Capelin Roe)"
    }

Assuming that you have Elasticsearch installed, running the :func:`blueprint()` command described above should build the knowledge base for the food ordering app by creating the two indexes and importing all the necessary data. To verify that the knowledge base has been set up correctly, use the :doc:`Question Answerer <../userguide/kb>` to query the indexes.

.. warning::

   Make sure that Elasticsearch is running in a separate shell before invoking the QuestionAnswerer.

.. code:: python

   from mindmeld.components.question_answerer import QuestionAnswerer
   qa = QuestionAnswerer(app_path='food_ordering')
   qa.get(index='menu_items')[0]

.. code-block:: console

    {
       "size_group": "pizzasize",
       "menu_id": "57572a43-f9fc-4a1c-96fe-788d544b1f2d",
       "restaurant_id": "B01DEEGQBK",
       "size_prices": [
          {
             "name": "12\" Small",
             "price": 13.99,
             "id": "B01N9YUFMX"
          },
          {
             "name": "14\" Medium",
             "price": 16.99,
             "id": "B01MRDF6V1"
          },
          {
             "name": "16\" Large",
             "price": 17.99,
             "id": "B01MUI0ZGE"
          },
          {
             "name": "18\" X-Large",
             "price": 19.99,
             "id": "B01N7ZK0ZR"
          }
       ],
       "option_groups": [
          "pizzawhole"
       ],
       "img_url": null,
       "description": null,
       "id": "B01NB08SGM",
       "popular": false,
       "name": "Cheese Pizza",
       "category": "Pizzas",
       "price": 13.99
    }

.. admonition:: Exercise

   The blueprint comes with a pre-configured, pre-populated knowledge base to help you get up and running quickly. Read the User Guide section on :doc:`Question Answerer <../userguide/kb>` to learn how to create knowledge base indexes from scratch. Then, try creating one or more knowledge base indexes for your own data.


6. Training Data
^^^^^^^^^^^^^^^^

The labeled data for training our NLP pipeline was created using both in-house data generation and crowdsourcing techniques. See :doc:`Step 6 <../quickstart/06_generate_representative_training_data>` of the Step-By-Step Guide for a full description of this highly important, multi-step process. Be aware that at minimum, the following data generation tasks are required:

+--------------------------------------+-----------------------------------------------------------------------------+
| | Purpose                            | | Question (for crowdsourced data generators)                               |
| |                                    | | or instruction (for annotators)                                           |
+======================================+=============================================================================+
| | Exploratory data generation        | | "How would you talk to a conversational app                               |
| | for guiding the app design         | | to place orders for food?"                                                |
+--------------------------------------+-----------------------------------------------------------------------------+
| | Targeted query generation          | | ``build_order``: "What would you say to the app                           |
| | for training the Intent Classifier | | to make food or restaurant selections and create                          |
| |                                    | | your delivery order?"                                                     |
+--------------------------------------+-----------------------------------------------------------------------------+
| | Targeted query annotation          | | ``build_order``: "Annotate all occurrences of restaurant,                 |
| | for training the Entity Recognizer | | cuisine, category, dish and option names                                  |
| |                                    | | in the given query."                                                      |
+--------------------------------------+-----------------------------------------------------------------------------+
| | Targeted synonym generation        | | ``restaurant``: "What are the different ways in which                     |
| | for training the Entity Resolver   | | you would refer to this restaurant location?"                             |
| |                                    | |                                                                           |
| |                                    | | ``dish``: "What names would you use to refer                              |
| |                                    | | to this item on the restaurant menu?"                                     |
+--------------------------------------+-----------------------------------------------------------------------------+
| | Annotate queries for               | | ``sys_number``: "Annotate all entities with their                         |
| | training the Role Classifier       | | corresponding roles, e.g. ``num_orders`` and ``item_size``."              |
| |                                    | |                                                                           |
+--------------------------------------+-----------------------------------------------------------------------------+

The ``domains`` directory contains the training data for intent classification and entity recognition. The ``entities`` directory contains the data for entity resolution. Both directories are at root level in the blueprint folder.

.. admonition:: Exercise 1

   - Study the best practices around training data generation and annotation for conversational apps in :doc:`Step 6 <../quickstart/06_generate_representative_training_data>` of the Step-By-Step Guide. Following those principles, create additional labeled data for all the intents in this blueprint. Read more about :doc:`NLP model evaluation and error analysis <../userguide/nlp>` in the User Guide. Then apply what you have learned in evaluating your app, using your newly-created labeled data as held-out validation data.

   - Complete the following exercise if you are extending the blueprint to build your own food ordering app. For app-agnostic, generic intents like ``greet``, ``exit``, and ``help``, start by simply reusing the blueprint data to train NLP models for your food ordering app. For ``build_order`` and any other app-specific intents, gather new training data tailored to the relevant entities (restaurants, dishes, etc.). Apply the approach you learned in :doc:`Step 6 <../quickstart/06_generate_representative_training_data>`.

7. Training the NLP Classifiers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train a baseline NLP system for the blueprint app. The :meth:`build()` method of the :class:`NaturalLanguageProcessor` class, used as shown below, applies MindMeld's default machine learning settings.

.. code:: python

   from mindmeld.components.nlp import NaturalLanguageProcessor
   import mindmeld as mm
   mm.configure_logs()
   nlp = NaturalLanguageProcessor(app_path='food_ordering')
   nlp.build()

.. code-block:: console

   Fitting intent classifier: domain='ordering'
   Loading queries from file ordering/build_order/train.txt
   Loading queries from file ordering/exit/train.txt
   Loading queries from file ordering/greet/train.txt
   Loading queries from file ordering/help/train.txt
   Loading queries from file ordering/place_order/train.txt
   Loading queries from file ordering/start_over/train.txt
   Loading queries from file ordering/unsupported/train.txt
   Selecting hyperparameters using k-fold cross-validation with 10 splits
   Best accuracy: 98.11%, params: {'C': 100, 'class_weight': {0: 1.7987394957983194, 1: 3.0125475285171097, 2: 0.89798826487845773, 3: 4.4964705882352938, 4: 2.5018518518518515, 5: 1.7559183673469387, 6: 0.46913229018492181}, 'fit_intercept': True}
   Fitting entity recognizer: domain='ordering', intent='place_order'
   Fitting entity recognizer: domain='ordering', intent='unsupported'
   Fitting entity recognizer: domain='ordering', intent='greet'
   Fitting entity recognizer: domain='ordering', intent='exit'
   Fitting entity recognizer: domain='ordering', intent='build_order'
   Selecting hyperparameters using k-fold cross-validation with 5 splits
   Best accuracy: 92.46%, params: {'C': 1000000, 'penalty': 'l2'}
   Fitting entity recognizer: domain='ordering', intent='start_over'
   Fitting entity recognizer: domain='ordering', intent='help'

.. tip::

  During active development, it's helpful to increase the :doc:`MindMeld logging level <../userguide/getting_started>` to better understand what's happening behind the scenes. All code snippets here assume that logging level has been set to verbose.

To see how the trained NLP pipeline performs on a test query, use the :meth:`process()` method.

.. code:: python

   nlp.process("I'd like a mujaddara wrap and two chicken kebab from palmyra")

.. code-block:: console

        { 'domain': 'ordering',
          'entities': [ { 'role': None,
                          'span': {'end': 24, 'start': 11},
                          'text': 'mujaddara wrap',
                          'type': 'dish',
                          'value': [ { 'cname': 'Mujaddara Wrap',
                                       'id': 'B01DEFNIRY',
                                       'score': 169.67435,
                                       'top_synonym': 'Mujaddara Wrap'},
                                     { 'cname': "Veggie Lover's Wrap",
                                       'id': 'B01D0QQNQA',
                                       'score': 22.034618,
                                       'top_synonym': 'veggie wrap'},
                                     { 'cname': 'Beef Shawarma Wrap',
                                       'id': 'B01DEFMQR2',
                                       'score': 22.034618,
                                       'top_synonym': 'beef wrap'},
                                     { 'cname': 'Sahara',
                                       'id': 'B01MDU5D1V',
                                       'score': 21.205025,
                                       'top_synonym': 'falafel wrap'},
                                     { 'cname': 'Falafel Wrap',
                                       'id': 'B01CRF8HUU',
                                       'score': 19.44212,
                                       'top_synonym': 'Falafel Wrap'},
                                     { 'cname': 'Falafel Wrap',
                                       'id': 'B01DEFMLMC',
                                       'score': 19.44212,
                                       'top_synonym': 'Falafel Wrap'},
                                     { 'cname': 'Chicken Wrap',
                                       'id': 'B01CK514OI',
                                       'score': 19.44212,
                                       'top_synonym': 'Chicken Wrap'},
                                     { 'cname': 'Veggie Wrap',
                                       'id': 'B01CUUBRZY',
                                       'score': 19.218494,
                                       'top_synonym': 'Veggie Wrap'},
                                     { 'cname': 'Kebab Wrap',
                                       'id': 'B01CRF8ULQ',
                                       'score': 19.218494,
                                       'top_synonym': 'meat wrap'},
                                     { 'cname': 'Kofte Wrap',
                                       'id': 'B01DEFNGEY',
                                       'score': 19.218494,
                                       'top_synonym': 'Kofte Wrap'}]},
                        { 'role': 'num_orders',
                          'span': {'end': 32, 'start': 30},
                          'text': 'two',
                          'type': 'sys_number',
                          'value': [{'value': 2}]},
                        { 'children': [ { 'role': 'num_orders',
                                          'span': {'end': 32, 'start': 30},
                                          'text': 'two',
                                          'type': 'sys_number',
                                          'value': [{'value': 2}]}],
                          'role': None,
                          'span': {'end': 46, 'start': 34},
                          'text': 'chicken kebab',
                          'type': 'dish',
                          'value': [ { 'cname': 'Chicken Kebab',
                                       'id': 'B01DEFMUSW',
                                       'score': 147.06445,
                                       'top_synonym': 'Chicken Kebab'},
                                     { 'cname': 'Chicken Tikka Kabab',
                                       'id': 'B01DN5635O',
                                       'score': 99.278786,
                                       'top_synonym': 'chicken kebab'},
                                     { 'cname': 'Chicken Kebab Plate',
                                       'id': 'B01CK4ZQ7U',
                                       'score': 93.68581,
                                       'top_synonym': 'Chicken Kebab'},
                                     { 'cname': 'Chicken Shish Kebab Plate',
                                       'id': 'B01N9Z1K2O',
                                       'score': 31.71228,
                                       'top_synonym': 'chicken kebab plate'},
                                     { 'cname': 'Kebab Wrap',
                                       'id': 'B01CRF8ULQ',
                                       'score': 21.39855,
                                       'top_synonym': 'Kebab Wrap'},
                                     { 'cname': 'Kofte Kebab',
                                       'id': 'B01N4VET8U',
                                       'score': 21.39855,
                                       'top_synonym': 'beef kebab'},
                                     { 'cname': 'Lamb Kebab',
                                       'id': 'B01DEFNQIA',
                                       'score': 21.39855,
                                       'top_synonym': 'Lamb Kebab'},
                                     { 'cname': 'Kebab Platter',
                                       'id': 'B01CRF8A7U',
                                       'score': 21.290503,
                                       'top_synonym': 'Kebab Dish'},
                                     { 'cname': 'Beef Kebab',
                                       'id': 'B01DEFL75Y',
                                       'score': 21.290503,
                                       'top_synonym': 'Beef Kebab'},
                                     { 'cname': 'Prawns Kebab',
                                       'id': 'B01DEFO4ZY',
                                       'score': 21.290503,
                                       'top_synonym': 'Prawns Kebab'}]},
                        { 'role': None,
                          'span': {'end': 59, 'start': 53},
                          'text': 'palmyra',
                          'type': 'restaurant',
                          'value': [ { 'cname': 'Palmyra',
                                       'id': 'B01DEFLJIO',
                                       'score': 52.51769,
                                       'top_synonym': 'Palmyra'}]}],
          'intent': 'build_order',
          'text': "I'd like a mujaddara wrap and two chicken kebab from palmyra"}

For the data distributed with this blueprint, the baseline performance is already high. However, when extending the blueprint with your own custom food ordering data, you may find that the default settings may not be optimal and you could get better accuracy by individually optimizing each of the NLP components.

Start by inspecting the baseline configurations that the different classifiers use. The User Guide lists and describes the available configuration options. As an example, the code below shows how to access the model and feature extraction settings for the Intent Classifier.

.. code:: python

   ic = nlp.domains['ordering'].intent_classifier
   ic.config.model_settings['classifier_type']

.. code-block:: console

   'logreg'

.. code-block:: python

   ic.config.features


.. code-block:: console

   {
    'bag-of-words': {'lengths': [1]},
    'freq': {'bins': 5},
    'in-gaz': {},
    'length': {}
   }

You can experiment with different learning algorithms (model types), features, hyperparameters, and cross-validation settings by passing the appropriate parameters to the classifier's :meth:`fit()` method. Here are a few examples.

Change the feature extraction settings to use bag of bigrams in addition to the default bag of words:

.. code:: python

   features = {
               'bag-of-words': {'lengths': [1, 2]},
               'freq': {'bins': 5},
               'in-gaz': {},
               'length': {}
              }
   ic.fit(features=features)

.. code-block:: console

   Fitting intent classifier: domain='ordering'
   Selecting hyperparameters using k-fold cross-validation with 10 splits
   Best accuracy: 98.36%, params: {'C': 10000, 'class_weight': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}, 'fit_intercept': False}

Change the classification model to random forest instead of the default logistic regression:

.. code:: python

   ic.fit(model_settings={'classifier_type': 'rforest'}, param_selection={'type': 'k-fold', 'k': 10, 'grid': {'class_bias': [0.7, 0.3, 0]}})

.. code-block:: console

   Fitting intent classifier: domain='ordering'
   Selecting hyperparameters using k-fold cross-validation with 10 splits
   Best accuracy: 94.12%, params: {'class_weight': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}}

You can use similar options to inspect and experiment with the Entity Recognizer and the other NLP classifiers. Finding the optimal machine learning settings is a highly iterative process involving several rounds of model training (with varying configurations), testing, and error analysis. See the User Guide for more about training, tuning, and evaluating the various MindMeld classifiers.

.. admonition:: Exercise

   Experiment with different models, features, and hyperparameter selection settings to see how they affect classifier performance. Maintain a held-out validation set to evaluate your trained NLP models and analyze misclassified test instances. Then, use observations from the error analysis to inform your machine learning experimentation. See the :doc:`User Guide <../userguide/nlp>` for examples and discussion.


.. _food_ordering_parser:

8. Parser Configuration
^^^^^^^^^^^^^^^^^^^^^^^

Once the NLP classification models are trained, you can configure and run the MindMeld :doc:`Language Parser <../userguide/parser>` to link related entities into meaningful entity groups. The application configuration file, ``config.py``, at the top level of the blueprint folder, contains the following parser configuration:

.. code:: javascript

   PARSER_CONFIG = {
       'dish': {
           'option': {'linking_words': {'with'}},
           'sys_number': {'max_instances': 1, 'right': False}
       },
       'option': {
           'sys_number': {'max_instances': 1, 'right': False}
       }
   }

This configuration specifies that a dish entity can have a numeric quantity entity and an option entity as its attributes. An option entity, in turn, can have its own associated quantity entity. These are the *head-dependent relations* between the entities. The configuration also defines constraints which improve parsing accuracy by helping to eliminate potentially incorrect parse hypotheses. These constraints may include the number of allowed dependents of a certain kind, the allowed attachment directions, and so on. See the :doc:`User Guide <../userguide/parser>` for a full list of configurable constraints.

The parser runs as the last step in the NLP pipeline. The easiest way to test it is to use the Natural Language Processor's :meth:`process()` method.

.. code:: python

   query = "Two chicken kebab and a kibbi platter with a side of mujadara from palmyra"
   entities = nlp.process(query)['entities']

You can then look at the ``children`` property of each entity to see its dependent entities. For example, you can verify that the numeric quantity "two" gets attached to the dish "chicken kebab":

.. code:: python

   entities[1]

.. code-block:: console

       {'text': 'chicken kebab',
        'type': 'dish',
        'role': None,
        'value': [{'cname': 'Chicken Kebab',
          'score': 147.06445,
          'top_synonym': 'Chicken Kebab',
          'id': 'B01DEFMUSW'},
         {'cname': 'Chicken Tikka Kabab',
          'score': 99.278786,
          'top_synonym': 'chicken kebab',
          'id': 'B01DN5635O'},
         {'cname': 'Chicken Kebab Plate',
          'score': 93.68581,
          'top_synonym': 'Chicken Kebab',
          'id': 'B01CK4ZQ7U'},
         {'cname': 'Chicken Shish Kebab Plate',
          'score': 31.71228,
          'top_synonym': 'chicken kebab plate',
          'id': 'B01N9Z1K2O'},
         {'cname': 'Kebab Wrap',
          'score': 21.39855,
          'top_synonym': 'Kebab Wrap',
          'id': 'B01CRF8ULQ'},
         {'cname': 'Kofte Kebab',
          'score': 21.39855,
          'top_synonym': 'beef kebab',
          'id': 'B01N4VET8U'},
         {'cname': 'Lamb Kebab',
          'score': 21.39855,
          'top_synonym': 'Lamb Kebab',
          'id': 'B01DEFNQIA'},
         {'cname': 'Kebab Platter',
          'score': 21.290503,
          'top_synonym': 'Kebab Dish',
          'id': 'B01CRF8A7U'},
         {'cname': 'Beef Kebab',
          'score': 21.290503,
          'top_synonym': 'Beef Kebab',
          'id': 'B01DEFL75Y'},
         {'cname': 'Prawns Kebab',
          'score': 21.290503,
          'top_synonym': 'Prawns Kebab',
          'id': 'B01DEFO4ZY'}],
        'span': {'start': 4, 'end': 16},
        'children': [{'text': 'Two',
          'type': 'sys_number',
          'role': 'num_orders',
          'value': [{'value': 2}],
          'span': {'start': 0, 'end': 2}}]}

Similarly, the option "mujadara" should apply to the second dish, "kibbi platter":

.. code:: python

   entities[2]

.. code-block:: console

       {'text': 'kibbi platter',
        'type': 'dish',
        'role': None,
        'value': [{'cname': 'Kibbi Platter',
          'score': 173.3674,
          'top_synonym': 'Kibbi Platter',
          'id': 'B01DEFLCL8'},
         {'cname': 'Kibbi',
          'score': 32.295372,
          'top_synonym': 'Kibbi',
          'id': 'B01DEFLNKS'},
         {'cname': 'Salad Platter',
          'score': 20.573273,
          'top_synonym': 'Salad Platter',
          'id': 'B01CUUCX0M'},
         {'cname': 'Beef Shawarma Platter',
          'score': 20.469078,
          'top_synonym': 'beef platter',
          'id': 'B01DEFNZ9K'},
         {'cname': 'Seafood Trio Platter',
          'score': 19.566566,
          'top_synonym': 'Seafood platter',
          'id': 'B01ENMO9Z2'},
         {'cname': 'Lamb Sawarma Platter',
          'score': 19.033337,
          'top_synonym': 'lamb wrap platter',
          'id': 'B01DEFLX3U'},
         {'cname': 'Kebab Platter',
          'score': 18.580826,
          'top_synonym': 'Kebab Platter',
          'id': 'B01CRF8A7U'},
         {'cname': 'Falafel Platter',
          'score': 18.580826,
          'top_synonym': 'Hummus platter',
          'id': 'B01DEFNGVC'},
         {'cname': 'Mixed Appetizer Platter',
          'score': 18.471098,
          'top_synonym': 'appetizer platter',
          'id': 'B01MYFKR8J'},
         {'cname': 'Meza Appetizer Plate',
          'score': 18.099669,
          'top_synonym': 'app platter with a pita',
          'id': 'B01CK4ZK6M'}],
        'span': {'start': 24, 'end': 36},
        'children': [{'text': 'mujadara',
          'type': 'option',
          'role': None,
          'value': [{'cname': 'Mujadara',
            'score': 127.824684,
            'top_synonym': 'Mujadara',
            'id': 'B01DEFLSN0'},
           {'cname': 'Marinara',
            'score': 13.329033,
            'top_synonym': 'Marinara',
            'id': 'B01ENMOGXW'},
           {'cname': 'Marinara',
            'score': 12.622546,
            'top_synonym': 'Marinara',
            'id': 'B01ENTRJNO'},
           {'cname': 'Cheddar',
            'score': 12.622546,
            'top_synonym': 'Cheddar',
            'id': 'B01CH0S4MG'},
           {'cname': 'Cheddar',
            'score': 12.622546,
            'top_synonym': 'Cheddar',
            'id': 'B01MRRXEW5'},
           {'cname': 'Caramelized Onions',
            'score': 12.330592,
            'top_synonym': 'Caramelized Onions',
            'id': 'B01MXSPQLC'},
           {'cname': 'Cheddar Cheese',
            'score': 12.330592,
            'top_synonym': 'Cheddar Cheese',
            'id': 'B01N59TWZ1'},
           {'cname': 'Cheddar',
            'score': 11.7442465,
            'top_synonym': 'Cheddar',
            'id': 'B01M7XF00X'},
           {'cname': 'Cheddar',
            'score': 11.7442465,
            'top_synonym': 'Cheddar',
            'id': 'B01CH0SM7S'},
           {'cname': 'Cheddar Cheese',
            'score': 11.689834,
            'top_synonym': 'Cheddar Cheese',
            'id': 'B01CH0RY9K'}],
          'span': {'start': 53, 'end': 60}}]
        }

Lastly, the restaurant "Palmyra" is a standalone entity without any dependents and hence has no ``children``:

.. code:: python

   entities[4]

.. code-block:: console

       {'text': 'palmyra',
        'type': 'restaurant',
        'role': None,
        'value': [{'cname': 'Palmyra',
          'score': 52.51769,
          'top_synonym': 'Palmyra',
          'id': 'B01DEFLJIO'}],
        'span': {'start': 67, 'end': 73}
       }

When extending the blueprint to your custom application data, the parser should work fine out-of-the-box for most queries, provided that head-dependent relations are properly set in the configuration file. To improve its accuracy further, experiment with the parser constraints, optimizing them for what makes the best sense for your data. See the :doc:`Language Parser <../userguide/parser>` section of the User Guide for details.

.. admonition:: Exercise

   - Experiment with the different constraints in the parser configuration and observe how it affects parsing accuracy.

   - Think of any additional entity relationships you might want to capture when extending the blueprint with new entity types for your own use case. For instance, the blueprint treats ``restaurant`` as a standalone entity. However, you might want to introduce related entities like ``location`` (to search for restaurants by geographical area or address) or ``price_level`` (the number of "dollar signs" or average price per person at a restaurant). In that case, you would need to update the parser configuration to extract these new relations.


9. Using the Question Answerer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :doc:`Question Answerer <../userguide/kb>` component in MindMeld is mainly used within dialogue state handlers for retrieving information from the knowledge base. For example, in our ``welcome`` dialogue state handler, we use the Question Answerer to retrieve the top three entries in our ``restaurants`` index and present their names as suggestions to the user.

.. code:: python

   from mindmeld.components.question_answerer import QuestionAnswerer
   qa = QuestionAnswerer(app_path='food_ordering')
   restaurants = qa.get(index='restaurants')[0:3]
   [restaurant['name'] for restaurant in restaurants]

.. code-block:: console

   [
    "Firetrail Pizza",
    "Grandma's Deli & Cafe",
    "The Salad Place"
   ]

The ``build_order`` handler retrieves details about the user's restaurant and dish selections from the knowledge base, and uses the information to

  #. Suggest restaurants to the user that offer their requested dishes
  #. Resolve the requested dish name to the most likely entry on a restaurant's menu
  #. Verify that a requested dish is offered at the selected restaurant
  #. Verify that a requested option is applicable for the selected dish
  #. Get pricing for the requested dish and options

Look at the ``build_order`` implementation in ``__init__.py`` to better understand the different ways you can leverage the knowledge base and Question Answerer to provide intelligent responses to the user. See the :doc:`User Guide <../userguide/kb>` for an explanation of the retrieval and ranking mechanisms that the Question Answerer offers.

.. admonition:: Exercise

   - Use the Question Answerer within the ``build_order`` state handler to add support for searching for restaurants by ``cuisine`` and searching for dishes by ``category``.

   - When customizing the blueprint for your own app, consider adding location information (for restaurants) and popularity (for both restaurants and dishes) in the knowledge base. You could then use the Question Answerer to rank restaurant and dish results using those factors, and evaluate whether this provides a more relevant list of suggestions to the user.

   - Think of other important data that would be useful to have in the knowledge base for a food ordering use case. Identify the ways that data could be leveraged to provide a more intelligent user experience.


10. Testing and Deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^

Once all the individual pieces (NLP, Question Answererer, Dialogue State Handlers) have been trained, configured, or implemented, use the :class:`Conversation` class in MindMeld to perform an end-to-end test of your conversational app.

For instance:

.. code:: python

   from mindmeld.components.dialogue import Conversation
   conv = Conversation(nlp=nlp, app_path='food_ordering')
   conv.say("Get me a pad thai and pinapple fried rice from thep phanom thai")

.. code-block:: console

   ['Sure, I have 1 order of 62. Pad Thai and 1 order of 66. Pineapple Fried Rice from Thep Phanom Thai Restaurant for a total price of $34.00. Would you like to place the order?']

The :meth:`say()` method packages the input text in a user request object and passes it to the MindMeld Application Manager to simulate a user interacting with the application. The method then outputs the textual part of the response sent by the app's Dialogue Manager. In the above example, we requested dishes from a restaurant, in a single query. The app responded, as expected, with a preview of the order details and a confirmation prompt.

You can also try out multi-turn dialogues:

.. code:: python

   >>> conv = Conversation(nlp=nlp, app_path='food_ordering')
   >>> conv.say('Hi there!')
   ['Hello. Some nearby popular restaurants you can order delivery from are Curry Up Now, Ganim's Deli, Firetrail Pizza.]
   >>> conv.say("I'd like to order from Saffron 685 today")
   ['Great, what would you like to order from Saffron 685?', 'Listening...']
   >>> conv.say("I would like two dolmas and a meza appetizer plate")
   ['Sure, I got 2 Dolmas, 1 Meza Appetizer Plate from Saffron 685 for a total price of $18.75. Would you like to place the order?', 'Listening...']
   >>> conv.say("I almost forgot! Could you also add a baklava please?")
   ['Sure, I got 2 Dolmas, 1 Meza Appetizer Plate, 1 Baklava from Saffron 685 for a total price of $22. Would you like to place the order?', 'Listening...']
   >>> conv.say("Yes")
   ['Great, your order from Saffron 685 will be delivered in 30-45 minutes.']
   >>> conv.say("Thank you!")
   ['Have a nice day.']

.. admonition:: Exercise

   Test the app multiple times with different conversational flows. Keep track of all cases where the response does not make good sense. Then, analyze those cases in detail. You should be able to attribute each error to a specific step in our end-to-end processing (e.g., incorrect intent classification, missed entity recognition, unideal natural language response, and so on). Categorizing your errors in this manner helps you understand the strength of each component in your conversational AI pipeline and informs you about the possible next steps for improving the performance of each individual module.


Refer to the User Guide for tips and best practices on testing your app before launch.

.. Once you're satisfied with the performance of your app, you can deploy it to production as described in the :doc:`deployment <../userguide/deployment>` section of the User Guide.
