Food Ordering
=============

1. The Use Case
^^^^^^^^^^^^^^^

This application provides a conversational interface for users to order food delivery from a service like Amazon Restaurants, Grubhub or DoorDash. It allows users to place these orders in a hands-free manner from the convenience of their homes, while still providing the same experience they would get when conversing with a restaurant waiter or a drive-through attendant.


2. Example Dialogue Interactions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The conversational user flows for a food ordering app can get highly complex, depending on the envisioned functionality and the amount of user guidance required at each step. This design exercise usually requires multiple iterations to finalize and enumerate all the possible user interactions. Below are examples of scripted dialogue interactions for a couple of possible user flows.

.. image:: /images/food_ordering_interactions.png
    :width: 700px
    :align: center


3. Domain-Intent-Entity Hierarchy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The NLP model hierarchy for our food ordering application is illustrated below.

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
   - ``option`` — The name of an option used for customizing a dish
   - ``sys:number`` — The quantity of a given dish, captured by the ``number`` :doc:`system entity <../userguide/entity_recognition>`

Our application requires labeled training data covering all of the above intents and entities in order to train the intent classifier and entity recognizer components in the NLP pipeline. To download the required data and accompanying code for running this blueprint, run this command in a directory of your choice.

.. code-block:: console

    $ python -c "import mmworkbench as wb; wb.blueprint('food_ordering');"

This should create a Workbench project folder called ``food_ordering`` in your current directory with the following structure:

.. image:: /images/food_ordering_directory.png
    :width: 250px
    :align: center


4. Dialogue States
^^^^^^^^^^^^^^^^^^

To capture the functionality we envision, our app needs seven dialogue states, one for each intent:

   - ``welcome`` — Begins an interaction and welcomes the user
   - ``build_order`` — Guides the user to make selections and build up the delivery order
   - ``place_order`` — Places the order to complete the transaction
   - ``start_over`` — Cancels the ongoing transaction and prompts the user for a new request
   - ``say_goodbye`` — Ends the current interaction
   - ``provide_help`` — Provides help information in case the user gets stuck
   - ``default`` — Prompts the user to get back to food ordering in case he goes off topic

All of the dialogue states and their associated handlers are defined in the :keyword:`app.py` application container file at the top level of the blueprint folder. For many of our dialogue states, namely, ``welcome``, ``start_over``, ``say_goodbye``, ``provide_help`` and ``default``, the handler logic is fairly straightforward. It mostly involves choosing from a set of pre-scripted natural language responses and replying to the user.

For example, here's the ``say_goodbye`` state handler, where we clear the :doc:`dialogue frame <../userguide/dialogue_manager>` and use the :doc:`responder <../userguide/dialogue_manager>` object to reply with one of our scripted "goodbye" responses:

.. code:: python

    @app.handle(intent='exit')
    def say_goodbye(context, slots, responder):
        """
        When the user ends a conversation, clear the dialogue frame and say goodbye.
        """
        # Clear the dialogue frame to start afresh for the next user request.
        context['frame'] = {}

        # Respond with a random selection from one of the canned "goodbye" responses.
        responder.reply(['Bye!', 'Goodbye!', 'Have a nice day.', 'See you later.'])

The core business logic for our application mainly resides in the ``build_order`` and ``place_order`` dialogue state handlers, where we use the :doc:`Question Answerer <../userguide/question_answering>` and external API calls to process the transaction.

Here is a simplistic implementation of the ``build_order`` handler for illustrative purposes:

.. code:: python

    @app.handle(intent='build_order')
    def build_order(context, slots, responder):
        """
        When the user expresses an intent to make food selections, build up the order by 
        adding the requested dishes to their "check-out" basket.
        """
        # Get the first recognized restaurant entity in the user query.
        rest_entity = next(e for e in context['entities'] if e['type'] == 'restaurant')

        # Resolve the restaurant entity to a specific entry in the knowledge base (KB).
        selected_restaurant = _get_restaurant_from_kb(rest_entity['value'][0]['id'])

        # Next, get all the recognized dish entities in the user query.
        dish_entities = [e for e in context['entities'] if e['type'] == 'dish']

        # Add dishes one by one to the "check-out" shopping basket.
        selected_dishes = list()
        for entity in dish_entities:
            # Resolve the dish entity to a KB entry using restaurant information.
            selected_dishes.append(_resolve_dish(entity, selected_restaurant))

        # Store dish and restaurant selections in the dialogue frame.
        context['frame']['restaurant'] = selected_restaurant
        context['frame']['dishes'] = selected_dishes

        # Respond with a preview of the current basket and prompt for order confirmation.
        slots['restaurant_name'] = selected_restaurant['name']
        slots['dish_names'] = ', '.join([dish['name'] for dish in selected_dishes])
        slots['price'] = sum([dish['price'] for dish in selected_dishes])
        responder.prompt('Sure, I got {dish_names} from {restaurant_name} for a total '
                         'price of ${price:.2f}. Would you like to place the order?')

The code above assumes that every user query contains a ``restaurant`` entity and at least one ``dish`` entity. It uses the Question Answerer (within the :keyword:`_get_restaurant_from_kb()` and :keyword:`_resolve_dish()` methods not shown above) to select the most likely restaurant and dishes requested by the user. That information is then saved in the dialogue frame for use in future conversational turns and also presented to the user via the responder object.

For a more realistic implementation of ``build_order`` that deals with varied user flows and the full code behind all the dialogue state handlers, see the :keyword:`app.py` file in the blueprint folder. 


5. Knowledge Base
^^^^^^^^^^^^^^^^^

Our food ordering app leverages publicly available information about San Francisco restaurants, scraped from the `Amazon Restaurants <https://primenow.amazon.com/restaurants>`_ website. Specifically, the knowledge base comprises of two indexes in `Elasticsearch <https://www.elastic.co/products/elasticsearch>`_:

   - ``restaurants`` — Each entry describes a unique restaurant location
   - ``menu_items`` — Each entry describes a unique dish on a specific restaurant's menu

For example, here's the knowledge base entry for a Thai restaurant in San Francisco named "Thoughts Style Cuisine Showroom" in the ``restaurants`` index:

.. code:: javascript

    {
        'categories': ['Drinks', 'Watery', 'Beginnings', 'Salads', 'Fried Rice', 'Significant', 'Noodles', 'Supper Sizzles', 'Sugary'],
        'cuisine_types': ['Thai'],
        'id': 'B01DUUMTLY',
        'image_url': 'https://images-na.ssl-images-amazon.com/images/G/01/ember/restaurants/SanFrancisco/ThoughtsStyleCuisineShowroom/logo_232x174._CB295406843_SX600_QL70_.png',
        'menus': [{'id': '4b999943-a3d6-4af1-b7ab-fbd56094c40d',
                   'option_groups': [{'id': 'Alacarte2',
                     'max_selected': 1,
                     'min_selected': 0,
                     'name': 'Make It A La Carte',
                     'options': [{'description': None,
                       'id': 'B01ERURPOM',
                       'name': 'Make It A La Carte',
                       'price': 4.0}]},
                    {'id': 'Alacarte',
                     'max_selected': 1,
                     'min_selected': 0,
                     'name': 'Make It A La Carte',
                     'options': [{'description': None,
                       'id': 'B01DWWSZN6',
                       'name': 'Make It A La Carte',
                       'price': 2.0}]}],
                   'size_groups': []}],
        'name': 'Thoughts Style Cuisine Showroom',
        'num_reviews': None,
        'price_range': 2.0,
        'rating': None
    }

Similarly, here's an example of a knowledge base entry in the ``menu_items`` index for a specific dish at the above restaurant.

.. code:: javascript

    {
        'category': 'Fried Rice',
        'description': None,
        'id': 'B01DWWTMGK',
        'img_url': None,
        'menu_id': '4b999943-a3d6-4af1-b7ab-fbd56094c40d',
        'name': 'Basil Fried Rice with Crispy Pork Shoulder',
        'option_groups': [],
        'popular': False,
        'price': 13.0,
        'restaurant_id': 'B01DUUMTLY',
        'size_group': None,
        'size_prices': []}
    }

Assuming you have Elasticsearch installed on your machine, running the :keyword:`blueprint()` command described above should build the knowledge base for your app by creating the two indexes and importing all the necessary data. To verify that the knowledge base has been set up correctly, you can use the Question Answerer to query your indexes.

For example:

.. code:: python

  >>> from mmworkbench.components.question_answerer import QuestionAnswerer
  >>> qa = QuestionAnswerer(app_path='food_ordering')
  >>> qa.get(index='menu_items')[0]
  {
    'category': 'Signature Pizza',
    'description': 'Fresh mushroom, red onion, artichoke heart, green pepper, vine tomato, broccoli, fresh basil, tomato sauce, mozzarella & sprinkle of cheddar',
    'id': 'B06XB2DFDV',
    'img_url': None,
    'menu_id': 'f5f5e585-d56b-45de-b592-c453eaf1f082',
    'name': 'Drag It Thru The Garden',
    'option_groups': ['crust', 'signature toppings2'],
    'popular': False,
    'price': 10.95,
    'restaurant_id': 'B06WRPJ21G',
    'size_group': 'Size',
    'size_prices': [{'id': 'B06X9XWPTV', 'name': 'Indee-8', 'price': 10.95},
     {'id': 'B06XB3FXNZ', 'name': 'Medium-12', 'price': 21.95},
     {'id': 'B06X9ZX74N', 'name': 'Large-14', 'price': 25.95},
     {'id': 'B06XB12GH5', 'name': 'Xlarge-16', 'price': 29.95},
     {'id': 'B06X9XZPJ1', 'name': 'Huge-18', 'price': 33.95}]
  }


6. Training Data
^^^^^^^^^^^^^^^^

The labeled data for training our NLP pipeline was created using a combination of in-house data generation and crowdsourcing techniques. This is an iterative process that is described in more detail in the :doc:`user guide <../userguide/training_data>`. But briefly, it requires at least the following data generation tasks:

1. Exploratory data generation for guiding the app design

.. code:: text

   "How would you talk to a conversational app to place orders for food delivery?"

2. Targeted query generation for training the Intent Classifier

.. code:: text

   (build_order) "What would you say to the app to make food or restaurant selections and
                  create your delivery order?"

   (start_over) "How would you ask the app to cancel your current selections and start over?"


3. Targeted query annotation for training Entity Recognizer

.. code:: text

   (build_order) "Annotate all occurrences of restaurant, cuisine, category, dish and
                  option names in the given query."

4. Targeted synonym generation for training Entity Resolver

.. code:: text

   (restaurant) "What are the different ways in which you would refer to this
                 restaurant location?"

   (dish) "What names would you use to refer to this dish on a restaurant's menu?"

The training data for intent classification and entity recognition can be found in the :keyword:`domains` directory, whereas the data for entity resolution is in the :keyword:`entities` directory, both at the root level of the blueprint folder.


7. Training the NLP Classifiers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^




