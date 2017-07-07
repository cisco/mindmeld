Food Ordering
=============

This page documents the Workbench blueprint for a conversational application that allows users to order food for delivery from nearby restaurants.


Quick Start
-----------

|

1. Download
^^^^^^^^^^^

Open a python shell and type the following commands to download and set up the food ordering blueprint application.

.. code:: python

   >>> import mmworkbench as wb
   >>> wb.configure_logs()
   >>> wb.blueprint('food_ordering')


2. Build
^^^^^^^^

Build the Natural Language Processing models that power the app.

.. code:: python

   >>> from mmworkbench.components import NaturalLanguageProcessor
   >>> nlp = NaturalLanguageProcessor('food_ordering')
   >>> nlp.build()


3. Run
^^^^^^

Interact with the app in the python shell using the commands below. Try out the examples shown here as well as some queries of your own.

.. code:: python

   >>> from mmworkbench.components.dialogue import Conversation
   >>> conv = Conversation(nlp=nlp, app_path='food_ordering')
   >>> conv.say('Hello!')
   ['Hello. Some nearby popular restaurants you can order delivery from are Curry Up Now, Ganim's Deli, Firetrail Pizza.]
   >>> conv.say("Get me a saag paneer and garlic naan from urban curry")
   ['Sure, I got Saag Paneer, Garlic Naan from Urban Curry for a total price of $14.70. Would you like to place the order?']
   >>> conv.say("Bye")
   ['Goodbye!']


Deep Dive
---------

|

1. The Use Case
^^^^^^^^^^^^^^^

This application provides a conversational interface for users to order food delivery from a service like Amazon Restaurants, Grubhub or DoorDash. It allows users to place these orders in a hands-free manner from the convenience of their homes, while still providing the same experience they would get when conversing with a restaurant waiter or a drive-through attendant.


2. Example Dialogue Interactions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The conversational user flows for a food ordering app can get highly complex, depending on the envisioned functionality and the amount of user guidance required at each step. This design exercise usually requires multiple iterations to finalize and enumerate all the possible user interactions. Below are examples of scripted dialogue interactions for a couple of possible user flows.

.. image:: /images/food_ordering_interactions.png
    :width: 700px
    :align: center

.. admonition:: Exercise

   Pick a representation (textual or graphical) that's convenient to you and try to design as many user flows as you can, in each case, capturing the entire dialogue from start to finish. Think of scenarios other than the examples above, such as a user asking to order from a specific restaurant without choosing a dish, requesting a dish that is not available at the selected restaurant, asking for a restaurant location that doesn't exist, choosing a customization option that is not applicable for the chosen dish, etc.


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
   - ``option`` — The name of an available option (customization, add-on, etc.) for a dish
   - ``sys_number`` — The quantity of a given dish, captured by the :doc:`number system entity <../userguide/entity_recognizer>`

.. admonition:: Exercise

   While the intents and entities in the blueprint provide a good starting point, you may need additional ones to cover the desired scope of your app. Enumerate some other intents (e.g. ``check_order_status``, ``get_calories``, etc.) and entities (e.g. ``location``, ``price_level``, etc.) you may need in a food ordering use case.

Our application requires labeled training data covering all of the above intents and entities in order to train the different machine learning models in the NLP pipeline. To download the required data and accompanying code for running this blueprint, run this command in a directory of your choice.

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

All of the dialogue states and their associated handlers are defined in the ``app.py`` application container file at the top level of the blueprint folder. For many of our dialogue states, namely, ``welcome``, ``start_over``, ``say_goodbye``, ``provide_help`` and ``default``, the handler logic is fairly straightforward. It mostly involves choosing from a set of pre-scripted natural language responses and replying to the user.

For example, here's the ``say_goodbye`` state handler, where we clear the :doc:`dialogue frame <../userguide/dm>` and use the :doc:`responder <../userguide/dm>` object to reply with one of our scripted "goodbye" responses:

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

The core business logic for our application mainly resides in the ``build_order`` and ``place_order`` dialogue state handlers, where we use the :doc:`Question Answerer <../userguide/kb>` and external API calls to process the transaction.

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

The code above assumes that every user query contains a ``restaurant`` entity and at least one ``dish`` entity. It uses the Question Answerer (within the :func:`_get_restaurant_from_kb()` and :func:`_resolve_dish()` functions not shown above) to select the most likely restaurant and dishes requested by the user. That information is then saved in the dialogue frame for use in future conversational turns and also presented to the user via the responder object.

For a more realistic implementation of ``build_order`` that deals with varied user flows and the full code behind all the dialogue state handlers, see the ``app.py`` file in the blueprint folder.

.. admonition:: Exercise

   Extend the ``build_order`` dialogue state handler in ``app.py`` to handle more user flows or handle the existing ones in a smarter way. There are many suggestions for improvements in the comments accompanying the code in the ``app.py`` file. Here are a few more:

   - Add support to select restaurants by ``cuisine`` or to search for dishes by ``category``. These are already modeled as entities in the blueprint and are also available as part of the restaurant and dish metadata stored in the knowledge base. But ``build_order`` needs some additional code to handle queries containing these entities.

   - After providing restaurant suggestions to a user based on a dish they requested, do not ask them to repeat their dish selection from scratch in the next turn. Instead, keep track of the dish they were originally interested in and directly add that to the check-out basket when the user makes a restaurant selection.


5. Knowledge Base
^^^^^^^^^^^^^^^^^

Our food ordering app leverages publicly available information about San Francisco restaurants, scraped from the `Amazon Restaurants <https://primenow.amazon.com/restaurants>`_ website. Specifically, our knowledge base comprises of two indexes in `Elasticsearch <https://www.elastic.co/products/elasticsearch>`_:

   - ``restaurants`` — Stores information about restaurant locations
   - ``menu_items`` — Stores information about dishes on different restaurants' menus

For example, here's the knowledge base entry in the ``restaurants`` index for a Thai restaurant in San Francisco named "Thoughts Style Cuisine Showroom":

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

Assuming you have Elasticsearch installed on your machine, running the :func:`blueprint()` command described above should build the knowledge base for the food ordering app by creating the two indexes and importing all the necessary data. To verify that the knowledge base has been set up correctly, you can use the :doc:`Question Answerer <../userguide/kb>` to query the indexes.

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

.. admonition:: Exercise

   The blueprint comes with a pre-configured, pre-populated knowledge base to help you get up and running with an end-to-end working application quickly. To learn how you can set up knowledge base indexes from scratch for your own data, read the user guide section on :doc:`Question Answerer <../userguide/kb>`.


6. Training Data
^^^^^^^^^^^^^^^^

The labeled data for training our NLP pipeline was created using a combination of in-house data generation and crowdsourcing techniques. This is a highly important multi-step process that is described in more detail in :doc:`Step 6 <../quickstart/06_generate_representative_training_data>` of the Step-By-Step Guide. But briefly, it requires at least the following data generation tasks:

+--------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| Purpose                                                      | Question posed to data annotators                                                                                       |
+==============================================================+=========================================================================================================================+
| Exploratory data generation for guiding the app design       | "How would you talk to a conversational app to place orders for food delivery?"                                         |
+--------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| Targeted query generation for training the Intent Classifier | ``build_order``: "What would you say to the app to make food or restaurant selections and create your delivery order?"  |
|                                                              |                                                                                                                         |
|                                                              | ``start_over``: "How would you ask the app to cancel your current selections and start over?"                           |
+--------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| Targeted query annotation for training the Entity Recognizer | ``build_order``: "Annotate all occurrences of restaurant, cuisine, category, dish and option names in the given query." |
+--------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| Targeted synonym generation for training the Entity Resolver | ``restaurant``: "What are the different ways in which you would refer to this restaurant location?"                     |
|                                                              |                                                                                                                         |
|                                                              | ``dish``: "What names would you use to refer to this item on the restaurant menu?"                                      |
+--------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+

The training data for intent classification and entity recognition can be found in the ``domains`` directory, whereas the data for entity resolution is in the ``entities`` directory, both located at the root level of the blueprint folder.

.. admonition:: Exercise

   - Read :doc:`Step 6 <../quickstart/06_generate_representative_training_data>` of the Step-By-Step Guide for best practices around training data generation and annotation for conversational apps. Following those principles, create additional labeled data for all the intents in this blueprint and use them as held-out validation data for evaluating your app. You can read more about :doc:`NLP model evaluatation and error analysis <../userguide/nlp>` in the user guide.

   - To train NLP models for your own food ordering app, you can start by reusing the blueprint data for generic intents like ``greet``, ``exit`` and ``help``. However, for core intents like ``build_order``, it's recommended that you collect new training data that is tailored towards the entities (restaurants, dishes, etc.) that your app needs to support. Follow the same approach to gather new training data for the ``build_order`` intent or any additional intents and entities needed for your app.


7. Training the NLP Classifiers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To put the training data to use and train a baseline NLP system for your app using Workbench's default machine learning settings, use the :meth:`build()` method of the :class:`NaturalLanguageProcessor` class:

.. code:: python

   >>> from mmworkbench.components.nlp import NaturalLanguageProcessor
   >>> nlp = NaturalLanguageProcessor(app_path='food_ordering')
   >>> nlp.build()
   Fitting intent classifier: domain='ordering'
   Loading queries from file ordering/build_order/train.txt
   Loading queries from file ordering/exit/train.txt
   Loading queries from file ordering/greet/train.txt
   Loading queries from file ordering/help/train.txt
   Loading queries from file ordering/place_order/train.txt
   Loading queries from file ordering/start_over/train.txt
   Loading queries from file ordering/unsupported/train.txt
   Selecting hyperparameters using k-fold cross validation with 10 splits
   Best accuracy: 98.11%, params: {'C': 100, 'class_weight': {0: 1.7987394957983194, 1: 3.0125475285171097, 2: 0.89798826487845773, 3: 4.4964705882352938, 4: 2.5018518518518515, 5: 1.7559183673469387, 6: 0.46913229018492181}, 'fit_intercept': True}
   Fitting entity recognizer: domain='ordering', intent='place_order'
   Fitting entity recognizer: domain='ordering', intent='unsupported'
   Fitting entity recognizer: domain='ordering', intent='greet'
   Fitting entity recognizer: domain='ordering', intent='exit'
   Fitting entity recognizer: domain='ordering', intent='build_order'
   Selecting hyperparameters using k-fold cross validation with 5 splits
   Best accuracy: 92.46%, params: {'C': 1000000, 'penalty': 'l2'}
   Fitting entity recognizer: domain='ordering', intent='start_over'
   Fitting entity recognizer: domain='ordering', intent='help'

.. tip::

  During active development, it's helpful to increase the :doc:`Workbench logging level <../userguide/getting_started>` to better understand what's happening behind the scenes. All code snippets here assume that logging level has been set to verbose.

You should see a cross validation accuracy of around 98% for the :doc:`Intent Classifier <../userguide/intent_classifier>` and about 92% for the :doc:`Entity Recognizer <../userguide/entity_recognizer>`. To see how the trained NLP pipeline performs on a test query, use the :meth:`process()` method.

.. code:: python

   >>> nlp.process("I'd like a mujaddara wrap and two chicken kebab from palmyra")
   {
    'domain': 'ordering',
    'entities': [{'role': None,
      'span': {'end': 24, 'start': 11},
      'text': 'mujaddara wrap',
      'type': 'dish',
      'value': [{'cname': 'Mujaddara Wrap', 'id': 'B01DEFNIRY'}]},
     {'confidence': 0.15634607039069398,
      'role': None,
      'span': {'end': 32, 'start': 30},
      'text': 'two',
      'type': 'sys_number',
      'value': {'value': 2}},
     {'children': [{'confidence': 0.15634607039069398,
        'role': None,
        'span': {'end': 32, 'start': 30},
        'text': 'two',
        'type': 'sys_number',
        'value': {'value': 2}}],
      'role': None,
      'span': {'end': 46, 'start': 34},
      'text': 'chicken kebab',
      'type': 'dish',
      'value': [{'cname': 'Chicken Kebab', 'id': 'B01DEFMUSW'}]},
     {'role': None,
      'span': {'end': 59, 'start': 53},
      'text': 'palmyra',
      'type': 'restaurant',
      'value': [{'cname': 'Palmyra', 'id': 'B01DEFLJIO'}]}],
    'intent': 'build_order',
    'text': "I'd like a mujaddara wrap and two chicken kebab from palmyra"
   }

For the data distributed with this blueprint, the baseline performance is already high. However, when extending the blueprint with your own custom food ordering data, you may find that the default settings may not be optimal and you could get better accuracy by individually optimizing each of the NLP components.

A good place to start is by inspecting the baseline configuration used by the different classifiers. The user guide lists and describes all of the available configuration options in detail. As an example, the code below shows how to access the model and feature extraction settings for the Intent Classifier.

.. code:: python

   >>> ic = nlp.domains['ordering'].intent_classifier
   >>> ic.config.model_settings['classifier_type']
   'logreg'
   >>> ic.config.features
   {
    'bag-of-words': {'lengths': [1]},
    'freq': {'bins': 5},
    'in-gaz': {},
    'length': {}
   }

You can experiment with different learning algorithms (model types), features, hyperparameters and cross-validation settings by passing the appropriate parameters to the classifier's :meth:`fit()` method. Here are a couple of examples.

Change the feature extraction settings to use bag of bigrams in addition to the default bag of words:

.. code:: python

   >>> features = {
   ...             'bag-of-words': {'lengths': [1, 2]},
   ...             'freq': {'bins': 5},
   ...             'in-gaz': {},
   ...             'length': {}
   ...            }
   >>> ic.fit(features=features)
   Fitting intent classifier: domain='ordering'
   Selecting hyperparameters using k-fold cross validation with 10 splits
   Best accuracy: 98.36%, params: {'C': 10000, 'class_weight': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}, 'fit_intercept': False}

Change the classification model to random forest instead of the default logistic regression:

.. code:: python

   >>> ic.fit(model_settings={'classifier_type': 'rforest'})
   Fitting intent classifier: domain='ordering'
   Selecting hyperparameters using k-fold cross validation with 10 splits
   Best accuracy: 97.31%, params: {'max_features': 'auto', 'n_estimators': 10, 'n_jobs': -1}

Similar options are available for inspecting and experimenting with the Entity Recognizer and other NLP classifiers as well. Finding the optimal machine learning settings is a highly iterative process involving several rounds of model training (with varying configurations), testing and error analysis. Refer to the appropriate sections in the user guide for a detailed discussion on training, tuning and evaluating the various Workbench classifiers.

.. admonition:: Exercise

   Experiment with different models, features and hyperparameter selection settings to see how they affect the classifier performance. It's helpful to have a held-out validation set to evaluate your trained NLP models and analyze the misclassified test instances. You could then use observations from the error analysis to inform your machine learning experimentation. For more examples and discussion on this topic, refer to the :doc:`user guide <../userguide/nlp>`.


8. Parser Configuration
^^^^^^^^^^^^^^^^^^^^^^^

Once the NLP classification models are trained, you can configure and run the Workbench :doc:`Language Parser <../userguide/parser>` to link related entities into meaningful entity groups. The application configuration file, ``config.py``, at the top level of blueprint folder contains the following parser configuration:

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

In simple terms, the configuration for our food ordering app specifies that a dish entity can have a numeric quantity entity and an option entity as its attributes, and an option can in turn have another quantity entity associated with it. In addition to defining the head - dependent relations between the entities, the config also defines constraints such as the number of allowed dependents of a certain kind, the allowed attachment directions, etc. These constraints improve parsing accuracy by helping to eliminate potentially incorrect parse hypotheses. A full list of configurable constraints can be found in the :doc:`user guide <../userguide/parser>`.

Since the parser runs as the last step in the NLP pipeline, the easiest way to test it is using the Natural Language Processor's :meth:`process()` method.

.. code:: python

   >>> query = "Two chicken kebab and a kibbi platter with a side of mujadara from palmyra"
   >>> entities = nlp.process(query)['entities']

You can then look at the ``children`` property of each entity to see its dependent entities. For example, you can verify that the numeric quantity "two" gets attached to the dish "chicken kebab":

.. code:: python

   >>> entities[1]
   {
    'children': [{
        'confidence': 0.15634607039069398,
        'role': None,
        'span': {'end': 2, 'start': 0},
        'text': 'Two',
        'type': 'sys_number',
        'value': {'value': 2}
    }],
    'role': None,
    'span': {'end': 16, 'start': 4},
    'text': 'chicken kebab',
    'type': 'dish',
    'value': [{'cname': 'Chicken Kebab', 'id': 'B01DEFMUSW'}]
   }

Similarly, the option "mujadara" should apply to the second dish, "kibbi platter":

.. code:: python

   >>> entities[2]
   {
    'children': [{
        'role': None,
        'span': {'end': 60, 'start': 53},
        'text': 'mujadara',
        'type': 'option',
        'value': [{'cname': 'Mujadara', 'id': 'B01DEFLSN0'}]
    }],
    'role': None,
    'span': {'end': 36, 'start': 24},
    'text': 'kibbi platter',
    'type': 'dish',
    'value': [{'cname': 'Kibbi Platter', 'id': 'B01DEFLCL8'}]
   }

Lastly, the restaurant "Palmyra" is a standalone entity without any dependents and hence has no ``children``:

.. code:: python

   >>> entities[4]
   {
    'role': None,
    'span': {'end': 73, 'start': 67},
    'text': 'palmyra',
    'type': 'restaurant',
    'value': [{'cname': 'Palmyra', 'id': 'B01DEFLJIO'}]
   }

When extending the blueprint to your custom application data, the parser should work fine out-of-the-box for most queries as long as the head - dependent relations are properly set in the configuration file. Generally speaking, you should be able to improve its accuracy even further by experimenting with the parser constraints and optimizing them for what makes the best sense for your data. Read the :doc:`Language Parser user guide <../userguide/parser>` for a more detailed discussion.

.. admonition:: Exercise

   - Experiment with the different constraints in the parser configuration and observe how it affects the parsing accuracy.

   - Think of any additional entity relationships you might want to capture when extending the blueprint with new entity types for your own use case. For instance, ``restaurant`` is a standalone entity in the blueprint. However, when you introduce related entities like ``location`` (to search for restaurants by geographical area or address) or ``price_level`` (the number of "dollar signs" or average price per person at a restaurant), you would have to update the parser configuration to extract these new relations.


9. Using the Question Answerer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :doc:`Question Answerer <../userguide/kb>` component in Workbench is mainly used within dialogue state handlers for retrieving information from the knowledge base. For example, in our ``welcome`` dialogue state handler, we use the Question Answerer to retrieve the top three entries in our ``restaurants`` index and present their names as suggestions to the user.

.. code:: python

   >>> restaurants = app.question_answerer.get(index='restaurants')[0:3]
   >>> [restaurant['name'] for restaurant in restaurants]
   [
    'Curry Up Now',
    "Ganim's Deli",
    'Firetrail Pizza'
   ]

The ``build_order`` handler retrieves details about the user's restaurant and dish selections from the knowledge base, and uses the information to:

  #. Suggest restaurants to the user that offer their requested dishes.
  #. Resolve the requested dish name to the most likely entry on a restaurant's menu.
  #. Verify that a requested dish is offered at the selected restaurant.
  #. Verify that a requested option is applicable for the selected dish.
  #. Get pricing for the requested dish and options.

Look at the ``build_order`` implementation in ``app.py`` to better understand the different ways in which the knowledge base and Question Answerer can be leveraged to provide intelligent responses to the user. Also refer to the :doc:`user guide <../userguide/kb>` for an in-depth explanation of the retrieval and ranking mechanisms offered by the Question Answerer.

.. admonition:: Exercise

   - Use the Question Answerer within the ``build_order`` state handler to add support for searching for restaurants by ``cuisine`` and searching for dishes by ``category``.

   - When customizing the blueprint for your own app, consider adding location information (for restaurants) and popularity (for both restaurants and dishes) in the knowledge base. You could then use the Question Answerer to rank restaurant and dish results using those factors to provide a more relevant list of suggestions to the user.

   - Think of other important data that would be useful to have in the knowledge base for a food ordering use case and how it could be leveraged to provide a more intelligent user experience.


10. Testing and Deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^

Once all the individual pieces (NLP, Question Answererer, Dialogue State Handlers) have been trained, configured or implemented, you can do an end-to-end test of your conversational app using the :class:`Conversation` class in Workbench.

For instance:

.. code:: python

   >>> from mmworkbench.components.dialogue import Conversation
   >>> conv = Conversation(nlp=nlp, app_path='food_ordering')
   >>> conv.say("Get me a saag paneer and garlic naan from urban curry")
   ['Sure, I got Saag Paneer, Garlic Naan from Urban Curry for a total price of $14.70. Would you like to place the order?']

The :meth:`say()` method packages the input text in a :doc:`user request <../userguide/interface>` object and passes it to the Workbench :doc:`Application Manager <../userguide/application_manager>` to a simulate an external user interaction with the application. It then outputs the textual part of the response sent by the app's Dialogue Manager. In the above example, we requested a couple of dishes from a restaurant and the app responded, as expected, with a preview of the order details and a confirmation prompt.

You can also try out multi-turn dialogues:

.. code:: python

   >>> conv.say('Hi there!')
   ['Hello. Some nearby popular restaurants you can order delivery from are Curry Up Now, Ganim's Deli, Firetrail Pizza.]
   >>> conv.say("I'd like to order from Saffron 685 today")
   ['Great, what would you like to order from Saffron 685?']
   >>> conv.say("I would like two dolmas and a meza appetizer plate")
   ['Sure, I got 2 Dolmas, 1 Meza Appetizer Plate from Saffron 685 for a total price of $18.75. Would you like to place the order?']
   >>> conv.say("I almost forgot! Could you also add a baklava please?")
   ['Sure, I got 2 Dolmas, 1 Meza Appetizer Plate, 1 Baklava from Saffron 685 for a total price of $22. Would you like to place the order?']
   >>> conv.say("Yes")
   ['Great, your order from Saffron 685 will be delivered in 30-45 minutes.']
   >>> conv.say("Thank you!")
   ['Have a nice day.']

.. admonition:: Exercise

   Test the app multiple times with different conversational flows and keep track of all the cases where the response doesn't make sense. Then, analyze those cases in detail to attribute each error to a specific step in our end-to-end processing (e.g. incorrect intent classification, missed entity recognition, unideal natural language response, etc.). Categorizing your errors in this manner helps in understanding the strength of each component in your conversational AI pipeline and informs you about the possible next steps for improving the performance of each individual module.


Refer to the user guide for tips and best practices on testing your app before launch. Once you're satisfied with the performance of your app, you can deploy it to production using MindMeld's cloud deployment offerings. Read more about the different available options in :doc:`deployment <../userguide/deployment>` section of the user guide.
