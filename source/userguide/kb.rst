Working with the Knowledge Base and Question Answerer
=====================================================

The knowledge base

  - is a comprehensive repository that contains the universe of helpful information needed to understand requests and answer questions
  - reflects, in the way we structure its content, our understanding of domain- and application-specific concepts and relationships
  - enables us to optimize natural language processing and question answering to achieve human-like accuracy

The question answerer

 - answers and validates questions, disambiguates entities, and suggests alternatives
 - is designed to work with custom knowledge bases derived from large content catalogs, for production-grade applications

Not every MindMeld application needs these two components. An app always has either both or neither one.

.. note::

    This is an in-depth tutorial to work through from start to finish. Before you begin, read the :ref:`Step-by-Step Guide <quickstart>`, paying special attention to Steps :doc:`5 <../quickstart/05_create_the_knowledge_base>` and :doc:`9 <../quickstart/09_optimize_question_answering_performance>`.

.. warning::

   The QuestionAnswerer requires Elasticsearch. Make sure that Elasticsearch is running in a separate shell before invoking the QuestionAnswerer.

Decide Whether your App Needs a Knowledge Base
----------------------------------------------

The first question to answer before you start building your application is: *Do we need a knowledge base or not?*

For applications with large- or medium-vocabulary content, having a knowledge base is critical and can be the differentiating factor for the best conversational applications. For example, video discovery applications typically need to find movie or TV shows from a catalog containing hundred of thousands or even millions of records. The knowledge base provides information about important entities to fulfill users' requests.

For applications with small-vocabulary content, it is not necessary to have a knowledge base, especially when ranking and disambiguating entities is not required. One example is a home assistant for smart home automation.

In its most basic form, a knowledge base is just a repository of objects of specified types. For example, a knowledge base that contains a large entertainment content catalog could include ``movie`` and ``tv_show`` among the types of its objects. A knowledge base that contains a directory of local businesses could have object types like ``restaurant``, ``hardware_store``, and so on. Objects typically have attributes which capture salient aspects of the concepts they describe. For example, a ``restaurant`` object might have attributes for ``address`` and ``phone_number``; a ``movie`` object might have attributes like ``cast_list``, ``runtime``, and ``release_date``.

Decide How to Use Question Answering in your Application
--------------------------------------------------------

The purpose of the MindMeld question answerer is to map natural language queries to a query which is then run over a structured database. (This is called *knowledge-based question answering*.) The question answerer is designed to work with custom knowledge bases derived from large content catalogs, for production-grade applications.

Catalogs containing even up to hundreds of millions of unique, domain-specific entity objects can easily be handled by MindMeld in production. The number of unique entities typical scales up to tens of thousands for restaurant menus processed by food delivery services, millions for product retail catalogs, and tens or hundreds of millions for media entertainment libraries. The MindMeld knowledge base and question answering systems have been successfully applied to all of the above use cases.

The question answerer can be used in a variety of ways, but in practice, conversational applications rely on this module and its underlying knowledge base for the four primary purposes listed below. Decide which purposes are most important to your use case and incorporate those priorities into the way you code your dialogue state handlers.

1. Answer Questions
^^^^^^^^^^^^^^^^^^^

Providing relevant results as answers to user requests is a basic use case for the question answerer. Knowledge base searches can be constructed using the entities extracted from NLP pipeline. The form of the answer depends on the nature of the application. For instance, in a food ordering app the answer could be a dish with attributes matching the criteria provided by users, while in a video discovery app the answer could be a list of movies or TV shows that best match the user request.

2. Validate Questions
^^^^^^^^^^^^^^^^^^^^^

The knowledge base can contain information to help inform users when requests are out of scope. For example, if a user orders pizza from a coffee shop in a food ordering app, the question answerer would be unable to find an entity matching the user request, but could provide information to help steer the conversation in the right direction. Other examples of this kind include the user asking for a customization option that is not available for the desired dish, or asking for campsite information from a travel app that only offers hotels and motels.

3. Disambiguate Entities
^^^^^^^^^^^^^^^^^^^^^^^^

We often need to disambiguate entities based on application requirements and context. For example, a food ordering app could offer dozens of ``Pad Thai`` dishes from a number of restaurants. In a travel booking app it is impossible to book the right flight without knowing which of multiple cities named ``Springfield`` the user is asking for. In a music discovery app a request for ``thriller`` could refer to a song or an album. In such cases we are unable to retrieve the exact entities that the users mean without taking into account contextual information like entity relationships and hierarchy, user preferences, or application business logic.

This disambiguation task can be formulated as a knowledge base search with constraints derived from contextual information. For the food ordering example, the selected restaurant can be added as a filter to the knowledge base search to find the best-matching dishes from that restaurant.

4. Suggest Alternatives
^^^^^^^^^^^^^^^^^^^^^^^

The question answerer can suggest the closest matches when exactly correct matches cannot be found. For example, if a user requests 'Star Wars Rogue One' and that movie is not available, the knowledge base could suggest other available Star Wars titles. The question answerer uses scoring factors including text relevance and location proximity to retrieve and suggest the most relevant information from the knowledge base.

The question answerer can also suggest alternatives based on custom logic in the application. Taking food ordering application as an example, we may want to:

	* suggest dishes in nearby restaurants when they cannot be found in the desired restaurant
	* suggest alternative dishes in the desired restaurant when the specified dishes cannot be found

In such cases, you would use the question answerer to formulate knowledge base searches with modified constraints to find and suggest the best matches.

Prepare your Data for the Knowledge Base
----------------------------------------

Building a custom knowledge base using application content data is straightforward in MindMeld. The content data can be restaurant menus, retailing product catalogs, or any custom data that users would like to interact with through conversational interfaces. This data is often stored in large-scale databases with application-specific data models. The question answerer can build a knowledge base using either (1) data dumps from a database, or (2) the output of a data pipeline which handles more complex data transformations.

The question answerer takes in data files that contain knowledge base objects, which are the basic unit of information in knowledge base indexes. Each data file contains objects of a specified type.

Each object has:

  - an ``id`` field as the unique identifier
  - an optional ``location`` field for location information if available, and
  - a list of arbitrary data fields of different types that contain information about the object, or about relationships with other object types.

To efficiently and accurately retrieve the most relevant information, the question answerer creates optimized indexes for objects. The question answerer processes all fields in the data, determining the data field types and indexing them accordingly.

The following data types are supported:

+----------------+--------------------------+-------------------------------------------------------------+
| Field Type     | Data Format              | Description or Examples                                     |
+================+==========================+=============================================================+
|   **id**       | string                   | unique identifier                                           |
+----------------+--------------------------+-------------------------------------------------------------+
|   **text**     | string                   | full-text values, for example dish names like ``Pad Thai``  |
|                |                          | or movie names like ``Star Wars``                           |
+----------------+--------------------------+-------------------------------------------------------------+
|   **number**   | ``long``, ``integer``,   | numeric values                                              |
|                | ``short``, ``byte``,     |                                                             |
|                | ``double``, or ``float`` |                                                             |
+----------------+--------------------------+-------------------------------------------------------------+
|   **date**     | string                   | formatted dates like "2017-07-31" or "2017/07/31 12:10:30"  |
+----------------+--------------------------+-------------------------------------------------------------+
|                | long number              | milliseconds-since-the-epoch                                |
+----------------+--------------------------+-------------------------------------------------------------+
|                | integer                  | seconds-since-the-epoch                                     |
+----------------+--------------------------+-------------------------------------------------------------+
|   **location** | string                   | latitude/longitude pair like "37.77,122.41"                 |
+----------------+--------------------------+-------------------------------------------------------------+
|                | array                    | latitude/longitude pair like [37.77, 122.41]                |
+----------------+--------------------------+-------------------------------------------------------------+
|                | object                   | latitude/longitude pair like {"lat": 37.77, "lon": 122.41}  |
+----------------+--------------------------+-------------------------------------------------------------+


.. note:: Note that the location information of a knowledge base object is specified in the ``location`` field. Specifying additional location fields or using custom field names for location are currently not supported.

The question answerer supports whatever data model the application uses. Developers must decide what data model suits their apps best. For example, a food ordering app could model ``cuisine`` types either as information-rich objects with a list of attributes, or simply as a string (the cuisine type name) which is an attribute of dish objects.

Mapping between NLP entity types and knowledge base objects is often application-specific, and knowledge base searches need to be formulated accordingly. For example, a food ordering application may have ``restaurant`` and ``menu_item`` objects, while a video discovery application has ``cast`` and ``title`` objects. The ``menu_item`` object in the food ordering app could have the following fields:

+---------------+---------------+------------------------------------+
| Field         | Type          | Description                        |
+===============+===============+====================================+
| id            | id            | unique identifier string           |
+---------------+---------------+------------------------------------+
| name          | text          | name of the dish                   |
+---------------+---------------+------------------------------------+
| description   | text          | description of the dish            |
+---------------+---------------+------------------------------------+
| category      | text          | dish category                      |
+---------------+---------------+------------------------------------+
| price         | number        | dish price                         |
+---------------+---------------+------------------------------------+
| img_url       | text          | dish image URL                     |
+---------------+---------------+------------------------------------+
| restaurant_id | text          | ID of the restaurant               |
+---------------+---------------+------------------------------------+

The JSON data file for a ``menu_item`` object of this kind would look like the following:

.. code-block:: javascript

  {
    "category": "Makimono-Sushi Rolls (6 Pcs)",
    "description": "Makimono-Sushi Rolls (6 Pcs)\nDeep-fried shrimp, avocado, cucumber",
    "price": 6.5,
    "restaurant_id": "B01N97KQNJ",
    "img_url": null,
    "id": "B01N0KXELH",
    "name": "Shrimp Tempura Roll"
  },
  {
    "category": "Special Rolls",
    "description": "California roll topped w/ cooked salmon, mayo and masago",
    "price": 9.95,
    "restaurant_id": "B01N97KQNJ",
    "img_url": null,
    "id": "B01MYTS7W4",
    "name": "Pink Salmon Roll"
  }
  ...

Verify that the Data is Clean
-----------------------------

For the question answerer to achieve the best possible performance, it's critical to have clean data in the knowledge base. You should not assume that the generic text processing and normalization that the MindMeld knowledge base performs is sufficient. Domain- or application-specific normalizations are often necessary. For example, in a food ordering app, menus from different restaurants could differ in format and the conventions they use.

Good practice dictates that you inspect the data to identify noise and inconsistency in the dataset, then clean up and normalize the data as needed. In order for your app to achieve high accuracy, it's important to do this as a pre-processing task.

Import the Data into the Knowledge Base
---------------------------------------

.. note::

   For this tutorial, we will be using the ``food_ordering`` blueprint application. You can download this application by running these commands:

   .. code:: python

      import mindmeld as mm
      mm.configure_logs()
      mm.blueprint('food_ordering')

The :meth:`load_kb()` API loads data into the knowledge base from JSON-formatted data files. If the index of specified objects is already present in the knowledge base, the new objects are imported into the existing index. If no index for the specified objects exists, the question answerer creates one.

Use :meth:`load_kb()` to load a data file from a path and create an index for the objects in the data file. Substitute a data file, index name, and app name for those used in the example.

.. code:: python

	from mindmeld.components import QuestionAnswerer
	qa = QuestionAnswerer(app_path='food_ordering')
	qa.load_kb(app_namespace='food_ordering', index_name='restaurants', data_file='food_ordering/data/restaurants.json')

Alternatively, use the MindMeld command line tool to perform the same operation.

.. code-block:: console

	python -m food_ordering load-kb my_app restaurants food_ordering/data/restaurants.json

Verify that the index was created successfully using the :meth:`get()` method of the question answerer:

.. code:: python

   restaurants = qa.get(index='restaurants')
   restaurants[0]

.. code-block:: console

  [
    {
      'categories': ['Beverages', 'Pizzas', 'Sides', 'Popular Dishes'],
      'cuisine_types': ['Pizza'],
      'id': 'B01CT54GYE',
      'image_url': 'https://images-na.ssl-images-amazon.com/images/G/01/ember/restaurants/SanFrancisco/FiretrailPizza/logo_232x174._CB295435423_SX600_QL70_.png',
      'menus': [
        {
          'id': '127c097e-2d9d-4880-99ac-f1688909af07',
          'option_groups': [
            {
              'id': 'ToppingsGF',
              'max_selected': 9,
              'min_selected': 0,
              'name': 'Add Some Extra Toppings',
              'options': [
                {
                  'description': None,
                  'id': 'B01D8TDFV0',
                  'name': 'Goat Cheese',
                  'price': 2.0
                },
                {
                  'description': None,
                  'id': 'B01D8TCH3M',
                  'name': 'Olives',
                  'price': 1.0
                },
                ...
              ]
            }
            ...
          ]
        }
      ],
      'name': 'Firetrail Pizza',
      'num_reviews': 13,
      'price_range': 2.0,
      'rating': 4.1
    },
    ...
  ]

Perform Simple Searches with the ``get()`` API
----------------------------------------------

The :meth:`get()` method is the basic API for searching the knowledge base, with a simple and intuitive interface similar to those of common web search interfaces.

As arguments, :meth:`get()` takes in one or more *knowledge base field/text query* pairs. The knowledge base fields to be used depend on the mapping between NLP entity types and knowledge base objects, which is often application-specific since it depends, in turn, on the data model of the application. For example, in a food ordering app, a ``cuisine`` entity type could be mapped to a knowledge base object or to an attribute of a knowledge base object.

Use :meth:`get()` to retrieve a knowledge base object whose ID is already known:

.. code:: python

   from mindmeld.components import QuestionAnswerer
   qa = QuestionAnswerer(app_path='food_ordering')
   qa.get(index='menu_items', id='B01N97KQNJ')

.. code-block:: console

		[{'category': 'Hawaiian Style Poke (HP)',
		  'description': None,
		  'id': 'B01CGKGQ40',
		  'img_url': None,
		  'menu_id': '78eb0100-029d-4efc-8b8c-77f97dc875b5',
		  'name': 'Spicy Creamy Salmon Poke',
		  'option_groups': [],
		  'popular': False,
		  'price': 6.5,
		  'restaurant_id': 'B01N97KQNJ',
		  'size_group': None,
		  'size_prices': [],
		  'syn_whitelist': [{'name': 'special fish'}]}]

Use :meth:`get()` to search the knowledge base for objects that best match all of several *knowledge base field/text query* pairs.  In the following example we try to find dishes (that is, items in the ``menu_items`` index) whose name matches ``fish and chips`` from a restaurant whose ID matches ``B01DEEGQBK``:

.. code:: python

	from mindmeld.components import QuestionAnswerer
	qa = QuestionAnswerer(app_path='food_ordering')
	results = qa.get(index='menu_items', name='fish and chips', restaurant_id='B01N97KQNJ')

.. code-block:: console

    [{'size_group': None,
      'img_url': None,
      'price': 6.5,
      'restaurant_id': 'B01N97KQNJ',
      'name': 'Soyu Salmon Poke',
      'description': None,
      'id': 'B01N9BO8RC',
      'category': 'Hawaiian Style Poke (HP)',
      'popular': False,
      'menu_id': '78eb0100-029d-4efc-8b8c-77f97dc875b5'},
     {'size_group': None,
      'img_url': None,
      'price': 3.95,
      'restaurant_id': 'B01N97KQNJ',
      'name': 'Maguro (Tuna)',
      'description': 'Nigiri Sushi',
      'id': 'B01MZZCKDX',
      'category': 'Nigiri Sushi (2 Pcs)',
      'popular': False,
      'menu_id': '78eb0100-029d-4efc-8b8c-77f97dc875b5'},
     {'size_group': None,
      'img_url': None,
      'price': 4.95,
      'restaurant_id': 'B01N97KQNJ',
      'name': 'Unagi (Sea Eel)',
      'description': 'Nigiri Sushi',
      'id': 'B01MYTS99Z',
      'category': 'Nigiri Sushi (2 Pcs)',
      'popular': False,
      'menu_id': '78eb0100-029d-4efc-8b8c-77f97dc875b5'},
	  ...

The :meth:`get()` method also supports custom sort criteria, limited to the following:

==================== ===
**_sort**            the knowledge base field on which to sort
**_sort_type**       type of sort to perform. valid values are ``asc``, ``desc`` and ``distance``. ``asc`` and ``desc`` specifies the sort order for sorting on number or date fields, while ``distance`` indicates sorting by distance based on ``location`` field.
**_sort_location**   origin location for sorting by distance
==================== ===

When you use custom sort criteria, ranking is a optimized blend of sort score and text relevance scores.

Use :meth:`get()` to search the knowledge base for objects that best match all of several *knowledge base field/text query* pairs, including custom sort criteria. The example below shows a search for ``menu_items`` objects that best match ``fish and chips`` on ``name``, ``B01CGKGQ40`` on ``restaurant_id``, ordered from lowest to highest price.

.. code:: python

	from mindmeld.components import QuestionAnswerer
	qa = QuestionAnswerer(app_path='food_ordering')
	qa.get(index='menu_items', name='fish and chips', restaurant_id='B01CGKGQ40', _sort='price', _sort_type='asc')

.. code-block:: console

	[{'category': 'Appetizers and Side Orders',
	  'description': None,
	  'id': 'B01N3BB0PK',
	  'img_url': None,
	  'menu_id': '57572a43-f9fc-4a1c-96fe-788d544b1f2d',
	  'name': 'Fish and Chips',
	  'option_groups': [],
	  'popular': False,
	  'price': 9.99,
	  'restaurant_id': 'B01DEEGQBK',
	  'size_group': None,
	  'size_prices': []},
	 {'category': 'Appetizers and Side Orders',
	  'description': None,
	  'id': 'B01N9Z38XT',
	  'img_url': None,
	  'menu_id': '57572a43-f9fc-4a1c-96fe-788d544b1f2d',
	  'name': 'Chicken Tenders and Chips',
	  'option_groups': [],
	  'popular': False,
	  'price': 9.99,
	  'restaurant_id': 'B01DEEGQBK',
	  'size_group': None,
	  'size_prices': []}]
	  ...

Use :meth:`get()` to search the knowledge base for objects that best match all of several *knowledge base field/text query* pairs, including a custom sort criterion of distance from a specified location. The example below shows a search for the restaurant closest to the center of San Francisco:

.. code:: python

	from mindmeld.components import QuestionAnswerer
	qa = QuestionAnswerer(app_path='food_ordering')
	qa.get(index='restaurants', _sort='location', _sort_type='distance', _sort_location='37.77,122.41')

.. code-block:: console

	  [
	    {
	      'categories': ['Beverages', 'Pizzas', 'Sides', 'Popular Dishes'],
	      'cuisine_types': ['Pizza'],
	      'id': 'B01CT54GYE',
	      'image_url': 'https://images-na.ssl-images-amazon.com/images/G/01/ember/restaurants/SanFrancisco/FiretrailPizza/logo_232x174._CB295435423_SX600_QL70_.png',
	      'menus': [
	        {
	          'id': '127c097e-2d9d-4880-99ac-f1688909af07',
	          'option_groups': [
	            {
	              'id': 'ToppingsGF',
	              'max_selected': 9,
	              'min_selected': 0,
	              'name': 'Add Some Extra Toppings',
	              'options': [
	                {
	                  'description': None,
	                  'id': 'B01D8TDFV0',
	                  'name': 'Goat Cheese',
	                  'price': 2.0
	                },
	                {
	                  'description': None,
	                  'id': 'B01D8TCH3M',
	                  'name': 'Olives',
	                  'price': 1.0
	                },
	                ...
	              ]
	            }
	            ...
	          ]
	        }
	      ],
	      'name': 'Firetrail Pizza',
	      'num_reviews': 13,
	      'price_range': 2.0,
	      'rating': 4.1,
	      'location': [37.77, 122.39]
	    },
	    ...
	  ]

Perform Advanced Searches with the ``build_search()`` API
---------------------------------------------------------

While the basic search API described above covers the most common use cases in conversational applications, it can't help in scenarios like the following:

  * sorting on more than one custom criterion
  * filtering on number or date ranges
  * controlling search behavior in a fine-grained manner

To support these more complex knowledge base searches, the question answerer provides advanced search APIs which allow you to create a Search object, which is an abstraction of a knowledge base search. You then apply text query, text and range filters, and custom sort criteria using the Search object's own APIs, which are chainable. This approach offers powerful and precise search in a compact and readable syntax.

Use the :meth:`build_search()` API to create a Search object.

.. code:: python

	from mindmeld.components import QuestionAnswerer
	qa = QuestionAnswerer(app_path='food_ordering')
	s = qa.build_search(index='menu_items')

Query
^^^^^

Use the :meth:`query()` API to run an advanced text query search. For each query, specify a text field in the knowledge base, and a query string for the text relevance match. The question answerer ranks results using several factors including exact matches, phrase matches, and partial matches on the text.

In the following example, the question answerer returns the dishes that best match the name ``fish and chips``. We specify the query string ``fish and chips`` on the knowledge base field ``name`` in the ``menu_items`` index (which contains all available dishes). The top two results are from two different restaurants, and both match the name ``fish and chips`` exactly:

.. code:: python

	from mindmeld.components import QuestionAnswerer
	qa = QuestionAnswerer(app_path='food_ordering')
	s = qa.build_search(index='menu_items')
	s.query(name='fish and chips').execute()

.. code-block:: console

	[{'category': 'Appetizers and Side Orders',
	  'description': None,
	  'id': 'B01N3BB0PK',
	  'img_url': None,
	  'menu_id': '57572a43-f9fc-4a1c-96fe-788d544b1f2d',
	  'name': 'Fish and Chips',
	  'option_groups': [],
	  'popular': False,
	  'price': 9.99,
	  'restaurant_id': 'B01DEEGQBK',
	  'size_group': None,
	  'size_prices': []},
	 {'category': 'Entrees',
	  'description': None,
	  'id': 'B01CH0SUMA',
	  'img_url': 'http://g-ec2.images-amazon.com/images/G/01/ember/restaurants/SanFrancisco/V_Cafe/VCafe_FishandChips_640x480._V286448998_.jpg',
	  'menu_id': '17612bcf-307a-4098-828e-329dd0962182',
	  'name': 'Fish and Chips',
	  'option_groups': ['dressing'],
	  'popular': True,
	  'price': 13.0,
	  'restaurant_id': 'B01CH0RZOE',
	  'size_group': None,
	  'size_prices': []},
	  ...


Filter
^^^^^^

Use the :meth:`filter()` API to add filters to an advanced text query search.

Two types of filters are supported: **text filter** and **range filter**. For text filter, specify a knowledge base text field name and the filtering text string. The text string is normalized and the entire text string is used to filter the results, like SQL predicates in relational databases. For example, in food ordering apps, users often request dishes of a particular cuisine type or from a specific restaurant.

In the example below we try to find dishes (that is, items in the ``menu_items`` index) whose names best match ``fish and chips``, from a restaurant whose ID matches ``B01DEEGQBK``:

.. code:: python

	from mindmeld.components import QuestionAnswerer
	qa = QuestionAnswerer(app_path='food_ordering')
	s = qa.build_search(index='menu_items')
	s.query(name='fish and chips').filter(restaurant_id='B01DEEGQBK').execute()

.. code-block:: console

	[{'category': 'Appetizers and Side Orders',
	  'description': None,
	  'id': 'B01N3BB0PK',
	  'img_url': None,
	  'menu_id': '57572a43-f9fc-4a1c-96fe-788d544b1f2d',
	  'name': 'Fish and Chips',
	  'option_groups': [],
	  'popular': False,
	  'price': 9.99,
	  'restaurant_id': 'B01DEEGQBK',
	  'size_group': None,
	  'size_prices': []},
	  ...

Use the :meth:`filter()` API to apply filters on number or date ranges in an advanced search.

To define a filter on ranges, specify a knowledge base field and one or more of the following range operators:

======== ===
**gt**   greater than
**gte**  greater than or equal to
**lt**   less than
**lte**  less than or equal to
======== ===

Use cases for this kind of filtering include finding products within certain price ranges in a retailing app, and finding movies released in the past five years in a video discovery app.

In the example below we filter on price range to find dishes priced below five dollars:

.. code:: python

	from mindmeld.components import QuestionAnswerer
	qa = QuestionAnswerer(app_path='food_ordering')
	s = qa.build_search(index='menu_items')
	s.filter(field='price', lte=5).execute()

.. code-block:: console

	[{'category': 'Makimono-Sushi Rolls (6 Pcs)',
	  'description': 'Makimono-Sushi Rolls (6 Pcs)',
	  'id': 'B01MXSBGG0',
	  'img_url': None,
	  'menu_id': '78eb0100-029d-4efc-8b8c-77f97dc875b5',
	  'name': 'Sake Maki-Salmon',
	  'option_groups': [],
	  'popular': False,
	  'price': 3.95,
	  'restaurant_id': 'B01N97KQNJ',
	  'size_group': None,
	  'size_prices': []},
	 {'category': 'Popular Dishes',
	  'description': None,
	  'id': 'B01CUUCX7K',
	  'img_url': 'http://g-ec2.images-amazon.com/images/G/01/ember/restaurants/SanFrancisco/TheSaladPlace/TheSaladPlace_Potatosalad_640x480._V295354393_.jpg',
	  'menu_id': '1e6f9732-4d87-4e08-ac8c-c6198b2645cc',
	  'name': 'Potato',
	  'option_groups': [],
	  'popular': True,
	  'price': 3.95,
	  'restaurant_id': 'B01CUUBQC8',
	  'size_group': 'SaladSize',
	  'size_prices': [{'id': 'B01CUUC10O', 'name': 'Small', 'price': 3.95},
	   {'id': 'B01CUUBPYM', 'name': 'Medium', 'price': 4.95},
	   {'id': 'B01CUUD9FA', 'name': 'Large', 'price': 5.95}]},
	   ...

.. note::

   Range filters are only valid for number and date knowledge base fields.

Sort
^^^^

Use the :meth:`sort()` API to add one or more custom sort criteria to an advanced search.

Custom sort

 - can be used with number, date or location knowledge base fields
 - takes in three parameters: ``field``, ``sort_type``, and ``location``

The ``field`` parameter specifies the knowledge base field for sort, the ``sort_type`` parameter can be either ``asc`` or ``desc`` to indicate sort order for number or date fields and ``distance`` to indicate sorting by distance using location field, and the ``location`` field parameter specifies the origin location when sorting by distance.

The custom sort can be applied to any number or date fields desirable and the score for ranking will be a optimized blend of sort score with other scoring factors including text relevance scores when available.

In the example below, we search for ``menu_item`` objects that best match the text query ``fish and chips``, priced from lowest to highest. TO do this, we combining the text relevance and sort scores on the ``price`` field:

.. code:: python

	from mindmeld.components import QuestionAnswerer
	qa = QuestionAnswerer(app_path='food_ordering')
	s = qa.build_search(index='menu_items')
	s.query(name='fish and chips').sort(field='price', sort_type='asc').execute()

.. code-block:: console

	[{'category': 'Appetizers and Side Orders',
	  'description': None,
	  'id': 'B01N3BB0PK',
	  'img_url': None,
	  'menu_id': '57572a43-f9fc-4a1c-96fe-788d544b1f2d',
	  'name': 'Fish and Chips',
	  'option_groups': [],
	  'popular': False,
	  'price': 9.99,
	  'restaurant_id': 'B01DEEGQBK',
	  'size_group': None,
	  'size_prices': []},
	 {'category': 'Entrees',
	  'description': None,
	  'id': 'B01CH0SUMA',
	  'img_url': 'http://g-ec2.images-amazon.com/images/G/01/ember/restaurants/SanFrancisco/V_Cafe/VCafe_FishandChips_640x480._V286448998_.jpg',
	  'menu_id': '17612bcf-307a-4098-828e-329dd0962182',
	  'name': 'Fish and Chips',
	  'option_groups': ['dressing'],
	  'popular': True,
	  'price': 13.0,
	  'restaurant_id': 'B01CH0RZOE',
	  'size_group': None,
	  'size_prices': []},
	  ...

Use the :meth:`sort()` API to sort by distance in an advanced text search.

To define sorting by distance, specify ``location`` as a sort field, with ``distance`` as the ``sort_type`` parameter, and the origin location latitude and longitude in the object passed to the ``location`` parameter. Proximity is an important sorting factor for conversational applications designed for use on the go.

In the example below, we search for restaurants whose names best match ``firetrail``, in order of nearest to to furthest from the center of San Francisco:

.. code:: python

	from mindmeld.components import QuestionAnswerer
	qa = QuestionAnswerer(app_path='food_ordering')
	s = qa.build_search(index='restaurants')
	s.query(name='firetrail').sort(field='location', sort_type='distance', location='37.77,122.41').execute()


.. code-block:: console

	[
	    {
		  'categories': ['Beverages', 'Pizzas', 'Sides', 'Popular Dishes'],
		  'cuisine_types': ['Pizza'],
		  'id': 'B01CT54GYE',
		  'image_url': 'https://images-na.ssl-images-amazon.com/images/G/01/ember/restaurants/SanFrancisco/FiretrailPizza/logo_232x174._CB295435423_SX600_QL70_.png',
		  'menus': [{'id': '127c097e-2d9d-4880-99ac-f1688909af07',
		    'option_groups': [{'id': 'ToppingsGF',
			  'max_selected': 9,
			  'min_selected': 0,
			  'name': 'Add Some Extra Toppings',
			  'options': [{'description': None,
			    'id': 'B01D8TDFV0',
			    'name': 'Goat Cheese',
			    'price': 2.0},
			   {'description': None,
			    'id': 'B01D8TCH3M',
			    'name': 'Olives',
			    'price': 1.0},
			   ...
		  'name': 'Firetrail Pizza',
		  'num_reviews': 13,
		  'price_range': 2.0,
		  'rating': 4.1,
		  'location': [37.77, 122.39]
		},
	  	...
	  ]
