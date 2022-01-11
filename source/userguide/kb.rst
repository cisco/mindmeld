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

   The QuestionAnswerer by default uses `Elasticsearch <https://www.elastic.co/products/elasticsearch>`_ full-text search and analytics engine for information retrieval, unless explicitly specified in configs to not to use it. In which case, make sure that Elasticsearch is running in a separate shell before invoking the QuestionAnswerer. If you do not wish to use Elasticsearch, you can disable it following the :ref:`QuestionAnswerer without Elasticsearch backend <non_elasticsearch_question_answerer>` section below.

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

The :meth:`load_kb()` API loads data into the knowledge base from JSON-formatted data files. If the index of specified objects is already present in the knowledge base, the new objects are imported into the existing index. If no index for the specified objects exists, the question answerer creates one. Remember that in absence of any custom config, an Elasticsearch-backed QuestionAnswerer is loaded by default. If you do not wish to use Elasticsearch, you can use an alternate choice following the :ref:`QuestionAnswerer without Elasticsearch backend <non_elasticsearch_question_answerer>` section below.

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

.. _import_kb:

To fully recreate an existing index from scratch, you can run a clean load as follows.

.. code:: python

  from mindmeld.components import QuestionAnswerer
  qa = QuestionAnswerer(app_path='food_ordering')
  qa.load_kb(app_namespace='food_ordering', index_name='restaurants', data_file='food_ordering/data/restaurants.json', clean=True)

This will internally delete the existing index, create a new index and load the specified objects.

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

By default, the :meth:`get()` method will return a maximum list of 10 records per search. We can change the number of records per search by setting the ``size`` parameter.

.. code:: python

	from mindmeld.components import QuestionAnswerer
	qa = QuestionAnswerer(app_path='food_ordering')
	results = qa.get(index='restaurants', size=20, _sort='location', _sort_type='distance', _sort_location='37.77,122.41')
	len(results)

.. code-block:: console

	20

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

Similarly to the :meth:`get()` method, the :meth:`query()` method by default will return a list of up to 10 records. We can set the ``size`` parameter of the :meth:`execute()` method to specify a different maximum number of records.

.. code:: python

	from mindmeld.components import QuestionAnswerer
	qa = QuestionAnswerer(app_path='food_ordering')
	s = qa.build_search(index='menu_items')
	results = s.query(name='fish and chips').execute(size=20)
	len(results)


.. code-block:: console

	20

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

   We can set the ``size`` parameter of the :meth:`execute()` method to specify the maximum number of records.

Sort
^^^^

Use the :meth:`sort()` API to add one or more custom sort criteria to an advanced search.

Custom sort

 - can be used with number, date or location knowledge base fields
 - takes in three parameters: ``field``, ``sort_type``, and ``location``

The ``field`` parameter specifies the knowledge base field for sort, the ``sort_type`` parameter can be either ``asc`` or ``desc`` to indicate sort order for number or date fields and ``distance`` to indicate sorting by distance using location field, and the ``location`` field parameter specifies the origin location when sorting by distance.

The custom sort can be applied to any number or date fields desirable and the score for ranking will be a optimized blend of sort score with other scoring factors including text relevance scores when available.

In the example below, we search for ``menu_item`` objects that best match the text query ``fish and chips``, priced from lowest to highest. To do this, we combine the text relevance and sort scores on the ``price`` field:

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

.. note::

   We can set the ``size`` parameter of the :meth:`execute()` method to specify the maximum number of records.


.. _question_answerer_config:

QuestionAnswerer Configurations
-------------------------------

To override MindMeld's default QuestionAnswerer configuration with custom settings, you can edit the app configuration file by adding a dictionary of custom setting named :data:`QUESTION_ANSWERER_CONFIG`. If no such dictionary is present in the ``config.py`` file, MindMeld loads a QuestionAnswerer with default settings. The following are the default settings:

.. code-block:: python

  QUESTION_ANSWERER_CONFIG = {
      'model_type': 'elasticsearch',
      'model_settings': {
          'query_type': 'keyword'
      }
  }

The ``query_type`` can be one of ``keyword`` or ``text``. While specifying the former optimizes your QuestionAnswerer search for keywords or short spans of text, the latter optimizes for searching on larger paragraphs or passages of unstructured text. In order to leverage embeddings-based semantic matching along with surface-level text features matching, you can specify one of the three embedder query types- ``embedder``, ``embedder_text``, or ``embedder_keyword``. For using embeddings of your choice, you can specify embedder configurations in the :data:`QUESTION_ANSWERER_CONFIG` within the key 'model_settings'. For full details about using embedders in QuestionAnswerer, check the :ref:`Leveraging semantic embeddings <semantic_embeddings>` section below. Note that the specified ``query_type`` is also the default ``query_type`` for all QuestionAnswerer calls in your application, but you can always pass in a different query type to your ``qa.get(query_type=...)`` command as desired.

.. _non_elasticsearch_question_answerer:

QuestionAnswerer without Elasticsearch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Search operations of QuestionAnswerer are by default backed by Elasticsearch. While it is recommended to use the Elasticsearch based QuestionAnswerer, it might not always be feasible to use it due to various constraints. In such cases, one can easily disable Elasticsearch and perform the search natively by setting `"model_type": "native"` in the ``QUESTION_ANSWERER_CONFIG`` as follows.

.. code-block:: python

  QUESTION_ANSWERER_CONFIG = {
      'model_type': 'native',
      'model_settings': {
          'query_type': 'keyword'
      }
  }

You can set this key irrespective of leveraging embedders or not. If unspecified, this key will be set to ``elasticsearch``. It is noteworthy that the search results obtained with and without Elasticsearch might have minor differences between them.

The native QuestionAnswerer, similar to the Elasticsearch backed QuestionAnswerer, is capable of doing both surface-level text features matching as well as semantic matching based on embeddings. Upon loading a Knowledge Base, the native QuestionAnswerer stores the KB data at the location '/Users/<username>/.cache/mindmeld/.generated/question_answerer/<indexname>.pkl'. If using an embedder, the embeddings of the required data fields will also be saved under the same '.generated' folder. Note that the matching scores of the Elasticsearch backed QuestionAnswerer are larger in magnitude whereas the scores from the native backend are generally fractions less than 1.


.. _unstructured_data:

Dealing with unstructured text
------------------------------
The knowledge bases described in the previous sections are comprised of clearly defined data types that are easily searchable. This kind of data is known as `structured data`.

In this section we describe how to use the question answerer on knowledge bases with long, free-form text or `unstructured data`. For example, a company's Human Resources (HR) team could have documents about general policies of the company. We can use this data to develop an HR assistant that automatically answers any policy-related questions.

To use the question answerer on unstructured text, we would follow the same steps mentioned before to prepare, index and load the knowledge base.

Here is an example of what the knowledge base can look like for the HR assistant use case. It consists of frequently asked question and answer pairs.

.. code:: python

  import mindmeld as mm
  from mm.components import QuestionAnswerer

  # Download hr assistant blueprint
  mm.configure_logs()
  bp_name = 'hr_assistant'
  mm.blueprint(bp_name)

  # Query KB
  qa = QuestionAnswerer(app_path='hr_assistant')
  qa.load_kb(app_namespace='hr_assistant', index_name='faq_data', data_file='hr_assistant/data/hr_faq_data.json')
  qa.get(index='faq_data')

.. code-block:: console

	[{
	    'question': 'What if I did not receive a form W2?',
	    'answer': 'W2s are mailed to home addresses of employees as of a date in mid-January each year. If you did not receive a form W2, you may access it online (beginning with 2008 W2s), similar to employee Pay Statements. Search for and click on the W-2 task. Enter the year for the W-2 you are wanting to locate and the last 4 digits of your Social Security Number. You then will have access to the W-2 in a format that can be used for filing paper versions of federal and state tax returns.',
	    'id': 'hrfaq6'
	 },
	 {
	    'question': 'Are employers expected to hire the less qualified over the more qualified to meet affirmative action goals?',
	    'answer': 'Employers are not expected to establish any hiring practices that conflict with the principles of sound personnel management. No one should be hired unless there is a basis for believing the individual is the best-qualified candidate. In fact, affirmative action calls for the hiring of qualified people.',
	    'id': 'hrfaq12'
	 },
	 ...
	]

The difference between a structured knowledge base and an unstructured one is in how MindMeld handles the search query. Internally, while the ranking algorithm remains the same for both cases, the features extracted for ranking are different and are optimized to handle long text passages rather than keyword phrases.

To search the knowledge base for the answer to a policy question, we will use the :meth:`get()` API as before with one small modification. We specify that the query is against unstructured text by setting the ``query_type`` parameter to `text` (by default the ``query_type`` is set to `keyword`).

.. code:: python

	from mindmeld.components import QuestionAnswerer
	qa = QuestionAnswerer(app_path='hr_assistant')
	query = 'what is phase 2 of the review cycle'
	qa.get(index='faq_data', query_type='text', question=query, answer=query, size=1)

.. code-block:: console

	[{
	    'question': 'What is the performance cycle?',
	    'answer': 'The intent of the performance cycle is to identify the key parts of each employee’s job, identify what it looks like when that is done well (meets your expectations as a manager), and how both you as manager and your employee will know when that is achieved (measurements).Phase 1 - Planning:  Creating goals and expectations between the employee and manager for the current year. Phase 2 - Check-Ins:  Giving ongoing feedback throughout the year; identifying acomplishments, areas for improvement and adjusting the goals/expectations as necessary. Phase 3 - Review:  Reviewing the year at the end of the performance period.',
	    'id': 'hrfaq24'
	 }
	]

In the above example, we try to find the best answer for the given user query by matching against both the `question` and `answer` field of the knowledge base. Using the `answer` field enables the question answerer to find matches even when the query does not have matches against the `question` field.

We can perform the same search using the :meth:`query()` API as well.

.. code:: python

	from mindmeld.components import QuestionAnswerer
	qa = QuestionAnswerer(app_path='hr_assistant')
	s = qa.build_search(index='faq_data')

	query = 'when do i get my w2 form'
	s.query(query_type='text', question=query, answer=query).execute()

.. code-block:: console

	[{
	    'question': 'When can I expect my annual form W2?',
	    'answer': 'All employee W2s are mailed on or around January 31 for the prior calendar year.',
	    'id': 'hrfaq5'
	 },
	 {
	    'question': 'What if I did not receive a form W2?',
	    'answer': 'W2s are mailed to home addresses of employees as of a date in mid-January each year. If you did not receive a form W2, you may access it online (beginning with 2008 W2s), similar to employee Pay Statements. Search for and click on the W-2 task. Enter the year for the W-2 you are wanting to locate and the last 4 digits of your Social Security Number. You then will have access to the W-2 in a format that can be used for filing paper versions of federal and state tax returns.',
	    'id': 'hrfaq6'
	 }
	 ...
	]

.. note::

	For knowledge bases indexed prior to MindMeld 4.2, you will have to delete and reindex all the data to use this feature.

	To delete and reindex your data, follow the steps mentioned :ref:`here <import_kb>`.


.. _semantic_embeddings:

Leveraging semantic embeddings
------------------------------

.. note::

    If you are working with QuestionAnswerer's default Elasticsearch backend, in order to use the embedding features, you must be on ElasticSearch 7 or above. If you are upgrading to ES7 from an older version, we recommend you delete and recreate all of your indexes.

.. note::

    To use the BERT embedder, make sure you've installed the extra requirement with the command ``pip install mindmeld[bert]``.

The question answerer capabilities described so far rely purely on text-based retrieval. Deep learning based dense embeddings (character, word, or sentence) are in many cases better at capturing semantic information than traditional sparse vectors. Pretrained or fine-tuned embeddings can be used to find the best match in the knowledge base even if the search token wasn’t present in the uploaded data.

To leverage semantic embeddings in search, the first step is to generate the embeddings for your desired knowledge base fields. You can use one of the provided embedders or use your own. If your app mainly consists of standard English vocabulary, one of the provided embedders may work well enough, but if the text you are searching against has quite a bit of domain-specific vocabulary, you may benefit from training or fine tuning your own embedder on your data.

The settings for semantic embeddings are part of the ``QUESTION_ANSWERER_CONFIG`` in the app configuration file, ``config.py``. To use semantic embeddings, you need to specify a supported ``query_type``,  the ``model_settings``, and the fields you would like to generate embeddings for in ``embedding_fields``. The ``embedding_fields`` parameter takes a dictionary where the key is the name of your index, and the value is a list of field names or regexes to match against the field names for that index.

Using the HR Assistant blueprint as an example, here is what the question answerer config could look like.

.. code:: python

  QUESTION_ANSWERER_CONFIG = {
      "model_type": "elasticsearch",
      "model_settings": {
          "query_type": "embedder",
          "embedder_type": "bert",
          "embedding_fields": {
              "faq_data": ["question", "answer"]
          }
      }
  }

There are three available query types which leverage semantic embedders. The specified ``query_type`` is also the default ``query_type`` for all question answering calls for your application, but you can always pass in a different query type to your ``qa.get()`` command as desired.

  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | Query type            | Description                                                                                                |
  +=======================+============================================================================================================+
  | embedder              | Only leverage deep semantic embedders. This option allows for using deep semantic embedders like           |
  |                       | Sentence-BERT or GloVe for doing vector-based retrieval.                                                   |
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | embedder_keyword      | Leverage a combination of deep semantic embedders and text signals in a way that's optimized for search on |
  |                       | keywords or short spans of text. GloVe may be preferable for this use case.                                |
  +-----------------------+------------------------------------------------------------------------------------------------------------+
  | embedder_text         | Leverage a combination of deep semantic embedders and text signals in a way that's optimized for search on |
  |                       | larger paragraphs or passages of unstructured text. Sentence-BERT is preferable for this use case.         |
  +-----------------------+------------------------------------------------------------------------------------------------------------+


The two included embedder types are `Sentence-BERT <https://github.com/UKPLab/sentence-transformers>`_ and `GloVe <https://nlp.stanford.edu/projects/glove/>`_.

Sentence-BERT is a modification of the pretrained BERT network that uses siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity. To use this embedder type, specify ``bert`` as the ``embedder_type`` in the ``model_settings``.

Multiple models fine-tuned on BERT / RoBERTa / DistilBERT / ALBERT / XLNet are provided. You can view the full list `here <https://github.com/UKPLab/sentence-transformers#english-pre-trained-models>`_, and specify your choice of pre-trained model via the ``model_name`` parameter in the ``model_settings`` section of the config. The default is ``bert-base-nli-mean-tokens`` which is a BERT model trained on the `SNLI <https://nlp.stanford.edu/projects/snli/>`_ and `MultiNLI <https://cims.nyu.edu/~sbowman/multinli/>`_ datasets with mean-tokens pooling.

The provided GloVe embeddings are word vectors trained on the `Wikipedia <https://dumps.wikimedia.org/>`_ and `Gigaword 5 <https://catalog.ldc.upenn.edu/LDC2011T07>`_ datasets. To use this embedder type, specify ``glove`` as the ``embedder_type`` in the ``model_settings``.

The 50, 100, 200, and 300 dimension word embeddings are supported. The desired dimension can be specified via the ``token_embedding_dimension`` in the ``model_settings`` section of the config (defaults to 300). For fields with more than one token, word vector averaging is used.

Once your config file is specified, you can load the Knowledge Base and the embeddings will automatically be generated as specified in the config and stored in your index as dense vectors. Note that you must pass the application path in order for the config to be processed. This can be done via the CLI command with the ``--app-path`` option or via the ``app_path`` parameter for the ``load_kb`` method.


.. code-block:: console

  mindmeld load-kb hr_assistant faq_data data/hr_faq_data.json --app-path .

All of the vectors generated at load time will be cached for faster retrieval at inference time and for future loads. It is stored with other generated data in the generated folder under the provided model name. It’s important to update the mode name when updating the model settings to maintain consistency with the cache.

.. code-block:: console

  .generated/indexes/<model_name>_cache.pkl

If our built-in embedders don't fit your use case and you would like to use your own embedder, you can use the provided ``Embedder`` abstract class. You need to implement two methods: ``load`` and ``encode``. The load method will load and return your embedder model. The encode method will take a list of text strings and return a list of numpy vectors. You can register your class for use with MindMeld via the ``register_embedder`` method as shown below. This code can be added to any new file, say ``custom_embedders.py``. You will then need to import it your application's ``__init__.py`` file.

.. code-block:: python

  from mindmeld.models import Embedder, register_embedder

  class MyEmbedder(Embedder):
      def load(self, **kwargs):
          # return the loaded model
      def encode(self, text_list):
          # encodes each query in the list

  register_embedder('my_embedder_name', MyEmbedder)


.. code-block:: python

   import hr_assistant.custom_embedders


In many cases, you may like to provide some parameters to your ``load`` method to initialize your model in a certain way. The ``model_settings`` dictionary in the config will be passed to the load method as ``kwargs``, so any needed parameters can be specified there.


.. code-block:: python

  QUESTION_ANSWERER_CONFIG = {
      "model_type": "elasticsearch",
      "model_settings": {
          "query_type": "embedder",
          "embedder_type": "my_embedder_name",
          "some_model_param": "my_model_param",
          "embedding_fields": {"faq": ['question', 'answer']}
      }
  }


Once your knowledge base has been created, to search it while leveraging vector similarity, we will use the :meth:`get()` API as before with one small modification. We set the ``query_type`` parameter to ``embedder``, ``embedder_keyword``, or ``embedder_text``. This will automatically find results for which the embedded search fields are close in cosine similarity to the embedded query.

.. code:: python

  answers = qa.get(index=index_name, query_type='embedder',
                   question=query)

You can search against multiple fields.

.. code:: python

  answers = qa.get(index=index_name, query_type='embedder',
                   question=query, answer=query)


And you can use a combination of embedder and text based search.

.. code:: python

  answers = qa.get(index=index_name, query_type='embedder_keyword',
                   question=query, answer=query)

  answers = qa.get(index=index_name, query_type='embedder_text',
                 question=query, answer=query)
