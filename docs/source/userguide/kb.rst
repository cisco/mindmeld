Working with the Knowledge Base and Question Answerer
=====================================================

What is Knowledge Base?
-----------------------
A knowledge base is a comprehensive repository that contains the universe of helpful information needed to understand requests and answer questions. It is the key component which understands domain and application specific concept and relationships. It allows us to optimize the natural language processing and question answering to achieve human-like accuracy.

In its most basic form, a knowledge base is simply a repository of objects of specified types. For example, a knowledge base that contains a large entertainment content catalog could include ``movie`` and ``tv_show`` among the types of its objects. A knowledge base that contains a directory of local businesses could have object types like ``restaurant``, ``hardware_store``, and so on. Objects typically have attributes which capture salient aspects of the concepts they describe. For example, a ``restaurant`` object might have attributes for ``address`` and ``phone_number``; a ``movie`` object might have attributes like ``cast_list``, ``runtime``, and ``release_date``.

Do we need a Knowledge Base?
----------------------------
The first question to answer before we start to build the application is - do we need a knowledge base or not? Knowledge base is critical and necessary component for application with large or medium vocabulary content and is usually the differentiating factor for the best conversational application. For example, video discovery application typically need to find the movie or TV shows from a huge catalog containing hundred of thousands or even millions of records and the knowledge base provides information about important entities to fulfill users' requests. On the other hand, the applications with small vocabulary such as home assistant for smart home automation genearally don't require ranking and disambiguating entities and may not need a knowledge base. 

Question Answering in Conversational Applications
-------------------------------------------------
The idea of knowledge-based question answering is mapping natural language queries to a query over structured database. The Workbench Question Answerer is specifically designed for custom knowledge bases involving large content catalogs, for production-grade applications. Catalogs containing even up to hundreds of millions of unique, domain-specific entity objects can easily be handled by Workbench in production. Typically, Food delivery service processes restaurant menus range in the tens of thousands entities and specific product retail catalogs go into the millions, while media entertainment libraries scale into the tens or hundreds of millions of unique entities. Workbench's Knowledge Base and Question Answering systems have been successfully applied to all of the above use cases. The Question Answerer can be used in a variety of ways, but in practice, conversational applications rely on this module and its underlying knowledge base for the four primary purposes listed below.

1. Answer Questions
```````````````````

One of the main use cases of question answerer in conversational applications is to provide one or more relevant results as answers to users' requests. Knowledge base searches can be constructed using the entities extracted from NLP pipeline. The answers can be in various forms depending on the nature of the applications. For instance, in a food ordering application the answer is usually a dish with attributes matching the criteria provided by users, while in video discovery application the answer is more often a list of best matching movies or TV shows.

2. Validate Questions 
`````````````````````

The knowledge base can also contain information to help inform user that the request is out of scope. For example, if a user orders pizza from a coffee shop in a food ordering application the Question Answerer would not be able to find the entity matching user's request and can provide information to help steer the conversation to the right direction. There are other examples that fall into this category such as asking a dish that is not available in the specified restaurant or asking for a customization option that is not available for the specified dish.

3. Disambiguate Entities
````````````````````````

It's common that we need to disambiguate entities based on application requirements and context. For example, in a food ordering application there could be hundreds of ``Pad Thai`` dishes offered from a number of restaurants. In a travel booking application it is impossible to book the right flight without knowing exactly which destination city ``Springfield`` user is asking for. (In Massachusetts or Illinois?) In a music discovery application when user asks for ``thriller`` they could be referring to a song or an album. In these cases we would not be able to retrieve the exact entities that users are referring to without having the contextual information taken into account. The contextual info may be the entity relationships and hierarchy, user preferences or application business logic. 

This disambiguation task can be formulated as a knowledge base search with constraints from contextual information. For the food ordering example the selected restaurant can be added as a filter to the knowledge base search to find the best matching dishes within that particular restaurant.

4. Suggest Alternatives
```````````````````````

Workbench Question Answerer can be used to suggest closest matches when the correct matches could not be found. For example, if a user requests 'Star Wars Rogue One' and that movie is not available, the knowledge base could suggest other available Star Wars titles. The Question Answerer uses a number of scoring factors including text relevance, location proximity among others to retrieve most relevant information from knowledge base as suggestions. 

There are other cases where the Question Answerer can be used to suggest alternative based on application custom logic. Take food ordering application as example, we may want to 

	* suggesting dishes in nearby restaurants when they could not be found in selected restaurant.
	* suggesting other dishes in the selected restaurant when the specified dishes could not be found.

This can be done by using Question Answerer to formulate knowledge base searches with modified constraints to find best matches as suggestions.

Prepare Data for Knowledge Base
-------------------------------
The Workbench Question Answerer makes it starightforward to build custom knowledge base using application content data. The content data can be restaurant menus, retailing product catalogs or any custom data that users would like to interact with through conversational interfaces. They are often stored in large scale databases with application specific data models. The Question Answerer can build knowledge base using data dumps from databases or output of data pipelines which handles more complex data transformations if necessary.  

The Question Answerer takes in data files containing knowledge base objects which is the basic unit of information in knowledge base index. Each data file contains objects of a specified type. Each object has an ``id`` field as the unique identifier, an optional ``location`` field for location information if available and a list of arbitrary data fields of different types that contain information about the object or about the relationship with other object types. The Question Answerer creates optimized indexes for objects to efficiently and accurately retrieve most relevant information. It processes all data fields to determine the data field types and index them accordingly.  

The Question Answerer supports the following data types.

==================== ===
**id**               unique identifier string.
**text**             full-text strings, e.g. dish names like ``Pad Thai`` or movie names like ``Star Wars``.
**number**           numerics values in one of the supported formats: ``long``, ``integer``, ``short``, ``byte``, ``double``, ``float``. 
**date**             date value in one of the supported formats described in the table below.
**location**         location value in one of the supported formats described in the table below.
==================== ===

.. note:: Note that the location information of a knowledge base object needs to be specified using ``location`` field and it is currently not supported to specify additional location fields or use custom field name for location.

For date and location data types the following value formats are supported.

==================== ===
**date**             * strings containing formatted dates, e.g. "2017-07-31" or "2017/07/31 12:10:30".
                     * a long number representing milliseconds-since-the-epoch.
                     * an integer representing seconds-since-the-epoch.
**location**         * an object containing latitude and longitude: 
                       
                       .. code-block:: javascript

	                       {
	                       	  "lat": 37.77, 
	                       	  "lon": 122.41
	                       }

                     * geo-point as a string, e.g. "37.77,122.41"
                     * geo-point as an array, e.g. [37.77, 122.41]  
==================== ===

The Question Answerer supports any data model that applications choose to use. Applications may prefer using certain data models over the other for various reasons. For example, in certain food ordering applications the ``cuisine`` types can have richer information and be modeled as an object with a list of attributes or it can simply be a string for cuisine type name and be modeled as an attribute of dish objects. The mapping between NLP entity type and knowledge base objects is often application specific and the knowledge base searches will need to be formulated accordingly.

For example, a food ordering application may have ``restaurant`` and ``menu_item`` objects, while a video discovery application may have ``cast`` and ``title`` objects. The ``menu_item`` object in food ordering application may have the following fields:

+---------------+---------------+------------------------------------+
| Field         | Type          | Description                        |
+===============+===============+====================================+
| id            | id            | unique identifier string.          |
+---------------+---------------+------------------------------------+
| name          | text          | name of the dish.                  |
+---------------+---------------+------------------------------------+
| description   | text          | description of the dish.           |
+---------------+---------------+------------------------------------+
| category      | text          | dish category.                     |
+---------------+---------------+------------------------------------+
| price         | number        | dish price.                        |
+---------------+---------------+------------------------------------+
| img_url       | text          | dish image URL.                    |
+---------------+---------------+------------------------------------+
| restaurant_id | text          | ID of the restaurant.              |
+---------------+---------------+------------------------------------+

And the JSON data file for the ``menu_item`` object may look like the following:

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

It's critical to have clean data in knowledge base for question answerer to achieve the best possible performance. While Workbench knowledge base performs generic text processing and normalization it's common that some necessary normalizations are rather domain or application specific and it's often a good practice to inspect the data to identify noise and inconsistency in the dataset and perform necessary clean-up and normalization as pre-processing. For example, in a food ordering application it's possible that the menus from different restaurant can have different formats and use different conventions. This pre-processing task is very important to ensure high accuracy.

Import Data into Knowledge Base
-------------------------------
The Question Answerer provides APIs to load data into knowledge base. The :meth:`load_kb()` API loads data from JSON-formatted data file to create an index for the specified objects in the knowledge base. The index will be created if it does not exist, otherwise the objects will be imported into existing index.

In the following example :meth:`load_kb()` is used to load data file from path ``my_app/data/restaurants.json`` and create an index called ``restaurants`` for all restaurant objects specified in the data file: 

.. code:: python

	>>> from mmworkbench.components import QuestionAnswerer
	>>> qa = QuestionAnswerer(app_path='my_app')
	>>> qa.load_kb(app_name='my_app', index_name='restaurants', data_file='my_app/data/restaurants.json')

Alternatively the Workbench command line tool can be used to create knowledge base indexes.

.. code-block:: console

	$ python app.py load-kb my_app restaurants my_app/data/restaurants.json

To check that your knowledge base was created successfully, use the Question Answerer to retrieve restaurant information from your index:

.. code:: python

  >>> restaurants = qa.get(index='restaurants')
  >>> restaurants[0]
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

Knowledge Base Search
---------------------

The Question Answerer provides easy-to-use flexible APIs to retrieve relevant information from knowledge base.

Basic Search
````````````

The Question Answerer provides basic search API - :meth:`get()` method for simple knowledge base searches. It has a simple and intuitive interface and can be used in a similar way as in common Web search interfaces. It takes in a list of (knowledge base field, text query) pairs to find best matches. The knowledge base fields to be used depend on the mapping between NLP entity types and corresponding knowledge base objects. For example, in a food ordering application ``cuisine`` entity type can be mapped to a knowledge base object or an attribute of a knowledge base object. The mapping is often application specific and is dependent on the data model of the application. 

The basic search API can be used to retrieve a particular knowledge base object using ID when the exact ID of the object is already identified.   

.. code:: python
	
	>>> from mmworkbench.components import QuestionAnswerer
	>>> qa = QuestionAnswerer(app_path='my_app')
	>>> qa.get(index='menu_items', id='B01CGKGQ40')
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

It also supports knowledge base searches with a list of text queries. The text query strings are specified like keywords accompanied with corresponding knowledge base field and the best results matching all queries specified are returned. In the following example we try to find the dishes that have name matching ``fish and chips`` and the restaurant ID matching ``B01DEEGQBK``:

.. code:: python
	
	>>> from mmworkbench.components import QuestionAnswerer
	>>> qa = QuestionAnswerer(app_path='my_app')
	>>> results = qa.get(index='menu_items', name='fish and chips', restaurant_id='B01DEEGQBK')
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

It's also possible to specify custom sort criteria in basic search API. The following parameters are supported to specify custom sort criteria.

==================== ===
**_sort**            the knowledge base field used for sorting.
**_sort_type**       valid values are ``asc``, ``desc`` and ``distance``. ``asc`` and ``desc`` specifies the sort order for sorting on number or date fields, while ``distance`` indicates sorting by distance based on ``location`` field.
**_sort_location**   specify origin location for sorting by distance.
==================== ===

In the following example Question Answerer finds ``menu_items`` objects that best match ``fish and chips`` on ``name``, ``B01CGKGQ40`` on ``restaurant_id`` and have cheaper price. Note that the score for ranking is a optimized blend of sort score and text relevance scores:

.. code:: python
	
	>>> from mmworkbench.components import QuestionAnswerer
	>>> qa = QuestionAnswerer(app_path='my_app')
	>>> results = qa.get(index='menu_items', name='fish and chips', restaurant_id='B01CGKGQ40', _sort='price', _sort_type='asc')
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

To define sorting by distance based on certain origin location we can specify the ``_sort_type`` parameter to be ``distance`` and specify origin location in ``_sort_location`` parameter. In the following example we try to find the closest restaurant from the center of San Francisco:

.. code:: python

	>>> from mmworkbench.components import QuestionAnswerer
	>>> qa = QuestionAnswerer(app_path='my_app')
	>>> qa.get(index='restaurants', _sort='location', _sort_type='distance', _sort_location='37.77,122.41')
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


Question Answerer provides advanced search API for more advanced use case which require more fine-grained control of the knowledge base search behavior. The advanced search APIs are described in the next section.

Advanced Search
```````````````

Workbench Question Answerer provides advanced search APIs to support more complex knowledge base searches. It allows a list of text queries, filters and custom sort criteria to be specified for having fine-grained control on knowledge base search behavior.

.. code:: python

	>>> from mmworkbench.components import QuestionAnswerer
	>>> qa = QuestionAnswerer(app_path='my_app')
	>>> s = qa.build_search(index='menu_items')

:meth:`build_search()` API creates a Search object which is an abstraction of a knowledge base search. It provides several APIs for specifying text query, text or range filters and custom sort criteria. The APIs are chainable to provide a compact and readable syntax.  

Query
'''''

:meth:`query()` API can be used to add text queries to the knowledge base search. For each query a knowledge base field and query string are specified for text relevance match. Workbench Question Answerer ranks results using several ranking factors on textual information including exact matches, phrase matches and partial matches to find best matching results. Note that Question Answerer expects the queries to be specified on knowledge base text fields.

In the following example Question Answerer returns best matching dishes with the name ``fish and chips``. We specify the query string ``fish and chips`` on the knowledge base field ``name`` in ``menu_items`` index which contains all available dishes. The top two results have the name exactly as ``fish and chips`` from different restaurants:

.. code:: python

	>>> from mmworkbench.components import QuestionAnswerer
	>>> qa = QuestionAnswerer(app_path='my_app')
	>>> s = qa.build_search(index='menu_items')
	>>> s.query(name='fish and chips').execute()
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
''''''

:meth:`filter()` API can be used to add filters to the knowledge base search. There are two types of filters supported: **text filter** and **range filter**. For text filter a knowledge base text field name and the filtering text string are specified. The text string is normalized and the entire text string is used to filter the results like SQL predicates in relational databases. For example, in food ordering applications it's common that users would want to find dishes of a particular cuisine type or from a specific restaurant they had in mind. In the following example we try to find the best matching ``fish and chips`` dishes within restaurant with ID ``B01DEEGQBK``:

.. code:: python

	>>> from mmworkbench.components import QuestionAnswerer
	>>> qa = QuestionAnswerer(app_path='my_app')
	>>> s = qa.build_search(index='menu_items')
	>>> s.query(name='fish and chips').filter(restaurant_id='B01DEEGQBK').execute()
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

Question Answerer also allows applying filters on number or date ranges. Some example use cases are finding products within certain price ranges in retailing application and finding movies released in the past five 5 years in video discovery application. 

To define a filter on ranges we specify a knowledge base field and one or more range operators. The supported range operators are described below.

======== ===
**gt**   greater than
**gte**  greater than or equal to
**lt**   less than
**lte**  less than or equal to
======== ===

In the example below we filter on price range to find the dishes priced below 5 dollars:

.. code:: python

	>>> from mmworkbench.components import QuestionAnswerer
	>>> qa = QuestionAnswerer(app_path='my_app')
	>>> s = qa.build_search(index='menu_items')
	>>> s.filter(field='price', lte=5).execute()
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

.. note:: Note that the range filters are only valid for number and date knowledge base fields. 

Sort
''''

:meth:`sort()` API can be used to add one or more custom sort criteria to a knowledge base search. Custom sort can be used with number, date or location knowledge base fields. It takes in three parameters: ``field``, ``sort_type`` and ``location``. The ``field`` parameter specifies the knowledge base field for sort, the ``sort_type`` parameter can be either ``asc`` or ``desc`` to indicate sort order for number or date fields and ``distance`` to indicate sorting by distance using location field, and the ``location`` field parameter specifies the origin location when sorting by distance. 

The custom sort can be applied to any number or date fields desirable and the score for ranking will be a optimized blend of sort score with other scoring factors including text relevance scores when available. In the following example Question Answerer finds the best ``menu_item`` objects matching text query ``fish and chips`` with cheaper price by combining the text relevance score and sort score on ``price`` field:

.. code:: python

	>>> from mmworkbench.components import QuestionAnswerer
	>>> qa = QuestionAnswerer(app_path='my_app')
	>>> s = qa.build_search(index='menu_items')
	>>> s.query(name='fish and chips').sort(field='price', sort_type='asc').execute()
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

It's also fairly common to use proximity as sorting factor when using conversational applications on the go. To define sorting by distance ``location`` needs to be specified as sort field with ``distance`` for sort_type parameter and the origin location latitude and longitude for location parameter. In the example below Question Answerer provides a list of best restaurant options that match ``firetrail`` on restaurant name and close to center of San Francisco:

.. code:: python

	>>> from mmworkbench.components import QuestionAnswerer
	>>> qa = QuestionAnswerer(app_path='my_app')
	>>> s = qa.build_search(index='restaurants')
	>>> s.query(name='firetrail').sort(field='location', type='distance', location='37.77,122.41').execute()
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


When to use Basic Search vs Advanced Search?
`````````````````````````````````````````````
The basic search API is designed to cover the most common use cases in conversational applications, while the advanced search API provides additional capabilities for building more complex knowledge base searches. Generally the advanced search API is needed in the following scenarios. 

	* need more than one custom sort criteria
	* need to filter on number or date ranges
	* need fine-grained control of the search behavior


