Knowledge Base and Question Answerer
====================================

What is Knowledge Base?
-----------------------
A knowledge base is a comprehensive repository that contains the universe of helpful information needed to understand requests and answer questions. It is the key component which understands domain and application specific concept and relationships. It allows us to optimize the NLP process and question answering to achieve human like accuracy.

In its most basic form, a knowledge base is simply a repository of objects of specified types. For example, a knowledge base that contains a large entertainment content catalog could include ``movie`` and ``tv_show`` among the types of its objects. A knowledge base that contains a directory of local businesses could have object types like ``restaurant``, ``hardware_store``, and so on. Objects typically have attributes which capture salient aspects of the concepts they describe. For example, a ``restaurant`` object might have attributes for ``address`` and ``phone_number``; a ``movie`` object might have attributes like ``cast_list``, ``runtime``, and ``release_date``.

Do we need a Knowledge Base?
----------------------------
The first question to answer before we start to build the application is do we need a knowledge base or not? Knowledge base is critical and necessary component for application with large or medium vocabulary content and is usually the differentiating factor for the best conversational application. For example, video discovery application typically need to find the movie or TV shows from a huge catalog containing hundred of thousands or even millions of records and the knowledge base provides information about important entities to fulfill users' requests. On the other hand application with small vocabulary may not require a knowledge base. 

Question Answering in Conversational Applications
-------------------------------------------------
The foundations of Question Answering began back in the 60's and early 70's with knowledge bases targeted at narrow domains of knowledge. With computational linguistics emerging and evolving over the decades, modern day Question Answering systems have relied on statistical processing of broader domain, large vocabulary document collections. These systems especially started gaining popularity in mainstream media in the last 8-10 years. The IBM Watson Jeopardy! challenge revealed the beginnings of the true potential of broad vocabulary Question Answering systems that could be deployed in real world production applications.

Several academic research initiatives continue to hallmark the field of Question Answering. The `OAQA <https://oaqa.github.io/>`_ initiative is a collaboration between `academia <http://www.cs.cmu.edu/~ehn/>`_, `industry <https://www.research.ibm.com/deepqa/question_answering.shtml>`_ and the open source community. MIT CSAIL has a research thrust in Question Answering with the `START <http://start.csail.mit.edu/index.php>`_ project. A highly regarded conference specialized to the field of Question Answering is the `TREC <http://trec.nist.gov/>`_ - Text REtrieval Conference. Several academic datasets and related publications are available there for background reading.

Knowledge-based question answering is the idea of mapping natural language queries to a query over structured database. The MindMeld Workbench Question Answerer is specifically designed for custom knowledge bases involving large content catalogs, for production-grade applications. Catalogs containing even upto hundreds of millions of unique, domain-specific entity objects can easily be handled by MindMeld Workbench in production. Typically, Quick Service Restaurant menus range in the hundreds of thousands entities and specific product retail catalogs go into the millions, while media entertainment libraries scale into the tens or hundreds of millions of unique entities. MindMeld Workbench's Knowledge Base and Question Answering systems have been successfully been applied to all of the above use cases. The Question Answerer can be used in a variety of ways, but in practice, conversational applications rely on this module and its underlying knowledge base for the four primary purposes listed below.

Answer Questions
````````````````

One of the main use cases of question answerer in conversational applications is to provide one or more relevant documents as answer to users' requests. A knowledge base search can be constructed using the entities from NLP pipeline. The answers can be in various forms depending on the nature of the applications. For example, in a food ordering application the answer is usually the canonical dish with attributes necessary to complete the dialogue flow, while in video discovery application the answer is more often a list of best matching movies or TV shows.

Validate Questions 
``````````````````

The knowledge base can also contain information to help inform user that the request is out of scope. For example, a knowledge base of food ordering application understands that the movie title ``Star Wars`` is not one of the entities supported by the application and can provide information to help steer the conversation to the right direction.

Disambiguate Entities
`````````````````````

It's common that we need to disambiguate entities based on application requirements. For example, in a food ordering application there could be hundreds of ``Pad Thai`` dishes offered from a number of restaurants. We would not be able to retrieve the canonical entities that users are referring to without having the contextual information taken into account. The contextual info may be the entity relationships and hierarchy, user preferences or business logic in the application. 

This task can be formulated to a knowledge base search with constraints from contextual information. For the food ordering example the selected restaurant can be added as a filter to the knowledge base search to find the best matching dishes within that particular restaurant.

Suggest Alternatives
````````````````````

Workbench Question Answerer uses a number of scoring factors for knowledge base search. It suggests closest matches when the correct matches could not be found. For example, if a user requests 'Star Wars Rogue One' and that movie is not available, the knowledge base could suggest other, available Star Wars titles.

Prepare Data for Knowledge Base
-------------------------------
In knowledge base various objects of different types are stored in one or more indices. Each object can have a list of attributes which contain information about the object or about the relationship with another object type. 

For example, the knowledge base data could look like the following in a food ordering application.

.. code-block:: javascript

  {
    "category": "Makimono-Sushi Rolls (6 Pcs)",
    "menu_id": "78eb0100-029d-4efc-8b8c-77f97dc875b5",
    "description": "Makimono-Sushi Rolls (6 Pcs)\nDeep-fried shrimp, avocado, cucumber",
    "price": 6.5,
    "option_groups": [],
	"restaurant_id": "B01N97KQNJ",
	"size_prices": [],
	"size_group": null,
	"popular": false,
	"img_url": null,
	"id": "B01N0KXELH",
	"name": "Shrimp Tempura Roll"
  },
  {
    "category": "Special Rolls",
	"menu_id": "78eb0100-029d-4efc-8b8c-77f97dc875b5",
	"description": "California roll topped w/ cooked salmon, mayo and masago",
	"price": 9.95,
	"option_groups": [],
	"restaurant_id": "B01N97KQNJ",
	"size_prices": [],
	"size_group": null,
	"popular": false,
	"img_url": null,
	"id": "B01MYTS7W4",
	"name": "Pink Salmon Roll"
  }
  ...

[TODO: add details about location field value format]

It's critical to have clean data in knowledge base for question answerer to achieve the best possible performance. While Workbench knowledge base performs generic text processing and normalization it's common that some necessary normalizations are rather application specific and it's often a good practice to inspect the data to identify noise and incosistency in the dataset and perform necessary clean-up and normalization as pre-processing before knowledge base data import. For example, in a food ordering application it's possible that the menus from different restaurant can have in different formats and use different conventions. It is very important to have pre-processing for this to avoid potential issues down the road.

Import Data into Knowledge Base
-------------------------------
Workbench Question Answerer provides APIs to load data into knowledge base. Currently Workbench expects knowledge base data in JSON format.

.. code:: python

	>>> from mmworkbench.components import QuestionAnswerer
	>>> qa = QuestionAnswerer(app_path='my_app')
	>>> qa.load_kb('my_app', 'stores', 'my_app/data/stores.json')

See API :doc:`documentation<link>` for more details.

The knowledge base data import can also be done via Workbench command-line tool ``mmworkbench``.

.. code-block:: console
	>>> mmworkbench load_kb


Knowledge Base Search
---------------------

Workbench Question Answerer module provides APIs to retrieve relevant information from knowledge base.

Basic Search
````````````

Question Answerer provides basic search API - ``get()`` method for simple knowledge base searches. It has a simple and intuitive interface and can be used in a similar way as seen in common web search interfaces. It allows developers to specify a list of text query and knowledge base field pairs to find best matches. The knowledge base fields to be used depend on the mapping between NLP entity types and corresponding knowledge base objects, For example, in a food ordering domain application ``cuisine`` entity type can be mapped to a knowledge base object or an attribute of a knowledge base object. The mapping is often application specific and is dependent on the data model developers choose to use when building knowledge base. 

The basic search API can retrieve a particular knowledge base object using ID

.. code:: python
	
	>>> from mmworkbench.components import QuestionAnswerer
	>>> qa = QuestionAnswerer(app_path='my_app')
	>>> qa.get(index='menu_items', id='B01CGKGQ40')

It also supports knowledge base search using a list of text queries

.. code:: python
	
	>>> from mmworkbench.components import QuestionAnswerer
	>>> qa = QuestionAnswerer(app_path='my_app')
	>>> qa.get(index='menu_items', name='pork and shrimp', restaurant_id='B01CGKGQ40')

When using the basic search API the text query strings are specified like keywords accompanied with the corresponding knowledge base field. In the example above we have a query string ``pork and shrimp`` to search against knowledge base field ``name``. Filter conditions can also be specified as queries in basic search API. In the example above the filter condition using ID on ``restaurant_id`` field are specified the same way as text queries. It automatically figures out the exact matches to be the important ranking factor for matching on filter criteria to find the best matching objects.

It's also possible to specify one optional custom sort criteria with the basic search API. The following parameters are supported for controlling custom sort behavior.

	* **_sort_field**: the knowledge base field used for sorting. 
	* **_sort_type**: specify the sort order (asc vs desc) or sort by distance. The sort order can be specified for number and date field, while the sort by distance can be specified for location field.
	* **_sort_location**: optional parameter to specify origin for sorting by distance.

.. code:: python
	
	>>> from mmworkbench.components import QuestionAnswerer
	>>> qa = QuestionAnswerer(app_path='my_app')
	>>> qa.get(index='menu_items', name='pork and shrimp', restaurant_id='B01CGKGQ40', _sort='price', _sort_type='asc')

It's often desirable to sort by distance to find best matches with user's current location taken into account.

	>>> from mmworkbench.components import QuestionAnswerer
	>>> qa = QuestionAnswerer(app_path='my_app')
	>>> qa.get(index='menu_items', name='pork and shrimp', _sort='location', _sort_type='distance', _sort_location='33.14,123.15')

The basic search API is designed to have an intuitive interface that works for the most common use cases. It has certain limitations to keep the interface clean and simple including:
	* filters based on number or date ranges are not supported.
	* Only one custom sort criteria is allowed.

Question Answerer provides advanced search API for more advanced use case which require more fine-grained control of the knowledge base search behavior. The advanced search APIs are described in the next section.

Advanced Search
```````````````

Workbench Question Answerer provides advanced search APIs to support more complex knowledge base searches. It allows a list of text queries, filters and custom sort criteria to be specified for having fine-grained control on knowledge base search behavior.

.. code:: python
	>>> from mmworkbench.components import QuestionAnswerer
	>>> qa = QuestionAnswerer(app_path='my_app')
	>>> s = qa.build_search()

build_search() API creates a Search object which is an abstraction of a knowledge base search. It provides several APIs for specifying text query, text or range filters and custom sort criteria.

Query
'''''

``query()`` API can be used to add text queries to the knowledge base search. For each query a knowledge base field and query string are specified for text relevance match. Several ranking factors including exact matches, phrase matches and partial matches are used to calculate text relevance scores and find best matching documents.

.. code:: python
	>>> from mmworkbench.components import QuestionAnswerer
	>>> qa = QuestionAnswerer(app_path='my_app')
	>>> s = qa.build_search()
	>>> s = s.query(dish_name='fish and chips')

Filter
''''''

``filter()`` API can be used to add filters to the knowledge base search. There are two types of filters supported: text filter and range filter. For text filter a knowledge base text field name and the filtering text string are specified. The text string is normalized and the entire text string is used to filter the documents like SQL predicates in RDBMS. For example, in food ordering applications we can filter dishes using selected restaurant ID. 

.. code:: python
	>>> from mmworkbench.components import QuestionAnswerer
	>>> qa = QuestionAnswerer(app_path='my_app')
	>>> s = qa.build_search()
	>>> s = s.filter(restaurant_id='B01CGKGQ40')

Range filter is used to filter based on number or date ranges. It's created by specifying knowledge base field and one or more range operators. The supported range operators are described below.

	* ``gt``: greater than
	* ``gte``: greater than or equal to
	* ``lt``: less than
	* ``lte``: less than or equal to

.. code:: python
	>>> from mmworkbench.components import QuestionAnswerer
	>>> qa = QuestionAnswerer(app_path='my_app')
	>>> s = qa.build_search()
	>>> s = s.filter(field='price', lte=25)

Note that the range filters are only valid for number and date knowledge base fields. 

Sort
''''

``sort()`` API can be used to add custom sort criteria for a knowledge base search. To define a custom sort criteria a knowledge base field, sorting types (``asc``, ``desc``, or ``distance``) and origin location for sorting by distance.

Custom sort can only be used with number, date and location knowledge base fields. For number and date fields the sort type can simply be either ``asc`` or ``desc`` to determine sort order. Some example use cases are finding most popular items, cheapest items and most recently released items and etc.

.. code:: python
	>>> from mmworkbench.components import QuestionAnswerer
	>>> qa = QuestionAnswerer(app_path='my_app')
	>>> s = qa.build_search()
	>>> s = s.sort(field='popularity', type='desc')

As mentioned in previous section the requirement of sorting by distance is also very common in many applications. The sort by distance criteria can be applied to knowledge base location field by specifying the field name with the sort type ``distance`` and sort location parameter to indicate the origin location. 

.. code:: python
	>>> from mmworkbench.components import QuestionAnswerer
	>>> qa = QuestionAnswerer(app_path='my_app')
	>>> s = qa.build_search()
	>>> s = s.sort(field='location', type='distance', location='33.14,-123.15')

When to use Basic Search vs Advanced Search?
`````````````````````````````````````````````
The basic search API is designed to cover the most common use cases in conversational applications. The advanced search API provides additional capability for building more complex knowledge base searches. Generally the advanced search API is needed in the following scenarios. 

	* need more than one custom sort criteria
	* need to filter on ranges (number or date)
	* need finer control of the search behavior


