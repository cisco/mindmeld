Step 9: Optimize Question Answering Performance
===============================================

The Workbench question answerer is a powerful component which streamlines the development of applications that need to answer questions in addition to understanding user requests. The question answerer relies on a knowledge base which encompasses all of the important world knowledge for a given application use case. For example, the question answerer might rely on a knowledge base which knows details about every product in a product catalog. Alternately, the question answerer might have a knowledge base containing detailed information about every song or album in a music library.

To leverage the Workbench question answerer in your application, you must first create your knowledge base, as described in :doc:`Step 5 </create_the_knowledge_base>`. With the knowledge base created, your dialogue state handlers can invoke the question answerer, as illustrated in :doc:`Step 4 </define_the_dialogue_handlers>`, to find answers, validate questions, and suggest alternatives.  For example, a simple dialogue handler which finds nearby Kwik-E-Mart store locations might resemble the snippet below. Notice that the application imports the :keyword:`QuestionAnswerer` component.

.. code:: python

  from mmworkbench import Application

  app = Application(__name__)

  @app.handle(intent='find_nearest_store')
  def send_nearest_store(context, slots, responder):
      user_location = context['request']['session']['location']
      stores = app.question_answerer.get(index='stores', sort='location', location=user_location)
      target_store = stores[0]
      slots['store_name'] = target_store['store_name']

      context['frame']['target_store'] = target_store
      responder.reply('Your nearest Kwik-E-Mart is located at {store_name}.')

Assuming you have already created an index, such as ``stores``, and uploaded the knowledge base data, the :keyword:`get()` method provides a flexible mechanism for retrieving relevant results.

.. code:: python

  >>> from mmworkbench.components import QuestionAnswerer
  >>> qa = QuestionAnswerer('my_app')
  >>> stores = qa.get(index='stores')
  >>> stores[0]
  {
    "store_name": "23 Elm Street",
    "open_time": "7am",
    "close_time": "9pm",
    "address": "100 Central Plaza, Suite 800, Elm Street, Capital City, CA 10001",
    "phone_number": "(+1) 415-555-1100",
    "score": 1.0
  }

Similarly, to retrieve store locations on Market Street, you could use something like:

.. code:: python

  >>> stores = qa.get(index='stores', 'market')
  >>> stores[0]
  {
    "store_name": "Pine and Market",
    "open_time": "6am",
    "close_time": "10pm",
    "address": "750 Market Street, Capital City, CA 94001",
    "phone_number": "(+1) 650-555-4500"
  }

By default, the :keyword:`get()` method uses a baseline ranking algorithm which displays the most relevant documents based on text similarity. Each result includes the relevance score in the :keyword:`score` property. For some applications, the baseline ranking is sufficient. For others, the Workbench question answerer provides flexible options for customizing relevance.

Custom Ranking Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider an application where we want to show only the least expensive products to users. For example, a user might ask 'show me your cheapest items', and your application then displays products in ascending order by price. Let's say we have the following objects in the knowledge base for the question answerer index ``products``:

.. code-block:: javascript

  {
    "id": 1,
    "name": "Pink Donut",
    "price": 1.29
  },
  {
    "id": 2,
    "name": "Green Donut",
    "price": 0.99
  },
  {
    "id": 3,
    "name": "Purple Squishee",
    "price": 0.89
  },
  {
    "id": 4,
    "name": "Yellow Donut",
    "price": 1.09
  }
  ...

To retrieve the all products sorted in ascending order by price, you can specify the ranking configuration and then retrieve results as follows.

.. code:: python

  >>> product_index = qa.indexes['products']
  >>> product_index.config({'price': 'asc'})
  >>> products = product_index.get()
  >>> products[0]
  {
    "id": 3,
    "name": "Purple Squishee",
    "price": 0.89,
    "score": 0.89
  }

As you can see, when we configure the ranking algorithm to return the least expensive products first, the item with the lowest price appears at the top of the list.

While a single-field sort operation is very straightforward, most applications require a more sophisticated ranking algorithm which blends many different signals to determine relevance. For example, suppose that your user is looking for the least expensive donut. In this case, a simple sort by price will not work. Instead, you need to return inexpensive products that can also be described as 'donuts.' In other words, the ideal ranking algorithm should blend both price and the text relevance of the term 'donut'. The Workbench question answerer makes it easy to configure ranking algorithms which blend signals from many different knowledge base fields, as shown below.

.. code:: python

  >>> product_index = qa.indexes['products']
  >>> product_index.config({'price': 'asc', 'name': 'desc'})
  >>> products = product_index.get('donut')
  >>> products[0]
  {
    "id": 2,
    "name": "Green Donut",
    "price": 0.99,
    "score": 0.946598
  }

Now the least expensive donut in the catalog is returned as the top result. Once you find a ranking configuration for an index that serves your needs, save it to file as follows.

.. code:: python

  >>> qa.indexes['products'].dump()

Similarly, to load a previously saved ranking configuration, you can use:

.. code:: python

  >>> qa.indexes['products'].load()

See the :ref:`User Guide <userguide>` for more about how to specify custom ranking configurations.


Proximity-Based Ranking
~~~~~~~~~~~~~~~~~~~~~~~

Location-based ranking is fairly common in mobile applications. We have already seen an intent designed to provide the nearest retail locations for a given user in our Kwik-E-Mart example. Going further, to support proximity-based ranking, is straightforward to accomplish using the Workbench question answerer.

First, let's assume that you have created a knowledge base for the ``stores`` index, which contains every retail location. Each store object also has a :keyword:`location` field which contains latitude and longitude coordinates for each store.

.. code-block:: javascript

  {
    "store_name": "23 Elm Street",
    "open_time": "7am",
    "close_time": "9pm",
    "address": "100 Central Plaza, Suite 800, Elm Street, Capital City, CA 10001",
    "phone_number": "(+1) 415-555-1100",
    "location": {"latitude": 37.790683, "longitude": -122.403889}
  },
  {
    "store_name": "Pine and Market",
    "open_time": "6am",
    "close_time": "10pm",
    "address": "750 Market Street, Capital City, CA 94001",
    "phone_number": "(+1) 650-555-4500",
    "location": {"latitude": 37.790426, "longitude": -122.405752}
  }
  ...

We can now retrieve the nearest stores as follows.

.. code:: python

  >>> store_index = qa.indexes['stores']
  >>> store_index.config({'location': 'asc'})
  >>> my_loc = {"latitude": 37.790415, "longitude": -122.405218}
  >>> stores = store_index.get(current_location=my_loc)
  >>> stores[0]
  {
    "store_name": "Pine and Market",
    "open_time": "6am",
    "close_time": "10pm",
    "address": "750 Market Street, Capital City, CA 94001",
    "phone_number": "(+1) 650-555-4500",
    "location": {"latitude": 37.790426, "longitude": -122.405752},
    "distance": 0.231543
  }

Each result includes a :keyword:`distance` field that says how far from the user the store is located (in kilometers). Equivalently, you can also use the :keyword:`sort` argument of the :keyword:`get()` method to explictly define the sort operation without relying on configuration beforehand.

.. code:: python

  >>> store_index = qa.indexes['stores']
  >>> my_loc = {"latitude": 37.790415, "longitude": -122.405218}
  >>> stores = store_index.get(sort='location', current_location=my_loc)
  >>> stores[0]
  {
    "store_name": "Pine and Market",
    "open_time": "6am",
    "close_time": "10pm",
    "address": "750 Market Street, Capital City, CA 94001",
    "phone_number": "(+1) 650-555-4500",
    "location": {"latitude": 37.790426, "longitude": -122.405752},
    "distance": 0.231543
  }


Machine-Learned Ranking
~~~~~~~~~~~~~~~~~~~~~~~

State-of-the-art information retrieval systems such as the Bing and Google search engines rely on sophisticated AI-powered ranking algorithms. These ranking algorithms leverage `machine learning <https://en.wikipedia.org/wiki/Learning_to_rank>`_ in order to learn the optimal ranking formula based on training data collected from live user traffic. For large knowledge domains which may contain millions or even billions of objects in a knowledge base, machine-learned ranking is typically the most effective path for delivering optimal ranking. The MindMeld question answerer component provides the capability not only to handle large knowledge bases but also to train machine-learned ranking algorithms.

The training data for machine-learned ranking is captured in the index ranking files discussed in :doc:`Step 6 </generate_representative_training_data>`. These index ranking files specify the ideal rank for a knowledge base object given a specific query. For example, for the ``stores`` index, the training data file might look something like:

.. code-block:: javascript

  [
    {
      'query': 'Kwik-E-Marts in Springfield',
      'id': '152323',
      'rank': 3
    },
    {
      'query': 'Kwik-E-Marts in Springfield',
      'id': '102843',
      'rank': 1
    },
    {
      'query': 'stores downtown',
      'id': '207492',
      'rank': 1
    },
    ...

  ]
  ...

These training data examples can be generated using manual QA where human graders subjectively score the relevance of the knowledge base results for a set of reference queries. Alternately, for applications with live production traffic, this training data can often be generated by observing how actual users interact with knowledge base results in the application. If sufficient representative training data is available, the Workbench question answerer makes it straightforward to train and evaluate a custom ranking model.

.. code-block:: python

  >>> from mmworkbench import QuestionAnswerer
  >>> qa = QuestionAnswerer()
  >>> store_index = qa.indexes['stores']
  >>>
  >>> # Fit the ranking model using training data available in the application directory.
  ... store_index.fit()

  >>> # Now retrieve results using the new ranking model.
  ... stores = store_index.get('ferry bldg')
  >>> stores[0]
  {
    "store_name": "Ferry Building Market",
    "open_time": "6am",
    "close_time": "10pm",
    "address": "Pier 1, The Embarcadero, SF, CA 94001",
    "score": 0.874098
    ...
  }

  >>> # To save the model to file.
  ... store_index.dump()

For more about how to train and evaluate machine-learned ranking models, see the :ref:`User Guide <userguide>`.

