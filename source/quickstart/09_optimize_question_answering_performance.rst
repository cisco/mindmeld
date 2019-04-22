Step 9: Optimize Question Answering Performance
===============================================

The MindMeld question answerer is a powerful component which streamlines the development of applications that need to answer questions in addition to understanding user requests. The question answerer relies on a knowledge base which encompasses all of the important world knowledge for a given application use case. For example, the question answerer might rely on a knowledge base which knows details about every product in a product catalog. Alternately, the question answerer might have a knowledge base containing detailed information about every song or album in a music library.

To leverage the MindMeld question answerer in your application, you must first create your knowledge base, as described in :doc:`Step 5 <05_create_the_knowledge_base>`. With the knowledge base created, your dialogue state handlers can invoke the question answerer, as illustrated in :doc:`Step 4 <04_define_the_dialogue_handlers>`, to find answers, validate questions, and suggest alternatives.  For example, a simple dialogue handler which finds nearby Kwik-E-Mart store locations might resemble the snippet below. Notice that the application imports the :class:`QuestionAnswerer` component.

.. code:: python

  from mindmeld import Application

  app = Application(__name__)

  @app.handle(intent='find_nearest_store')
  def send_nearest_store(request, responder):
      try:
          user_location = request.context['location']
      except KeyError:
          responder.reply("I'm not sure. You haven't told me where you are!")
          responder.suggest([{'type': 'location', 'text': 'Share your location'}])
          return

      stores = app.question_answerer.get(index='stores', _sort='location', _sort_type='distance',
                                         _sort_location=user_location)
      target_store = stores[0]
      responder.slots['store_name'] = target_store['store_name']

      responder.frame['target_store'] = target_store
      responder.reply('Your nearest Kwik-E-Mart is located at {store_name}.')

Assuming you have already created an index, such as ``stores``, and uploaded the knowledge base data, the :meth:`get()` method provides a flexible mechanism for retrieving relevant results.

.. code-block:: shell

   cd $MM_APP_ROOT
   python

.. code:: python

   from mindmeld.components import QuestionAnswerer
   qa = QuestionAnswerer('.')
   stores = qa.get(index='stores')
   stores[0]

.. code:: console

  {
    "store_name": "23 Elm Street",
    "open_time": "7am",
    "close_time": "9pm",
    "address": "100 Central Plaza, Suite 800, Elm Street, Capital City, CA 10001",
    "phone_number": "(+1) 415-555-1100"
  }

Similarly, to retrieve store locations on Market Street, you could use something like:

.. code:: python

   stores = qa.get('market', index='stores')
   stores[0]

.. code:: console

   {
     "store_name": "Pine and Market",
     "open_time": "6am",
     "close_time": "10pm",
     "address": "750 Market Street, Capital City, CA 94001",
     "phone_number": "(+1) 650-555-4500"
   }

By default, the :meth:`get()` method uses a baseline ranking algorithm which displays the most relevant documents based on text similarity.

Proximity-Based Ranking
~~~~~~~~~~~~~~~~~~~~~~~

Location-based ranking is fairly common in mobile applications. We have already seen an intent designed to provide the nearest retail locations for a given user in our Kwik-E-Mart example. Going further, to support proximity-based ranking, is straightforward to accomplish using the MindMeld question answerer.

First, let's assume that you have created a knowledge base for the ``stores`` index, which contains every retail location. Each store object also has a ``location`` field which contains latitude and longitude coordinates for each store.

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

We can now retrieve the nearest stores using the ``sort`` argument of the :meth:`get()` method
as follows:

.. code:: python

   my_loc = {"latitude": 37.790415, "longitude": -122.405218}
   stores = qa.get(index='stores', location=my_loc, sort='location')
   stores[0]

.. code:: console

   {
     "store_name": "Pine and Market",
     "open_time": "6am",
     "close_time": "10pm",
     "address": "750 Market Street, Capital City, CA 94001",
     "phone_number": "(+1) 650-555-4500",
     "location": {"latitude": 37.790426, "longitude": -122.405752}
   }

See the :doc:`User Guide <../userguide/kb>` for more about how to use the Question Answerer to find answers to questions, validate user requests, disambiguate entities, and offer alternative suggestions.
