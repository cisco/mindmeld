Step 5: Create the Knowledge Base
===================================================

Every one of the smartest and most useful intelligent assistants in widespread use today relies on an underlying *knowledge base*. A knowledge base is a comprehensive repository that contains the universe of helpful information needed to understand requests and answer questions. The knowledge base is often the secret ingredient that makes a conversational interface appear surprisingly intelligent and versatile. For example, when asked to play a song, Alexa relies on a knowledge base which contains information about every track, artist and album accessible from your music streaming service. When you asked to send a text message, Siri enlists a knowledge base which knows about all of your important contacts. When IBM's Watson bested Jeopardy! grand champions, it leveraged an extensive knowledge base of Wikipedia facts. While some tasks, like setting a thermostat or appending a to-do list, may not warrant a knowledge base, the lion's share of today's commercial conversational applications owe their intelligence and utility to an underlying base of global knowledge.

In its most basic form, a knowledge base is simply a repository of objects of specified types. For example, a knowledge base that contains a large entertainment content catalog could include ``movie`` and ``tv_show`` among the types of its objects. A knowledge base that contains a directory of local businesses could have object types like ``restaurant``, ``hardware_store``, and so on. Objects typically have attributes which capture salient aspects of the concepts they describe. For example, a ``restaurant`` object might have attributes for ``address`` and ``phone_number``; a ``movie`` object might have attributes like ``cast_list``, ``runtime``, and ``release_date``.

MindMeld makes it straightforward to leverage a custom knowledge base in any application. The Question Answerer module of MindMeld provides powerful capabilities for creating a knowledge base in order to demonstrate intelligent behavior in your application. The Question Answerer can be used in a variety of ways, but in practice, conversational applications rely on this component and its underlying knowledge base for the four primary purposes listed below.

============================ ===
**Answer Questions**         The primary purpose of the Question Answerer is to identify and rank candidate answers to user questions. For example, if a user asks about good, nearby Italian restaurants, a knowledge base of local restaurants provides the best options.
**Validate Questions**       The knowledge base can also be used to inform a user if her question is out of scope. For example, if a user mistakenly asks to order a pizza from a coffee shop assistant, the knowledge base can help steer the user in the right direction.
**Disambiguate Entities**    Vague user requests often require clarification. A knowledge base can help disambiguate similar concepts. For example, if a user says 'play Thriller,' the Question Answerer could ask the user whether he means the bestselling album or the hit song.
**Suggest Alternatives**     When an exact answer cannot be found, the knowledge base can sometimes offer relevant suggestions. For example, if a user requests 'Star Wars Rogue One' and that movie is not available, the knowledge base could suggest other, available Star Wars titles.
============================ ===

Creating the knowledge base is the first step in utilizing the Question Answerer capabilities in MindMeld. The knowledge base contains one or more indexes. Each index holds a collection of objects of the same type. For example, one index may contain a collection of retail store locations and another index might contain a collection of products in a product catalog. Each index is built using data from one or more JSON files. These JSON data files can be stored either locally or remotely, for example in an `AWS S3 <https://aws.amazon.com/s3/>`_ bucket.

As shown in :doc:`Step 4 <04_define_the_dialogue_handlers>`, our example application provides information about Kwik-E-Mart stores, relying on a knowledge base which contains information about all retail store locations. In our example, let's assume that each store object contains the following attributes:

    * ``store_name``
    * ``open_time``
    * ``close_time``
    * ``address``
    * ``phone_number``

The corresponding JSON data file could look like this:

.. code-block:: javascript

  [
      {
        "store_name": "23 Elm Street",
        "open_time": "7am",
        "close_time": "9pm",
        "address": "100 Central Plaza, Suite 800, Elm Street, Capital City, CA 10001",
        "phone_number": "(+1) 415-555-1100"
      },
      {
        "store_name": "Pine and Market",
        "open_time": "6am",
        "close_time": "10pm",
        "address": "750 Market Street, Capital City, CA 94001",
        "phone_number": "(+1) 650-555-4500"
      }
  ]

Assuming that this file is named ``stores.json`` and is in the ``data`` subdirectory of the application root directory, you would create the knowledge base as follows.

.. warning::

   The QuestionAnswerer requires Elasticsearch. Make sure that Elasticsearch is running in a separate shell before invoking the QuestionAnswerer.

.. code-block:: shell

   cd $MM_APP_ROOT
   python

.. code:: python

   from mindmeld.components import QuestionAnswerer
   qa = QuestionAnswerer('.')
   qa.load_kb('my_app', 'stores', './data/stores.json')

This code loads the Question Answerer module from MindMeld, then loads the ``data/stores.json`` JSON file into the index named ``stores``. To check that your knowledge base was created successfully, use the Question Answerer to retrieve store information from your index:

.. code:: python

  stores = qa.get(index='stores')
  stores[0]


.. code:: console

  {
    "store_name": "Central Plaza Store",
    "open_time": "0800 hrs",
    "close_time": "1800 hrs",
    "address": "100 Central Plaza, Suite 800, Elm Street, Capital City, CA 10001",
    "phone_number": (+1) 100-100-1100
  }

Now that your knowledge base is created, the Question Answerer can leverage it in your dialogue state handling logic. See the :doc:`User Guide <../userguide/kb>` for more about how to use the Question Answerer to find answers to questions, validate user requests, disambiguate entities, and offer alternative suggestions.


