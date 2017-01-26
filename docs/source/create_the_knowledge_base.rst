Step 5: Create the Question Answerer Knowledge Base
===================================================

The smartest and most useful intelligent assistants in widespread use today all rely on an underlying knowledge base. A knowledge base is comprehensive repository containing the universe of helpful information which is essential to understand requests and answer questions. In many cases, the knowledge base is the secret ingredient which makes a conversational interface appear surprisingly intelligent and versatile. For example, when you ask Alexa to play a song, it relies on a knowledge base which contains information about every track, artist and album in your music streaming service. When you ask Siri to send a text message, it enlists a knowledge base which knows about all of your important contacts. When IBM's Watson bested Jeopardy! grand champions, it leveraged an extensive knowledge base of Wikipedia facts. While some tasks, like setting a thermostat or appending a to-do list, may not warrant a knowledge base, the lion's share of today's commercial conversational applications owe their intelligence and utility to an underlying base of global knowledge.

In its most basic form, a knowledge base is simply a repository of objects of specified types. For example, each object could represent a film in a large content catalog or a restaurant in a directory of local businesses. Each object typically has multiple attributes which capture important properties associated with each object. For example, a restaurant object might have attributes for the address and phone number; a film object might have attributes which list the cast members, the runtime and the release date.

MindMeld Workbench makes it straightforward to leverage a custom knowledge base in any application. The Question Answerer module of Workbench provides a set of powerful capabilities for creating a knowledge base in order to demonstrate intelligent behavior in your application. The Question Answerer can be used in a variety of ways, but in practice, conversational applications rely on this component and its underlying knowledge base for the four primary purposes listed below. 

============================ ===
**Answer Questions**         The primary purpose of the Question Answerer is to identify and rank candidate answers for user questions. For example, if a user asks about good, nearby Italian restaurants, a knowledge base of local restaurants provides the best options.
**Validate Questions**       The knowledge base can also be used to inform a user if their question is invalid. For example, if a user mistakenly asks to order a pizza from a coffee shop assistant, the knowledge base can help steer the user in the right direction.
**Disambiguate Entities**    Vague user requests might often require clarification. A knowledge base can help disambiguate similar concepts. For example, if a user says 'play Thriller', the Question Answerer could ask the user if they mean the bestselling album or the hit song.
**Suggest Alternatives**     When an exact answer cannot be found, the knowledge base can sometimes offer relevant suggestions. For example, if a user requests 'Star Wars Rogue One' and it is not yet available, the knowledge base could suggest other available Star Wars titles.
============================ ===

Creating the knowledge base is the first step in utilizing the Question Answerer capabilities in Workbench. The knowledge base can contain one or more indexes. Each index is intended to hold a collection of objects of the same type. For example, one index may contain a collection of retail store locations and another index might contain a collection of products in a product catalog. Each index is built using data from one or more JSON files. These JSON data files can be either stored locally or available remotely in an `AWS S3 <https://aws.amazon.com/s3/>`_ bucket, for example.

As we saw in the preceding section, our example application can provide information about Kwik-E-Mart stores, and it relies on a knowledge base which contains information about all retail store locations. In our example, let's assume that each store object contains the following attributes

    * store_name
    * open_time
    * close_time
    * address
    * phone_number

In this case, the corresponding JSON data file could be represented as shown below.

.. code-block:: javascript

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
  ...

Assuming this file is named 'stores.json' and it is located in the 'data' subdirectory of the root directory, MindMeld Workbench can create the knowledge graph as follows.

.. code:: python

  > from mmworkbench import QuestionAnswerer
  > qa = QuestionAnswerer('stores', 'data/stores.json')

This code snippet simply loads the Question Answerer module from Workbench and then loads the :keyword:`data/stores.json` JSON file into the index named :keyword:`stores`. To check that your knowledge base was created successfully, you can use the Question Answerer to retrieve store information from your index:

.. code:: python

  > stores = qa.get(index='stores')
  > stores[0]
  {
    "store_name": "Central Plaza Store",
    "open_time": 0800 hrs,
    "close_time": 1800 hrs,
    "address": "100 Central Plaza, Suite 800, Elm Street, Capital City, CA 10001",
    "phone_number": (+1) 100-100-1100
  }

As you can see, your knowlege base is now created and it can be leveraged by the Question Answerer in your dialogue state handling logic. Refer to the :ref:`User Guide <userguide>` for more detailed information on how the Question Answerer can be used to find answers to questions, validate user requests, disambiguate entities and offer alternative suggestions.     

