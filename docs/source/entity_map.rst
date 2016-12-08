Entity Map
=======================

.. raw:: html

    <style> .red {color:red} </style>

.. role:: red

.. _Building The Entity Recognizer: entity_recognition.html

Named Entity Recognition (NER) identifies portions of natural language input that are important to your application. For example, in a music-assistant app, any **artist**, **track**, **album**, and **playlist** might need to be recognized as an entity of type **document**.

Before you can train ML models for building Entity Recognizers (see `Building The Entity Recognizer`_ ), you must define an **Entity Map** in order to

* specify a schema for the types and names of entities to recognize

* map synonyms — :red:`"let's hear some yeezy"` should search for :red:`"Kanye West"`

* specify which database fields to query given certain entity types

  + e.g., query the **pubdate** fields given a **time-range** entity

* optionally assign "roles" to entities (see the next section)

Entities And Roles
------------------

.. raw:: html

    <style> .orange {background-color:orange} </style>

.. raw:: html

    <style> .aqua {background-color:#00FFFF} </style>

.. role:: orange
.. role:: aqua

Consider this example in a **change-alarm** intent:

* Change my alarm from :orange:`6 am` to :aqua:`7 am`

Here, :orange:`6 am` and :aqua:`7 am` are both **time** entities, but they literally play different roles in constructing the meaning of the sentence.

Classifying :orange:`6 am` with the role of :orange:`oldtime` and :aqua:`7 am` with the role :aqua:`newtime` captures the distinction between them.

Taking this extra step is preferable to treating :orange:`oldtime` and :aqua:`newtime` as separate kinds of entities. This is because all time entities strongly share features, which means that the NER parser performs better when they all belong to one class.

Be aware that some NER models (such as the **memm**) utilize local structure better than global (by global we mean depending on words/entities far away in the sentence). We can recognize :aqua:`7 am` as a :aqua:`newtime` with higher accuracy/confidence if we have a feature indicating that there is another time entity in the sentence located before it, separated by "to".

Apply these principles when assigning roles:

* Avoid splitting training data unnecessarily.
* Consider the interaction between roles, features, and NER model performance characteristics.

Structure Of The Entity Map
---------------------------

At the top-level, the **"entity-map.json"** file is structured as follows -

.. code-block:: javascript

  {
    "entities": [
      {
        "entity-name": "type",
         ...
      },
      {
        "entity-name": "title",
         ...
      },
      {
        "entity-name": "artist",
         ...
      }
    ]
  }

Entity configuration objects contain the following fields -

  +---------------+------------------------------------------------------------------------------+
  | Field         | Description                                                                  |
  +===============+==============================================================================+
  | entity-name   | corresponds to an entity name in labeled queries                             |
  +---------------+------------------------------------------------------------------------------+
  | mode          | search, filter, sort, range, or no-kb, according to the purpose of the entity|
  +---------------+------------------------------------------------------------------------------+
  | one-per-query | boolean that means “for any query, there can only be one entity of this type”|
  +---------------+------------------------------------------------------------------------------+
  | numeric       | corresponds to a :doc:`Duckling </mallard>` (numerical parser) type          |
  +---------------+------------------------------------------------------------------------------+
  | roles         | array of role objects                                                        |
  +---------------+------------------------------------------------------------------------------+
  | text-map      | maps raw text from the input query into canonical form                       |
  +---------------+------------------------------------------------------------------------------+
  | clause-map    | maps a language pattern to a template for creating knowledge base queries    |
  +---------------+------------------------------------------------------------------------------+

Entity objects with a TextMap -

.. code-block:: javascript

  {
    "entity-name": "action",
    "mode": "no-kb",
    "text-map": {
      "when did": "show-year",
      "tell me the year": "show-year",
      ...
      "who": "show-artist",
      "which singer": "show-artist",
      ...
    }
  }

Entity objects with a ClauseMap -

.. code-block:: javascript

  {
    "entity-name": "type",
    "one-per-query": true,
    "mode": "filter",
    "clause-map": {
      "cover": "category:track",
      "cover song": "category:track",
       ...
    }
  }

TextMap and ClauseMap apply to roles in a similar vein -

.. code-block:: javascript

  {
    "entity-name": "hits",
    "roles": [
      {
        "name": "popularity",
        "mode": "sort",
        "clause-map": {
          "popular": "popularity:desc",
          "most obscure": "popularity:asc",
           ...
        }
      },
      ...
    ]
  }

For a mapping that applies to all entity values in the entity block, a **`*`** wildcard can be used to encode the mapping -

.. code-block:: javascript

  "clause-map": {
    "*": "artist:{entity}"
  }
