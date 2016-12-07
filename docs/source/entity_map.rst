Entity Map
=======================

.. raw:: html

    <style> .red {color:red} </style>

.. role:: red

.. _Building The Entity Recognizer: entity_recognition.html

Named Entity Recognition (NER) is a critical step in Deep-Domain Conversational AI. NER is used to highlight important portions of natural language input. Steps for training ML models for building Entity Recognizers are detailed in the `Building The Entity Recognizer`_ section. However, an important precursor to training Entity Recognition models is to define an **Entity Map**. The Entity Map allows us to -

* Specify a schema for the types and names of entities we wish to recognize.

  + In the music-assistant app, we might want to support "document-type" entities such as **artists**, **tracks**, **albums**, and **playlists**

* Map synonyms - :red:`"let's hear some yeezy"` should search for :red:`"Kanye West"`

* Specify which database fields to query given certain entity types.

  + E.g. query the **pubdate** fields given a **time-range** entity

* Optionally assign "roles" to entities for Role Classification.

Entities And Roles
------------------

.. raw:: html

    <style> .orange {background-color:orange} </style>

.. raw:: html

    <style> .aqua {background-color:#00FFFF} </style>

.. role:: orange
.. role:: aqua

Giving an entity a role is an optional additional classification step. Let's consider an example in the **change-alarm** intent -

* Change my alarm from :orange:`6 am` to :aqua:`7 am`

Here, :orange:`6 am` and :aqua:`7 am` are both **time** entities, but :orange:`6 am` has the role :orange:`oldtime` and :aqua:`7 am` has the role :aqua:`newtime`. The key principles in assigning roles (rather than treating them as separate entities) are -

* Avoiding splitting training data unnecessarily. In effect, all time entities strongly share features, and so the NER parser will perform better having them as one class.
* Some NER models (such as the **memm**) utilize local structure better than global (by global we mean depending on words/entities far away in the sentence). We can recognize :aqua:`7 am` as a :aqua:`newtime` with higher accuracy/confidence if we have a feature indicating that there is another time entity in the sentence located before it, separated by "to".

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
