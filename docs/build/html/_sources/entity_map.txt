Defining The Entity Map
=======================

Named Entity Recognition (NER) is a critical step in Deep-Domain Conversational AI. NER is used to highlight important portions of natural language input. Steps for training ML models for building Entity Recognizers are detailed in the `Building The Entity Recognizer` section. An important precursor to training those models is to define an "Entity Map". The Entity Map allows us to -

* Map synonyms: "let's hear some yeezy" should search for "Kanye West"
* Specify which database fields to query given certain entity types. E.g. for movies, query the pubdate fields given a time-range entity
* Add small-vocabulary gazetteer entries for named entity extraction. E.g. document type names for music search: artists, tracks, albums, and playlists
* Optionally assign "roles" to entities for Semantic Role Labeling.

Entities And Roles
******************

Giving an entity a role is an optional additional classification step.

* Change my alarm from `6 am` to `7 am`

Here, `6 am` and `7 am` are both `time` entities, but `6 am` has the role `oldtime` and `7 am` has the role `newtime`. The key principles in assigning roles (rather than treating them as separate entities) include -

* Avoiding splitting training data unnecessarily. In effect, All time entities strongly share features, and so the NER parser will perform better having them as one class.
* Most NER models utilize local structure better than global (by global we mean depending on words/entities far away in the sentence). We can recognize `7 am` as a `newtime` with higher accuracy/confidence if we have a feature indicating that there is another time entity in the sentence located before it, separated by "to"

Structure Of The Entity Map
***************************

At the top-level, the entity-map.json file structured as follows -

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
  | entity-name   | corresponds to a facet name in labeled queries                               |
  +---------------+------------------------------------------------------------------------------+
  | mode          | search, filter, sort, range, or no-kb, according to the purpose of the facet |
  +---------------+------------------------------------------------------------------------------+
  | one-per-query | boolean that means “for any query, there can only be one entity of this type”|
  +---------------+------------------------------------------------------------------------------+
  | numeric       | corresponds to a Duckling (numerical parser) type                            |
  +---------------+------------------------------------------------------------------------------+
  | roles         | array of role objects                                                        |
  +---------------+------------------------------------------------------------------------------+
  | text-map      | maps raw text from the input query into canonical form                       |
  +---------------+------------------------------------------------------------------------------+
  | clause-map    | maps a language pattern to a template for creating knowledge base queries    |
  +---------------+------------------------------------------------------------------------------+
  | conversions   | dictionary which maps conversion names to operations                         |
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

For a mapping that applies to all entity values in the entity block, a `*` wildcard can be used to encode the mapping -

.. code-block:: javascript

  "clause-map": {
    "*": "artist:{entity}"
  }
