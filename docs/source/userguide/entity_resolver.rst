.. meta::
    :scope: private

Entity Resolver
===============

The MindMeld Entity Resolver takes the entities recognized by the Entity Recognizer and transforms them into canonical forms that can be looked up in a Knowledge Base. For instance, the extracted entity "lemon bread" may get resolved to "Iced Lemon Pound Cake" and "SF" may get resolved to "San Francisco". In NLP literature, Entity Resolution or `Entity Linking <https://en.wikipedia.org/wiki/Entity_linking>`_ is the problem of identifying all terms that refer to the same real world entity. In Workbench, these real world entities are identified by either a unique id or a canonical name in both the Entity Map and Knowledge Base.

For many entity types, an entity's canonical form corresponds with a document in the knowledge base. When this is the case, the goal of the Entity Resolver is to resolve to the specific document id so that as the developer, you can directly use that id to query the Knowledge Base or make an API call. For example, consider the *dish* entity type for the food ordering use case. The entity "stir-fried Thai noodles" would be resolved to {cname: "Pad Thai", id: 123}. Then the developer can use the dish id 123 to query the knowledge base, display results, or make an API call to place an order.

It is important to note that there are multiple restaurants which serve a dish called "Pad Thai". Each one of these "Pad Thai" dishes have a different id in the knowledge base. When canonical names are the same but ids are different, the Entity Resolver can rank one item above the other based on: 

1. **Synonym lists.** Entities with the same canonical name may have different properties (e.g. "House Salad" may be a "spinach salad" at one restaurant but a "tropical fruit salad" at another restaurant). These differences can be captured in the synonym list of each entry which is used by the Entity Resolver to select the appropriate result.

2. **A numeric value.** Textual similarity is the primary factor in entity resolution, but when there are many items with similar textual similarity, the numeric value is used to boost the items that the user is most likely referring to. A document with a higher numeric value will be preferred, but the meaning of the numeric value differs across applications. For example, in a food ordering application, the score may be the rating of a restaurant. In a music discovery application, the score may be number of listens for an album.

For some entity types, the entity's canonical form does not correspond with a knowledge base document, but is simply a name that can be used to filter results or in natural language responses. For example, in the food ordering use case the *cuisine* entity type doesn't correspond to specific documents in the knowledge base. But resolving to the cuisine type "Thai" allows the developer to do a filter search against the knowledge base to select a list of relevant restaurants.

To train the Entity Resolver, you must generate Entity Mapping files which include a synonym set for each entity as well as the optional numeric value. The details of the Entity Mappings and guidelines on synonym data collection are described in the following sections.


Entity Mapping
--------------

For each entity type, it is up to the developer to generate an Entity Mapping file which is used to train the Entity Resolver. The Entity Mapping is a json file with a list of documents, one for each real world entity that could be resolved to. Each document refers to a single real world entity and contains:

==================== ===
**canonical name**   The name used to refer to the real world entity. Note if canonical names are unique a separate unique id may not be needed. Textual similarity with the canonical name is one of the primary factors of entity resolution.

**unique id**        A unique identifier (optional). If a corresponding entry exists in the knowledge base, this id should be the same as the KB document id. In cases where there are no corresponding documents in the knowledge base and there are no duplicate canonical names, an id is not needed.  

**whitelist**        A list of synonyms. The whitelist is the most important component of the entity mapping file, because it allows the resolver to consistently resolve to a given entity that it is often referred to by different terms. Textual similarity with synonyms in the whitelist is one of the primary factors of entity resolution.

**numeric value**    An optional numeric value. Entities with a higher numeric value will be ranked above those with a lower value and similar textual similarity.
==================== ===

In the food ordering blueprint where a dish is an entity type, the dish entity mapping file contains a list of all possible dishes that a user could order. Here is an example of what a couple of entries in the dish entity mapping file may look like.

.. code-block:: json

    [
        {
            "cname": "Baigan Bharta",
            "id": "B01DN55TFO",
            "whitelist": [
                "Mashed eggplant and spiced tomato",
                "Specialty Spiced Eggplant Curry",
                "Seasoned Roasted Eggplant Mash",
                "Eggplant curry",
                "Spicy roasted eggplant dish"
            ],
            "value": 3.5
        },
        {
            "cname": "Keema Naan",
            "id": "B01DN56EN0",
            "whitelist": [
                "keema stuffed naan",
                "lamb naan",
                "lamb stuffed naan"
            ],
            "value": 4.2
        },
        ...
    ]

This file should be saved as *mapping.json* and exist in the corresponding entity folder. For example, the mapping.json file for the *store_name* entity should exist in the following location:

.. image:: /images/directory5.png
    :width: 500px
    :align: center


Data Collection
---------------

The most important component of developing a production quality entity resolver is collecting a high quality and dense set of synonyms. These synonyms allow the resolver to consistently resolve to a given entity that it is often referred to by different terms. Synonyms can be generated in house or by using a crowdsourcing tool such as Mechanical Turk. For some use cases you may also be able to find existing synonym data sets. An important question is - what makes a synonym high quality? Here we will give some general synonym generation guidelines.

1. The best synonyms are textually different but semantically similar. For example, *Beef rice bowl* as a synonym for *Gyudon* 

2. Include synonyms that are common alternate names for a given entity. For example, *phone* as a synonym for *cell*

3. Add synonyms that include any useful information that is not reflected in more generic canonical names. For example, for a particular restaurant *Spinach Tomato Salad* would be a good synonym for *House Salad*

4. Don’t worry about generating exhaustive lists of possible misspellings or pluralization, since the resolver will handle those cases


Entity Resolution Configuration
-------------------------------

There are two options for entity resolution:

1. Use an advanced text similarity model (strongly recommended, requires Elasticsearch)
2. Use a baseline exact match model (no requirements)

Elasticsearch is a full-text search and analytics engine that the Entity Resolver leverages for information retrieval. For more details on setting up Elasticsearch consult the :doc:`Getting Started guide <getting_started>`. **If Elasticsearch is set up, Workbench's advanced information retrieval based entity resolver is used by default, there is nothing you have to do.**

If you don't want to use Elasticsearch, Workbench provides a simple baseline version of Entity Resolution which only resolves to a document if the text is an exact match on the canonical name or one of its synonyms. To use this version, add the following to your *config.py* located in the top level of your app folder:

.. code-block:: python

    ENTITY_RESOLUTION_CONFIG = {
        'model_type': 'exact_match'
    }

Again, the above exact match model is *not* recommended as Workbench will use the more advanced text relevance based Entity Resolver by default which significantly improves performance. However, if you have no way of getting Elasticsearch set up it is a possible alternative.

Trying it out
-------------

Once all of the Entity Mapping files are generated, **nlp.build()** will build the entity resolver. Note that the first time you build the Entity Resolver, it may take some time if your data set is large and your Elasticsearch server is not on the same machine as your code.

Then, **nlp.process()** will include a list of resolved entities

TODO: finish this section



