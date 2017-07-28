Entity Resolver
===============

The :ref:`Entity Resolver <arch_resolver>` takes the entities recognized by the Entity Recognizer and transforms them into canonical forms that can be looked up in a Knowledge Base. For instance, the extracted entity "lemon bread" may get resolved to "Iced Lemon Pound Cake" and "SF" may get resolved to "San Francisco". In NLP literature, Entity Resolution or `Entity Linking <https://en.wikipedia.org/wiki/Entity_linking>`_ is the problem of identifying all terms that refer to the same real world entity. In Workbench, these real world entities are identified by a unique ID or a canonical name in the Entity Mapping files and the Knowledge Base.

For many entity types, an entity's canonical form corresponds with an object in the knowledge base. When this is the case, the goal of the entity resolver is to resolve to the specific object ID so that as the developer, you can directly use that ID to perform relevant actions. For example, consider the ``dish`` entity type for the food ordering use case. The entity "stir-fried Thai noodles" would be resolved to {cname: "Pad Thai", id: 123}. Then the developer can use the dish ID 123 to query the knowledge base, display results, or make an API call to place an order.

In many cases, a knowledge base could have multiple unique entries having the same canonical name, each of them with their own different IDs. This may arise in the music discovery use case when there are two different songs sung by different artists with the same song name. Similarly in the food ordering application, there may be multiple restaurants which serve a dish called "Pad Thai". Each one of these "Pad Thai" dishes would have a different ID in the Knowledge Base. When canonical names are the same but IDs are different, the entity resolver can rank one item above the other based on: 

1. **Synonym lists.** Entities with the same canonical name may have different properties (e.g. "House Salad" may be a "spinach salad" at one restaurant but a "tropical fruit salad" at another restaurant). These differences can be captured in the synonym list of each entry which is used by the entity resolver to select the appropriate result.

2. **A numeric sort factor.** Textual similarity is the primary factor in entity resolution, but when there are many items with similar textual similarity, a provided numeric value called the 'sort factor' is used to boost items that the user is most likely referring to. An object with a higher sort factor will be preferred, but the meaning of the sort factor differs across applications. For example, in a food ordering application, the score may be the rating of a restaurant. In a music discovery application, the score may be number of listens for an album.

For some entity types, the entity's canonical form does not correspond with a knowledge base (KB) object, but is simply a name that can be used to filter results or in natural language responses. In most cases, the entity mapping files for these entity types do not include an ID field since there is no connection to the KB. When no ID is provided in the entity mapping files, the resolution value from the entity resolver does not include an ID. Often times, these entity types refer to attributes of objects in the KB. For example, in the food ordering use case the ``cuisine`` entity type doesn't correspond to specific objects in the knowledge base. But resolving to the cuisine type "Thai" allows the developer to do a filter search on the cuisine attribute of the ``restaurant`` objects in the KB to select a list of relevant restaurants. 

To train the entity resolver, you must generate entity mapping files which include a synonym set for each entity as well as the optional sort factor. The details of the entity mappings and guidelines on synonym data collection are described in the following sections.


Entity Mapping
--------------

For each entity type, it is up to the developer to generate an entity mapping file which is used to train the entity resolver. The entity mapping is a json file with a list of dictionaries, one for each possible real world entity of the given entity type. Each dictionary refers to a single real world entity and contains:

==================== ===
**canonical name**   The standardized or official name of the real world entity. Textual similarity with the canonical name is one of the primary factors used in entity resolution.

**unique ID**        An optional unique identifier. If there are multiple entries in the mapping file with the same canonical name, the ID is necessary for uniquely identifying each entry. If an entity has a corresponding entry in the Knowledge Base, this ID should be the same as the ID of the KB entry. You can then use the resolved ID to query the KB for the appropriate entry.

**whitelist**        A list of synonyms. The whitelist is the most important component of the entity mapping file, because it allows the resolver to consistently resolve to a given entity that is often referred to by different terms.

**sort factor**      An optional numeric value. Entities with a higher sort factor will be ranked above those with a lower value and similar textual similarity.
==================== ===

In the food ordering blueprint where a ``dish`` is an entity type, the ``dish`` entity mapping file contains a list of all possible dishes that a user could order. Here is an example of what a couple of entries in the ``dish`` entity mapping file may look like.

.. code-block:: javascript

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
            "sort_factor": 3.5
        },
        {
            "cname": "Keema Naan",
            "id": "B01DN56EN0",
            "whitelist": [
                "keema stuffed naan",
                "lamb naan",
                "lamb stuffed naan"
            ],
            "sort_factor": 4.2
        },
        ...
    ]

This file should be saved as ``mapping.json`` in the corresponding entity folder. For example, the ``mapping.json`` file for the ``category`` entity should exist in the following location:

.. image:: /images/food_ordering_directory3.png
    :width: 300px
    :align: center


Collect the Data
----------------

The most important component of developing a production quality entity resolver is collecting a high quality and comprehensive set of synonyms. These synonyms allow the resolver to consistently resolve to a given entity that it is often referred to by different terms. Synonyms can be generated in-house or by using a crowdsourcing tool such as Mechanical Turk. For some use cases you may also be able to find existing synonym data sets. An important question is - what makes a synonym high quality? Here are some general synonym generation guidelines:

1. The best synonyms are textually different but semantically similar. For example, *Beef rice bowl* as a synonym for *Gyudon*. 

2. Include synonyms that are common alternate names for a given entity. For example, *phone* as a synonym for *cell*.

3. Add synonyms that include any useful information that is not reflected in more generic canonical names. For example, for a particular restaurant *Spinach Tomato Salad* would be a good synonym for *House Salad*.

4. Synonyms should be commonplace enough to be used in a conversational setting, rather than those which are highly contrived. For example, *cinnamon bun* may be a good synonym for *cinnamon roll*, but *cinnamon sugar sprinkled yeast-leavened dough in spiral form* would not be.

5. Don’t worry about generating exhaustive lists of possible misspellings or pluralization, since the resolver will handle those cases.

Collecting or generating sort factors is largely app specific. Use what makes the most sense for your use case. In most cases, these values are part of an existing dataset. For example, for the food ordering use case it could be something like the rating of a restaurant, the number of reviews for a restaurant, or the proximity of the restaurant to some location. Often times the sort factor is a value that can be scraped from a publically available dataset.

The metric you decide to use as a sort factor will be scaled differently for different apps. For example, a restaurant rating will be between 1 and 5, but the number of song listens may be between 1 and over a hundred million. If you notice that the sort value is outweighting good text relevance matches for your use case, you may want to scale the sort factor to a lower max value. On the other hand, you can slightly boost the weight of the sort factor in ranking by scaling to a higher max value.

Configure the Entity Resolver (optional)
----------------------------------------

There are two options for entity resolution:

1. Use an advanced text similarity model (strongly recommended, requires Elasticsearch)
2. Use a simple exact match model (no requirements)

Elasticsearch is a full-text search and analytics engine that the Entity Resolver leverages for information retrieval. For more details on setting up Elasticsearch consult the :doc:`Getting Started guide <getting_started>`.

.. note::

   If Elasticsearch is set up, Workbench's advanced information retrieval based entity resolver is used by default, there is nothing you have to do.

If you don't want to use Elasticsearch, Workbench provides a simple baseline version of Entity Resolution which only resolves to an object if the text is an exact match on the canonical name or one of its synonyms. To use this version, add the following to your app config (``config.py``) located in the top level of your app folder:

.. code-block:: python

    ENTITY_RESOLUTION_CONFIG = {
        'model_type': 'exact_match'
    }

It is highly recommended that you install Elasticsearch to leverage Workbench's default entity resolution model which uses advanced text relevance techniques to guarantee a production-level accuracy. The exact match model is merely provided as a fall-back option to get an end-to-end app running without Elasticsearch. However, this approach isn't optimal, and hence not recommended for a broad vocabulary conversational app.

Run the Entity Resolver
-----------------------

Once all of the Entity Mapping files are generated, it will be trained and used as part of the NLP pipeline. Using :meth:`nlp.build()` will fit the resolver and :meth:`nlp.process()` will include the resolved entities in the result. To try out the resolver as a stand alone component, you can train it as shown below.

.. code-block:: python

  >>> from mmworkbench import configure_logs; configure_logs() 
  >>> from mmworkbench.components.nlp import NaturalLanguageProcessor
  >>> nlp = NaturalLanguageProcessor(app_path='food_ordering')
  >>> nlp.domains['ordering'].intents['build_order'].build()
  >>> er = nlp.domains['ordering'].intents['build_order'].entities['dish'].entity_resolver
  >>> er.fit()

When using the fit method for the first time, the Elasticsearch index will be created and all of the objects will be uploaded, so this may take some time depending on the size of your data, your network speed, and whether your code and Elasticsearch server are running on the same machine. Subsequent calls to *er.fit()* will update the existing index rather than creating a new one from scratch to improve speed. This means that new objects will be added, and objects with the same ID will but updated, but no objects will be deleted. If you would like to delete objects, you can fully recreate the index from scratch by running a clean fit as follows.

.. code-block:: python

  >>> er.fit(clean=True)

Unlike the other NLP components, *er.dump()* and *er.load()* do not do anything since there are no model weights to be saved to disk. Everything needed exists in the Elasticsearch index and the entity mapping files.

Once the resolver is fit, you can pass Entity objects to test the Entity Resolver as follows.

.. code-block:: python

  >>> from mmworkbench.core import Entity
  >>> er.predict(Entity(text='gluten free pepperoni pizza', entity_type='dish'))

    [{'cname': 'Pepperoni Pizza (Gluten Free)',
      'id': 'B01D8TCLJ2',
      'score': 119.62746,
      'top_synonym': 'gluten free pepperoni pizza'},
     {'cname': 'Margherita Pizza (Gluten Free)',
      'id': 'B01D8TCRWI',
      'score': 38.989628,
      'top_synonym': 'gluten-free margherita pizza'},
     {'cname': 'Barbecued Chicken Pizza (Gluten Free)',
      'id': 'B01D8TCCK0',
      'score': 35.846962,
      'top_synonym': 'gluten-free barbeque chicken pizza'},
     {'cname': 'Plain Cheese Pizza (Gluten Free)',
      'id': 'B01D8TCJEE',
      'score': 35.43069,
      'top_synonym': 'cheese pizza gluten free'},
     {'cname': 'Sausage and Mushroom Pizza (Gluten Free)',
      'id': 'B01D8TD5T2',
      'score': 35.094833,
      'top_synonym': 'gluten-free sausage and mushroom pizza'},
     {'cname': 'Four Cheese White Pizza (Gluten Free)',
      'id': 'B01D8TD9DO',
      'score': 31.833534,
      'top_synonym': 'Four Cheese White Pizza (Gluten Free)'},
     {'cname': 'The Truck Stop Burger',
      'id': 'B01DWO5N5W',
      'score': 28.069,
      'top_synonym': 'gluten free burger'},
     {'cname': 'Pesto with Red Pepper and Goat Cheese (Gluten Free)',
      'id': 'B01D8TCA48',
      'score': 28.018322,
      'top_synonym': 'Pesto with Red Pepper and Goat Cheese (Gluten Free)'},
     {'cname': 'Gluten Free Waffle',
      'id': 'B01GXT877O',
      'score': 27.94693,
      'top_synonym': 'Gluten Free Waffle'},
     {'cname': 'Lamb Platter',
      'id': 'B01CRF8WAK',
      'score': 27.913887,
      'top_synonym': 'gluten free lamb platter'}]

The Entity Resolver returns a ranked list of the top 10 canonical forms for each recognized entity. For most cases, taking the top 1 is sufficient, but in some cases it may be beneficial to look at other options if there are other constraints that the top few do not satisfy. The resolver returns:

==================== ===
**canonical name**   The name used to refer to the real world entity.

**unique ID**        The ID as listed in the entity mapping file which should correspond with a Knowledge Base object.

**score**            A score which indicates the strength of the match. This score is a relative value (higher scores are better). It is not normalized accross all entity types or queries.

**top synonym**      The synonym in the whitelist of this canonical form that most closely matched the user's query.

**sort factor**      If the sort factor was provided in the entity mapping file, it will also be returned.
==================== ===
