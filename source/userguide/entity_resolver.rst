Working with the Entity Resolver
================================

The :ref:`Entity Resolver <arch_entity_model>`

 - is run as the fifth step in the :ref:`natural language processing pipeline <instantiate_nlp>`
 - uses information retrieval techniques to perform entity resolution (also called entity linking)
 - is trained per entity type, using an entity mapping file that contains synonyms for all possible entity instantiations

Every MindMeld app has one entity resolver for every entity type. The mapping files are found in their respective entity folders.

.. note::

    This is an in-depth tutorial to work through from start to finish. Before you begin, read the :ref:`Step-by-Step Guide <quickstart>`, paying special attention to the :ref:`Entity Resolution <entity_resolution>` section.

Understanding the entity resolver
---------------------------------

In NLP literature, entity resolution or `entity linking <https://en.wikipedia.org/wiki/Entity_linking>`_ is the problem of identifying all terms that refer to the same real-world entity.

In MindMeld, two different kinds of entities require resolution:

  - *entities that refer to an object* in the knowledge base — these resolve to an *id, canonical name* pair consisting of the unique ID of the object in the knowledge base, and the :term:`canonical name` for the object, respectively
  - *entities that refer to a property* of an object in the knowledge base — these resolve to a *id, canonical name* pair consisting of an optional user-specified ID and the :term:`canonical name` for the property respectively

For example, in a food-ordering app:

  - the entity resolver could determine that the entity 'lemon bread' signifies the 'Iced Lemon Pound Cake' on the Buttery Bakery menu, which is represented by the object whose unique ID is '4567.' The result is that 'lemon bread' resolves to the dictionary ``{ 'id': '4567', cname: 'Iced Lemon Pound Cake' }``.
  - the entity 'Siamese' could resolve to the dictionary ``{ 'id': None, cname: 'Thai' }``. Multiple 'restaurant' objects could have that property/value combination. The property has no ID, but in the 'restaurant' index in the knowledge base, every object does have an ID. See the `Food Ordering blueprint <https://mindmeld.com/docs/blueprints/food_ordering.html#knowledge-base>`_.

This discussion assumes that the app has a knowledge base. For apps with no knowledge base, entity resolution is similar but no objects are involved.

The Entity Resolver has two main tools at its disposal:

1. **Synonym lists**

    Even entities of the same type and with the same canonical name can differ in their properties. For example, there could be two dishes whose canonical name is 'House Salad,' but they are from different restaurants, and while one is a 'spinach salad,' the other is a 'tropical fruit salad.' For these two entities, then, the ``restaurant`` and ``salad_type`` properties would differ. MindMeld captures these differences in *synonym lists* (described in the next section) which the entity resolver uses to select the most appropriate result for a given query.

2. **A numeric sort factor**

    Textual similarity is the primary factor in entity resolution. When many entities are highly similar textually, MindMeld uses a numeric value called the *sort factor* to boost the likelier ones. The sort factor means different things in different applications. A food ordering application might use restaurant ratings, while a music discovery application might use number of listens for an album, and so on.

    For music discovery, the sort factor could help the app choose between two different songs, sung by different artists, that have the same song title. For food ordering application, the sort factor could help the app choose between multiple 'Pad Thai' dishes, each from a different restaurant, and each with a different ID in the Knowledge Base.

These tools help when the knowledge base has multiple unique entries with different IDs but the same canonical name, and the entity resolver needs a way to make the best choice among them.

In the knowledge base, multiple objects can have the same canonical name, but every object has a unique ID. When an entity resolves to an object ID, the developer can use that ID directly to perform actions like querying the knowledge base, displaying results, or making an API call to place an order.

An example of an entity type that works this way is the ``dish`` entity type for the food ordering use case. The entity 'stir-fried Thai noodles,' an entity of type ``dish``, might resolve to the object ``{cname: "Pad Thai", id: 123}``. Then the developer can use the dish ID, ``123``, to perform desired actions.

When an entity resolves to a particular text value (contained in the cname field) for an object property, the developer can simply treat the entity as a name for use in natural language responses or in filtering results.

For example, in food ordering, the ``cuisine`` entity type does not correspond to any objects in the knowledge base. But resolving to the cuisine type 'Thai' allows the developer to do a filter search on the cuisine attribute of the ``restaurant`` objects to obtain a list of relevant restaurants. In this scenario, there is no need for the entity resolver to include an ID in the resolution value.

The entity resolver is trained using *entity mapping files*, described in the next section.

Prepare data for the Entity Resolver
------------------------------------

The most important task when you are developing a production-quality entity resolver is to collect a high-quality and comprehensive set of synonyms, and place those synonyms into entity mapping files, which is where MindMeld can use them.

This section explains how entity mapping files and synonym files are structured, and how they work. Each sub-section concludes with instructions for generating the files.

Entity mapping files come first, since they provide the framework into which the synonyms fit. 

Generate entity mapping files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every MindMeld app has an entity mapping file for each entity type.

An entity mapping file is a JSON file containing a dictionary whose purpose is to provide information about entities of the given type, and, where applicable, about how the entity type corresponds to a knowledge base object type. Entity mapping files are used to train the entity resolver.

At the top level, the entity mapping file has the three key/value pairs described in the table below.

===================================== =================== ===
**parameter name**                    **JSON field name** **description**

entities                              ``entities``        A list of dictionaries, one for each possible real-world instance of the given entity type. See the table below for more details.

knowledge base index name (optional)  ``kb_index_name``   Name of the knowledge base index that contains information about this entity type. For example, information about the ``dish`` entity type may be stored in the objects in the ``menu_items`` index in the knowledge base.

knowledge base field name (optional)  ``kb_field_name``   Name of the knowledge base field name that this entity type corresponds to. In other words, the entity text is contained in this field of a knowledge base object. For example, the text span captured in entity type ``dish`` describes the dish name in a user's request and corresponds to the knowledge base text field ``name`` of the index ``menu_items``. The knowledge base index and the knowledge base field parameters together describe how the NLP entity type corresponds to a knowledge base object type. When specified, the synonym whitelist for such knowledge base-linked entities is accessible by the Question Answerer. It can then leverage that information when formulating knowledge base filtered searches to disambiguate entities with custom constraints. See `Consider context-aware entity resolution`_ section

===================================== =================== ===

One key, ``entities``, has a list of dictionaries for its value. Each of those dictionaries refers to a single real-world entity, and has the attributes described in the table below.

====================== =================== ===
canonical name         ``cname``           The standardized or official name of the real-world entity. Textual similarity with the canonical name is one of the primary factors used in entity resolution.

unique ID              ``id``              An optional unique identifier. If there are multiple entries in the mapping file with the same canonical name, the ID is necessary for uniquely identifying each entry. If an entity has a corresponding entry in the Knowledge Base, this ID should be the same as the ID of the KB entry. You can then use the resolved ID to query the KB for the appropriate entry.

whitelist              ``whitelist``       A list of synonyms. The whitelist is the most important component of the entity mapping file, because it allows the resolver to consistently resolve to a given entity that is often referred to by different terms.

sort factor (optional) ``sort_factor``     An optional numeric value. Entities with a higher sort factor will be ranked above those with a lower value and similar textual similarity.
====================== =================== ===

In the following excerpt of an entity mapping file from the food ordering blueprint, ``dish`` is an entity type, and each dictionary in the entity mapping file refers to a dish that users can order.

.. code-block:: javascript

    {
      "kb_index_name": "menu_items",
      "kb_field_name": "name",
      "entities": [

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
    }

Create an entity mapping file for each entity type in your app. When you first create them, the ``whitelist`` property of each entity object will have an empty list as its value. In the next section, you will populate that list with synonyms.

Save each entity mapping file as ``mapping.json`` in the corresponding entity folder. For example, the ``mapping.json`` file for the ``category`` entity should exist in the following location:

.. image:: /images/food_ordering_directory3.png
    :width: 300px
    :align: center

Generate synonyms
^^^^^^^^^^^^^^^^^

Synonyms enable the resolver to consistently resolve to an entity that users refer to using varied language. You can generate synonyms in-house, or crowdsource them from a service like Mechanical Turk. For some use cases, existing synonym data sets may be available.

To ensure that you get high-quality synonyms, observe the following guidelines:

1. The best synonyms are textually different but semantically similar. For example, *beef rice bowl* as a synonym for *gyudon*.

2. Include synonyms that are common alternate names for a given entity. For example, *phone* as a synonym for *cell*.

3. Add synonyms that include any useful information that more generic canonical names lack. For example, *Spinach Tomato Salad* would be a good synonym for *House Salad*.

4. Synonyms should be commonplace enough to be used in a conversational setting. Avoid contrived-sounding synonyms. For example, *cinnamon bun* may be a good synonym for *cinnamon roll*, but *cinnamon sugar sprinkled yeast-leavened dough in spiral form* would not be.

5. Don’t worry about generating exhaustive lists of possible misspellings or pluralization. The resolver will handle those cases.

Create the synonyms and make them available to your MindMeld app, going entity type by entity type. The general process to follow (subject to refinement and even some automation) is this:

#. Choose an entity type

#. Open the ``mapping.json`` file for the entity type in a text editor

    a. Choose a target entity from ``entities`` listed in the file

    b. Locate the ``whitelist`` key in the dictionary that represents the target entity

    c. Create synonyms for the target entity

    d. Add each synonym to the list which is the value for ``whitelist``

    e. Choose a new target entity (continue until you have covered all the entities for the entity type)

#. Choose a new entity type (stop when you have covered all the entity types for the app) 


Train the entity resolver
-------------------------

The Entity Resolver uses the `Elasticsearch <https://www.elastic.co/products/elasticsearch>`_ full-text search and analytics engine for information retrieval. This is required by the advanced text similarity model MindMeld uses by default. Once Elasticsearch is up and running, no configuration needed is needed within MindMeld. To learn how to set up Elasticsearch, see the :doc:`Getting Started guide <getting_started>`.

.. note::

   If you choose not to use Elasticsearch (not recommended), MindMeld provides a simple baseline version of entity resolution as a fallback. See :ref:`About the Exact Match text similarity model <exact_match>`.

Once all of the entity mapping files are generated, you can either (1) train the resolver as a standalone component, or (2) build the whole NLP pipeline, which trains the resolver along with the other components.

Train the resolver alone
^^^^^^^^^^^^^^^^^^^^^^^^

To train the resolver as a standalone component, adapt the following snippet to the particulars of your app:

.. code-block:: python

  from mindmeld import configure_logs; configure_logs()
  from mindmeld.components.nlp import NaturalLanguageProcessor
  nlp = NaturalLanguageProcessor(app_path='food_ordering')
  nlp.domains['ordering'].intents['build_order'].build()
  er = nlp.domains['ordering'].intents['build_order'].entities['dish'].entity_resolver
  er.fit()

Train the resolver by building the NLP pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Build the NLP pipeline using :meth:`NaturalLanguageProcessor.build()`. Among its other tasks, this fits the resolver. When you run :meth:`NaturalLanguageProcessor.process()`, the NLP pipeline includes the resolved entities in its results.

About the `EntityResolver.fit()` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When :meth:`EntityResolver.fit()` method is run for the first time, MindMeld creates the Elasticsearch index and uploads all the objects. How long this takes depends on the size of your data, your network speed, and whether your code and Elasticsearch server are running on the same machine.

For the sake of speed, subsequent calls to :meth:`EntityResolver.fit()` update the existing index rather than creating a new one from scratch. This means that new objects are added, and objects with existing IDs are updated, but no objects are deleted. If you wish to delete objects, fully recreate the index from scratch by running a clean fit as follows.

.. code-block:: python

   er.fit(clean=True)

Unlike the other NLP components, *EntityResolver.dump()* and *EntityResolver.load()* do not do anything since there are no model weights to be saved to disk. Everything needed exists in the Elasticsearch index and the entity mapping files.

Run the entity resolver
-----------------------

Once the resolver has been fit, test the entity resolver by passing ``Entity`` objects as follows.

.. code-block:: python

  from mindmeld.core import Entity
  er.predict(Entity(text='gluten free pepperoni pizza', entity_type='dish'))


.. code-block:: console

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

Each entry in the list of resolved entities contains:

==================== ===
**canonical name**   The name used to refer to the real-world entity.

**unique ID**        The ID as listed in the entity mapping file which should correspond with a Knowledge Base object.

**score**            A score which indicates the strength of the match. This score is a relative value (higher scores are better). It is not normalized accross all entity types or queries.

**top synonym**      The synonym in the whitelist of this canonical form that most closely matched the user's query.

**sort factor**      If the sort factor is provided in the entity mapping file, it is also returned.
==================== ===

The Entity Resolver returns a ranked list of the top ten canonical forms for each recognized entity. Taking the top-ranked value usually works, but in some cases it's preferable to look at other options in the ranked list. Here are two examples:

1. When building a browsing functionality in your app, you might want to offer the user a choice of the top three resolved values.
2. Suppose the user has provided some constraints in a previous query. The entity resolver has no access to this previous context at resolution time, so the top-ranked result may not satisfy previously defined constraints. Here, you may want to look deeper into the ranked list.

The next section explains how to handles scenarios like this.

Consider context-aware entity resolution
----------------------------------------

While the Entity Resolver finds the best-matching canonical values based on text relevance and numeric sort factors, you might sometimes want application-specific constraints to influence how the recognized entities are resolved. These constraints could come from proximity information, business logic, the data model hierarchy, or elsewhere. For instance:

* resolve to the dish whose name best matches ``Pad Thai``, within a selected restaurant
* resolve to the nearest ``Best Buy``
* resolve to the product that best matches ``cotton long sleeve shirts``, and that is on sale

These scenarios are examples of *context-aware resolution*. Deciding whether context-aware resolution is required, and if so under what circumstances, is important aspect of designing your app.

There are two ways to accomplish context-aware resolution in MindMeld:

 - In your :doc:`dialogue state handlers <../quickstart/04_define_the_dialogue_handlers>`, iterate through the ranked list to find the first entry that satisfies the constraints.
 - Use the :doc:`Question Answerer <kb>` to do a filtered search against the knowledge base to disambiguate entities with contextual constraints.

.. _exact_match:

About the Exact Match text similarity model
-------------------------------------------

If you choose not to use Elasticsearch, MindMeld provides a simple baseline version of entity resolution. This Exact Match Model only resolves to an object when the text exactly matches a canonical name or synonym. To use the Exact Match Model, add the following to your app config (``config.py``) located in the top level of your app folder:

.. code-block:: python

    ENTITY_RESOLVER_CONFIG = {
        'model_type': 'exact_match'
    }

This is merely a fall-back option, for when you need to get an end-to-end app running without Elasticsearch. However, this approach is not optimal, and unsuitable for a broad-vocabulary conversational app.
