Working with the Language Parser
================================

The :ref:`Language Parser <arch_parser>`

 - is run as the sixth and final step in the :ref:`natural language processing pipeline <arch_nlp>`
 - is a heuristic-driven `chart parser <https://en.wikipedia.org/wiki/Chart_parser>`_ that extracts the relationships between :term:`entities <entity>` in a given :term:`query <query / request>`
 - models the `dependencies <https://en.wikipedia.org/wiki/Dependency_grammar>`_ between different entity types in an application, based on a developer-provided configuration
 - clusters the :doc:`recognized entities <entity_recognizer>` in a query together, grouping them into a meaningful hierarchy (called a :term:`entity group`) that captures how different entities relate to each other

Not every MindMeld app needs a parser.

.. note::

    This is an in-depth tutorial to work through from start to finish. Before you begin, read the :ref:`Step-by-Step Guide <quickstart>`, paying special attention to :doc:`Step 8 <../quickstart/08_configure_the_language_parser>`.

Some parsing terminology
-------------------------

In this guide, the terms *language parser* and *parser* are interchangeable. To understand how the parser works, you need to understand **entity groups**.

The main entity in an entity group is the **head** entity. The other, **dependent** entities, are attributes of the main entity. The terms, *head* and *dependent*, reflect the `linguistic (syntactic) function <https://en.wikipedia.org/wiki/Dependency_grammar>`_ of the entities within the group.

**Head** and **dependent** entities are also called **parent** and **child** entities, respectively. This is because the hierarchy inherent in any entity group can be seen as a tree data structure, where **parent** and **child** denote the relationship between the nodes in the `tree representation <https://en.wikipedia.org/wiki/Tree_(data_structure)>`_.

Figure out whether your app needs a parser
------------------------------------------

Your MindMeld app needs a language parser only if **both** the conditions in the table below are true.

 +--------------+------------------------------------------------------------------------------------+
 | Condition #1 | The app has one or more :term:`dependent (child) <dependent / child>`              |
 |              | entity types that describe a                                                       |
 |              | :term:`head (parent) <head / parent>` entity type.                                 |
 +--------------+------------------------------------------------------------------------------------+
 | Condition #2 | The app supports queries with multiple head entities of the same type.             |
 +--------------+------------------------------------------------------------------------------------+

If the conditions are both true, your app needs a parser to identify the head-dependent relationships between the recognized entities in the query, and to cluster them together into meaningful entity groups.

Consider the following explanatory examples.

**a. Kwik-E-Mart**

 +--------------+---+
 | Condition #1 | ✗ |
 +--------------+---+

In the simple :ref:`Kwik-E-Mart store assistant app <model_hierarchy>`, neither of the two entity types (``store_name``, ``sys_time``) describes the other. No entity of one type is directly related to any entity of the other.

The first condition is false, so there is no need for a parser.

**b. Home assistant**

 +--------------+---+
 | Condition #1 | ✓ |
 +--------------+---+

The ``turn_appliance_on`` and ``turn_appliance_off`` intents in the :ref:`home assistant blueprint <home_model_hierarchy>` have a pair of related entity types named ``location`` and ``appliance``. The ``location`` entity describes where a specific ``appliance`` is located. This makes the ``location`` entity a dependent of the ``appliance`` head entity.

 +--------------+---+
 | Condition #2 | ✗ |
 +--------------+---+

The home assistant blueprint app is only designed to support the operation of one appliance per query. Queries in which a user references two different appliances, e.g., "Switch off the {living room|location} {tv|appliance} and turn on the {bedroom|location} {tv|appliance}," are not supported. For this reason, the app assumes that all dependents must refer to a single head entity in any query.

Only one condition is true, so there is no need for a parser.

**c. Food ordering**

 +--------------+---+
 | Condition #1 | ✓ |
 +--------------+---+

In the :ref:`food ordering blueprint <food_ordering_parser>`, entities of type ``option`` and ``sys_number`` are dependents of ``dish`` entities, since they provide information about the ``dish`` being ordered.

 +--------------+---+
 | Condition #2 | ✓ |
 +--------------+---+

Users can order multiple dishes in the same query, e.g., "Two hamburgers with extra cheese, an order of garlic fries, and a diet coke." The app must determine which options and quantities apply to which dishes, and group them sensibly to ensure that the correct order is placed.

Both conditions are true, so the app needs a parser.

.. note::

   The rest of this chapter assumes that your app needs a parser. If not, you can skip to the next chapter.

.. _simple_parser_config:

How the configuration instructs the parser
------------------------------------------

What lets MindMeld know about the head and dependent entity types for your application is the :data:`PARSER_CONFIG` dictionary in ``config.py``, your app configuration file. In :data:`PARSER_CONFIG`, the keys are the head entity types, and the values capture information about the corresponding dependent entity types.

Using the head-dependent relationships defined in the configuration, the parser analyzes the detected entities in a query and hypothesizes different potential ways of grouping the entities together. Each such grouping is called a candidate **parse**. After generating these parse hypotheses, the parser uses a set of linguistically-motivated heuristics to pick the most likely candidate.

MindMeld supports both simple and advanced forms of parser configuration. Recommended practice is to get the parser up and running with the simple configuration. If the app achieves satisfactory accuracy, you do not need to move on to the advanced configuration. If, however, you want to experiment with fine-tuned parsing, the advanced configuration makes that possible.

Think about whether your app must support queries where (1) there are multiple head entities of the same type, and (2) those head entities have many potential dependents. For example, "Get me a pepperoni pizza with extra cheese, a calzone, and two diet cokes." Such a query is inherently ambiguous because there is more than one way to group its entities that satisfies the head-dependent relationships a simple configuration can define. For apps that deal with queries like this, fine-tuning the settings available in the advanced configuration is highly recommended. :ref:`Later in this chapter <food_parser_advanced_config>`, we explore this issue in detail.

.. _food_simple_parser_config:

Learn how to create a simple parser configuration
-------------------------------------------------

The first step toward running the parser is creating the simple configuration described in this section.

Simple configuration structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The configuration is a dictionary where each key-value pair instructs the parser to (1) look for a specific head entity in the query, and (2) cluster that head entity with one or more specified dependent entities to form an entity group.

  - The key is a string describing the entity type and optionally, the role type of the head entity. Whereas the key ``'dish'`` matches all entities of the type ``dish``, the key ``'dish|beverage'`` only matches ``dish`` entities with a ``beverage`` role.

  - The value is a list of strings, where each string describes the entity type and optionally, the role type of a dependent entity. The value ``['size', 'option|beverage']`` instructs the parser to consider all ``size`` entities, and ``option`` entities with a ``beverage`` role, as potential dependents for the head entity.

  - The point of defining roles is to prevent the parser from grouping incompatible options and dishes together, for example "extra cheese" with a "mocha" or "whipped cream" with a "lasagna".

Simple configuration example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's define a :data:`PARSER_CONFIG` to see how to apply these principles. This example is for the :doc:`food ordering blueprint <../blueprints/food_ordering>`.

First, specify that a ``dish`` entity can have an ``option`` entity and a numeric quantity entity (``sys_number``) as its dependents. An ``option`` entity, in turn, can have its own associated quantity entity.

.. code-block:: python

   PARSER_CONFIG = {
       'dish': ['option', 'sys_number'],
       'option': ['sys_number']
   }

If the blueprint were to be extended with role types for subclassifying some of the entity types, you can additionally specify the :term:`role` for some head and dependent entities. We can say that ``option`` entities with a ``beverage`` role type can only be grouped with ``dish`` entities that also have a ``beverage`` role. Likewise, ``option`` entities with a ``baked_good`` role type can only be grouped with ``dish`` entities that also have a ``baked_good`` role. Now the parser will only group options with compatible dishes.

.. code-block:: python

   PARSER_CONFIG = {
       'dish|beverage': ['option|beverage', 'sys_number'],
       'dish|baked_good': ['option|baked_good', 'sys_number'],
       'option': ['sys_number']
   }


.. _define_config:

Define a simple parser configuration
------------------------------------

Write out (or better yet, diagram) the entities in your app and how they are related. Then think through how to capture those relationships in a :data:`PARSER_CONFIG` dictionary that maps each head entity type to a list of related dependent entity types, as explained in the previous section.

Define the simple :data:`PARSER_CONFIG` dictionary and add it to your ``config.py``.


.. _load_config:

Load the parser configuration
-----------------------------

Load the configuration by calling the :meth:`build` method on the :class:`NaturalLanguageProcessor` class.

.. code-block:: python

   from mindmeld import configure_logs; configure_logs()
   from mindmeld.components.nlp import NaturalLanguageProcessor
   nlp = NaturalLanguageProcessor(app_path='food_ordering')
   nlp.build()

.. _run_parser:

Run the parser
--------------

The parser runs as the last step in the NLP pipeline, building on top of the information provided by all the previous NLP models. The most convenient way to run a configured parser is to use the :meth:`NaturalLanguageProcessor.process` method, which sends the query for processing by each NLP pipeline component in sequence, then returns the aggregated output from all the components.

In the output of :meth:`NaturalLanguageProcessor.process`:

 - The entry relevant to the parser is the ``'entities'`` field.
 - Each recognized entity is represented as a dictionary with entity-specific properties like the entity text, the entity type, the role type, and so on.
 - For any entity the parser detects as a head, it adds a 'children' key whose value is a list of all the head's dependent entities.
 - An entity and its children together form an entity group.
 - Each childless entity is considered to be in a singleton group of its own.

For more about this output dictionary, see :ref:`Run the NLP pipeline <run_nlp>`.

Here's an example from the :ref:`food ordering <food_ordering_parser>` blueprint.

.. code:: python

   nlp.process("I'd like a mujaddara wrap and two chicken kebab from palmyra")

.. code-block:: console

   {
    'domain': 'ordering',
    'entities': [
      {
        'role': None,
        'span': {'end': 24, 'start': 11},
        'text': 'mujaddara wrap',
        'type': 'dish',
        'value': [{'cname': 'Mujaddara Wrap', 'id': 'B01DEFNIRY'}]
      },
      {
        'role': None,
        'span': {'end': 32, 'start': 30},
        'text': 'two',
        'type': 'sys_number',
        'value': {'value': 2}
      },
      {
        'children': [
          {
            'role': None,
            'span': {'end': 32, 'start': 30},
            'text': 'two',
            'type': 'sys_number',
            'value': {'value': 2}
          }
        ],
        'role': None,
        'span': {'end': 46, 'start': 34},
        'text': 'chicken kebab',
        'type': 'dish',
        'value': [{'cname': 'Chicken Kebab', 'id': 'B01DEFMUSW'}]
      },
      {
        'role': None,
        'span': {'end': 59, 'start': 53},
        'text': 'palmyra',
        'type': 'restaurant',
        'value': [{'cname': 'Palmyra', 'id': 'B01DEFLJIO'}]
      }
    ],
    'intent': 'build_order',
    'text': "I'd like a mujaddara wrap and two chicken kebab from palmyra"
   }

Inspect an individual entity. We'll choose the only entity which has a dependent: "chicken kebab" (``dish``), whose dependent is the entity "two" (``sys_number``).

.. code:: python

   results = nlp.process("I'd like a mujaddara wrap and two chicken kebab from palmyra")
   results['entities'][2]

.. code-block:: console
   :emphasize-lines: 2-6

   {'text': 'chicken kebab',
    'children': [{'text': 'two',
      'type': 'sys_number',
      'role': 'num_orders',
      'value': [{'value': 2}],
      'span': {'start': 30, 'end': 32}}],
    'type': 'dish',
    'role': None,
    'value': [{'cname': 'Chicken Kebab',
      'score': 147.06445,
      'top_synonym': 'Chicken Kebab',
      'id': 'B01DEFMUSW'},
     {'cname': 'Chicken Tikka Kabab',
      'score': 99.278786,
      'top_synonym': 'chicken kebab',
      'id': 'B01DN5635O'},
     {'cname': 'Chicken Kebab Plate',
      'score': 93.68581,
      'top_synonym': 'Chicken Kebab',
      'id': 'B01CK4ZQ7U'},
     {'cname': 'Chicken Shish Kebab Plate',
      'score': 31.71228,
      'top_synonym': 'chicken kebab plate',
      'id': 'B01N9Z1K2O'},
     {'cname': 'Kebab Wrap',
      'score': 21.39855,
      'top_synonym': 'Kebab Wrap',
      'id': 'B01CRF8ULQ'},
     {'cname': 'Kofte Kebab',
      'score': 21.39855,
      'top_synonym': 'beef kebab',
      'id': 'B01N4VET8U'},
     {'cname': 'Lamb Kebab',
      'score': 21.39855,
      'top_synonym': 'Lamb Kebab',
      'id': 'B01DEFNQIA'},
     {'cname': 'Kebab Platter',
      'score': 21.290503,
      'top_synonym': 'Kebab Dish',
      'id': 'B01CRF8A7U'},
     {'cname': 'Beef Kebab',
      'score': 21.290503,
      'top_synonym': 'Beef Kebab',
      'id': 'B01DEFL75Y'},
     {'cname': 'Prawns Kebab',
      'score': 21.290503,
      'top_synonym': 'Prawns Kebab',
      'id': 'B01DEFO4ZY'}],
    'span': {'start': 34, 'end': 46}
   }

The remaining entities in the query, "mujaddara wrap" (``dish``) and "palymra" (``restaurant``), are childless because the parser found no dependent entities for them. This example thus has three entity groups in total: {"two", "chicken kebab"}, {"mujaddara wrap"}, and {"palmyra"}.

.. note::

   The parser does not assign a 'children' property to an entity when any of the following are true:

   #. The entity type is a potential head according to the configuration, but the parser finds no compatible dependents in the query.

   #. The entity type is not specified as a potential head in the configuration. By definition, the parser never attaches dependents to such entities.

   #. The entity type is absent from the configuration altogether. The parser ignores such entities.

Test out the preconfigured parser and experiment with different configuration settings, using the :doc:`food ordering blueprint <../blueprints/food_ordering>` as a sandbox.
Study the blueprint's application file (``__init__.py``) for examples on how to use parser output within :term:`dialogue state handlers <dialogue state handler>`. Continue this exercise until you the language parser and its capabilities feel familiar to you.

Your app should exhibit decent baseline parsing accuracy out-of-the-box using default parser settings. To improve its accuracy further, you can optimize the parser settings for the nature of your data. If you decide to experiment in this way, run your parser using an advanced configuration as described in the next section.

.. _advanced_parser_config:

Learn how to create an advanced parser configuration (optional)
---------------------------------------------------------------

MindMeld offers an advanced parser configuration format to provide fine-grained control over parser behavior. While both the simple and the advanced configurations define head-dependent relationships, in the advanced configuration you can specify that *only a dependent entity that satisfies certain constraints* can be attached to a compatible head entity. If chosen well, these constraints can help eliminate potentially incorrect parse hypotheses, resulting in significantly improved parsing accuracy.

Advanced configuration structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The advanced configuration, like the simple configuration, is a dictionary where each key-value pair instructs the parser to (1) look for a specific head entity in the query, and (2) cluster that head entity with one or more specified dependent entities to form an entity group. The advanced configuration also specifies constraints that a dependent must satisfy to be attached to a compatible head entity.

  - In both simple and advanced configurations, the key is a string describing the entity type and optionally, the role type of the head entity.

  - In the advanced configuration, the value is a rich object that maps each potential dependent to a configuration dictionary whose key/value pairs specify constraints to apply to candidate parses. We call this dictionary the *per-dependent configuration.*

The table below enumerates the settings you can define in the per-dependent configuration.

+---------------------+-----------------+------------------------------------------------------------------------------------------------------+
| Key                 | Value type      | Value                                                                                                |
+=====================+=================+======================================================================================================+
| ``'left'``          | :class:`bool`   | Whether to allow attachment in the left direction. If ``True``, a dependent entity of this type is   |
|                     |                 | permitted to attach to an instance of the head entity type on its left (as determined by their       |
|                     |                 | relative positions in the query text). If ``False``, the parser disallows any candidate parses where |
|                     |                 | this dependent type is grouped with the head entity to its left.                                     |
|                     |                 |                                                                                                      |
|                     |                 | Default: ``True``.                                                                                   |
+---------------------+-----------------+------------------------------------------------------------------------------------------------------+
| ``'right'``         | :class:`bool`   | Whether to allow attachment in the right direction (analogous to the ``'left'`` setting above).      |
|                     |                 |                                                                                                      |
|                     |                 | Default: ``True``.                                                                                   |
+---------------------+-----------------+------------------------------------------------------------------------------------------------------+
| ``'min_instances'`` | :class:`int`    | The minimum number of dependent entities of this type that must be grouped with the head entity for  |
|                     |                 | a successful parse. The parser will not create an entity group unless it can link the required       |
|                     |                 | number of dependents to the head.                                                                    |
|                     |                 |                                                                                                      |
|                     |                 | Default: 0.                                                                                          |
+---------------------+-----------------+------------------------------------------------------------------------------------------------------+
| ``'max_instances'`` | :class:`int`    | The maximum number of dependent entities of this type that can be grouped with the head entity. If   |
|                     | or ``NoneType`` | the value is ``None``, the parser does not impose any limits on the number of dependents of this     |
|                     |                 | type that can link with the head entity.                                                             |
|                     |                 |                                                                                                      |
|                     |                 | Default: ``None``.                                                                                   |
+---------------------+-----------------+------------------------------------------------------------------------------------------------------+
| ``'precedence'``    | :class:`str`    | The preferred direction of attachment for dependent entities of this type. The preferred direction   |
|                     |                 | determines the head to attach to, if there are **equidistant** compatible head entities in the query |
|                     |                 | on either side of the dependent . Accepted values are ``'left'``, to prefer the head to the left     |
|                     |                 | of the dependent entity, or ``'right'``, to choose the one on the right.                             |
|                     |                 |                                                                                                      |
|                     |                 | Default: 'left'.                                                                                     |
+---------------------+-----------------+------------------------------------------------------------------------------------------------------+
| ``'linking_words'`` | :class:`set`    | A set of words, whose occurence between two entities increases the chance of the entities being      |
|                     |                 | in the specified head-dependent relationship. These linking words provide hints to the parser to     |
|                     |                 | prefer candidate parses where one of these words is present in the query text between a dependent    |
|                     |                 | entity of this type and the head entity.                                                             |
|                     |                 |                                                                                                      |
|                     |                 | Default: ``set()`` (an empty set).                                                                   |
+---------------------+-----------------+------------------------------------------------------------------------------------------------------+

.. _food_parser_advanced_config:

Advanced configuration example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's define an advanced parser configuration that sets up the same head-dependent relationships as the :ref:`simple configuration <food_simple_parser_config>` for :ref:`food ordering <food_ordering_parser>` in the previous section, but that defines constraints for each dependent.

The desired per-dependent constraints are:

  - 'with' should be treated as a linking word between ``option`` and ``dish`` entities.

  - Only one quantity (``sys_number``) can be associated with a ``dish``, and the quantity entity must be to its left.

  - Only one quantity (``sys_number``) can be associated with an ``option``, and the quantity entity must be to its left.

.. We omit the roles seen in the simple example because ...?

.. code:: python

   PARSER_CONFIG = {
       'dish': {
           'option': {'linking_words': {'with'}},
           'sys_number': {'max_instances': 1, 'right': False}
       },
       'option': {
           'sys_number': {'max_instances': 1, 'right': False}
       }
   }

The first constraint is motivated by natural language constructs like "a burger `with` a side of fries" or "chicken biriyani `with` cucumber raita" where the intervening word "with" implies a ``dish``-``option`` relationship. The other two settings embody real-world constraints (a thing cannot be described by more than one quantifying adjective) and English grammar rules (an adjective generally appears before the noun it describes).

The syntactic and semantic cues that these constraints provide help the parser weed out nonsensical parses. To see this, consider three possible candidate parses for a sample food ordering query:

.. image:: /images/candidate_parses.png
    :align: center

A baseline parser using the :ref:`simple configuration <food_simple_parser_config>` will reject the incorrect third candidate and choose the second hypothesis, which is better, but still not fully correct. A parser configured using the :ref:`per-dependent settings <food_parser_advanced_config>`, on the other hand, will correctly choose the first parse based on its knowledge of the linking word, "with".

What is significant about the query above is that it contains multiple head entities of the same type with many potential dependents. It is inherently ambiguous because there is more than one way to group its entities that satisfies the head-dependent relationships a simple configuration can define. To optimize parser performance, we needed to define constraints that help the parser eliminate nonsensical candidate parses.

Experimenting with an advanced configuration (optional)
-------------------------------------------------------

Once you have defined an advanced configuration, :ref:`load the configuration <load_config>`, and :ref:`run the parser <run_parser>`. Observe the effects of your per-dependent configuration settings on parser accuracy, and if desired, iterate on the whole process.
