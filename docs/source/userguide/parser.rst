.. meta::
    :scope: private

Working with the Language Parser
================================

The :ref:`Language Parser <arch_parser>` (often referred to simply as the Parser) is run as the sixth and final step in the :ref:`natural language processing <arch_nlp>` pipeline to extract the relationships between :term:`entities <entity>` in a given :term:`query <query / request>`. It is a heuristic-driven `chart parser <https://en.wikipedia.org/wiki/Chart_parser>`_ that is configured to model the `dependencies <https://en.wikipedia.org/wiki/Dependency_grammar>`_ between different entity types in an application. The parser uses a developer-provided configuration to cluster the :doc:`recognized entities <entity_recognizer>` in a query together and group them into a meaningful hierarchy (called :term:`entity group`) that captures how different entities relate to each other.

.. note::

   **Recommended prior reading:**

   - :doc:`Step 8: Configure the Language Parser<../quickstart/08_configure_the_language_parser>` (Step-By-Step Guide)
   - :doc:`Natural Language Processor <nlp>` (User Guide)
   - :doc:`Entity Recognizer <entity_recognizer>` (User Guide)


Do you need a parser?
---------------------

Not all Workbench apps need a language parser. The following two conditions need to be met to necessitate the use of a parser:

  1. The app has one or more :term:`dependent (child) <dependent / child>` entity types that describe a :term:`head (parent) <head / parent>` entity type.

  2. The app supports queries with multiple head entities of the same type.

If your app satisfies both of the above conditions, you need a parser to identify the head-dependent relationships between the recognized entities in the query and cluster them together into meaningful entity groups.

Let's take three examples to understand this better.

**a. Kwik-E-Mart**

============ =
Condition #1 ✗
============ =

In the simple :ref:`Kwik-E-Mart store assistant app <model_hierarchy>`, neither entity (``store_name``, ``sys_time``) describes the other or is directly related in any way. Since the two entities are independent of each other, a parser is not required.

**b. Home assistant**

============ =
Condition #1 ✓
Condition #2 ✗
============ =

The ``turn_appliance_on`` and ``turn_appliance_off`` intents in the :ref:`home assistant blueprint <home_model_hierarchy>` have a pair of related entity types named ``location`` and ``appliance``. The ``location`` entity identifies the specific ``appliance`` of interest by describing where it is located. The ``location`` entity is thus a dependent of the ``appliance`` head entity.

However, the home assistant blueprint app is only designed to support the operation of one appliance per query. In other words, the user cannot reference two different appliances in the same query such as "Switch off the {living room|location} {tv|appliance} and turn on the {bedroom|location} {tv|appliance}." A parser is therefore not required since the app already knows that all dependents must refer to the only head entity present in the query.

**c. Food ordering**

============ =
Condition #1 ✓
Condition #2 ✓
============ =

In the :ref:`food ordering blueprint <food_ordering_parser>`, the ``option`` and ``sys_number`` entities are dependents of the ``dish`` entity, since they provide more information about the ``dish`` being ordered. Also, users often order multiple dishes in the same query. E.g., "Two hamburgers with extra cheese, an order of garlic fries, and a diet coke."  Here, the app needs to determine which options and quantities apply to which dishes, and group them sensibly to ensure that the correct order is placed. A parser is therefore required for this app.


.. note::

   The main entity in an entity group can be interchangbly referred to as the **parent** or **head** entity. The other entities in the group that are attributes of the main entity are correspondingly called **child** or **dependent** entities.

   The terms **head** and **dependent** reflect the `linguistic (syntactic) function <https://en.wikipedia.org/wiki/Dependency_grammar>`_ of the different entities within the group.

   Every entity group has an inherent hierarchy that can be represented as a tree data structure. In this context, **parent** and **child** denote the relationship between the different nodes in the `tree representation <https://en.wikipedia.org/wiki/Tree_(data_structure)>`_.


Configure the parser
--------------------

Before you can use the language parser, Workbench needs to know about the head and dependent entity types for your application. These are defined in a dictionary named :data:`PARSER_CONFIG` in your application configuration file, ``config.py``. The configuration gets loaded by the :class:`NaturalLanguageProcessor` when the :meth:`build` method is called.

.. code-block:: python

   >>> from mmworkbench import configure_logs; configure_logs()
   >>> from mmworkbench.components.nlp import NaturalLanguageProcessor
   >>> nlp = NaturalLanguageProcessor(app_path='food_ordering')
   >>> nlp.build()

The dictionary defining the parser configuration contains the head entity types as keys and information about the corresponding dependent entity types as values. Workbench supports two configuration formats that are described below. Your can choose the one that better suits your needs.

.. _simple_parser_config:

Simple parser configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

As described in :doc:`Step 8 <../quickstart/08_configure_the_language_parser>`, the fastest way to configure the language parser is by defining a simple dictionary that maps each head entity type to a list of related dependent entity types.

.. _food_simple_parser_config:

Here is an example from the :doc:`food ordering blueprint <../blueprints/food_ordering>`:

.. code-block:: python

   PARSER_CONFIG = {
       'dish': ['option', 'sys_number'],
       'option': ['sys_number']
   }

This configuration specifies that a ``dish`` entity can have an ``option`` entity and a numeric quantity entity (``sys_number``) as its dependents. An ``option`` entity, in turn, can have its own associated quantity entity.

Here's a slightly more complicated example where the configuration also specifies the :term:`role` types for some of its head and dependent entities:

.. code-block:: python

   PARSER_CONFIG = {
       'dish|beverage': ['option|beverage', 'sys_number'],
       'dish|baked_good': ['option|baked_good', 'sys_number'],
       'option': ['sys_number']
   }

In this example, ``option`` entities with a ``beverage`` or ``baked_good`` role type can only be grouped with ``dish`` entities having the same ``beverage`` or ``baked_good`` role, respectively. This ensures that the parser doesn't group incompatible options and dishes together, such as "extra cheese" with a "mocha" or "whipped cream" with a "lasagna".

Each key-value pair in the configuration instructs the parser to look for a specific head entity in the query and cluster it with one or more of the specified dependent entities to form an entity group.

  - The key is a string describing the entity type and optionally, the role type of the head entity. E.g., ``'dish'`` matches all entities of the type ``dish``, whereas ``'dish|beverage'`` only matches ``dish`` entities with a ``beverage`` role.

  - The value is a list of strings, with each string describing the entity type and optionally, the role type of a dependent entity. E.g., ``['size', 'option|beverage']`` instructs the parser to consider all ``size`` entities, and ``option`` entities with a ``beverage`` role as potential dependents for the head entity.

Using the head-dependent relationships defined in the configuration, the parser analyzes the detected entities in a query and hypothesizes different potential ways of grouping the entities together. Each such grouping is called a candidate **parse**. After generating these parse hypotheses, the parser uses a set of linguistically-motivated heuristics to pick the most likely candidate.

Workbench's default settings for the parser should give you a decent baseline parsing accuracy out-of-the-box. To improve its accuracy further, you can experiment with the parser settings, optimizing them for what makes the best sense for your data. See the next section for more details.


Advanced parser configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Workbench's advanced parser configuration format gives you a finer-grained control over the parser's behavior. In addition to defining the head-dependent relationships, it allows you to to specify constraints that must be satisfied for a dependent entity to be attached to a compatible head entity. If chosen correctly, these additional constraints can significantly improve the parsing accuracy by helping to eliminate potentially incorrect parse hypotheses.

Similar to the :ref:`simple format <simple_parser_config>`, each key-value pair in the advanced configuration instructs the parser to look for a specific head entity and group it with one or more of the specified dependent entities. Just as in the simple format, the key is a string describing the entity type and optionally, the role type of the head entity. The value, however, is a much richer object, mapping each potential dependent to a per-dependent configuration dictionary.

The table below enumerates the different settings that can be defined in the per-dependent configuration.

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

Here's an example of an advanced parser configuration from the :ref:`food ordering blueprint <food_ordering_parser>`:

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

It sets up the same head-dependent relationships as the :ref:`simple configuration <food_simple_parser_config>` in the previous section, but defines some additional settings for each dependent:

  - 'with' should be treated as a linking word between ``option`` and ``dish`` entities.

  - A ``dish`` can have only one quantity (``sys_number``) associated with it, and the quantity entity must be to its left.

  - An ``option`` can have only one quantity (``sys_number``) associated with it, and the quantity entity must be to its left.

The first setting is motivated by natural language constructs like "a burger `with` a side of fries" or "chicken biriyani `with` cucumber raita" where the intervening word "with" implies a ``dish``-``option`` relationship. The last two settings are due to real-world constraints (a thing can only have one quantifying adjective describing it) and English grammar rules (an adjective generally appears before the noun it describes). These settings provide useful syntactic and semantic cues to help the parser weed out non-sensical parses.

For example, here are three possible candidate parses for a sample food ordering query:

.. image:: /images/candidate_parses.png
    :align: center

A baseline parser using the :ref:`simple configuration <food_simple_parser_config>` will reject the incorrect third candidate and choose the second hypothesis, which is better, but still not fully correct. A parser configured using the :ref:`per-dependent settings <food_parser_advanced_config>`, on the other hand, will correctly choose the first parse by leveraging its knowledge of the linking word, "with".

Queries like the one above which contain multiple head entities of the same type with many potential dependents are inherently ambiguous. In other words, there is more than one way to generate an entity grouping for such queries that satisfies the specified head-dependent relationships. If you expect your app to deal with queries like this, it's highly recommended that you fine-tune the settings available in the advanced configuration format to optimize your parser's performance.


Run the parser
--------------

The parser runs as the last step in the NLP pipeline, building on top of the information provided by all the previous NLP models. Since running the previous components is a prerequisite for parsing, the most convenient way to run a configured parser on a test query is by using the :meth:`NaturalLanguageProcessor.process` method. As described in the chapter on :ref:`Natural Language Processor <run_nlp>`, the :meth:`process` method sends the query for sequential processing by each component in the NLP pipeline and returns the aggregated output from all of them.

Here's an example from the :ref:`food ordering <food_ordering_parser>` blueprint:

.. code:: python

   >>> nlp.process("I'd like a mujaddara wrap and two chicken kebab from palmyra")
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

To interpret all the items in the returned dictionary, refer to the chapter on :ref:`Natural Language Processor <run_nlp>`. The entry relevant to the parser is the ``'entities'`` field. Each recognized entity is represented as a dictionary with entity-specific properties like the entity text, the entity type, the role type, and so on. Additionally, for any entity that is detected as a head (parent), the parser adds a 'children' key, whose value is a list of all the dependent (child) entities related to this entity.

For instance, the entity, "chicken kebab" (``dish``), has the entity "two" (``sys_number``), as its dependent:

.. code:: python
   :emphasize-lines: 4-13

   >>> results = nlp.process("I'd like a mujaddara wrap and two chicken kebab from palmyra")
   >>> results['entities'][2]
   {
     'children': [
       {
         'confidence': 0.15634607039069398,
         'role': None,
         'span': {'end': 32, 'start': 30},
         'text': 'two',
         'type': 'sys_number',
         'value': [{'value': 2}]
       }
     ],
     'role': None,
     'span': {'end': 46, 'start': 34},
     'text': 'chicken kebab',
     'type': 'dish',
     'value': [ ... ]
   }

The remaining entities in the query, "mujaddara wrap" (``dish``) and "palymra" (``restaurant``), are childless since the parser did not find any dependent entities that are related to them.

More generally, an entity is not assigned a 'children' property by the parser if any of the following is true:

  #. The entity type is a potential head, according to the configuration, but the parser did not find any compatible dependents in the query that could attach to it.

  #. The entity type is not specified as a potential head in the configuration. By definition, the parser does not attach any dependents to such entities.

  #. The entity type is absent from the configuration altogether. The parser leaves such entities alone.

An entity, together with its children forms an entity group. The entity groups in the above example are {"mujaddara wrap"}, {"two", "chicken kebab"} and {"palmyra"}. Childless entities are considered to be in a singleton group of their own.

To better familiarize yourself with the language parser, use the :doc:`food ordering blueprint <../blueprints/food_ordering>` as a sandbox to test out the preconfigured parser and experiment with different configuration settings. Also, refer to the blueprint's application file (``app.py``) for examples on how to use the parser output within your :term:`dialogue state handlers <dialogue state handler>`.