.. meta::
    :scope: private

Language Parser
===============

The :ref:`Language Parser <arch_parser>` (often referred to simply as the Parser) is run as the sixth and final step in the :ref:`natural language processing <arch_nlp>` pipeline to extract the relationships between :term:`entities <entity>` in a given :term:`query <query / request>`. It is a heuristic-driven `chart parser <https://en.wikipedia.org/wiki/Chart_parser>`_ that is configured to model the `dependencies <https://en.wikipedia.org/wiki/Dependency_grammar>`_ between different entity types in an application. The parser uses a developer-provided configuration to cluster the :doc:`recognized entities <entity_recognizer>` in a query together and group them into a meaningful hierarchy (:term:`entity group`) that captures how different entities relate to each other.

.. note::

   **Recommended prior reading:**

   - :doc:`Step 8: Configure the Language Parser<../quickstart/08_configure_the_language_parser>` (Step-By-Step Guide)
   - :doc:`Natural Language Processor <nlp>` (User Guide)
   - :doc:`Entity Recognizer <entity_recognizer>` (User Guide)


Do you need a parser?
---------------------

Not all Workbench apps need a language parser. The following two conditions need to be met to necessitate the use of a parser:

  1. The app has one or more :term:`dependent <dependent / child>` entity types that further describe a :term:`head <head / parent>` entity type.

  2. The app supports queries with multiple head entities of the same type.

If your app satisfies both of the above conditions, you need a parser to identify the head - dependent relationships between the recognized entities in the query and cluster them together into meaningful entity groups.

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

However, the home assistant blueprint app is only designed to support the operation of one appliance per query. In other words, the user cannot reference two different appliances in the same query such as "Switch off the {living room|location} {tv|appliance} and turn on the {bedroom|location} {tv|appliance}." A parser is therefore not required since the app already knows the one and only head entity that all the dependent entities must refer to.

**c. Food ordering**

============ =
Condition #1 ✓
Condition #2 ✓
============ =

In the :ref:`food ordering blueprint <food_ordering_parser>`, the ``option`` and ``sys_number`` entities are dependents of the ``dish`` entity, since they provide more information about the ``dish`` being ordered. Also, users often order multiple dishes in the same query. E.g., "Two hamburgers with extra cheese, an order of garlic fries, and a diet coke."  Here, the app needs to determine which options and quantities apply to which dishes, and group them sensibly to ensure that the correct order is placed. A parser is therefore required for this app.




