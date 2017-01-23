Step 8: Configure the Language Parser
=====================================

The last component within the Natural Language Processor is the **Language Parser**. Its job is to find relations between the extracted entities and group them into meaningful entity groups. The Parser analyzes the information provided by all the previous NLP models and outputs a data structure called the parse tree, that represents how different entites relate to each other. 

Consider the use case where you not only want to check the hours of your local Kwik-e-Mart, but also order items to pick up from the store in advance. To handle this additional functionality, we would have to add a new entity type called ``product`` which is the name of the food item and other associated entities like ``size`` and ``quantity`` which provide more information about the order. The Language Parser takes these entities that are detected by the Entity Recognizer, and `parses <https://en.wikipedia.org/wiki/Parsing>`_ them into entity groups representing distinct meaningful real-world objects or concepts.

Here's an example:

.. image:: images/parse_example.png

The parse tree is a collection of entity groups, with each group having a main entity and optionally, a set of related entities that further describe the main entity. In linguistics, the main entity is called the `Head <https://en.wikipedia.org/wiki/Head_(linguistics)>`_ and the related entities are called `Dependents <https://en.wikipedia.org/wiki/Dependency_grammar>`_. 

The example query above contains three main pieces of information - the two products that the user wants to order and the name of the store they want place the order at. Correspondingly, we have three entity groups, two of them with ``product`` entities as heads and one with the ``store_name`` entity. The ``product`` entity has attributes like ``quantity`` and ``size`` that `modify <https://en.wikipedia.org/wiki/Grammatical_modifier>`_ it, and are hence grouped together with the head as its dependents.

`Natural Language Parsing <https://en.wikipedia.org/wiki/Natural_language_parsing>`_ is a long-studied problem in Computer Science and there are different approaches used, depending on the end goal and the depth of linguistic analysis required. The methods range from simple ones like rule-based and regex-based parsing to more sophisticated techniques like `Syntactic Parsing <http://spark-public.s3.amazonaws.com/nlp/slides/Parsing-Intro.pdf>`_ and `Semantic parsing <https://web.stanford.edu/class/cs224u/materials/cs224u-2016-intro-semparse.pdf>`_. While parsing remains an active area of research, commercial applications like Siri, Cortana, Alexa and Google Assistant use fairly standard conventional approaches that work well in practice and are easy to build, debug and maintain. Such systems almost always have a good rule-based parser, optionally augmented by a statistical parser if enough human-annotated parsed data is available for training.

The Language Parser in Workbench is versatile enough to support everything from the simplest rule-based systems to state-of-the-art syntactic and semantic parsers. Out of the box, Workbench comes with a config-driven `Dependency Parser <http://spark-public.s3.amazonaws.com/nlp/slides/Parsing-Dependency.pdf>`_ (a kind of Syntactic Parser), similar to what's used in most commercial conversational applications today. Getting started with the parser merely requires creating a configuration file named ``parser_config.json`` that defines the expected dependency relations between your different entities. 

Here is the ``parser_config.json`` file that instructs the Parser to extract the entity groups described in the figure above.

.. code-block:: javascript

  {
    "product": ["quantity", "size"]
  }

This tells the Parser that ``product`` is a head entity and ``quantity`` and ``size`` are its dependents. By default, the Parser creates a distinct entity group for any entity that it can't find a head for. Hence the ``store_name`` entity gets its own group. The :doc:`Language Parser User Manual </language_parsing>` has more details on the different options available to fine-tune the behavior of the config-driven parser. It also covers how to define your own custom parsing logic and train a state-of-the-art statistical parser using annotated data.

The Language Parser completes the query understanding process by identifying the heads, their dependents and linking them together with into a number of logical units (entity groups) that can be used by downstream components to take appropriate actions and generate the responses necessary to fulfill the user's request. However, it's worth mentioning that not every scenario may need the Language Parser. For instance, in our simple "Kwik-e-Mart Store Information" app, there are only two kinds of entities - ``date`` and ``store_name``, which are distinct and unrelated pieces of information. Thus, running the parser would just yield two singleton entity groups having heads, but no dependents. The Parser becomes more crucial when you have a complex app that supports complicated natural language queries like the example we discussed above.
