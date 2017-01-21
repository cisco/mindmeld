Configure the Language Parser
=============================

The last component within the Natural Language Processor is the **Language Parser**. Its job is to find relations between the extracted entities and group them into meaningful entity groups. The Parser analyzes the information provided by all the previous NLP models and outputs data structures called parse trees, that represent how different entites relate to each other. The figure below shows the Language Parser in action on a sample input.

.. image:: images/parse_example.png

Each parse tree has a main entity as its root node and any related entities that describe the main entity further, as the root's children. In linguistics, the main entity is called the `Head <https://en.wikipedia.org/wiki/Head_(linguistics)>`_ and the related entities are called `Dependents <https://en.wikipedia.org/wiki/Dependency_grammar>`_. In the figure above, the input query has two main pieces of information - the product information and the store information. Correspondingly, we have two parse trees, one with the ``Product`` entity type as its head and the other with the ``Store`` entity type. The ``Product`` entity has attributes like ``Quantity`` and ``Size`` that `modify <https://en.wikipedia.org/wiki/Grammatical_modifier>`_ it, and hence become its dependents in the tree. Similarly, the ``Store`` entity has ``Location`` as a dependent.

The Language Parser thus completes the query understanding process by identifying the heads, their dependents and linking them together with into a number of logical units (parse trees) that can be used by downstream components to take appropriate actions and generate the responses necessary to fulfill the user's request. However, it's worth mentioning that not every scenario may need the Language Parser. For instance, in our simple "Store Information" app, there are only two kinds of entities - ``Date`` and ``Name``, which are distinct and unrelated pieces of information. Thus, running the parser would just yield two singleton parse trees having heads, but no dependents. The Parser becomes more crucial when you have a complex app that supports complicated natural language queries like the example in the figure above. 

`Parsing <https://en.wikipedia.org/wiki/Parsing>`_ is a well-studied problem in Computer Science and there are several approaches used in practice, depending on the end goal and the depth of linguistic analysis required. The methods range from simple ones like rule-based and regex-based parsing to more sophisticated techniques like `Syntactic Parsing <http://spark-public.s3.amazonaws.com/nlp/slides/Parsing-Intro.pdf>`_ and `Semantic parsing <https://web.stanford.edu/class/cs224u/materials/cs224u-2016-intro-semparse.pdf>`_. 

The Language Parser in Workbench is a `Dependency Parser <http://spark-public.s3.amazonaws.com/nlp/slides/Parsing-Dependency.pdf>`_ (a kind of Syntactic Parser) which could either be trained statistically with annotated data or run in a config-driven rule-based fashion in the absence of training data. The latter is usually the quickest way to get started since it merely requires creating parser configuration files that define the expected dependency relations between your different entities. These files must be created per instance and named ``parser.config``. They are placed alongside the ``labeled_queries`` folder for that intent in your data directory.

Below is an example config file that instructs the Parser to extract the trees described in the figure above.

.. code-block:: text

  tree:
    name:'product_info'
    head:
      type: 'product'
    dependent:
      type: 'quantity'
    dependent:
      type: 'size'

  tree:
    name: 'store_info'
    head:
      type: 'store'
    dependent:
      type: 'location'

Finally, Workbench also offers the flexibility to define your own custom parsing logic that can be run instead of the default config-driven dependency parser. The :doc:`Language Parser User Guide </language_parsing>` in Section 3 has more details on the different options for our config-driven parser and how to implement your own custom parser.


