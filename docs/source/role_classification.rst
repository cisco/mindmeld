Role Classifier
===============

Roles are named after the "Semantic Role Labeling" (SRL) task in NLP. Semantic Role Labeling is the task of identifying predicates and predicate arguments. For example,

.. raw:: html

    <style> .yellow {background-color:yellow} </style>

.. raw:: html

    <style> .orange {background-color:orange} </style>

.. raw:: html

    <style> .skin {background-color:#FAEBD7} </style>

.. raw:: html

    <style> .aqua {background-color:#00FFFF} </style>

.. raw:: html

    <style> .red {color:red} </style>

.. role:: yellow
.. role:: orange
.. role:: aqua
.. role:: skin
.. role:: red

* :yellow:`Mary` :orange:`sold` the :skin:`book` to :aqua:`John`

In the above example, the task would be to recognize the verb "to sell" as representing the predicate, "Mary" as representing the seller (agent), "the book" as representing the goods (theme), and "John" as representing the recipient. In an alternate form of the sentence - :red:`"The book was sold by Mary to John"` has a different syntactic form, but the same semantic roles.

.. _Defining The Entity Map: entity_map.html

In the context of Conversational NLU, a "role" defines how a named entity should be used to fulfill a query. In Workbench, Roles are defined as part of the Entity Map. See the `Defining The Entity Map`_ chapter for more details.

When To Use Roles
-----------------

Roles are not necessary in many cases, but can be tactically used to improve your end-to-end accuracy. In general, use roles -

* When different database columns share many entities and sometimes use similar language, e.g. **cast** vs **director**, **theme** vs **genre** vs **radio**.
* When you want to dynamically decide how entities are used to build queries based on context and the entity itself, e.g. "show me :skin:`Affleck` movies" searches **cast** and **director**, "show me :aqua:`Spielberg` movies" searches **director** only.
* When your entity map is bloated with entries to help distinguish how entities should be used.

When Not To Use Roles
^^^^^^^^^^^^^^^^^^^^^
When none of the above are true, and the assumption that named *entity* = *DB column* holds for your application.

Train The Model
---------------

.. code-block:: python

  import mindmeld as mm
  from mm.role_classification import RoleClassifier

  # Load the training data
  training_data = mm.load_data('/path/to/domain/intent/training_data.txt')

  # Load the gazetteer resources
  gazetteers = mm.load_gaz('/path/to/domain/intent/gazetteers')

  # Define the feature settings
  features = {
    'bag-of-words-before': {},
    'bag-of-words-after': {},
    'in-gaz': {},
    'other-entities': {},
    'operator-entities': {},
  }

  # Train the classifier
  role_model = RoleClassifier(model_type='maxent', features=features, gaz=gazetteers)
  role_model.fit(data=training_data, model='memm')

  # Evaluate the model
  eval_set = mm.load_data('/path/to/eval_set.txt')
  role_model.evaluate(data=eval_set)

Prediction
----------

.. code-block:: python

  q = "Play Black Sabbath by Black Sabbath from Black Sabbath"
  roles = role_model.predict(query=q)

Output:

.. code-block:: python

  # List of Roles (as defined in the Entity Map)
  [song, artist, album]