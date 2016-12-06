Building The Role Classifier
============================

Roles are named after the "Semantic Role Labeling" (SRL) task in NLP. Semantic Role Labeling is the task of identifying predicates and predicate arguments. For example,

.. raw:: html

    <style> .yellow {background-color:yellow} </style>

.. raw:: html

    <style> .orange {background-color:orange} </style>

.. raw:: html

    <style> .skin {background-color:#FAEBD7} </style>

.. raw:: html

    <style> .aqua {background-color:#00FFFF} </style>

.. role:: yellow
.. role:: orange
.. role:: aqua
.. role:: skin

:yellow:`Mary` :orange:`sold` the :skin:`book` to :aqua:`John`

In the context of Conversational NLU, a "role" defines how a named entity should be used to fulfill a query. Treating Named Entity Recognition (NER) and Semantic Role Labeling (SRL) as separate tasks has a few advantages -

* NER models are hurt by splitting examples across fairly similar categories. Grouping facets with significantly overlapping entities and similar surrounding natural language will lead to better parsing and let us use more powerful models.
* Joint NER & SRL needs global dependencies, but fast & good NER models only do local. NER models (MEMM, CRF) quickly become intractable with long-distance dependencies. Separating NER from SRL let us use local dependencies for NER and long-distance dependencies in SRL.
* Role labeling might be a multi-label problem. With multi-label roles, we can use the same entity to query multiple fields.

.. _Defining The Entity Map: entity_map.html

In Workbench, Roles are defined as part of the Entity Map. See the `Defining The Entity Map`_ chapter for more details.

When To Use Roles
*****************

Roles are not necessary in many cases, but can be tactically used to improve your end-to-end accuracy. In general, use roles -

* When different database columns share many entities and sometimes use similar language, e.g. **cast** vs **director**, **theme** vs **genre** vs **radio**.
* When you want to dynamically decide how entities are used to build queries based on context and the entity itself, e.g. "show me :skin:`Affleck` movies" searches **cast** and **director**, "show me :aqua:`Spielberg` movies" searches **director** only.
* When your entity map is bloated with entries to help distinguish how entities should be used, e.g. **billboard**, **grammys**.

When Not To Use Roles
^^^^^^^^^^^^^^^^^^^^^
When none of the above are true, and the assumption that named *entity* = *facet* = *DB column* holds for your application.

Train The Model
***************

A Maximum Entropy model can be used under the hood to define a Role Classification model. At the moment, no role config is required, and the default set of model + feature settings work pretty well.

.. code-block:: python

  import mindmeld as mm
  from mindmeld.role_classification import RoleClassifier

  # Load the training data
  training_data = mm.load_data(domain='clothing', 'intent'='search-products')

  # Train the classifier
  role_model = mm.RoleClassifier()
  role_model.fit(data=training_data, model='memm')

  # Evaluate the model
  role_model.evaluate(data='eval_set.txt')

Prediction
**********

.. code-block:: python

  q = "Play Black Sabbath by Black Sabbath from Black Sabbath"
  roles = role_model.predict(query=q)

Output:

.. code-block:: python
  
  # List of Roles (as defined in the Entity Map)
  [song, artist, album]