Generate Representative Training Data
=====================================

Most components in the Mindmeld Workbench Natural Language Processor utilize Supervised Learning models to analyze a user's query and derive meaning out of it. To train each of these components, we typically require thousands to millions of *labeled* queries to build powerful models. **It is critical that you obtain high-quality, representative training data** to ensure high accuracy. The training data serves as the ground truth for the models, so it is imperative that the ground truth data is clean and represents the exact use-case that you are training the model for.

Some strategies for collecting training data are -

#. Human Data Entry
#. Mining The Web
#. Crowdsourcing
#. Operational Logs (Customer Service, Search etc.)

In MindMeld Workbench, there are 5 components that need training data for a Machine Learning based Conversational Application. Typically, a given application would need training data for some subset of these components depending on the domain and core use-cases.

* Domain Classification
* Intent Classification
* Entity Recognition
* Role Classification
* Entity Resolution

We now describe the formats and structure of data required for training each of these components.

Domain Classification
~~~~~~~~~~~~~~~~~~~~~

In our example application of Kwik-E-Mart store information, Domain Classification is not needed since we have only one domain - **store_information**. In case we have additional domains (such as **weather** or **timers**), we would need separate sets of training queries for each domain. In such cases, MindMeld Workbench provides the facility of using queries from all the intents belonging to a domain as labeled queries for that domain. For example, training queries for the **store_information** domain would be the union of all queries in the *greet*, *get_close_time*, *get_open_time*, *get_nearest_store*, *get_is_open_now* and *exit* intents. The folder structure described in Section 1.3 provides an easy way of specifying your queries pertaining to a domain.

Intent Classification
~~~~~~~~~~~~~~~~~~~~~

For the **store_information** domain, here are snippets of training examples for a few intents for Intent Classification. We can define query sets for all other intents in a similar vein. These queries reside in *.txt* files under the **labeled_queries** folder of each intent directory as shown in Section 1.3.

* File .../greet/labeled_queries/**train_greet.txt**

.. code-block:: text

  Hi
  Hello
  Good morning
  ...

* File .../get_close_time/labeled_queries/**train_get_close_time.txt**

.. code-block:: text

  when does the elm street store close?
  what's the shut down time for pine & market store?
  ...

Entity Recognition
~~~~~~~~~~~~~~~~~~

To train the MindMeld Entity Recognizer, we need to label sections of the training queries with corresponding entity types. We do this by adding annotations to our training queries to identify all the entities. As a convenience in MindMeld Workbench, the training data for Entity Recognition and Role Classification are stored in the same files that contain queries for Intent Classification. To locate these files, please refer to the folder structure as specified in Section 1.3. For adding annotations for Entity Recognition, mark up the parts of every query that correspond to an entity in the following syntax -

* Enclose the entity in curly braces
* Follow the entity with its type
* Use the pipe character as separator

Example -

File .../get_is_open_now/labeled_queries/**train_get_is_open_now.txt**

.. code-block:: text

  Is the {Central Plaza|name} Kwik-E-Mart open {now|time}?
  The store near {Pine & Market|intersection} - is it open?
  Is the {Rockerfeller|name} Kwik-E-Mart on {30th Street|street} open for business?
  Can you check if the {Main St|street} store is open?

.. note::

  Pro Tip - We recommend using a popular text editor such as Vim, Emacs or Sublime Text 3 to create these annotations. This process is normally much faster than creating GUIs and point-and-click systems for annotating data at scale.

Role Classification
~~~~~~~~~~~~~~~~~~~

In some applications, a single entity can be used to cover multiple semantic roles. In our example of Kwik-E-Mart store information, a good candidate for Role Classification is the **time** entity type. Consider this example -

* Show me all Kwik-E-Mart stores open between 8 am and 6 pm.

Here, both *"8 am"* and *"6 pm"* are **time** entities, but they denote different semantic roles - *"open_time"* and *"close_time"* respectively.

For entities that have multiple semantic roles, a Role Classifier must be trained to accurately identify the semantic roles. To train a role classifier, label the respective entities in the training queries with their corresponding role labels. We can do this by adding additional annotations to the already labeled entities. Mark up the labeled entities with role annotations in the following syntax -

* Follow the labeled entity type with it's role label
* Use the pipe character as separator (similar to Entity training labels)

Examples -

.. code-block:: text

  Show me all Kwik-E-Mart stores open between {8 am|time|open_time} and {6 pm|time|close_time}
  Are there any Kwik-E-Mart stores open after {3 pm tomorrow|time|open_time}

Entity Resolution
~~~~~~~~~~~~~~~~~

Entity Resolution is the task of maping each entity to a unique and unambiguous concept, such as a product with a specific ID or an attribute with a specific SKU number. In MindMeld Workbench, this can usually be specified by a simple lookup dictionary in the Entity Map for all entity types. But for some applications, we need to specify thousands or even millions of mapping-pair examples that can be used to train a Machine Learning model.

In our Kwik-E-Mart store information example, a simple dictionary would be sufficient to map store names and other attributes to their respective constructs to retrieve corresponding Knowledge Base objects. For applications with catalogs such as Quick Service Restaurant menus or Product Information Catalogs, the MindMeld Entity Resolver needs a large number of "synonyms" for Product IDs or attribute SKUs. This is needed to ensure high accuracy on queries about the long-tail of products or attributes, when it is infeasible to map directly in a lookup dictionary.

Consider the following example of ordering items from Kwik-E-Mart stores. Lets assume there was a product named -

* *"Pink Frosted Sprinklicious Doughnut"*

in the menu catalog. However, there might be a multitude of ways users can refer to this particular product. For example, *"sprinkly doughnut"*, *"pink doughnut"*, *"frosty sprinkly doughnut"* could all be ways of referring to the same final product. In order to train the Entity Resolver to correctly resolve these utterances to their exact product ID, create a **synonyms.tsv** file that encodes various ways users refer to a specific product. The file is a TSV with 2 fields - the synonym and the final product/attribute name (as per the Knowledge Base object). Note that in the case where we don't need to train a Machine Learned Entity Resolver, this file would be optional. Locate the file in the folder structure as shown in Section 1.3.

Example -

File **synonyms.tsv**

.. code-block:: text

  sprinkly doughnut           Pink Frosted Sprinklicious Doughnut
  pink doughnut               Pink Frosted Sprinklicious Doughnut
  frosty sprinkly doughnut    Pink Frosted Sprinklicious Doughnut
  ...

.. note::

  Academic datasets (though instrumental in researching advanced algorithms), are not always reflective of real-world conversational data. Therefore, datasets from popular conferences such as TREC and ACM-SIGDIAL might not be the best choice for developing production applications.

For guidelines on collecting training data at scale, please refer to the User Guide chapter on Training Data. It has useful information on collecting a large amount of training data using relatively inexpensive and easy-to-implement crowdsourcing techniques.
