Building a Conversational Interface in 11 Steps
===============================================

Great conversational applications require both advanced technology as well as solid design judgement. The most widely used conversational applications today, such as Siri, Alexa, Google Assistant and Cortana, are all built using a similar set of techniques and technologies to ensure both high accuracy and utility. This section outlines the methodology that is used to build today's most advanced and useful conversational applications. Specifically, this section will walk through the steps that are required to build and deploy a conversational application for a simple use case. This simple example will highlight the key design steps and technology components which underpin any great conversational experience. Along the way, we will illustrate how MindMeld Workbench can be employed to streamline the task of building and deploying conversational interfaces. 

Taking a conversational application from conception to production typically involves the eleven implementation steps summarized below.

== ===
1  Select the right use case.
2  Script your ideal dialogue interactions.
3  Define the domain, intent, entity and role hierarchy.
4  Define the dialog state handlers.
5  Create the knowledge base.
6  Generate representative training data.
7  Train the natural language processing classifiers.
8  Train the entity resolvers.
9  Implement the language parser.
10 Optimize question answering performance.
11 Deploy trained models to production.
== ===

This section will illustrate each of these steps using an example of a simple conversational application.


Select the Right Use Case
-------------------------
Selecting the right use case is perhaps the most important step in building a conversational application that users will love. There are many use cases where a voice or chat conversation can simplify the task of finding information or accomplishing a task. There are also many use cases where a conversational interface can be inconvenient or frustrating. Selecting an unrealistic or incorrect use case will render even the smartest voice or chat assistant dead on arrival.

While there is no magic formula to determine which use case is ideal for a conversational interface, some best practices have begun to emerge to delineate good candidates from the bad. Before selecting a use case, it is important to ask the following questions to ensure that your conversational application will be practical to build and provide real value to users.

===================================================== ===
**Does it resemble a real-world human interaction?**  Conversational interfaces do not come with instruction manuals, and there is little opportunity to teach users about the supported functionality before they take it for a spin. The best use cases mimic an existing, familiar real-world human interaction so that users intuitively know what they can ask and how the service can help. For example, a conversational interface could mimic an interaction with a bank teller, a barista or a customer support rep.

**Does it save users time?**                          Conversational interfaces shine when they save users time. A conversational interface will be viewed as an unwelcome impediment when a well-designed GUI would be faster. The most useful conversational experiences often center around a use case where users are looking to accomplish a specific task and know how to articulate it. For example, simply saying 'play my jazz music playlist in the kitchen' can be much faster than launching an app and navigating to the equivalent option by touch.

**Is it more convenient for the user?**               Voice interfaces can be particularly useful when users' hands and attention are occupied or if a mobile device is not within reach. If you expect users will often enlist your application while driving, biking, walking, exercising, cooking or sitting on the couch, it is likely an excellent candidate for a conversational interface.

**Does it hit the Goldilocks zone?**                  The best conversational applications fall squarely into the 'Goldilocks zone'. They offer a range of functionality that is narrow enough to ensure the machine learning models have high accuracy. At the same time, they have functionality that is broad enough to ensure that users find the experience useful for a wide variety of tasks. Apps that are too narrow can be trivial and useless. Apps that are too broad can have hit-or-miss accuracy which can frustrate users.

**Is it possible to get enough training data?**       Even the best use cases will fail if it is not possible or practical to collect enough training data to illustrate the complete range of envisioned functionality. For ideal use cases, training data can be easily generated from production traffic or crowdsourcing techniques. If training data for your use case can only be sourced from a small number of hard-to-find human experts, it is likely not a good candidate for a conversational interface.
===================================================== ===

For this quickstart section, we will consider a simple conversational use case which can provide information about retail store locations for your neighborhood Kwik-E-Mart. For example, you could use this service to ask about store hours: 'When does the Kwik-E-Mart on Elm street close today?'. This rudimentary use case will serve as a reference example to highlight the key steps in building a useful conversational interface.


Script Your Ideal Dialogue Interactions
---------------------------------------

Once you have identified a good use case, the next step is to script your envisioned dialogue interactions. This design exercise details the conversational interaction flows which define the user experience for your application. It is important to think through not just the most obvious user flows, but also the corner case and exception user flows which may be encountered by users during an interaction. For example, the dialogue flows should illustrate how the application responds if the user request is beyond the application scope. Also, the flows should illustrate the interactions which will enable users to get help if they are stuck or gracefully leave the interaction. The dialogue flows should detail not only the various interaction pathways a conversation may traverse, but they should also illustrate the desired language and tone envisioned for the experience.

For developers familiar with graphical interface design, this step is comparable to the task of creating wireframe and pixel-perfect mockups. Like any design step, there will likely be many iterations required to work through usability issues and reach consensus on a final design. It is always wise to begin coding and implementation only after the dust has settled on the scripted dialogue interactions. Otherwise, much implementation work and effort may be wasted. 

For our simple use case, the following diagram illustrates the dialogue interactions we will strive to deliver in our final implementation.

.. image:: images/quickstart_interaction.png
    :width: 400px
    :align: center


Define the Domain, Intent, Entity and Role Hierarchy
-------------------------------------------------------

Conversational applications rely on a hierarchy of machine learning classifiers in order to model and understand natural language. Often called the Natural Language Processor, this family of machine learning models sits at the core all conversational assistants in widespread production use today. While there are many different ways that machine learning techniques can be enlisted to dissect and understand human language, a set of best practices has been adopted in recent years to systematize the sometimes challenging task of building accurate and useful natural language processing systems. Today, nearly all commercial conversational applications rely on a hierarchy of machine learning models illustrated below.

.. image:: images/hierarchy.png
    :width: 700px
    :align: center

The topmost layer in the model hierarchy is the domain classifier. The domain classifier is responsible for performing a first-pass classification to group incoming queries into set of pre-defined buckets or 'domains'. For any given domain, there may be one or more pre-defined intents. Each intent defines a specific action or answer type to invoke for a given request. The intent classifier models are responsible for deciding which intent is most likely associated with a given request. Once the request is categorized into a specific intent, the entity recognition models are employed to discern the important words and phrases in each query that must be identified in order to understand and fulfill the request. These identified words and phrases are called 'entities', and each intent may have zero or more types of entities which must be recognized. For some types of entities, a fourth and final classification step, called role classification, may be required. The role classifiers are responsible for adding differentiating labels to entities of the same type. Refer to the User Guide for a more in-depth treatment of the natural language processing classifier hierarchy utilized by MindMeld Workbench. 


Show the directory structure which captures this hierarchy for a simple example.

Developer creates a directory structure that implicitly defines the domain, intent and entity hierarchy.

Mention the concept of 'blueprints' (aka reference applications).


For example,

 - ``store_information`` Defines the domain.
 
   - ``greet`` Begins an interaction.
   - ``get_close_time`` Returns the close time for the requested store.
   - ``get_open_time`` Returns the open time for the requested store.
   - ``get_nearest_store`` Returns the closest store to the user.
   - ``get_is_open_now`` Returns yes or no if the requested store is open now.
   - ``exit`` Ends the current interaction.

.. note::

  By convention, intent names should always be verbs which describe what the user is trying accomplish.


Directory structure::

  my_app/
      my_app.py
      data/
          store_information/
              gazetteers/
              greet/
                  labeled_queries/
              get_store_close_time/
                  labeled_queries/
              get_store_open_time/
                  labeled_queries/
              get_nearest_store/
                  labeled_queries/
              get_is_open_now/
                  labeled_queries/
              exit/
                  labeled_queries/

Entities:

 - When does the store on ``Elm Street | NAME`` close ``today | DATE``?
 - When does that store open ``tomorrow | DATE``?
 - Is the ``Central Plaza Kwik-E-Mart | NAME`` open now?

.. note::

  By convention, entity names should always be nouns which describe the entity type.



Define the dialog state handlers
-----------------------------------
In my view, this is where we define the natural language response templates which should be returned at each dialogue state in an interaction. We should illustrate a simple flow in a flow chart and then in a snippet of python code which illustrates how the logic is implemented in the dialogue manager.

Create the python file which defines your application.

File my_app.py

.. code:: python

  from mmworkbench import Application
  from mmworkbench import context, slots
  import mmworkbench.KnowledgeBase as kb
  
  app = Application(__name__)
  
  @app.handle(intent='greet')
  def welcome():
      slots['name'] = context.request.session.user_name
      response = {
          'replies': [
              'Hello, {name}. I can help you find store hours ' +
              'for your local Kwik-E-Mart. How can I help?'
          ]
      }
      return response
  
  @app.handle(intent='get_store_close_time')
  def send_close_time():
      set_target_store(context)
      if context.frame.target_store:
          slots['time'] = context.frame.target_store['close_time']
          slots['store_name'] = context.frame.target_store['name']
          response = {
              'replies': [
                  'The {store_name} Kwik-E-Mart closes at {time}.'
              ]
          }
      else:
          response = {'replies': ['For which store?']}
      return response
  
  @app.handle(intent='get_store_open_time')
  def send_open_time():
      set_target_store(context)
      if context.frame.target_store:
          slots['time'] = context.frame.target_store['open_time']
          slots['store_name'] = context.frame.target_store['name']
          response = {
              'replies': [
                  'The {store_name} Kwik-E-Mart opens at {time}.'
              ]
          }
      else:
          response = {'replies': ['For which store?']}
      return response
  
  @app.handle(intent='get_nearest_store')
  def send_nearest_store():
      loc = context.request.session.location 
      stores = kb.get('store', sort='proximity', current_location=loc)
      slots['store_name'] = stores[0]['name']
      response = {
          'replies': [
              'your nearest Kwik-E-Mart is located at {store_name}.'
          ]
      }
      return response
  
  @app.handle(intent='exit')
  def say_goodbye():
      return {'replies': ['Bye', 'Goodbye', 'Have a nice day.']}
  
  def set_target_store(context):
      stores = [e.value for e in context.entities if e.type == 'name']
      if names: context.frame.target_store = stores[0]
  
  if __name__ == "__main__":
      app.run()



Create the Knowledge Base
----------------------------
A Knowledge Base is a repository for storing complex, real-world, structured and unstructured information relevant to a content catalog. In the context of Deep-Domain Conversational AI, a Knowledge Base comprises of a searchable repository of objects that encode entities and their associated attributes. 

Here are some examples of Knowledge Base object types for various applications -

* Product items in a **Products** catalog - with attributes *"sku_id"*, *"name"*, *"price"*
* Video objects in a **Video Content** library - with attributes *"description"*, *"cast"*, *"duration"*
* Menu items from a **Quick Service Restaurants** catalog - with attributes *"size"*, *"price"*, *"options"* etc.

In our example of store information on Kwik-E-Mart stores, we would have store objects with the following attributes -

* store_name
* open_time
* close_time
* address
* phone_number

Indexing
~~~~~~~~

A Knowledge Base can have one or more indexes. An index (short for "Inverted Index") is a data structure designed to allow very fast full-text searches. It consists of a list of unique words that appear in any document, and for each word, a list of the documents in which it appears. By default, every field in a document object is indexed, and thus is searchable. Different indexes can be used to map to objects of different types. In MindMeld Workbench, creating an index is as simple as specifying the index name while loading your data into the Knowledge Base. In our example of Kwik-E-Mart stores data, we would have just 1 index - *"stores"*.

Loading Data
~~~~~~~~~~~~

To load your content catalog into the MindMeld Knowledge Base, you can specify your catalog data as a JSON dump. The MindMeld Knowledge Base can read this JSON dump and extract all fields along with their types directly from the data.

Following is an example of JSON data containing document objects and their attributes for a few Kwik-E-Mart stores.

File **stores_data.json**

.. code-block:: javascript

  {
    "store_name": "Central Plaza Store",
    "open_time": 0800 hrs,
    "close_time": 1800 hrs,
    "address": "100 Central Plaza, Suite 800, Elm Street, Capital City, CA 10001",
    "phone_number": (+1) 100-100-1100
  },
  {
    "store_name": "Market Street Store",
    "open_time": 0900 hrs,
    "close_time": 2200 hrs,
    "adress": "750 Market Street, Capital City, CA 94001",
    "phone_number": (+1) 450-450-4500
  }
  ...

Once your catalog is tranformed to the above JSON format, you can load that data into a Knowledge Base. Following is a code snippet to create and load data into a Knowledge Base using MindMeld Workbench.

.. code-block:: python

  from mmworkbench.knowledge_base import KnowledgeBase

  # Initialize the KB
  kb = KnowledgeBase()

  # Load JSON Data into the KB
  kb.load(data_file='stores_data.json', index='stores')

If the specified index in the **load** method already exists, the index is recreated with the new data. If not, a new index with that name is created and tha data is loaded in.

To delete an index, simply use the **delete_index** method by specifying the index name to delete.

.. code-block:: python

  kb.delete_index(index='stores')

Retrieval
~~~~~~~~~

Once your data is loaded, you can use the **get** method to retrieve objects from the MindMeld Knowledge Base. The **get** method uses all information available in the query and entity mappings (passed through the context object) to retrieve documents. For String-valued fields, the Knowledge Base uses Full-Text Search for retrieval. When "range" entities are detected, the **get** method uses "greater-than" or "lesser-than" operations as applicable to the respective (real-valued) fields. More details on Sorting and Text Relevance strategies are available in Section 1.10 on "Optimizing Question Answering Performance".

Example use of **get** -

.. code-block:: python

  query = "Is the store on Elm Street open?"
  context = {
    'domain': 'store_information',
    'intent': 'get_is_store_open',
    'entities': [
      {
        'type': 'street',
        'mode': 'search',
        'text': 'Elm Street',
        'value': 'address:Elm Street',
        'chstart': 16,
        'chend': 25
      }
    ]
  }

  # Retrieve from the KB
  results = kb.get(index='stores', query, context)
  print results

Output -

.. code-block:: javascript

  {
    "store_name": "Central Plaza Store",
    "open_time": "8:00 am",
    "close_time": "6:00 pm",
    "address": "100 Central Plaza, Suite 800, Elm Street, Capital City, CA 10001",
    "phone_number": (+1) 100-100-1100
  }

The **get** method also supports pagination. You can use the *offset* and *num_docs* arguments to retrieve the required window of documents for the query. By default, the **get** method uses a value of *num_docs=10*.

.. code-block:: python

  # Retrieve documents numbers 11 to 30
  kb.get(index='stores', query, context, offset=10, num_docs=20)

Generate representative training data
----------------------------------------
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

  Pro Tip - Academic datasets (though instrumental in researching advanced algorithms), are not always reflective of real-world conversational data. Therefore, datasets from popular conferences such as TREC and ACM-SIGDIAL might not be the best choice for developing production applications.

For guidelines on collecting training data at scale, please refer to the User Guide chapter on Training Data. It has useful information on collecting a large amount of training data using relatively inexpensive and easy-to-implement crowdsourcing techniques.

Train the Natural Language Processing classifiers
---------------------------------------------------

The Natural Language Processor (NLP) component of Workbench is tasked with understanding the user's natural language input. It analyzes the input using a hierarchy of classification models, with each model assisting the next tier of models by narrowing the problem scope, or in other words, successively narrowing down the “search space”.

As introduced in [1.3. Define the Domain, Intent, Entity and Role Hierarchy], there are a total of four classifiers, applied in the following order:

#. **Domain Classifier**: For apps that handle conversations across varied topics having their own specialized vocabulary, the domain classifier provides the first level of categorization by classifying the input into one of the pre-defined set of conversational domains.

#. **Intent Classifier**: The intent classifier next determines what the user is trying to accomplish by categorizing the input into a set of user intents that the system can handle.

#. **Entity Recognizer**: The entity recognizer then extracts important words and phrases, called entities, that are required to fulfill the user's end goal.

#. **Role Classifier**: In cases where an entity of a particular type can have multiple meanings depending on the context, the role classifier can be used to provide another level of categorization and assign a differentiating label called "role" to the extracted entities.

To train the NLP classifiers for our "Store Information" app, we start by gathering the training data as described in [1.6 Generate representative training data] and placing them in the directory structure mentioned in [1.3. Define the Domain, Intent, Entity and Role Hierarchy]. For a quick start, we can train the necessary classifiers and save them to disk using these four simple lines of code:  

.. code-block:: python

  from mmworkbench import NLP

  # Instantiate MindMeld NLP by providing the app_data path.
  nlp = NLP('path_to_app_data_directory_root')

  # Train the NLP
  nlp.fit()

  # Save the trained NLP models to disk
  nlp.dump()

Based on the directory structure and the nature of your annotated data, the Natural Language Processor can automatically determine which classifiers need to be trained. In our case, the NLP will train an intent classifier for the ``store_information`` domain and entity recognizers for each of the intents under that domain, while ignoring the domain and role classifiers. The above code uses the default machine learning settings for each of the classifiers, which in most cases should train reasonable models. But to build a high quality production-ready conversational app, we need to carefully train, test and optimize each classification model individually, and that's where Workbench truly shines. 

We'll next take a closer look at what happens behind the scenes when you call ``nlp.fit()`` and understand two of the NLP steps - Intent Classification and Entity Recognition in more detail.


Intent Classifier
~~~~~~~~~~~~~~~~~

Intent Classifiers are text classifiers that are trained per domain using the data in each intent's ``labeled_queries`` folder. 

For our intent classifier, let's choose a `logistic regression <https://en.wikipedia.org/wiki/Logistic_regression>`_ model and use `Bag of Words <https://en.wikipedia.org/wiki/Bag-of-words_model>`_ and `Edge n-grams <https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-edgengram-tokenizer.html>`_ as features. Also, we would like to do `k-fold cross validation <https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation>`_  with 20 splits.

We start off by importing and instantiating an object of the Natural Language Processor (NLP) class by providing it the path to the root of our app data directory.

.. code-block:: python

  from mmworkbench import NLP

  # Instantiate MindMeld NLP by providing the app_data path.
  nlp = NLP('path_to_app_data_directory_root')

We next define the feature dictionary that lists all the feature types along with the feature-specific settings. E.g. We want bag of n-grams up to size 2 and similarly, edge-ngrams up to length 2.

.. code-block:: python

  # Define the feature settings
  feature_dict = {
    'bag-of-words': { 'lengths': [1, 2] },
    'edge-ngrams': { 'lengths': [1, 2] }
  }

Define the cross validation iterator with the desired number of splits.

.. code-block:: python

  # Define CV iterator
  kfold_cv = KFold(num_splits=20)

Finally, we fetch the domain we are interested in and call its ``fit_intent_model()`` method to train the intent classifier. We also use the ``dump_intent_model()`` method to persist the trained model to disk. By default, intent classifier models get saved to a ``models`` directory under their respective domains.

.. code-block:: python

  domain = nlp.get_domain('store_information')
  domain.fit_intent_model(model='logreg', features=feature_dict, cv=kfold_cv)
  domain.dump_intent_model()

We have now successfully trained an intent classifier for the ``store_information`` domain. If our app had more domains, we would follow the same steps for those other domains.

.. note::

  ``nlp.domains()`` returns an iterator over all domains.


Entity Recognizer
~~~~~~~~~~~~~~~~~

Entity Recognizers are sequence labeling models that are trained per intent using the annotated queries in each intent's ``labeled_queries`` folder. The task of the entity recognizer is both to detect the entities within a query and label them as one of the pre-defined entity types.

We'll use `Maximum Entropy Markov Models <https://en.wikipedia.org/wiki/Maximum-entropy_Markov_model>`_, which are a good choice for sequence labeling tasks in NLP. For features, one of the most helpful and commonly used sources of information in entity recognition models is a comprehensive list of entity names called a "`gazetteer <https://gate.ac.uk/sale/tao/splitch13.html#x18-32600013.1>`_". Each entity type has its own gazetteer. In our case, the gazetteer for the ``Name`` entity type would be a list of all the Kwik-e-Mart store names in our catalog. The list for the ``Date`` type could be a fairly small list: ['today', 'tomorrow', 'weekdays', 'weekends', ...]. Gazetteers can then be used to derive features based on full or partial match of words in the query against entries in the gazetteers. 

Apart from using gazetteer-based features, we'll use bag-of-words features like we did for intent classification. Length of the current token also ends up being a useful feature for entity recognition, so we'll add that too. Finally, we'll continuing using 20-fold cross validation like we did before.

Here's the code to instantiate an NLP object, define the features and initialize a k-fold iterator.

.. code-block:: python

  from mmworkbench import NLP

  # Instantiate MindMeld NLP by providing the app_data path.
  nlp = NLP('path_to_app_data_directory_root')

  # Define the feature settings
  feature_dict = {
    'in-gaz': {},
    'bag-of-words': { 'lengths': [1, 2] },
    'length': {}
  }

  # Define CV iterator
  kfold_cv = KFold(num_splits=20)

Now, let's train an entity recognizer for one of our intents and save it to disk. By default, entity recognizer models get saved to a ``models`` directory under their respective intents.

.. code-block:: python

  intent = nlp.get_domain('store_information').get_intent('get_open_time')
  intent.fit_entity_model(model='memm', features=feature_dict, cv=kfold_cv)
  intent.dump_entity_model()

We can similarly train the entity recognizers for other intents as well.

.. note::

  ``nlp.get_domain('xyz').intents()`` returns an iterator over all the intents for domain 'xyz'.

When we invoked ``nlp.fit()`` in the "quickstart" at the beginning of this section, we were essentially asking the Natural Language Processor  to do all these steps (``domain.fit_intent_model()``, ``domain.fit_entity_model()``, etc.) on our behalf using some default configuration for all the domains and intents in our hierarchy. However, we have seen that Workbench also offers the flexibility to define the model type, features and cross validation settings for each of its NLP classifiers. In addition, it's also possible to control various other aspects of the training algorithm such as hyperparameters and other model-specific settings (e.g. the kernel to use for an SVM). [3.9 The Natural Language Processor] in the Workbench User Guide has detailed documentation on all the NLP classifiers, along with the different configurations and options available for each. 


Train the entity resolvers
-----------------------------
Introduce the topic of loading training data, training entity resolution models, measuring CV and held-out performance, performing disambiguation.

Configure the Language Parser
--------------------------------

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


Optimize Question Answering Performance
---------------------------------------

The Question Answering module is responsible for ranking results retrieved from the Knowledge Base, based on some notion of relevance. The relevance of each document is represented by a positive floating point number - the ``score``. The higher the score, the more relevant the document. MindMeld Workbench offers a robust, built-in "ranking formula" for defining a general-purpose scoring function. However, in cases where the default ranking formula is not sufficient in ensuring good performance across a large number of test queries, MindMeld Workbench provides a facility for defining custom ranking formulae. The concept of "performance" is explained in Section 1.10.4 on "Evaluation Metrics".

Sorting
~~~~~~~

Among the various signals used in computing the relevance score, sorting is an important operation offered by MindMeld Workbench. Sorting is applicable on any real-valued field in the Knowledge Base (either ascending or descending order). The Question Answering module gets its cue to invoke the sorting function based on the presence of ``sort`` entities. If one or more sort entities are detected, the documents with resolved numerical field values corresponding to those entities will get a boost in the score function. Additionally, a decay is applied to this sorting boost to ensure a balance between the applied sort and other relevance signals.

For example, consider the following query:

* What are the cheapest doughnuts available in Kwik-E-Mart?

Let's say we have the following documents in the Knowledge Base:

.. code-block:: javascript

  {
    "item_id": 1,
    "item_name": "Pink Doughnut",
    "price": 20
  },
  {
    "item_id": 2,
    "item_name": "Green Doughnut",
    "price": 12
  },
  {
    "item_id": 3,
    "item_name": "Yellow Doughnut",
    "price": 15
  }
  ...

The Natural Language Processor would detect ``cheapest`` as a sort entity and populates the context object accordingly:

.. code-block:: python

  query = "What are the cheapest doughnuts available in Kwik-E-Mart?"
  context = {
    'domain': 'item_information',
    'intent': 'order_item',
    'entities': [
      {
        'type': 'item_name'
        'mode': 'search',
        'text': 'doughnut'
        'value': 'item_name:doughnut',
        'chstart': 22,
        'chend': 30
      },
      {
        'type': 'pricesort',
        'mode': 'sort',
        'text': 'cheapest',
        'value': 'price:asc',
        'chstart': 13,
        'chend': 20
      }
    ]
  }

  results = kb.get(index='items', query, context)
  print results

The final ranking that MindMeld Workbench returns is -

.. code-block:: javascript

  {item_id: 2},
  {item_id: 1},
  {item_id: 3}

Text Relevance
~~~~~~~~~~~~~~

In general, "Text Relevance" refers to the algorithm used to calculate how *similar* the contents of a full-text field are to a full-text query string. The Knowledge Base offered by MindMeld Workbench uses a standard similarity algorithm called the `TF_IDF <https://en.wikipedia.org/wiki/Tf-idf>`_ algorithm. Additionally, a *Field Length Norm* factor is applied, so longer field values are penalized.

Consider the following example documents on three different products:

.. code-block:: javascript

  { 
    "item_id": 1,
    "item_name": "Pink Frosty Doughnuts"
  },
  { 
    "item_id": 2,
    "item_name": "Pink Sprinklicious Doughnuts"
  },
  {
    "item_id": 3,
    "item_name": "Frosty Yellow Doughnuts With Frosty Sprinkles"
  }

For an incoming query like -

* "I want some frosty doughnuts"

The returned list of documents as per text relevance would be:

.. code-block:: javascript

  {item_id: 1},
  {item_id: 3},
  {item_id: 2} 

* {item_id: 1} is more relevant because it's ``item_name`` is short
* {item_id: 3} comes next because "frosty" appears twice and "doughnut" appears once
* {item_id: 2} is the last - only "doughnut" matched

If we want to specify a more stringent match criteria (E.g both "frosty" and "doughnut" must appear in the returned documents), we can use the ``minimum_should_match`` argument in the Knowledge Base **get** method. The ``minimum_should_match`` parameter specifies what percentage of query terms should match with the field value (at least).

.. code-block:: python

  # All query terms must match the terms in the field value
  kb.get(index='items', query, context, minimum_should_match=100)

The default value of the ``minimum_should_match`` parameter is set to 75%.

While the above example gives a glimpse of the text-matching strategies available in MindMeld Workbench, much more complex functionality (such as "Exact Matching" and "Boosting Query Clauses") is available in the User Guide chapter on Knowledge Base.

Advanced Settings
~~~~~~~~~~~~~~~~~

While creating the index, all fields in the data go through a process called "Analysis". Analyzers can be defined per field to define the following:

* Tokenizing a block of text into individual terms before adding to inverted index
* Normalizing these terms into a standard form to improve searchability

When we search on a full-text field, the query string is passed through the same analysis process, to ensure that we are searching for terms in the same form as those that exist in the index.

In MindMeld Workbench, you can optionally define custom analyzers per field by specifying an **es_mapping.json** file at the application root level. While the default MindMeld Workbench Analyzer uses a robust set of character filtering operations for tokenizing, custom analyzers can be handy for special character/token handling.

For example, lets say we have a store named *"Springfield™ store"*. We want the indexer to ignore characters like "™" and "®" since users never specify these in their queries. We need to define special ``char_filter`` and ``analyzers`` mappings as follows:

File **es_mapping.json** -

.. code-block:: javascript

  {
    "field_mappings": {
      "store_name": {
        "type": "string",
        "analyzer": "my_custom_analyzer"
      }
    },
    "settings": {
      "char_filter": {
        "remove_tm_and_r": {
            "pattern":"™|®",
            "type":"pattern_replace",
            "replacement":""
        }
      },
      "analyzers": {
        "my_custom_analyzer": {
          "type": "custom",
          "tokenizer": "whitespace",
          "char_filter": [
            "remove_tm_and_r"
          ]
        }
      }
    }
  }

More information on custom analyzers and the **es_mapping.json** file is available in the User Guide chapter on Knowledge Base. Example mapping files for a variety of use-cases and content types are also provided.

Evaluation Metrics
~~~~~~~~~~~~~~~~~~

In Information Retrieval, Top 1 accuracy, Top K accuracy, Precision, Recall and F1 scores are all great evaluation metrics to get started. To optimize Precision and Recall, you will need to create a "relevant set" of documents for each query in your test set. This relevant set is typically generated by a human expert, or by repeated error analysis.

.. code-block:: text

  Query                                   Relevant Set

  "get me a doughnut"                     1, 3, 28, 67, 253, 798
  "i want a lemon Squishee"               4, 363, 692
  "can I get a buzz cola"                 291
  "pink frosty sprinklicous doughnut"     67
  ...

For a thorough evaluation, it is advisable to create relevant sets for thousands of test queries for the initial pass. This bank of queries and their expected results should grow over time into hundreds of thousands, or even millions of query examples. This then becomes the golden set of data on which future models can be trained and evaluated.

Custom Ranking Functions
~~~~~~~~~~~~~~~~~~~~~~~~

In general, you should not have to worry about writing your own scoring function for ranking. MindMeld Workbench provides numerous knobs and dials for detailed, granular control over the built-in scoring function. However, in cases where the existing scoring function simply does not fit the needs of your application, you can specify your own custom scoring function for ranking. Define your custom ranking function in the **my_app.py** file as follows:

File **my_app.py** -

.. code-block:: python

  @app.kb.handle(domain='items')
  def items_ranking_fn(query, context, document):
    # Custom scoring logic goes here.
    score = compute_doc_score(query, context, document)
    return score

The custom ranking function can then be used in the **get** method of the Knowledge Base object.

.. code-block:: python

 # Assume KnowledgeBase object has been created and
 # the data is loaded into the 'items' index.

 # Get ranked results from KB
 ranked_results = kb.get(index='stores', query,
        context, ranking_fn=items_ranking_fn)

The function gets applied to each document in the retrieved set to compute their final scores, and the ranked set is then returned.

.. note::

  A note on system latency - In applications where hundreds or thousands of documents are retrieved on each query, applying a custom scoring function on each document can make the requests terribly slow, depending on how well the function is engineered. Please be mindful of request latencies and overall system performance when designing custom ranking functions.

Learning To Rank
~~~~~~~~~~~~~~~~

Given the right kind of training data (and lots of it), Machine Learning methods can be applied for ranking in a variety of ways. To learn how to develop a Machine Learning approach to ranking, i.e. `Learning To Rank <https://en.wikipedia.org/wiki/Learning_to_rank>`_, please refer to the guidelines on assembling the right kind of training data and building models in the User Guide chapter.

Deploy trained models to production
---------------------------------------
Show a simple example of the steps required to deploy to production