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

Conversational applications rely on a hierarchy of machine learning classifiers in order to model and understand natural language. Often called the Natural Language Processor, this family of machine learning models sits at the core all conversational assistants in widespread production use today. While there are many different ways that machine learning techniques can be enlisted to disect and understand human language, a set of best practices has been adopted in recent years to systematize the sometimes challenging task of building accurate and useful natural language processing systems.  

Today, nearly all commercial conversational applications rely on a hierarchy of machine learning models illustrated below.

.. image:: images/hierarchy.png
    :width: 600px
    :align: center

The topmost layer in the model hierarchy is the domain classifier. The domain classifier is responsible for performing a first-pass classification to group incoming queries into set of pre-defined buckets or 'domains'. For any given domain, there may be one or more pre-defined intents. Each intent defines a specific action or answer type to invoke for a given request. The intent classifier models are responsible for deciding which intent is most likely associated with a given request. Once the request is categorized into a specific intent, the entity recognition models are employed to discern the important words and phrases in each query that must be identified in order to understand and fulfill the request. These identified words and phrases are called 'entities', and each intent may have zero or more types of entities which must be recognized. For some types of entities, a fourth and final classification step, called role classification, may be required. The role classifiers are responsible for adding differentiating labels to entities of the same type. 

Refer to section 3 for a more in-depth treatment of the natural language processing classifier hierarchy utilized by MindMeld Workbench. 


In the past few years, some standard approaches have been adopted 


Today's commercial conversational applications rely on 

Show and describe a simple diagram which illustrates the domain, intent, entity and role hierarchy.  Show the directory structure which captures this hierarchy for a simple example.

Developer creates a directory structure that implicitly defines the domain, intent and entity hierarchy.

Intent names are always verbs which describe what the user is trying to accomplish.
Entity names are always nouns which describe the entity type.

For example,

 - ``store_information`` Defines the domain.
 
   - ``greet`` Begins an interaction.
   - ``get_close_time`` Returns the close time for the requested store.
   - ``get_open_time`` Returns the open time for the requested store.
   - ``get_nearest_store`` Returns the closest store to the user.
   - ``get_is_open_now`` Returns yes or no if the requested store is open now.
   - ``exit`` Ends the current interaction.


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

Following is an example of JSON data containing objects and their attributes for a few Kwik-E-Mart stores.

File **stores_data.json**

.. code-block:: text

  {
    "store_name": "Central Plaza Store", "open_time": 0800 hrs, "close_time": 1800 hrs,
    "address": "100 Central Plaza, Suite 800, Elm Street, Capital City, CA 10001",
    "phone_number": "(+1) 100-100-1100"
  },
  {
    "store_name": "Market Street Store", "open_time": 0900 hrs, "close_time": 2200 hrs,
    "adress": "750 Market Street, Capital City, CA 94001",
    "phone_number": "(+1) 450-450-4500"
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

Once your data is loaded, you can use the **get** method to retrieve objects from the MindMeld Knowledge Base. The **get** method uses various types information available in the query and entity mappings (passed in the context object) to retrieve documents. For String-valued fields, the Knowledge Base uses Full-Text Search for retrieval. When "range" entities are detected, the **get** method uses "greater-than" or "lesser-than" operations as applicable on the respective (real-valued) fields. More details on configuring Sorting and Text Relevance strategies are available in Section 1.10 and the User Guide chapter on Knowledge Base.

Example use of **get** -

.. code-block:: python

  # Get relevant objects from the KB
  query = "Is the store on Elm Street open?"
  context = {
    'domain': 'store_information',
    'intent': 'get_is_store_open',
    'entities': [
      {
        'type': 'street',
        'mode': 'search',
        'text': 'Elm Street',
        'value': 'Elm Street',
        'chstart': 16,
        'chend': 25
      }
    ]
  }
  results = kb.get(index='stores', query, context)
  print results

Output -

.. code-block:: text

  {
    "store_name": "Central Plaza Store", "open_time": "8:00 am", "close_time": "6:00 pm",
    "address": "100 Central Plaza, Suite 800, Elm Street, Capital City, CA 10001",
    "phone_number": "(+1) 100-100-1100"
  }

The **get** method also supports pagination. You can use the *offset* and *num_docs* arguments to retrieve the required window of documents for the query. By default, the **get** method uses a value of *num_docs=10*.

.. code-block:: python

  # Retrieve documents numbers 11 to 30
  kb.get(index='stores', query, context, offset=10, num_docs=20)

Advanced Settings
~~~~~~~~~~~~~~~~~

While creating the index, all fields in the data go through a process called "Analysis". "Analyzers" can be defined per field to define the following:

* Tokenizing a block of text into individual terms before adding to inverted index
* Normalizing these terms into a standard form to improve searchability

In MindMeld Workbench, you can optionally define custom analyzers per field by specifying an **es-mapping.json** file at the application root level. While the default MindMeld Workbench Analyzer uses a robust set of character filtering operations for tokenizing, custom analyzers can be handy for special character/token handling. For example, lets say we have a store named *"Springfield™ store"*. We want the indexer to ignore characters like "™" and "®" since users never specify these in their queries. We need to define a special character filter (*"char_filter"*) and analyzer mapping as follows:

.. code-block:: text

  {
    "mappings": {
      "properties": {
        "store_name": {
          "type": "string",
          "index_options": "docs",
          "analyzer": "keyword_with_folding_custom"
        }
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
        "keyword_with_folding_custom": {
          "type": "custom",
          "tokenizer": "keyword",
          "char_filter": [
            "remove_tm_and_r"
          ],
          "filter": [
            "lowercase",
            "asciifolding"
          ]
        }
      }
    }
  }

More information on custom analyzers and the **es_mapping.json** file is available in the User Guide chapter on the Knowledge Base. Example mapping files for a variety of use-cases and content types are also provided.


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

For the **store_information** domain, here are snippets of training examples for a few intents for Intent Classification. In a similar vein, we can define query sets for all other intents. These queries reside in *.txt* files under the **labeled_queries** folder of each intent directory as shown in Section 1.3.

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


Train the Natural Language Processing classifiers
---------------------------------------------------

The Natural Language Processor (NLP) is tasked with comprehending the user's natural language input. It analyzes the input using a hierarchy of classification models, with each model assisting the next tier of models by narrowing the problem scope, or in other words, successively narrowing down the “search space”.

There are a total of four classifiers, applied in the following order:

#. **Domain Classifier**: For apps that handle conversations across varied topics having their own specialized vocabulary, the domain classifier provides the first level of categorization by classifying the input into one of the pre-defined set of conversational domains.

#. **Intent Classifier**: The intent classifier next determines the specific informational or transactional user need by categorizing the input into a set of user intents that the system can handle.

#. **Entity Recognizer**: The entity recognizer then looks for relevant pieces of information in the input that are required to fulfill the user's end goal. It does this by extracting important words and phrases, called entities and assigning each of them a label that describes the type of information conveyed by it.

#. **Role Classifier**: In cases where an entity of a particular type can have multiple meanings depending on the context, the role classifier can be used to provide another level of categorization and assign semantic roles to the extracted entities.

These concepts are explained further in Chapter 3.4 and will also become clearer as you go over the walkthrough of an example app in Chapter 2.2. Chapter 3.9 describes these classifiers in detail along with the different configurations and options available for each. 

For an initial prototype of our "Store Information" app, we can get away with a simple Natural Language Processor that only uses the intent classifier and the entity recognizer. To make it easy for developers, the NLP class in Workbench makes this determination automatically based on your directory structure and the nature of your annotated data. As long as you have prepared your training data following the annotation guidelines in 2.1.6 and placed them in the directory structure described in 2.1.3, the NLP can figure out which classifiers need to be trained and which ones can be ignored.

In our case, we only have one de-facto domain called "store information" and correspondingly have only one folder at the "domain" level. But we do have multiple intent folders under that domain. Also, we annotated entities in the text along with their types, but did not specify any roles for the entities. Therefore, the NLP will only train an intent classifier and an entity recognizer.

Training the NLP classifiers for our app and persisting them to disk can be accomplished in these four simple lines of code:  

.. code-block:: python

  from mmworkbench import NLP

  # Instantiate MindMeld NLP by providing the app_data path.
  nlp = NLP('path_to_app_data_directory_root')

  # Train the NLP
  nlp.fit()

  # Save the trained NLP models to disk
  nlp.dump()

The code for training the NLP for an app that requires all our four classifiers would be exactly the same since the ``fit()`` method automatically makes that inference based on the format of the provided labeled training data. One other thing to note is that the above code will use the default machine learning model, feature extraction and hyper-parameter settings to train our classifiers. While that should be enough to give you a reasonable start, there are no shortcuts to creating a high quality conversational app. To get the best accuracy possible, you would need to understand each of the classifiers in depth and experiment with different classifier configurations to determine what's best for your particular scenario. Workbench enables this by making each of the individual classifiers configurable, so machine learning engineers can try out various model configurations, features, hyperparameters and cross validation settings.

For instance, you may want to specify that the intent classifier should use an SVM classifier instead of Logistic Regression (default) and additionally specify the model parameters that go with it. You may also want to increase the context window size (to 3 from the default 2) for the bag-of-word features computed and used by the entity recognizer. The code below shows how to accomplish this. Note that any settings left unspecified will use the Workbench default values.

.. code-block:: python

  from mmworkbench import NLP

  # Instantiate MindMeld NLP by providing the app_data path.
  nlp = NLP('path_to_app_data_directory_root')

  # Define model parameters for SVM training
  params = {
              "C": [5000],
              "class_bias": [0.5],
              "kernel": ["linear"],
              "probability": [true]
  }

  # Features for entity recognition
  entity_features = {
                      "bag-of-words": { "lengths": [1, 2, 3] },
                      "in-gaz": { "scaling": 10 },
                      "length": {}
  }

  # Train the NLP
  nlp.fit(intent_classifier_model='svm', 
          intent_classifier_params=params, 
          entity_recognizer_features=entity_features)

  # Save the trained NLP models to disk
  nlp.dump()

Refer to Chapter 3.9 of the Workbench User Guide for detailed documentation on all the NLP classifiers.

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


Optimize Question Answering
---------------------------

The Question Answering module is responsible for ranking results retrieved from the Knowledge Base, based on some notion of relevance. The MindMeld Knowledge Base offers a set of operators for ranking results retrieved. These operators are combined to define a "ranking formula". The ranking formula is a scoring function ("Function Score") that gets applied on each query as the metric for ranking Knowledge Base results. MindMeld Workbench provides a default implementation of the Function Score, which would work well for most applications.

The Function Score is a blend of **Text Relevance**, **Popularity** and **Sort** criteria (if present). If there are no sort entities present, then the Function Score blends the text relevance with descending popularity. The default implementation already considers the scaling factors and distributions of the text relevance scores to adjust the normalized popularity weight accordingly. If a sort entity is present, a decay function is applied to the corresponding sort field and combined with the scaled popularity and text relevance scores.

While the default ranking score implementation in MindMeld Workbench is well tuned and should work reasonably well for most applications, there is a flexible option to specify a custom ranking forumla if required. You need to produce a "ranking score" based on your choice of usage of the available arguments, which can then be applied as the scoring function after Knowledge Base retrieval.

Example -

File **app.py**

.. code-block:: python

  @app.kb.handle()
  def ranking_function_score():
    # Custom Ranking logic goes here. You can define arbitrary
    # logic for each of the scoring components.
    text_rel = compute_text_relevance_score(query, context)
    pop_score = compute_popularity_score(query, context)
    sort_factor = compute_sort_score(context.entities)

    # Combine the score factors as needed
    ranking_score = combine_factors(text_rel, pop_score, sort_factor)

    return ranking_score

The custom ranking function can then be used in the **get** method of the Knowledge Base object. 

.. code-block:: python

  # Assume KnowledgeBase object has been created and
  # the data is loaded into the 'stores' object.

  # Get ranked results from KB
  ranked_results = kb.get(index='stores', query,
              context, ranking_fn=ranking_function_score)

  print ranked_results

The process of fine tuning the scoring function can be mastered with more experience in building search ranking apps. But here are some general guidelines you can follow to optimize your ranking configuration -

#. Collect a set of few hundred (or thousand) diverse, representative queries
#. Run the queries through the parse + QA system with the default set of configurations
#. Analyze the results for Top 1 or Top K accuracy (depending on the use case)
#. Modify the ranking function to improve accuracy results for bulk of the misses (without compromising the correct results)
#. Repeat from Step 2

.. _Question Answering: question_answering.html

Detailed explanation on controlling Text Relevance is available in the User Guide chapter on `Question Answering`_. Also, if you would like to use a Machine Learning approach to ranking (Learning To Rank), more information on assembling the right kind of training data and building models is available in the User Guide chapter.

Deploy trained models to production
---------------------------------------
Show a simple example of the steps required to deploy to production