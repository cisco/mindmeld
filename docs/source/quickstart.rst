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

**Does it save users time?**                          Conversational interfaces shine when they save users time. A conversational interface that is never faster than a than a well-designed GUI will likely frustrate users. The most useful conversational experiences often center around a use case where users are looking to accomplish a specific task and know how to articulate it. For example, simply saying 'play my jazz music playlist in the kitchen' can be much faster than launching an app and navigating to the equivalent option by touch.

**Is it more convenient for the user?**               Voice interfaces can be particularly useful when users' hands and attention are occupied or if a mobile device is not within reach. If you expect users will often enlist your application while driving, biking, walking, exercising, cooking or sitting on the couch, it is likely an excellent candidate for a conversational interface.

**Does it hit the Goldilocks zone?**                  The best conversational applications fall squarely into the 'Goldilocks zone'. They offer a range of functionality that is narrow enough to ensure the machine learning models have high accuracy. At the same time, they have functionality that is broad enough to ensure that users find the experience useful for a wide variety of tasks. Apps that are too narrow can be trivial and useless. Apps that are too broad can have hit-or-miss accuracy which can frustrate users.

**Is it possible to get enough training data?**       Even the best use cases will fail if it is not possible or practical to collect enough training data to illustrate the complete range of envisioned functionality. For ideal use cases, training data can be easily generated from production traffic or crowdsourcing techniques. If training data for your use case can only be sourced from a small number of hard-to-find human experts, it is likely not a good candidate for a conversational interface.
===================================================== ===



Script your ideal dialogue interactions
------------------------------------------
Write down the conversational dialogue interactions in detail. This is equivalent to creating your wireframes and pixel-perfect mocks for a GUI.

  :User: Hello.
  :Bot: Hello, Pat. I can help you find store hours for your local Kwik-E-Mart. How can I help?
  :User: When does the store on Elm Street close today?
  :Bot: The 23 Elm Street Kwik-E-Mart at closes at 9pm today.
  :User: Where is the nearest Kwik-E-Mart?
  :Bot: Your nearest Kwik-E-Mart is located at Pine and Market.
  :User: When does that store open tomorrow?
  :Bot: The Pine and Market Kwik-E-Mart opens at 6am tomorrow.
  :User:  Is the Central Plaza Kwik-E-Mart open now?
  :Bot: Yes. The Kwik-E-Mart at Central Plaza closes at 8pm this evening.
  :User: Goodbye.
  :Bot: Have a nice day.

Define the domain, intent, entity and role hierarchy
-------------------------------------------------------
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

A Knowledge Base can have 1 or more indexes. An index is a logical namespace which maps to objects of a particular type in the Knowledge Base. Roughly, you can think of an index as a "database" in a relational database world. Different indexes can be used to map to objects of different types. In our example of Kwik-E-Mart stores data, we would have just 1 index - *"stores"*.

To setup your Knowledge Base using MindMeld Workbench, you can specify your content catalog as a JSON dump. The MindMeld Knowledge Base can read this JSON dump and extract all fields along with their types directly from the data. For example, if a field is a String, the Knowledge Base would read and store it as searchable strings. If a Float value is encountered, that attribute is stored as float values such that *"greater-than"* and *"lesser-than"* operators are applicable. 

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

If the specified index in the **load** method already exists, the data corresponding to that index will be overwritten. If not, a new index with that name is created and tha data is loaded in accordingly.

Once your data is loaded, you can use the **get** method to retrieve objects from the Knowledge Base -

.. code-block:: python

  # Get relevant objects from the KB
  query = "show me all stores on Elm Street that are open at 4 pm"
  results = kb.get(index='stores', query)

  print results

Output -

.. code-block:: text

  {
    "store_name": "Central Plaza Store", "open_time": "8:00 am", "close_time": "6:00 pm",
    "address": "100 Central Plaza, Suite 800, Elm Street, Capital City, CA 10001",
    "phone_number": "(+1) 100-100-1100"
  }

The MindMeld Knowledge Base offers a versatile set of ways to specify the **get** to retrieve results for a variety of increasingly complex inputs. Additionally, for granular control over the tokenization and text processing mechanisms for searching the Knowledge Base, more information is available in the User Guide chapter on the Knowledge Base.


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

Examples -

.. code-block:: text

  Is the {Central Plaza|name} Kwik-E-Mart open {now|time}?
  The store near {Pine & Market|intersection} - is it open?
  Is the {Rockerfeller|name} Kwik-E-Mart on {30th Street|street} open for business?
  Can you check if the {Market St|street} store is open at {6 pm tomorrow|time}?

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


Optimize question answering performance
-------------------------------------------
The Question Answering module is responsible for ranking results retrieved from the Knowledge Base, based on some notion of relevance. Just as in a relational database, MindMeld Workbench offers a set of operators for ranking results retrieved.

The ranking formula is a blend of text relevance, popularity and sort criteria (if any). MindMeld Workbench provides a default ranking function off-the-shelf that works well in most cases, but there is a flexible option to specify a custom ranking forumla if needed.

File **app.py**

.. code-block:: python

  @app.kb.handle()
  def custom_ranking_function():    
    # Custom Ranking logic goes here
    return text_relevance_coeff, popularity_coeff, sort_coeff

The custom ranking function can then be used in the **get** method of the Knowledge Base object.

.. code-block:: python

  # Assume KnowledgeBase object has been created and
  # the data is loaded into the 'stores' object.

  # Get ranked results from KB
  ranked_results = kb.get(index='stores', query,
              context, ranking_fn=custom_ranking_function)

  print ranked_results

.. _Question Answering: question_answering.html

Detailed explanations on custom ranking specifications are available in the User Guide chapter on `Question Answering`_.

Deploy trained models to production
---------------------------------------
Show a simple example of the steps required to deploy to production