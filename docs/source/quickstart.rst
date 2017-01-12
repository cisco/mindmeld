Building a Conversational Interface in 11 Steps
===============================================


This section outlines the step-by-step approach to building conversational interfaces using MindMeld workbench. This section provides four real-world examples (tutorials) to illustrate each of the steps in practice.


This section provides a high-level overview introducing the steps.

1. Select the right use case
----------------------------
Selecting the right use case is critical. Selecting an unrealistic or incorrect use case will render even the smartest voice or chat assistant dead on arrival. 
The best user cases today typically mimic an existing, familiar real-world interaction. This is the best way to ensure that users will know what they can ask.
Any use case must offer a practical path for collecting the training data required for the app.
The best use cases provide a clear value or utility where voice can be a faster way to find information or accomplish a task.

A good candidate use case will have the following:

- mimics a real-world human interaction (no instructions required)
- possible to collect sufficient training data
- save the user time
- when user knows how to articulate specifically what they want
- when the users hands are busy
- when no one else is around
- goldilocks domain - not too small to be trivial, but not to vast to be intractable



2. Script your ideal dialogue interactions
------------------------------------------
Write down the conversational dialogue interactions in detail.

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

3. Define the domain, intent, entity and role hierarchy
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


4. Define the dialog state handlers
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



5. Create the Knowledge Base
----------------------------
A Knowledge Base is a repository for storing complex, real-world, structured and unstructured information relevant to a content catalog. In the context of Deep-Domain Conversational AI, a Knowledge Base comprises of a searchable repository of objects that encode entities and their associated attributes. 

Here are some examples of Knowledge Base object types for various applications -

* Product objects in a **Products** catalog - Attributes include "sku_id", "name", "price"
* Video objects in a **Video Content** library - Attributes include "description", "cast", "duration"
* Menu items from a **Quick Service Restaurants** catalog - Attributes include "size", "price", "modifiers" etc.

In our example of store information on Kwik-E-Mart stores, we would have store objects with the following attributes -

* store_name
* open_time
* close_time
* address
* phone_number

A Knowledge Base can have 1 or more indexes. An "index" is a logical namespace which maps to objects of a particular type in the Knowledge Base. Roughly, you can think of an index as a "database" in a relational database world. Different indexes can be used to map to objects of different types. In our example of Kwik-E-Mart stores data, we would have just 1 index - *"stores"*.

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

The MindMeld Knowledge Base offers a versatile set of ways to specify the **get** to retrieve results for a variety of increasingly complex inputs. Further information is available in the User Guide chapter on the Knowledge Base.


6. Generate representative training data
----------------------------------------
Components in Mindmeld Workbench utilize Supervised Learning models to analyze a user's query and derive meaning out of it. To train each of these components, we typically require thousands to millions of *labeled* queries to build powerful models. **It is critical that you obtain high-quality, representative training data** to ensure high accuracy. The training data serves as the ground truth for the models, so it is imperative that the ground truth data is clean and represents the exact use-case that you are training the model for.

Some strategies for collecting training data are -

#. Human Data Entry
#. Mining The Web
#. Crowdsourcing
#. Operational Logs (Customer Service, Search etc.)

In MindMeld Workbench, there are 6 components that need training data for a Machine Learning based Conversational Application. Typically, a given application would need training data for some subset of these components depending on the domain and core use-cases.

* Domain Classification
* Intent Classification
* Entity Recognition
* Role Classification
* Entity Resolution
* Ranking

In our example application of Kwik-E-Mart store information, Domain Classification is not needed since we have only one domain - **store_information**. In case we have additional domains (such as **order_item**), we would need separate sets of training queries for each domain. In such cases, MindMeld Workbench provides the facility of using queries from all the intents belonging to a domain as labeled queries for that domain. For example, training queries for the **store_information** domain would be the union of all queries in the *greet*, *get_close_time*, *get_open_time*, *get_nearest_store*, *get_is_open_now* and *exit* intents. The folder structure described in Section 3 provides an easy way of specifying your queries pertaining to a domain.

Additionally, in this example application, Entity Resolution could be a simple map from the detected entity to it's construct of retrieving documents from the Knowledge Base, so no special training data is required for this step. Similarly, Role Classification and Ranking are not a must, so we don't need additional training data for those components. For the remaining components, we can specify the training labels in the following way -

Intent Classification -

For the **store_information** domain, here are snippets of training examples for each intent for Intent Classification -

* **greet**

.. code-block:: text

  Hi
  Hello
  Good morning
  ...

* **get_close_time**

.. code-block:: text

  when does the elm street store close?
  what's the shut down time for pine & market store?
  ...

In a similar vein, we can define query sets for all other intents.

Entity Recognition -

To train the MindMeld Entity Recognizer, we need to label sections of the training queries with corresponding entity types. We do this by adding annotations to our training queries to identify all the entities. As a convenience in MindMeld Workbench, the training data for Entity Recognition and Role Classification are stored in the same files that contain queries for Intent Classification. To locate these files, please refer to the folder structure as specified in Section 3. For adding annotations for Entity Recognition, mark up the parts of every query that correspond to an entity in the following syntax -

* Enclose the entity in curly braces
* Follow the entity with its type
* Use the pipe character as separator

We recommend using a popular text editor such as Vim, Emacs or Sublime Text 3 to create these annotations. This process is normally much faster than creating GUIs and point-and-click systems for annotating data at scale.

Examples -

.. code-block:: text

  Is the {Central Plaza|name} Kwik-E-Mart open {now|time}?
  The store near {Pine & Market|intersection} - is it open?
  Is the {Rockerfeller|name} Kwik-E-Mart on {30th Street|street} open for business?
  Can you check if the {Market St|street} store is open at {6 pm tomorrow|time}?

.. note::

  Pro tip - Academic datasets (though instrumental in researching advanced algorithms), are not always reflective of real-world conversational data. Therefore, datasets from popular conferences such as TREC and ACM-SIGDIAL might not be the best choice for developing production applications.


7. Train the Natural Language Processor classifiers
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

8. Train the entity resolvers
-----------------------------
Introduce the topic of loading training data, training entity resolution models, measuring CV and held-out performance, performing disambiguation.

9. Implement the semantic parser
--------------------------------
Introduce the topic of semantic and dependency parsing. Illustrate a simple example of a rule-based or grammar-based parser which groups entities into a tree data structure.

10. Optimize Question Answering
-------------------------------
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

11. Deploy trained models to production
---------------------------------------
Show a simple example of the steps required to deploy to production