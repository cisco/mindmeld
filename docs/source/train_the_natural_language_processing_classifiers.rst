Step 7: Train the Natural Language Processing Classifiers
=========================================================

The Natural Language Processor (NLP) in Workbench is tasked with understanding the user's natural language input. It analyzes the input using a hierarchy of classification models, with each model assisting the next tier of models by narrowing the problem scope, or in other words, successively narrowing down the 'solution space'.

As introduced in :doc:`step 3 </define_the_hierarchy>`, Workbench relies on four layers of classifiers, applied in the following order:

#. **Domain Classifier**: For apps that handle conversations across varied topics, each having their own specialized vocabulary, the domain classifier provides the first level of categorization by classifying the input into one of a pre-defined set of conversational domains.

#. **Intent Classifiers**: The intent classifier next determines what the user is trying to accomplish by assigning each input into one of a set of pre-defined intents that the system can handle.

#. **Entity Recognizers**: The entity recognizer then extracts important words and phrases, called entities, that are required to fulfill the user's end goal.

#. **Role Classifiers**: In cases where an entity of a particular type can have multiple meanings depending on the context, the role classifier can be used to provide another level of categorization and assign a differentiating label, called a 'role', to the extracted entities.

The first step towards training the NLP classifiers for our Kwik-E-Mart store information app is to gather the necessary training data as described in :doc:`step 6 </generate_representative_training_data>`. Once the data is ready, we can fire up a python shell and start building the different components of our natural language processor. 

.. code-block:: console

  $ cd $WB_APP_ROOT
  $ python

Once you are in the python interactive shell, the quickest way to train all the NLP classifiers together is using the :keyword:`nlp.build()` method, as shown below.

.. code-block:: python

  >>> from mmworkbench import NaturalLanguageProcessor as nlp
  >>> nlp.build()

This method will train all models in the specified NLP pipeline. Based on the directory structure and the annotations in the training data, the Natural Language Processor automatically infers which classifiers need to be trained. In our case, the NLP will train an intent classifier for the ``store_info`` domain and entity recognizers for each of the intents which contain labeled queries with entity annotations. This simple example did not include training data for domain classification or role classification, and consequently, these models will not be built. 

To run all of the trained models in the NLP pipeline, use the :keyword:`nlp.parse()` command as shown below.

.. code-block:: python

  >>> nlp.parse('When does One Market close?')
  {
    ...
    'domain': {'target': 'get_store_info', 'probs': []},
    'intent': {
      'target': 'get_store_hours', 
      'probs': [...]
    },
    'entities': [
      {
        'type': 'store_name',
        'span': {
          'raw': 'One Market',
          'norm': 'one market'
        },
        'value': 'One Market',
        'confidence': 0.934512
        ...
      }
    ]
    ...
  }
  
The :keyword:`nlp.parse()` command will return detailed information about the output of each of the trained NLP models. See the :ref:`User Manual <userguide>` for more details.

By default, the :keyword:`build()` method shown above will use the baseline machine learning settings for all classifiers, which in most cases should train reasonable models. To further optimize model performance, Workbench provides extensive capabilities to optimize individual model parameters and measure results. We'll next take a closer look at each of the NLP components and learn how to experiment with different settings for each of them individually.


Domain Classification
~~~~~~~~~~~~~~~~~~~~~

The domain classifier (also called the domain model) is a text classification model that is trained using the labeled queries across all domains. Our simple Kwik-E-Mart app only has one domain and hence does not need a domain classifier. However, complex conversational apps such as the popular virtual assistants on smartphones today have to handle queries from varied domains such as weather, navigation, sports, finance and music, among others. Such apps use domain classification as the first step to help narrow down the focus of the subsequent classifiers in the NLP pipeline.

The :keyword:`NaturalLanguageProcessor` class in Workbench exposes methods for training, testing and saving all the models in our classifier hierarchy, including the domain model. For the sake of example, let us suppose that we want to build a `support vector machine (SVM) <https://en.wikipedia.org/wiki/Support_vector_machine>`_ that does domain classification. In our python shell, we start off by importing an object of the :keyword:`NaturalLanguageProcessor` class. We then train the :keyword:`domain_classifier` model by calling its :keyword:`fit()` method.

.. code-block:: python

  >>> from mmworkbench import NaturalLanguageProcessor as nlp
  >>> nlp.domain_classifier.fit(model='svm')

The trained classifier can be tested on a new query using the :keyword:`predict()` method.

.. code-block:: python

  >>> nlp.domain_classifier.predict('Play my jazz playlist.')
  {
    'target': 'music', 
    'probs': [
      'music': 0.982342,
      'sports': 0.134523,
      'weather': 0.087263,
      'finance': 0.026346
    ]
  }

As you can see, the model output includes the predicted target domain as well as the classification probabilities associated with all of the available domains. In addition to the model type parameter that we've used above, the :keyword:`fit()` method also takes arguments for features, cross-validation settings and other model-specific configuration to improve upon the baseline SVM model trained by default. These are covered in detail in the :ref:`User Manual <userguide>`.

Intent Classification
~~~~~~~~~~~~~~~~~~~~~

Intent classifiers (also called intent models) are text classification models that are trained, one-per-domain, using the labeled queries in each intent folder. Our Kwik-E-Mart app supports multiple intents (e.g. ``greet``, ``get_store_hours``, ``find_nearest_store``, etc.) within the ``store_info`` domain. We will now go over the steps for training an intent classifier that can correctly map user queries to one of these supported intents.

We will train our intent model similarly to the domain model using the :keyword:`NaturalLanguageProcessor` class, but we will go a step further and explicitly define the features and cross validation settings we want to use this time. For our intent classifier, let us assume that we want to build a `logistic regression <https://en.wikipedia.org/wiki/Logistic_regression>`_ model and use `bag of words <https://en.wikipedia.org/wiki/Bag-of-words_model>`_ and `edge n-grams <https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-edgengram-tokenizer.html>`_ as features. Also, we would like to do `k-fold cross validation <https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation>`_  with 20 splits.

We start as before by importing and instantiating the :keyword:`NaturalLanguageProcessor` class. In addition, we import the :keyword:`KFold` module from the :keyword:`scikit-learn` library to define cross validation settings.

.. code-block:: python

  >>> from mmworkbench import NaturalLanguageProcessor as nlp
  >>> from sklearn.model_selection import KFold

Next, we define the feature dictionary that lists all the feature types along with the feature-specific settings. Let's say we want bag-of-n-grams up to size 2 and similarly, edge-ngrams up to length 2.

.. code-block:: python

  >>> feature_dict = {
  ...   'bag-of-words': { 'lengths': [1, 2] },
  ...   'edge-ngrams': { 'lengths': [1, 2] }
  ... }


We then define a cross-validation iterator with the desired number of splits.

.. code-block:: python

  >>> kf = KFold(n_splits=20)

Finally, we fetch the :keyword:`intent_classifier` for the domain we are interested in and call its :keyword:`fit()` method to train the model. The code below shows how to train an intent classifier for the ``store_info`` domain in our Kwik-E-Mart app.

.. code-block:: python

  >>> clf = nlp.domains['store_info'].intent_classifier
  >>> clf.fit(model='logreg', features=feature_dict, cv=kf)

We have now successfully trained an intent classifier for the ``store_info`` domain. If our app had more domains, we would follow the same steps for those other domains. We can test the trained intent model on a new query by calling its :keyword:`predict()` method.

.. code-block:: python

  >>> clf.predict('Where is my closest Kwik-e-Mart?')
  {
    'target': 'find_nearest_store', 
    'probs': [
      'find_nearest_store': 0.929584,
      'get_store_hours': 0.182523,
      'help': 0.097163,
      'greet': 0.010293,
      'exit': 0.009283
    ]
  }

Once we have experimented with different settings and have an optimized intent model that we are happy with, we can use the :keyword:`dump()` method to persist the trained model to file. 

.. code-block:: python

  >>> clf.dump()

Refer to the :ref:`User Manual <userguide>` for a comprehensive list of the different model, feature extraction and hyperparameter settings for training the domain and intent models. It also describes how to evaluate trained models using labeled test data.

Entity Recognition
~~~~~~~~~~~~~~~~~~

Entity recognizers (also called entity models) are `sequence labeling <https://en.wikipedia.org/wiki/Sequence_labeling>`_ models that are trained per intent using all the annotated queries in a particular intent folder in the :keyword:`domains` directory. The task of the entity recognizer is both to detect the entities within a query as well as label them as one of the pre-defined entity types.

From the model hierarchy we defined for our Kwik-E-Mart app in :ref:`step 3 <model_hierarchy>`, we can see that the ``get_store_hours`` intent depends on two types of entities. Of these, ``date`` is a 'system' entity that Workbench will recognize automatically. The ``store_name`` entity, on the other hand, will require custom training data and a trained entity model. We'll next take a look at how the :keyword:`NaturalLanguageProcessor` class can be used to train entity recognizers for detecting custom entities in user queries.

In this example. we will use a `Maximum Entropy Markov Model <https://en.wikipedia.org/wiki/Maximum-entropy_Markov_model>`_, which is a good choice for sequence labeling tasks like entity recognition. For features, one of the most powerful and commonly used sources of information in entity recognition models is a comprehensive list of popular entity names called a '`gazetteer <https://gate.ac.uk/sale/tao/splitch13.html#x18-32600013.1>`_'. For instance, the gazetteer for the ``store_name`` entity type could be a list of all the Kwik-E-Mart store names in our catalog. This list is stored in a text file called :keyword:`gazetteer.txt` and located in appropriate subdirectory of the :keyword:`entities` folder. An example of this file is shown below.

.. code-block:: text

  3rd Street
  Central Plaza
  East Oak Street
  Elm Street
  Evergreen Terrace
  Main Street
  Main and Market
  Market Square
  Shelbyville
  Spalding Way
  Springfield Mall
  ...

If we had more entity types, we would similarly have gazetteer lists for the other types as well. Workbench will automatically utilize gazetteers named :keyword:`gazetteer.txt` that are located in their respective entity folders. Gazetteers can be used to derive features based on full or partial match of words in the query against entries in the gazetteers. They are particularly helpful for detecting entities which might otherwise seem to be a sequence of common nouns (e.g. 'main street', 'main and market', etc.). Apart from using gazetteer-based features, we'll also use bag-of-words features like we did for our earlier classifiers. The length of the current token also ends up being a useful feature for entity recognition, so we'll add that too. Finally, we'll continue using 20-fold cross validation as before.

Below is the code to import the :keyword:`NaturalLanguageProcessor` object, define the features and initialize a k-fold iterator.

.. code-block:: python

  >>> from mmworkbench import NaturalLanguageProcessor as nlp
  >>> from sklearn.model_selection import KFold
  >>> feature_dict = {
  ...   'in-gaz': {},
  ...   'bag-of-words': { 'lengths': [1, 2] },
  ...   'length': {}
  ... }
  >>> kf = KFold(n_splits=20)

We then get the entity recognizer for the desired intent and invoke its :keyword:`fit()` method. We also serialize the trained model to disk for future use.

.. code-block:: python

  >>> recognizer = nlp.domains['store_info'].intents['get_store_hours'].entities['store_name'].recognizer
  >>> recognizer.fit(model='memm', features=feature_dict, cv=kfold_cv)
  >>> recognizer.dump()

We have thus trained and saved the ``get_name`` entity recognizer for the ``get_store_hours`` intent. Had additional entity recognizers been required, we would have repeated the same steps as above for each entity in each intent. The trained entity recognizer can be tested using its :keyword:`predict()` method.

.. code-block:: python

  >>> recognizer.predict(u'When does the store on Elm Street close?')
  [
    {
      'type': 'store_name',
      'span': {
        'raw': 'Elm Street',
        'norm': 'elm street'
      },
      'value': 'Elm Street',
      'confidence': 0.934512
      ...
    }
  ]
  
The :ref:`User Manual <userguide>` goes into more detail about all the available training and evaluation options for the entity recognizer.

Role Classification
~~~~~~~~~~~~~~~~~~~

Role Classifiers (also called role models) are trained per entity using all the annotated queries in a particular intent folder. Roles offer a way to assign an additional distinguishing label to entities of the same type. Our simple Kwik-e-Mart application does not need a role classification layer. However, consider a possible extension to our app, where users can search for stores that open and close at specific times. As we saw in the example in :ref:`step 6 <roles_example>`, this would require us to differentiate between the two ``sys:time`` entities by recognizing one as an ``open_time`` and the other as a ``close_time``. This can be accomplished by training an entity-specific role classifier that assigns the correct role label for each such ``sys:time`` entity detected by the Entity Recognizer.

Let us see how Workbench can be used for training a role classifier for the ``sys:time`` entity type. As with the previous classifiers, this involves the predictable workflow of instantiating a :keyword:`NaturalLanguageProcessor` object, accessing the classifier of interest (in this case, the :keyword:`role_classifier` for the ``sys:time`` entity), defining the machine learning settings and calling the :keyword:`fit()` method of the classifier. For this example, we will just train a baseline `Maximum Entropy model <http://repository.upenn.edu/cgi/viewcontent.cgi?article=1083&context=ircs_reports>`_ without specifying any additional training settings. Also, for the sake of code readability, we'll retrieve the classifier of interest in two steps - first get the object representing the current intent and then fetch the :keyword:`role_classifier` object of the appropriate entity under that intent.

.. code-block:: python

  >>> from mmworkbench import NaturalLanguageProcessor as NLP
  >>> nlp = NLP()
  >>> get_hours_intent = nlp.domains['store_information'].intents['get_store_hours']
  >>> clf = get_hours_intent.entities['sys:time'].role_classifier
  >>> clf.fit()

Once the classifier is trained, we can test it on a new query using the familiar :keyword:`predict()` method.

.. code-block:: python

  >>> predicted_roles = clf.predict(u'Show me stores open between 8 AM and 6 PM.')
  >>> predicted_roles
  {u'8 AM': u'open_time', u'6 PM': u'close_time'}

Our baseline role classifier can be further optimized using the various training and evaluation options detailed in the :ref:`User Manual <userguide>`

Entity Resolution
~~~~~~~~~~~~~~~~~

The entity resolver component of MindMeld Workbench is responsible for mapping each identified entity to a canonical  value. For example, if your application is used to browse TV shows, you may want to map both entity strings 'funny' and 'hilarious' to a pre-defined genre code like 'Comedy'. Similarly, in a music app, you may want to resolve both 'Elvis' and 'The King' to the known artist 'Elvis Presley (ID=20192)', while making sure not get confused by 'Elvis Costello (ID=139028)'. For some classes of entities, it can be pretty straightforward. For other entities, it can be quite complex and the dominant factor which may limit the overall accuracy of your application.

MindMeld Workbench provides advanced capabilities for building a state-of-the-art entity resolver. As discussed in 
:doc:`step 6 </generate_representative_training_data>`, each entity type can be associated with an optional entity mapping file. This file specifies, for each canonical concept, the possible alternate names or synonyms a user may express to refer to this concept. In the absence of an entity mapping file, the entity resolver simply assigns a value equivalent to the entity raw text span. For example, the following code illustrates the possible parse output of the natural language processor when an entity mapping data file is absent for the ``store_name`` entity:

.. code-block:: python

  >>> from mmworkbench import NaturalLanguageProcessor as nlp
  >>> nlp.build()
  >>> nlp.parse('When does One Market close?')
  {
    ...
    'entities': [
      {
        'type': 'store_name',
        'span': {
          'raw': 'One Market',
          'norm': 'one market'
        },
        'value': 'One Market',
        'confidence': 0.934512
        ...
      }
    ]
    ...
  }

If an entity mapping file is specified, as illustrated in :doc:`step 6 </generate_representative_training_data>`, the output of the natural language processor may be as follows

.. code-block:: python

  >>> from mmworkbench import NaturalLanguageProcessor as nlp
  >>> nlp.build()
  >>> nlp.parse('When does One Market close?')
  {
    ...
    'entities': [
      {
        'type': 'store_name',
        'span': {
          'raw': 'One Market',
          'norm': 'one market'
        },
        'value': {'id': 207492, 'cname': 'Market Square'},
        'confidence': 0.934512
        ...
      }
    ]
    ...
  }

Note that the :keyword:`value` attribute of the entity has resolved to an object with a defined id and canonical name. As with the other NLP components in Workbench, it is also possible to access the individual resolvers for each entity type. The code below illustrates how to train and evaluate the entity resolver model for the ``store_name`` entity.

.. code-block:: python

  >>> from mmworkbench import NaturalLanguageProcessor as nlp
  >>> resolver = nlp.domains[0].intents['get_store_hours'].entities['store_name'].resolver

  >>> # Train the resolver model using the mapping file, if available.
  ... resolver.fit()
  
  >>> # Run the model 
  ... resolver.predict('One Market')
  {'id': 207492, 'cname': 'Market Square'}

Refer to the :ref:`User Manual <userguide>` for more information about how to evaluation and optimize entity resolution models for your application.
