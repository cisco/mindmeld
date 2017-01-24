Step 7: Train the Natural Language Processing Classifiers
=========================================================

The Natural Language Processor (NLP) in Workbench is tasked with understanding the user's natural language input. It analyzes the input using a hierarchy of classification models, with each model assisting the next tier of models by narrowing the problem scope, or in other words, successively narrowing down the “search space”.

As introduced in :doc:`step 3 </define_the_hierarchy>`, Workbench relies on four layers of classifiers, applied in the following order:

#. **Domain Classifier**: For apps that handle conversations across varied topics having their own specialized vocabulary, the domain classifier provides the first level of categorization by classifying the input into one of a pre-defined set of conversational domains.

#. **Intent Classifiers**: The intent classifier next determines what the user is trying to accomplish by assigning each input into one of a set of pre-defined intents that the system can handle.

#. **Entity Recognizers**: The entity recognizer then extracts important words and phrases, called entities, that are required to fulfill the user's end goal.

#. **Role Classifiers**: In cases where an entity of a particular type can have multiple meanings depending on the context, the role classifier can be used to provide another level of categorization and assign a differentiating label called "role" to the extracted entities.

The first step towards training the NLP classifiers for our Kwik-E-Mart store information app is to gather the necessary training data as described in :doc:`step 6 </generate_representative_training_data>`. Once the data is ready, we can fire up a Python shell and start building the different components of our NLP. 

.. code-block:: console

  $ cd $WB_APP_ROOT
  $ python

.. attention::

  The rest of this step-by-step guide assumes that you are typing commands into a Python interactive shell launched from the app's root directory.

Once you are in the Python interactive shell, the quickest way to train all the NLP classifiers together is using the following lines of code.

.. code-block:: python

  >>> from mmworkbench import NaturalLanguageProcessor as NLP
  >>> nlp = NLP()
  >>> nlp.build()


Based on the directory structure and the annotations in the training data, the Natural Language Processor automatically infers which classifiers need to be trained. In our case, the NLP will train an intent classifier for the ``store_information`` domain and entity recognizers for each of the intents which contain labeled queries with entity annotations. This simple example did not include training data for domain classification or role classification, and consequently, these models will not be built.

By default, the ``build()`` method shown above will use the baseline machine learning settings for all classifiers, which in most cases should train reasonable models. To further optimize model performance, Workbench provides extensive capabilities to optimize individual model parameters and measure results. We'll next take a closer look at each of the NLP components and learn how to experiment with different settings for each of them individually.


Domain Classification
~~~~~~~~~~~~~~~~~~~~~

The Domain Classifier (also called the domain model) is a text classification model that is trained using the labeled queries across all domains. Our simple Kwik-e-Mart app only has one domain to support and hence does not need a domain classifier. However, complex conversational apps such as the virtual assistants on all smartphones today have to handle queries from varied domains such as "weather", "navigation", "sports", "finance" and "music" among others. Such apps use domain classification as the first step to help narrow down the focus of the subsequent classifiers in the NLP pipeline.

The ``NaturalLanguageProcessor`` class in Workbench exposes methods for training, testing and saving all the models in our classifier hierarchy, including the domain model. For the sake of example, let us suppose that we want to build a `support vector machine (SVM) <https://en.wikipedia.org/wiki/Support_vector_machine>`_ that does domain classification. In our Python shell, we start off by importing and instantiating an object of the ``NaturalLanguageProcessor`` class. We then train the ``domain_classifier`` by calling its ``fit()`` method.

.. code-block:: python

  >>> from mmworkbench import NaturalLanguageProcessor as NLP
  >>> nlp = NLP()
  >>> nlp.domain_classifier.fit(model='svm')

The trained classifier can be tested on a new query using the ``predict()`` method.

.. code-block:: python

  >>> predicted_domain = nlp.domain_classifier.predict(u'Play my jazz playlist.')
  >>> predicted_domain
  u'music'

In addition to the model type parameter that we've used above, the ``fit()`` method also takes arguments for features, cross-validation settings and other model-specific configuration to improve upon the baseline SVM model trained by default. These are covered in detail in the :ref:`User Guide <userguide>`.


Intent Classification
~~~~~~~~~~~~~~~~~~~~~

Intent Classifiers (also called intent models) are text classification models that are trained, one-per-domain, using the labeled queries in each intent folder. Our Kwik-e-Mart app supports multiple intents (e.g. ``greet``, ``get_store_hours``, ``find_nearest_store``, etc.) within the ``store_information`` domain. We will now go over the steps for training an intent classifier that can correctly map user queries to one of these supported intents.

We'll train our intent model similar to the domain model using the ``NaturalLanguageProcessor`` class, but we will go a step further and explicitly define the features and cross validation settings we want to use this time. For our intent classifier, let us assume that we want to build a `logistic regression <https://en.wikipedia.org/wiki/Logistic_regression>`_ model and use `bag of words <https://en.wikipedia.org/wiki/Bag-of-words_model>`_ and `edge n-grams <https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-edgengram-tokenizer.html>`_ as features. Also, we would like to do `k-fold cross validation <https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation>`_  with 20 splits.

We start as before by importing and instantiating the ``NaturalLanguageProcessor`` class. In addition, we import the ``KFold`` module from the ``scikit-learn`` library to define cross validation settings.

.. code-block:: python

  >>> from mmworkbench import NaturalLanguageProcessor
  >>> from sklearn.model_selection import KFold
  >>> nlp = NaturalLanguageProcessor()

Next, we define the feature dictionary that lists all the feature types along with the feature-specific settings. Let's say we want bag-of-n-grams up to size 2 and similarly, edge-ngrams up to length 2.

.. code-block:: python

  >>> feature_dict = {
  ... 'bag-of-words': { 'lengths': [1, 2] },
  ... 'edge-ngrams': { 'lengths': [1, 2] }
  ... }


We then define a cross validation iterator with the desired number of splits.

.. code-block:: python

  >>> kf = KFold(n_splits=20)

Finally, we fetch the ``intent_classifier`` for the domain we are interested in and call its ``fit()`` method to train the model. The code below shows how to train an intent classifier for the ``store_information`` domain in our Kwik-e-Mart app.

.. code-block:: python

  >>> clf = nlp.domains['store_information'].intent_classifier
  >>> clf.fit(model='logreg', features=feature_dict, cv=kf)

We have now successfully trained an intent classifier for the ``store_information`` domain. If our app had more domains, we would follow the same steps for those other domains. We can test the trained intent model on a new query by calling its ``predict()`` method.

.. code-block:: python

  >>> predicted_intent = clf.predict(u'Where is my closest Kwik-e-Mart?')
  >>> predicted_intent
  u'find_nearest_store'

Once we have experimented with different settings and have an optimized intent model that we are happy with, we can use the ``dump()`` method to persist the trained model to disk. By default, intent classifier models get saved to a ``models`` directory under their respective domains. 

.. code-block:: python

  >>> clf.dump()

The :ref:`User Guide <userguide>` has a comprehensive list of the different model, feature extraction and hyperparameter settings for training the intent models. It also describes how to evaluate a trained intent model using labeled test data.

Entity Recognition
~~~~~~~~~~~~~~~~~~

Entity Recognizers (also called entity models) are sequence labeling models that are trained per intent using the annotated queries in each entity folder. The task of the entity recognizer is both to detect the entities within a query and label them as one of the pre-defined entity types.

We'll again use Workbench's ``NLP`` class to train our entity recognizer. Let's use a `Maximum Entropy Markov Model <https://en.wikipedia.org/wiki/Maximum-entropy_Markov_model>`_, which is a good choice for sequence labeling tasks. For features, one of the most helpful and commonly used sources of information in entity recognition models is a comprehensive list of entity names called a "`gazetteer <https://gate.ac.uk/sale/tao/splitch13.html#x18-32600013.1>`_". Each entity type has its own gazetteer. In our case, the gazetteer for the ``store_name`` entity type would be a list of all the Kwik-e-Mart store names in our catalog. Gazetteers can then be used to derive features based on full or partial match of words in the query against entries in the gazetteers. 

[TODO: Add the location for the gazetteer file, and mention the file format (do we require a popularity field?)]

Apart from using gazetteer-based features, we'll use bag-of-words features like we did for intent classification. Length of the current token also ends up being a useful feature for entity recognition, so we'll add that too. Finally, we'll continue using 20-fold cross validation like we did before. Below is the code to instantiate an NLP object, define the features and initialize a k-fold iterator.

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

  intent = nlp.domains['store_information'].intents['get_open_time']
  intent.fit_entity_model(model='memm', features=feature_dict, cv=kfold_cv)
  intent.dump_entity_model()

We can similarly train the entity recognizers for other intents as well. The trained entity model can be tested using the ``predict_entities()`` method.

.. code-block:: python

  predicted_entities = intent.predict_entities(u'When does the Main Street store open?')

The :doc:`Entity Recognizer User Guide </entity_recognition>` goes into more detail about all the available training and evaluation options.

We have now looked at how to individually build the intent classification and entity recognition models for our "Kwik-e-Mart Store Information" app. Once we have experimented with different settings (model type, features, training parameters, etc.) for each of our classifiers and found the optimal configuration, we can save those settings in a build configuration file and have Workbench use it the next time we invoke the ``build`` command.

.. code-block:: text

  python my_app.py build --config build_config.json

This is the quickest way to retrain your classifiers in production (e.g. in case of a training data refresh) using the best known model configuration settings. For details on the configuration file format and a more in-depth treatment of the NLP classifiers in Workbench, refer to the :ref:`User Guide <userguide>`.

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

