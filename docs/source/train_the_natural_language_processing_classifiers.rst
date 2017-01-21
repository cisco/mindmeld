Train the Natural Language Processing Classifiers
=================================================

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

