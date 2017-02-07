Step 7: Train the Natural Language Processing Classifiers
=========================================================

The Natural Language Processor (NLP) in Workbench is tasked with understanding the user's natural language input. It analyzes the input using a hierarchy of classification models. Each model assists the next tier of models by narrowing the problem scope, or in other words successively narrowing down the 'solution space.'

As introduced in :doc:`Step 3 </define_the_hierarchy>`, Workbench applies four layers of classifiers in the following order:

#. **Domain Classifier** classifies input into one of a pre-defined set of conversational domains. Only necessary for apps that handle conversations across varied topics, each with its own specialized vocabulary.

#. **Intent Classifiers** determine what the user is trying to accomplish by assigning each input to one of the intents defined for your application.

#. **Entity Recognizers** extract the words and phrases, or *entities*, that are required to fulfill the user's end goal.

#. **Role Classifiers** assign a differentiating label, called a *role*, to the extracted entities. This level of categorization is only necessary where an entity of a particular type can have multiple meanings depending on the context.

To train the NLP classifiers for our Kwik-E-Mart store information app, we must first gather the necessary training data as described in :doc:`Step 6 </generate_representative_training_data>`. Once the data is ready, we open a Python shell and start building the components of our natural language processor.

.. code-block:: console

  $ cd $WB_APP_ROOT
  $ python

In the Python shell, the quickest way to train all the NLP classifiers together is to use the :keyword:`Nlp.build()` method.

.. code-block:: python

  >>> from mmworkbench import NaturalLanguageProcessor as Nlp
  >>> Nlp.build()

This method trains all models in the specified NLP pipeline. The Natural Language Processor automatically infers which classifiers need to be trained based on the directory structure and the annotations in the training data. In our case, the NLP will train an intent classifier for the ``store_info`` domain and entity recognizers for each intent that contains labeled queries with entity annotations. Domain classification and role classification models will not be built because our simple example did not include training data for them.

To run all of the trained models in the NLP pipeline, use the :keyword:`Nlp.parse()` command.

.. code-block:: python

  >>> Nlp.parse('When does One Market close?')
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



The :keyword:`Nlp.parse()` command returns detailed information about the output of each of the trained NLP models. See the :ref:`User Manual <userguide>` for details.

By default, the :keyword:`build()` method shown above uses the baseline machine learning settings for all classifiers, which should train reasonable models in most cases. To further improve model performance, Workbench provides extensive capabilities for optimizing individual model parameters and measuring results. We'll next explore how to experiment with different settings for each NLP component individually.


Domain Classification
~~~~~~~~~~~~~~~~~~~~~

The domain classifier (also called the domain model) is a text classification model that is trained using the labeled queries across all domains. Our simple Kwik-E-Mart app only has one domain and hence does not need a domain classifier. However, complex conversational apps such as the popular virtual assistants on smartphones today have to handle queries from varied domains such as weather, navigation, sports, finance, and music, among others. Such apps use domain classification as the first step to narrow down the focus of the subsequent classifiers in the NLP pipeline.

The :keyword:`NaturalLanguageProcessor` class in Workbench exposes methods for training, testing, and saving all the models in our classifier hierarchy, including the domain model. For example, suppose we want to build a `support vector machine (SVM) <https://en.wikipedia.org/wiki/Support_vector_machine>`_ that does domain classification. In our Python shell, we start off by importing an object of the :keyword:`NaturalLanguageProcessor` class. We then train the :keyword:`domain_classifier` model by calling its :keyword:`fit()` method.

.. code-block:: python

  >>> from mmworkbench import NaturalLanguageProcessor as Nlp
  >>> Nlp.domain_classifier.fit(model='svm')

We test the trained classifier on a new query using the :keyword:`predict()` method.

.. code-block:: python

  >>> Nlp.domain_classifier.predict('Play my jazz playlist.')
  {
    'target': 'music',
    'probs': [
      'music': 0.982342,
      'sports': 0.134523,
      'weather': 0.087263,
      'finance': 0.026346
    ]
  }

The model output includes the predicted target domain along with the classification probabilities associated with all available domains. In addition to the `model` parameter we used above, the :keyword:`fit()` method also takes parameters we can use to improve upon the baseline SVM model trained by default. These include parameters for features, cross-validation settings, and other model-specific configuration. See the :ref:`User Manual <userguide>` for details.

Intent Classification
~~~~~~~~~~~~~~~~~~~~~

Intent classifiers (also called intent models) are text classification models that are trained, one-per-domain, using the labeled queries in each intent folder. Our Kwik-E-Mart app supports multiple intents (e.g. ``greet``, ``get_store_hours``, ``find_nearest_store``, etc.) within the ``store_info`` domain. We will now see how to train an intent classifier that correctly maps user queries to one of these supported intents.

Training our intent model is similar to training the domain model using the :keyword:`NaturalLanguageProcessor` class, but this time we explicitly define the features and cross-validation settings we want to use. For our intent classifier, let us assume that we want to build a `logistic regression <https://en.wikipedia.org/wiki/Logistic_regression>`_ model and use `bag of words <https://en.wikipedia.org/wiki/Bag-of-words_model>`_ and `edge n-grams <https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-edgengram-tokenizer.html>`_ as features. Also, we would like to do `k-fold cross validation <https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation>`_  with 20 splits.

We start as before by importing and instantiating the :keyword:`NaturalLanguageProcessor` class. In addition, we import the :keyword:`KFold` module from the :keyword:`scikit-learn` library to define cross-validation settings.

.. code-block:: python

  >>> from mmworkbench import NaturalLanguageProcessor as Nlp
  >>> from sklearn.model_selection import KFold

Next, we define the feature dictionary that lists all the feature types along with the feature-specific settings. Let's say we want bag-of-n-grams up to size 2 and edge-ngrams up to length 2.

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

  >>> clf = Nlp.domains['store_info'].intent_classifier
  >>> clf.fit(model='logreg', features=feature_dict, cv=kf)

We have now successfully trained an intent classifier for the ``store_info`` domain. If our app had more domains, we would follow the same procedure for those other domains. We can test the trained intent model on a new query by calling its :keyword:`predict()` method.

.. code-block:: python

  >>> clf.predict('Where is my closest Kwik-E-Mart?')
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

Once we have experimented with different settings and have an optimized intent model that we are happy with, we persist the trained model to file using the :keyword:`dump()` method.

.. code-block:: python

  >>> clf.dump()

See the :ref:`User Manual <userguide>` for a comprehensive list of the different model, feature extraction and hyperparameter settings for training the domain and intent models. The :ref:`User Manual <userguide>` also describes how to evaluate trained models using labeled test data.

Entity Recognition
~~~~~~~~~~~~~~~~~~

Entity recognizers (also called entity models) are `sequence labeling <https://en.wikipedia.org/wiki/Sequence_labeling>`_ models that are trained per intent using all the annotated queries in a particular intent folder in the :keyword:`domains` directory. The entity recognizer detects the entities within a query, and labels them as one of the pre-defined entity types.

From the model hierarchy we defined for our Kwik-E-Mart app in :ref:`Step 3 <model_hierarchy>`, we can see that the ``get_store_hours`` intent depends on two types of entities. Of these, ``date`` is a system entity that Workbench recognizes automatically. The ``store_name`` entity, on the other hand, requires custom training data and a trained entity model. Let's look at how to use the :keyword:`NaturalLanguageProcessor` class to train entity recognizers for detecting custom entities in user queries.

In this example we use a `Maximum Entropy Markov Model <https://en.wikipedia.org/wiki/Maximum-entropy_Markov_model>`_, which is a good choice for sequence labeling tasks like entity recognition. The features we use include a *gazetteer* , which is a comprehensive list of popular entity names. `Gazetteers <https://gate.ac.uk/sale/tao/splitch13.html#x18-32600013.1>`_ are among the most powerful and commonly used sources of information in entity recognition models. Our example gazetteer for the ``store_name`` entity type is a list of all the Kwik-E-Mart store names in our catalog, stored in a text file called :keyword:`gazetteer.txt` and located in the appropriate subdirectory of the :keyword:`entities` folder. Workbench automatically utilizes any gazetteer named :keyword:`gazetteer.txt` that is located within an entity folder. The example gazetteer file looks like this:

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

If we had more entity types, we would have gazetteer lists for them, too.

When words in a query fully or partly match a gazetteer entry, that can be used to derive features. This makes gazetteers particularly helpful for detecting entities which might otherwise seem to be a sequence of common nouns, such as `main street`, `main and market`, and so on. Apart from using gazetteer-based features, we'll also use bag-of-words features like we did earlier. The length of the current token can also be a useful feature for entity recognition, so we'll add that too. Finally, we'll continue using 20-fold cross validation as before.

Below is the code to import the :keyword:`NaturalLanguageProcessor` object, define the features, and initialize a k-fold iterator.

.. code-block:: python

  >>> from mmworkbench import NaturalLanguageProcessor as Nlp
  >>> from sklearn.model_selection import KFold
  >>> feature_dict = {
  ...   'in-gaz': {},
  ...   'bag-of-words': { 'lengths': [1, 2] },
  ...   'length': {}
  ... }
  >>> kf = KFold(n_splits=20)

Next, we get the entity recognizer for the desired intent and invoke its :keyword:`fit()` method. We also serialize the trained model to disk for future use.

.. code-block:: python

  >>> recognizer = Nlp.domains['store_info'].intents['get_store_hours'].entities['store_name'].recognizer
  >>> recognizer.fit(model='memm', features=feature_dict, cv=kfold_cv)
  >>> recognizer.dump()

We have now trained and saved the ``get_name`` entity recognizer for the ``get_store_hours`` intent. If more entity recognizers were required, we would have repeated the same procedure for each entity in each intent. We test the trained entity recognizer using its :keyword:`predict()` method.

.. code-block:: python

  >>> recognizer.predict('When does the store on Elm Street close?')
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

See the :ref:`User Manual <userguide>` for more about entity recognizer training and evaluation options.

Role Classification
~~~~~~~~~~~~~~~~~~~

Role classifiers (also called role models) are trained per entity using all the annotated queries in a particular intent folder. Roles offer a way to assign an additional distinguishing label to entities of the same type. Our simple Kwik-E-Mart application does not need a role classification layer. However, consider a possible extension to our app, where users can search for stores that open and close at specific times. As we saw in the example in :ref:`Step 6 <roles_example>`, this would require us to differentiate between the two ``sys:time`` entities by recognizing one as an ``open_time`` and the other as a ``close_time``. This can be accomplished by training an entity-specific role classifier that assigns the correct role label for each such ``sys:time`` entity detected by the Entity Recognizer.

Let us see how Workbench can be used for training a role classifier for the ``sys:time`` entity type. As with the previous classifiers, this involves the predictable workflow of instantiating a :keyword:`NaturalLanguageProcessor` object, accessing the classifier of interest (in this case, the :keyword:`role_classifier` for the ``sys:time`` entity), defining the machine learning settings and calling the :keyword:`fit()` method of the classifier. For this example, we just train a baseline `Maximum Entropy model <http://repository.upenn.edu/cgi/viewcontent.cgi?article=1083&context=ircs_reports>`_ without specifying any additional training settings. For the sake of code readability, we retrieve the classifier of interest in two steps: first get the object representing the current intent, then fetch the :keyword:`role_classifier` object of the appropriate entity under that intent.

.. code-block:: python

  >>> from mmworkbench import NaturalLanguageProcessor as Nlp
  >>> get_hours_intent = Nlp.domains['store_info'].intents['get_store_hours']
  >>> clf = get_hours_intent.entities['sys:time'].role_classifier
  >>> clf.fit(model='memm')

Once the classifier is trained, we test it on a new query using the familiar :keyword:`predict()` method. The :keyword:`predict()` method of the role classifier requires both the full input query and the set of entities predicted by the entity recognizer.

.. code-block:: python

  >>> query = 'Show me stores open between 8 AM and 6 PM.'
  >>> recognizer = get_hours_intent.entities['sys:time'].recognizer
  >>> predicted_entities = recognizer.predict(query)
  >>> clf.predict(query, predicted_entities)
  {'8 AM': 'open_time', '6 PM': 'close_time'}

We can further optimize our baseline role classifier using the training and evaluation options detailed in the :ref:`User Manual <userguide>`.

Entity Resolution
~~~~~~~~~~~~~~~~~

The entity resolver component of MindMeld Workbench maps each identified entity to a canonical value. For example, if your application is used for browsing TV shows, you may want to map both entity strings `funny` and `hilarious` to a pre-defined genre code like `Comedy`. Similarly, in a music app, you may want to resolve both `Elvis` and `The King` to the artist `Elvis Presley (ID=20192)`, while making sure not to get confused by `Elvis Costello (ID=139028)`. Entity resolution can be straightforward for some classes of entities. For others, it can be complex enough to constitute the dominant factor limiting the overall accuracy of your application.

MindMeld Workbench provides advanced capabilities for building a state-of-the-art entity resolver. As discussed in :doc:`Step 6 </generate_representative_training_data>`, each entity type can be associated with an optional entity mapping file. This file specifies, for each canonical concept, the alternate names or synonyms with which a user may refer to this concept. In the absence of an entity mapping file, the entity resolver cannot resolve the entity. Instead, it simply assigns the entity raw text span to the :keyword:`value` attribute of the entity. For example, the following code illustrates the possible parse output of the natural language processor when an entity mapping data file is absent for the ``store_name`` entity:

.. code-block:: python

  >>> from mmworkbench import NaturalLanguageProcessor as Nlp
  >>> Nlp.build()
  >>> Nlp.parse('When does One Market close?')
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

If an entity mapping file is specified, as illustrated in :doc:`Step 6 </generate_representative_training_data>`, the entity resolver resolves the entity to a defined ID and canonical name. It assigns these to the :keyword:`value` attribute of the entity, in the form of an object. Then the output of the natural language processor could resemble the following.

.. code-block:: python

  >>> from mmworkbench import NaturalLanguageProcessor as Nlp
  >>> Nlp.build()
  >>> Nlp.parse('When does One Market close?')
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

As with the other NLP components in Workbench, you can access the individual resolvers for each entity type.

The code below illustrates how to train and evaluate the entity resolver model for the ``store_name`` entity.

.. code-block:: python

  >>> from mmworkbench import NaturalLanguageProcessor as Nlp
  >>> resolver = Nlp.domains[0].intents['get_store_hours'].entities['store_name'].resolver

  >>> # Train the resolver model using the mapping file, if available.
  ... resolver.fit()

  >>> # Run the model
  ... resolver.predict('One Market')
  {'id': 207492, 'cname': 'Market Square'}

See the :ref:`User Manual <userguide>` for more about how to evaluate and optimize entity resolution models.
