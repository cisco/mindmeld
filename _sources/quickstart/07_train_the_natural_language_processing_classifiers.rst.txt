Step 7: Train the Natural Language Processing Classifiers
=========================================================

The Natural Language Processor (NLP) in MindMeld is tasked with understanding the user's natural language input. It analyzes the input using a hierarchy of classification models. Each model assists the next tier of models by narrowing the problem scope, or in other words successively narrowing down the 'solution space.'

As introduced in :doc:`Step 3 <03_define_the_hierarchy>`, MindMeld applies four layers of classifiers in the following order:

#. **Domain Classifier** classifies input into one of a pre-defined set of conversational domains. Only necessary for apps that handle conversations across varied topics, each with its own specialized vocabulary.

#. **Intent Classifiers** determine what the user is trying to accomplish by assigning each input to one of the intents defined for your application.

#. **Entity Recognizers** extract the words and phrases, or *entities*, that are required to fulfill the user's end goal.

#. **Role Classifiers** assign a differentiating label, called a *role*, to the extracted entities. This level of categorization is only necessary where an entity of a particular type can have multiple meanings depending on the context.

.. note::

   The code examples in this chapter assume that you have installed the Kwik-E-Mart and Home
   Assistant blueprint applications. See the
   :doc:`blueprints overview page <../blueprints/overview>` for details on installing the apps.

To train the NLP classifiers for our Kwik-E-Mart store information app, we must first gather the necessary training data as described in :doc:`Step 6 <06_generate_representative_training_data>`. Once the data is ready, we open a Python shell and start building the components of our natural language processor.

.. code-block:: shell

   cd $MM_APP_ROOT
   python

In the Python shell, the quickest way to train all the NLP classifiers together is to use the :meth:`nlp.build()` method.

.. code-block:: python

   from mindmeld.components.nlp import NaturalLanguageProcessor
   nlp = NaturalLanguageProcessor('.')
   nlp.build()

This method trains all models in the specified NLP pipeline. The Natural Language Processor automatically infers which classifiers need to be trained based on the directory structure and the annotations in the training data. In our case, the NLP will train an intent classifier for the ``store_info`` domain and entity recognizers for each intent that contains labeled queries with entity annotations. Domain classification and role classification models will not be built because our simple example did not include training data for them.

To run all of the trained models in the NLP pipeline, use the :meth:`nlp.process()` command.

.. code-block:: python

   nlp.process('When does Elm Street close?')

.. code-block:: console

   {'text': 'When does Elm Street close?',
    'domain': 'store_info',
    'intent': 'get_store_hours',
    'entities': [{'text': 'Elm Street',
      'type': 'store_name',
      'role': None,
      'value': [{'cname': '23 Elm Street',
        'score': 44.777046,
        'top_synonym': 'Elm Street',
        'id': '1'},
       {'cname': '104 First Street',
        'score': 7.0927515,
        'top_synonym': '104 First Street',
        'id': '5'},
       {'cname': 'East Oak Street',
        'score': 7.0927515,
        'top_synonym': 'East Oak Street',
        'id': '12'},
       {'cname': '257th Street',
        'score': 6.958622,
        'top_synonym': '257th Street',
        'id': '18'},
       {'cname': 'D Street',
        'score': 6.7008686,
        'top_synonym': 'D Street',
        'id': '19'},
       {'cname': '181st Street',
        'score': 6.630241,
        'top_synonym': '181st Street',
        'id': '17'},
       {'cname': 'West Oak Street',
        'score': 6.249679,
        'top_synonym': 'West Oak Street',
        'id': '11'},
       {'cname': '156th Street',
        'score': 6.1613703,
        'top_synonym': '156th Street',
        'id': '15'},
       {'cname': 'Peanut Street',
        'score': 6.1613703,
        'top_synonym': 'Peanut Street',
        'id': '20'},
       {'cname': 'Little Italy Store',
        'score': 5.2708626,
        'top_synonym': 'Third Street',
        'id': '7'}],
      'span': {'start': 10, 'end': 19}}
      ]
   }

The :meth:`nlp.process()` command returns detailed information about the output of each of the trained NLP models. See the :doc:`User Guide <../userguide/nlp>` for details.

By default, the :meth:`build()` method shown above uses the baseline machine learning settings for all classifiers, which should train reasonable models in most cases. To further improve model performance, MindMeld provides extensive capabilities for optimizing individual model parameters and measuring results. We'll next explore how to experiment with different settings for each NLP component individually.

.. _domain_classification:

Domain Classification
~~~~~~~~~~~~~~~~~~~~~

The domain classifier (also called the domain model) is a text classification model that is trained using the labeled queries across all domains. Our simple app only has one domain and hence does not need a domain classifier. However, complex conversational apps such as the popular virtual assistants on smartphones and smart speakers today have to handle queries from varied domains such as weather, navigation, sports, finance, and music, among others. Such apps use domain classification as the first step to narrow down the focus of the subsequent classifiers in the NLP pipeline.

To see the domain classifier in action, you can download and try out the ``home_assistant`` blueprint application.

.. code-block:: python

   import mindmeld as mm
   mm.configure_logs()
   mm.blueprint('home_assistant')

The :class:`NaturalLanguageProcessor` class in MindMeld exposes methods for training, testing, and saving all the models in our classifier hierarchy, including the domain model. For example, suppose we want to build a `logistic regression classifier <https://en.wikipedia.org/wiki/Logistic_regression>`_ that does domain classification. In our Python shell, we start off by instantiating an object of the :class:`NaturalLanguageProcessor` class. We then train the :attr:`domain_classifier` model by calling its :meth:`fit()` method.

.. note::

   Since our simple Kwik-E-Mart app does not have a domain classifier, the example below uses the
   :doc:`../blueprints/home_assistant` blueprint to demonstrate the functionality.

.. code-block:: python

   from mindmeld.components.nlp import NaturalLanguageProcessor
   nlp = NaturalLanguageProcessor('home_assistant')
   nlp.domain_classifier.fit(model_settings={'classifier_type': 'logreg'})

We test the trained classifier on a new query using the :meth:`predict()` method.

.. code-block:: python

   nlp.domain_classifier.predict('close the kitchen door')

.. code-block:: console

   'smart_home'

To view the classification probabilities associated with all available domains, we can use the :meth:`predict_proba()` method.

.. code-block:: python

   nlp.domain_classifier.predict_proba('close the kitchen door')

.. code-block:: console

   [
    ('smart_home', 0.9999634367987815),
    ('times_and_dates', 1.81768265134388e-05),
    ('weather', 1.2388247900671112e-05),
    ('unknown', 4.110616819853133e-06),
    ('greeting', 1.8875099844624723e-06)
   ]

In addition to the `model` parameter we used above, the :meth:`fit()` method also takes parameters we can use to improve upon the baseline SVM model trained by default. These include parameters for features, cross-validation settings, and other model-specific configuration. See the :doc:`User Guide <../userguide/domain_classifier>` for details.

.. _intent_classification:

Intent Classification
~~~~~~~~~~~~~~~~~~~~~

Intent classifiers (also called intent models) are text classification models that are trained, one-per-domain, using the labeled queries in each intent folder. Our Kwik-E-Mart app supports multiple intents (e.g. ``greet``, ``get_store_hours``, ``find_nearest_store``, etc.) within the ``store_info`` domain. We will now see how to train an intent classifier that correctly maps user queries to one of these supported intents.

Training our intent model is similar to training the domain model using the :class:`NaturalLanguageProcessor` class, but this time we explicitly define the features and cross-validation settings we want to use. For our intent classifier, let us assume that we want to build a `logistic regression <https://en.wikipedia.org/wiki/Logistic_regression>`_ model and use `bag of words <https://en.wikipedia.org/wiki/Bag-of-words_model>`_ and `edge n-grams <https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-edgengram-tokenizer.html>`_ as features. Also, we would like to do `k-fold cross validation <https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation>`_  with 10 splits to find the ideal `hyperparameter <https://en.wikipedia.org/wiki/Hyperparameter_optimization>`_ values.

We demonstrate intent classification using the simpler Kwik-E-Mart application. We start as before by instantiating a :class:`NaturalLanguageProcessor` object.

.. code-block:: shell

   cd $MM_APP_ROOT
   python

.. code-block:: python

   from mindmeld.components.nlp import NaturalLanguageProcessor
   nlp = NaturalLanguageProcessor('.')

Next, we define the feature dictionary that lists all the feature types along with the feature-specific settings. Let's say we want bag-of-n-grams up to size 2 and edge-ngrams up to length 2.

.. code-block:: python

   feature_dict = {
       'bag-of-words': { 'lengths': [1, 2] },
       'edge-ngrams': { 'lengths': [1, 2] }
   }

We then define the hyperparameter selection settings.

.. code-block:: python

   search_grid = {
     'C': [0.01, 1, 10, 100, 1000],
     'class_bias': [0, 0.3, 0.7, 1]
   }

   hyperparam_settings = {
     'type': 'k-fold',
     'k': 10,
     'grid': search_grid
   }

Finally, we fetch the :attr:`intent_classifier` for the domain we are interested in and call its :meth:`fit()` method to train the model. The code below shows how to train an intent classifier for the ``store_info`` domain in our Kwik-E-Mart app.

.. code-block:: python

   clf = nlp.domains['store_info'].intent_classifier
   clf.fit(model_settings={'classifier_type': 'logreg'},
           features=feature_dict,
           param_selection=hyperparam_settings)


We have now successfully trained an intent classifier for the ``store_info`` domain. If our app had more domains, we would follow the same procedure for those other domains. We can test the trained intent model on a new query by calling its :meth:`predict()` and :meth:`predict_proba()` methods.

.. code-block:: python

   clf.predict('Where is my closest Kwik-E-Mart?')

.. code-block:: console

   'find_nearest_store'

.. code-block:: python

   clf.predict_proba('Where is my closest Kwik-E-Mart?')

.. code-block:: console

   [
       ('find_nearest_store', 0.999995),
       ('get_store_hours', 0.000005),
       ('greet', 0.000000),
       ('exit', 0.000000),
       ('help', 0.000000)
   ]


Once we have experimented with different settings and have an optimized intent model that we are happy with, we persist the trained model to a local file using the :meth:`dump()` method.

.. code-block:: python

   my_app_dump = 'models/experimentation/intent_model_logreg.pkl'
   clf.dump(my_app_dump)

See the :doc:`User Guide <../userguide/intent_classifier>` for a comprehensive list of the different model, feature extraction and hyperparameter settings for training the domain and intent models. The :doc:`User Guide <../userguide/intent_classifier>` also describes how to evaluate trained models using labeled test data.

.. _entity_recognition:

Entity Recognition
~~~~~~~~~~~~~~~~~~

Entity recognizers (also called entity models) are `sequence labeling <https://en.wikipedia.org/wiki/Sequence_labeling>`_ models that are trained per intent using all the annotated queries in a particular intent folder in the ``domains`` directory. The entity recognizer detects the entities within a query, and labels them as one of the pre-defined entity types.

From the model hierarchy we defined for our Kwik-E-Mart app in :ref:`Step 3 <model_hierarchy>`, we can see that the ``get_store_hours`` intent depends on two types of entities. Of these, ``sys_time`` is a system entity that MindMeld recognizes automatically. The ``store_name`` entity, on the other hand, requires custom training data and a trained entity model. Let's look at how to use the :class:`NaturalLanguageProcessor` class to train entity recognizers for detecting custom entities in user queries.

In this example we use a `Maximum Entropy Markov Model <https://en.wikipedia.org/wiki/Maximum-entropy_Markov_model>`_, which is a good choice for sequence labeling tasks like entity recognition. The features we use include a *gazetteer* , which is a comprehensive list of popular entity names. `Gazetteers <https://gate.ac.uk/sale/tao/splitch13.html#x18-32600013.1>`_ are among the most powerful and commonly used sources of information in entity recognition models. Our example gazetteer for the ``store_name`` entity type is a list of all the Kwik-E-Mart store names in our catalog, stored in a text file called ``gazetteer.txt`` and located in the appropriate subdirectory of the ``entities`` folder. MindMeld automatically utilizes any gazetteer named ``gazetteer.txt`` that is located within an entity folder. The example gazetteer file looks like this:

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

When words in a query fully or partly match a gazetteer entry, that can be used to derive features. This makes gazetteers particularly helpful for detecting entities which might otherwise seem to be a sequence of common nouns, such as `main street`, `main and market`, and so on. Apart from using gazetteer-based features, we'll use the bag of n-grams surrounding the token as additional features. Finally, we'll continue using 10-fold cross validation as before.

Below is the code to instantiate a :class:`NaturalLanguageProcessor` object, define the features, and the hyperparameter selection settings.

.. code-block:: python

   from mindmeld.components.nlp import NaturalLanguageProcessor
   nlp = NaturalLanguageProcessor('.')
   feature_dict = {
     'in-gaz-span-seq': {},
     'bag-of-words-seq':{
         'ngram_lengths_to_start_positions': {
             1: [-1, 0, 1],
             2: [-1, 0, 1]
         }
     }
   }
   search_grid = {
     'C': [0.01, 1, 10, 100, 1000],
     'penalty': ['l1', 'l2']
   }
   hyperparam_settings = {
     'type': 'k-fold',
     'k': 10,
     'grid': search_grid
   }

Next, we get the entity recognizer for the desired intent and invoke its :meth:`fit()` method. We also serialize the trained model to disk for future use.

.. code-block:: python

   recognizer = nlp.domains['store_info'].intents['get_store_hours'].entity_recognizer
   recognizer.fit(model_settings={'classifier_type': 'memm'},
                  features=feature_dict,
                  param_selection=hyperparam_settings)
   recognizer.dump('models/experimentation/entity_model_memm.pkl')

We have now trained and saved the entity recognizer for the ``get_store_hours`` intent. If more entity recognizers were required, we would have repeated the same procedure for each entity in each intent. We test the trained entity recognizer using its :meth:`predict()` method.

.. code-block:: python

   recognizer.predict('When does the store on Elm Street close?')

.. code-block:: console

  (<QueryEntity 'Elm Street' ('store_name') char: [23-32], tok: [5-6]>,)

See the :doc:`User Guide <../userguide/entity_recognizer>` for more about entity recognizer training and evaluation options.

.. _role_classification:

Role Classification
~~~~~~~~~~~~~~~~~~~

Role classifiers (also called role models) are trained per entity using all the annotated queries in a particular intent folder. Roles offer a way to assign an additional distinguishing label to entities of the same type. Our simple Kwik-E-Mart application does not need a role classification layer. However, consider a possible extension to our app, where users can search for stores that open and close at specific times. As we saw in the example in :ref:`Step 6 <roles_example>`, this would require us to differentiate between the two ``sys_time`` entities by recognizing one as an ``open_time`` and the other as a ``close_time``. This can be accomplished by training an entity-specific role classifier that assigns the correct role label for each such ``sys_time`` entity detected by the Entity Recognizer.

Let's walk through the process of using MindMeld to train a role classifier for the ``sys_time`` entity type. The workflow is just like the previous classifiers: instantiate a :class:`NaturalLanguageProcessor` object; access the classifier of interest (in this case, the :attr:`role_classifier` for the ``sys_time`` entity); define the machine learning settings; and, call the :meth:`fit()` method of the classifier. For this example, we will just use MindMeld's default configuration (Logistic Regression) to train a baseline role classifier without specifying any additional training settings. For the sake of code readability, we retrieve the classifier of interest in two steps: first get the object representing the current intent, then fetch the :attr:`role_classifier` object of the appropriate entity under that intent.

.. note::

   The Kwik-E-Mart blueprint distributed with MindMeld does not use role classification. The code
   snippet below shows a possible extension to the app where the ``sys_time`` entity is further
   classified into two different roles.

   For an example you can run readily, see the :ref:`Home Assistant example <ha_role_example>`
   further below.

.. code-block:: python

   from mindmeld.components.nlp import NaturalLanguageProcessor
   nlp = NaturalLanguageProcessor('.')
   get_hours_intent = nlp.domains['store_info'].intents['get_store_hours']
   # MindMeld doesn't know about entities until the training queries have been loaded.
   # Load queries for the relevant intent by calling build().
   get_hours_intent.build()
   # Get the role classifier for the 'sys_time' entity
   clf = get_hours_intent.entities['sys_time'].role_classifier
   clf.fit()

Once the classifier is trained, we test it on a new query using the familiar :meth:`predict()` method. The :meth:`predict()` method of the role classifier requires both the full input query and the set of entities predicted by the entity recognizer.

.. code-block:: python

   query = 'Show me stores open between 8 AM and 6 PM.'
   recognizer = get_hours_intent.entity_recognizer
   predicted_entities = recognizer.predict(query)
   clf.predict(query, predicted_entities, 0)

.. code-block:: console

   'open_time'

.. _ha_role_example:

Here is a different example of role classification from the :doc:`../blueprints/home_assistant`
blueprint. The home assistant app leverages roles to correctly implement the functionality of
changing alarms, e.g. "Change my 6 AM alarm to 7 AM".

First, we train the role classifier.

.. code-block:: python

   from mindmeld.components.nlp import NaturalLanguageProcessor
   nlp = NaturalLanguageProcessor(app_path='home_assistant')
   change_alarm_intent = nlp.domains['times_and_dates'].intents['change_alarm']
   change_alarm_intent.build()
   clf = change_alarm_intent.entities['sys_time'].role_classifier
   clf.fit()

We can then test the classifier on a new query.

.. code-block:: python

   query = 'Change my 6 AM alarm to 7 AM'
   recognizer = change_alarm_intent.entity_recognizer
   predicted_entities = recognizer.predict(query)
   clf.predict(query, predicted_entities, 0)

.. code-block:: console

   'old_time'

.. code-block:: python

   clf.predict(query, predicted_entities, 1)

.. code-block:: console

   'new_time'

We can further optimize our baseline role classifier using the training and evaluation options detailed in the :doc:`User Guide <../userguide/role_classifier>`.

.. _entity_resolution:

Entity Resolution
~~~~~~~~~~~~~~~~~

The entity resolver component of MindMeld maps each identified entity to a canonical value. For example, if your application is used for browsing TV shows, you may want to map both entity strings `funny` and `hilarious` to a pre-defined genre code like `Comedy`. Similarly, in a music app, you may want to resolve both `Elvis` and `The King` to the artist `Elvis Presley (ID=20192)`, while making sure not to get confused by `Elvis Costello (ID=139028)`. Entity resolution can be straightforward for some classes of entities. For others, it can be complex enough to constitute the dominant factor limiting the overall accuracy of your application.

MindMeld provides advanced capabilities for building a state-of-the-art entity resolver. As discussed in :doc:`Step 6 <06_generate_representative_training_data>`, each entity type can be associated with an optional entity mapping file. This file specifies, for each canonical concept, the alternate names or synonyms with which a user may refer to this concept. In the absence of an entity mapping file, the entity resolver cannot resolve the entity. Instead, it logs a warning and skips adding a :attr:`value` attribute to the entity. For example, the following code illustrates the output of the natural language processor when an entity mapping data file is absent for the ``store_name`` entity:

.. code-block:: python

   from mindmeld.components.nlp import NaturalLanguageProcessor
   nlp = NaturalLanguageProcessor('.')
   nlp.build()
   nlp.process("When does the one on elm open?")

.. code-block:: console

   Failed to resolve entity 'elm' for type 'store_name'
   {
     'domain': 'store_info',
     'entities': [
       {
         'role': None,
         'span': {'end': 23, 'start': 21},
         'text': 'elm',
         'type': 'store_name'
        }
     ],
     'intent': 'get_store_hours',
     'text': 'When does the one on elm open?'
   }

If an entity mapping file is specified, as illustrated in :doc:`Step 6 <06_generate_representative_training_data>`, the entity resolver resolves the entity to a defined ID and canonical name. It assigns these to the :attr:`value` attribute of the entity, in the form of an object. Then the output of the natural language processor could resemble the following.

.. code-block:: python

   from mindmeld.components.nlp import NaturalLanguageProcessor
   nlp = NaturalLanguageProcessor('.')
   nlp.build()
   nlp.process("When does the one on elm open?")

.. code-block:: console

   {
     'domain': 'store_info',
     'entities': [
       {
         'role': None,
         'span': {'end': 23, 'start': 21},
         'text': 'elm',
         'type': 'store_name',
         'value': [{'cname': '23 Elm Street', 'id': '1'}],
        }
     ],
     'intent': 'get_store_hours',
     'text': 'When does the one on elm open?'
   }

As with the other NLP components in MindMeld, you can access the individual resolvers for each entity type.

The code below illustrates how to train and evaluate the entity resolver model for the ``store_name`` entity.

.. code-block:: python

   from mindmeld.components.nlp import NaturalLanguageProcessor
   nlp = NaturalLanguageProcessor('.')
   # MindMeld doesn't know about entities until the training queries have been loaded.
   # Load queries for the relevant intent by calling build().
   nlp.domains['store_info'].intents['get_store_hours'].build()
   # Get the entity resolver for the entity type of interest.
   resolver = nlp.domains['store_info'].intents['get_store_hours'].entities['store_name'].entity_resolver

   # Train the resolver model using the mapping file, if available.
   resolver.fit()

   # Run the model on a detected entity
   recognizer = nlp.domains['store_info'].intents['get_store_hours'].entity_recognizer
   entities = recognizer.predict('When does the store on Elm Street close?')
   resolver.predict(entities[0])

.. code-block:: console

  [{'cname': '23 Elm Street', 'score': 40.69433, 'top_synonym': 'Elm Street', 'id': '1'}, ...]

See the :doc:`User Guide <../userguide/entity_resolver>` for more about how to evaluate and optimize entity resolution models.
