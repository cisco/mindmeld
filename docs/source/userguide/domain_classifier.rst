.. meta::
    :scope: private

Domain Classifier
=================

The Domain Classifier is run as the first step in the natural language processing pipeline to determine the target domain for a given query. It is a `text classification <https://en.wikipedia.org/wiki/Text_classification>`_ model that is trained using all of the labeled queries across all the domains in an application. The name of each domain folder serves as the label for the training queries contained within that folder. See :doc:`Step 6 <../quickstart/06_generate_representative_training_data>` for more details on training data preparation. A domain classifier is only needed for apps that support more than one domain.

.. note::

   For a quick introduction, refer to :ref:`Step 7 <domain_classification>` of the Step-By-Step Guide.


Training a baseline domain classifier
-------------------------------------

As described in :doc:`Step 7 <../quickstart/07_train_the_natural_language_processing_classifiers>`, the fastest way to train baseline versions of all the NLP classifiers is by using the ``nlp.build()`` method.

.. code-block:: python

  >>> from mmworkbench import configure_logs; configure_logs()
  >>> from mmworkbench.components.nlp import NaturalLanguageProcessor
  >>> nlp = NaturalLanguageProcessor(app_path='my_app')
  >>> nlp.build()
  >>> nlp.dump()

The above code trains all the components of the NLP pipeline using Workbench's default settings. To test 




To train a specific classifier individually, you can use the 


To only train the domain classifier, use  



We instantiate an object of the ``NaturalLanguageProcessor`` class by passing it the path to a Workbench project that contains all the training data for all our classifiers.

We then train the :keyword:`domain_classifier` model by calling its :keyword:`fit()` method.

.. code-block:: python



Once the data is ready, we open a Python shell and start building the components of our natural language processor.

.. code-block:: console

  $ cd $WB_APP_ROOT
  $ python

In the Python shell, the quickest way to train all the NLP classifiers together is to use the :keyword:`nlp.build()` method.

.. code-block:: python

  >>> from mmworkbench.components.nlp import NaturalLanguageProcessor
  >>> nlp = NaturalLanguageProcessor('my_app')
  >>> nlp.build()







The domain classifier (also called the domain model) is a text classification model that is trained using the labeled queries across all domains. Our simple Kwik-E-Mart app only has one domain and hence does not need a domain classifier. However, complex conversational apps such as the popular virtual assistants on smartphones today have to handle queries from varied domains such as weather, navigation, sports, finance, and music, among others. Such apps use domain classification as the first step to narrow down the focus of the subsequent classifiers in the NLP pipeline.

The :keyword:`NaturalLanguageProcessor` class in Workbench exposes methods for training, testing, and saving all the models in our classifier hierarchy, including the domain model. For example, suppose we want to build a `support vector machine (SVM) <https://en.wikipedia.org/wiki/Support_vector_machine>`_ that does domain classification. In our Python shell, we start off by instantiating an object of the :keyword:`NaturalLanguageProcessor` class. We then train the :keyword:`domain_classifier` model by calling its :keyword:`fit()` method.

.. code-block:: python

  >>> from mmworkbench.components.nlp import NaturalLanguageProcessor
  >>> nlp = NaturalLanguageProcessor('my_app')
  >>> nlp.domain_classifier.fit(model_settings={'classifier_type': 'svm'},
  ...                           params={'kernel': 'linear'})

We test the trained classifier on a new query using the :keyword:`predict()` method.

.. code-block:: python

  >>> nlp.domain_classifier.predict('Play my jazz playlist.')
  'music'

To view the classification probabilities associated with all available domains, we can use the :keyword:`predict_proba()` method.

.. code-block:: python

  >>> nlp.domain_classifier.predict_proba('Play my jazz playlist.')
  [
    ('music', 0.751868),
    ('sports', 0.134523),
    ('weather', 0.087263),
    ('finance', 0.026346)
  ]

In addition to the `model` parameter we used above, the :keyword:`fit()` method also takes parameters we can use to improve upon the baseline SVM model trained by default. These include parameters for features, cross-validation settings, and other model-specific configuration. See the :ref:`User Guide <userguide>` for details.







Introduce the general ML techniques and methodology common to all NLP classifiers:
Getting the right kind of training data using in-house data generation and crowdsourcing, QAing and analyzing the data
Training a Workbench classifier, using k-fold cross-validation for hyperparameter selection
Training with default settings
Training with different classifier configurations (varying the model type, features or hyperparameter selection settings)
Testing a Workbench classifier on a held-out validation set
Doing error analysis on the validation set, retraining based on observations from error analysis by adding more training examples or feature tweaks
Getting final evaluation numbers on an unseen “blind” test set
Saving models for production use 

Then, describe the above in more detail with specific code examples for each subcomponent:
4.6.1 The Domain Classifier
4.6.2 The Intent Classifier
4.6.3 The Entity Recognizer
Describe gazetteers.
4.6.4 The Role Classifier

Describe necessity of roles with examples.
4.6.5 The Entity Resolver

Describe collection of synonyms and the synonym mapping file.
4.6.6 The Language Parser

Describe our approach to language parsing, what a parser configuration looks like and how it can be used to improve parser accuracy.  Show code examples for parsing and how to inspect the parser output.
