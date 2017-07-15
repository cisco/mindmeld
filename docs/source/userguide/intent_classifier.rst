.. meta::
    :scope: private

Intent Classifier
=================

The Intent Classifier is run as the second step in the natural language processing pipeline to determine the target intent for a given query. It is a `text classification <https://en.wikipedia.org/wiki/Text_classification>`_ model that is trained using all of the labeled queries across all the intents in a given domain. The name of each intent folder serves as the label for the training queries contained within that folder. See :doc:`Step 6 <../quickstart/06_generate_representative_training_data>` for more details on training data preparation. Intent classification models are trained per domain. A Workbench app hence has one intent classifier for every domain with multiple intents.

.. note::

   For a quick introduction, refer to :ref:`Step 7 <intent_classification>` of the Step-By-Step Guide.
   
   Recommended prior reading: :doc:`Natural Language Processor <nlp>` chapter of the User Guide.


Access the intent classifier
----------------------------

Before using any of the NLP componenets, you need to generate the necessary training data for your app by following the guidelines in :doc:`Step 6 <../quickstart/06_generate_representative_training_data>`. You can then start by :ref:`instantiating an object <instantiate_nlp>` of the :class:`NaturalLanguageProcessor` (NLP) class.

.. code-block:: python

   >>> from mmworkbench.components.nlp import NaturalLanguageProcessor
   >>> nlp = NaturalLanguageProcessor(app_path='home_assistant')
   >>> nlp
   <NaturalLanguageProcessor 'home_assistant' ready: False, dirty: False>

Next, verify that the NLP has correctly identified all the domains for your app.

.. code-block:: python

   >>> nlp.domains
   {
    'smart_home': <DomainProcessor 'smart_home' ready: False, dirty: False>,
    'times_and_dates': <DomainProcessor 'times_and_dates' ready: False, dirty: False>,
    'unknown': <DomainProcessor 'unknown' ready: False, dirty: False>,
    'weather': <DomainProcessor 'weather' ready: False, dirty: False>
   }

Each domain has its own :class:`IntentClassifier` which can be accessed using the :attr:`intent_classifier` attribute of the corresponding domain.

.. code-block:: python

   >>> # Intent classifier for the 'smart_home' domain:
   >>> ic = nlp.domains['smart_home'].intent_classifier
   >>> ic
   <IntentClassifier ready: False, dirty: False>
   ...
   >>> # Intent classifier for the 'weather' domain:
   >>> ic = nlp.domains['weather'].intent_classifier
   >>> ic
   <IntentClassifier ready: False, dirty: False>


Training a baseline intent classifier
-------------------------------------



Introduce the general ML techniques and methodology common to all NLP classifiers:
Getting the right kind of training data using in-house data generation and crowdsourcing, QAing and analyzing the data
Training a Workbench classifier, using k-fold cross-validation for hyperparameter selection
Training with default settings
Training with different classifier configurations (varying the model type, features or hyperparameter selection settings)
Testing a Workbench classifier on a held-out validation set
Doing error analysis on the validation set, retraining based on observations from error analysis by adding more training examples or feature tweaks
Getting final evaluation numbers on an unseen “blind” test set
Saving models for production use 

