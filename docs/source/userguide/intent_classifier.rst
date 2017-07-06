.. meta::
    :scope: private

Intent Classifier
=================

The Intent Classifier is run as the second step in the natural language processing pipeline to determine the target intent for a given query. It is a `text classification <https://en.wikipedia.org/wiki/Text_classification>`_ model that is trained using all of the labeled queries across all the intents in a given domain. The name of each intent folder serves as the label for the training queries contained within that folder. See :doc:`Step 6 <../quickstart/06_generate_representative_training_data>` for more details on training data preparation. Intent classification models are trained per domain. A Workbench app hence has one intent classifier for every domain that has multiple intents.


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
