.. meta::
    :scope: private

Role Classifier
===============

The Role Classifier is run as the fourth step in the natural language processing pipeline to determine the target roles for entities in a given query. It is a machine-learned :doc:`classification <https://en.wikipedia.org/wiki/Statistical_classification>`_ model that is trained using all the labeled queries for a given intent. Labels are derived from the role types annotated within the training queries. See :doc:`Step 6 <../quickstart/06_generate_representative_training_data>` for more details on training data preparation. Role classification models are trained per entity type. A Workbench app hence has one role classifier for every entity type with associated roles.





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


===


Role Classification is the task of identifying predicates and predicate arguments. A **semantic role** in language is the relationship that a syntactic constituent has with a predicate. In Conversational NLU, a **role** represents the semantic theme a given entity can take. It can also be used to define how a named entity should be used for fulfilling a query intent. For example, in the query :red:`"Play Black Sabbath by Black Sabbath"`, the **title** entity :green:`"Black Sabbath"` has different semantic themes - **song** and **artist** respectively.

Treating Named Entity Recognition (NER) and Semantic Role Labeling (SRL) as separate tasks has a few advantages -

* NER models are hurt by splitting examples across fairly similar categories. Grouping entities with significantly overlapping entities and similar surrounding natural language will lead to better parsing and let us use more powerful models.
* Joint NER & SRL needs global dependencies, but fast & good NER models only do local. NER models (MEMM, CRF) quickly become intractable with long-distance dependencies. Separating NER from SRL let us use local dependencies for NER and long-distance dependencies in SRL.
* Role labeling might be a multi-label problem. With multi-label roles, we can use the same entity to query multiple fields.

===