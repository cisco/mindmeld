Deep Neural Networks in MindMeld
================================

The advent of `Deep learning <https://en.wikipedia.org/wiki/Deep_learning>`_ has led to substantial developments in the area of Conversational AI as well as Natural Langauge Processing (NLP) in general.
Specifically, the proliferation of deep neural models based on :wiki_api:`Convolutional neural networks (CNN) <Convolutional_neural_network#Natural_language_processing>`, :wiki_api:`Long short-term memory networks (LSTM) <Long_short-term_memory>` and :wiki_api:`Transformers <Transformer_(machine_learning_model)>`, and their rapid adaptation to wide range of NLP tasks has been a great success.
Thus extending the suite of traditional machine learning models (e.g. :sk_guide:`Logistic regression <linear_model.html#logistic-regression>`, :sk_guide:`Decision tree <tree.html#tree>`, etc.), MindMeld provides the option for users to train and use a variety of deep neural models with an array of configurable parameters.

The deep neural models are available both for text classification aka. sequence classification (:ref:`domain classification <domain_classification>` and :ref:`intent classification <intent_classification>`) as well as for token classification (:ref:`entity recognition <entity_recognition>`) tasks.
The models are implemented using `Pytorch <https://pytorch.org/>`_ framework and thus need extra installations before starting to use them in your chatbot application.

.. note::

   Please make sure to install the Pytorch requirement by running in the shell: :code:`pip install mindmeld[torch]`.

MindMeld supports the use of pretrained transformer models such as :wiki_api:`BERT <BERT_(language_model)>` through the popular `Huggingface Transformers <https://huggingface.co/docs/transformers/index>`_ library.
Several pretrained models from their `Models Hub <https://huggingface.co/models>`_ that can be used for sequence classification or token classification can be employed in your chatbot application.

.. note::

   Install the extra *transformers* requirement by running in the shell: :code:`pip install mindmeld[transformers]`.

Before proceeding to use the deep neural models, consider the following possible advantages and disadvantages of using them in place of traditional machine learning models.

Some advantages are:

- The deep models generally outperform traditional machine learning models on training sets with at least several hundreds or few thousands of queries when training from scratch, and with at least few hundreds if fine-tuning from a pretrained checkpoint.
- The deep models require little or no feature engineering work than traditional machine learning models, because they learn the input features (such as word embeddings) compared to several hundred engineered features (n-grams, system entities, and so on) of the latter.
- On GPU-enabled devices, the deep networks can achieve training time comparable to some of the traditional models in MindMeld.

Some disadvantages are:

- Training time on CPU-only machines is a lot slower than for traditional machine learning models.
- No automated hyperparameter tuning methods like :sk_api:`sklearn.model_selection.GridSearchCV <sklearn.model_selection.GridSearchCV.html>` are available for deep neural models.
- The deep neural models generally occupy more disk storage space compared to their counterparts.

In the following sections, different model architectures and their configurable parameters are outlined.

.. note::

   To use deep neural networks instead of traditional machine learning models, simply make few modifications to the classifier configuration dictionaries for all or selected classifiers in MindMeld's NLP hierarchy.

Domain and Intent classification (aka. Sequence Classification)
---------------------------------------------------------------

.. _dnns_sequence_classification:

Recall from :ref:`Working with the Domain Classifier <domain_classification>` and :ref:`Working with the Intent Classifier <intent_classification>` sections that a :ref:`Domain <domain_classifier_configuration>`/:ref:`Intent <intent_classifier_configuration>` classifier configuration consists of the keys ``'features'``, ``'param_selection'``, ``'model_settings'``, and ``'params'``, amongst other keys that do not have distinction between traditional models or deep neural models.
When working with deep neural models, the ``'features'`` and ``'param_selection'`` keys in classifier configuration are redundant as we neither have to handcraft any feature sets for modeling nor there is an automated hyperparameter tuning.
Thus, the only relevant keys to be configured are ``'model_settings'`` and ``'params'``.

The ``'model_settings'`` is a :class:`dict` with the single key ``'classifier_type'``, whose value specifies the machine learning model to use.
The allowed values that are backed by deep neural nets and are meant for sequence classification are:

+----------------+----------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------+
| Value          | Classifier                                                                                                                 | Reference for configurable parameters                                     |
+================+============================================================================================================================+===========================================================================+
| ``'embedder'`` | Pooled :wiki_api:`Token Embeddings <Word_embedding>` or :wiki_api:`Deep Contextualized Embeddings <BERT_(language_model)>` | :ref:`Embedder parameters <dnns_sequence_classification_models_embedder>` |
+----------------+----------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------+
| ``'cnn'``      | :wiki_api:`Convolutional neural networks (CNN) <Convolutional_neural_network#Natural_language_processing>`                 | :ref:`CNN parameters <dnns_sequence_classification_models_cnn>`           |
+----------------+----------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------+
| ``'lstm'``     | :wiki_api:`Long short-term memory networks (LSTM) <Long_short-term_memory>`                                                | :ref:`LSTM parameters <dnns_sequence_classification_models_lstm>`         |
+----------------+----------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------+

The ``'params'`` is also a :class:`dict` with several configurable keys, some of which are specific to the choice of classifier type and others common across all the above classifier types.
In the following section, list of allowed parameters related to each choice of classifier type are outlined.
See :ref:`Common Configurable Params <common_configurable_params>` section for list of configurable params that are not just specific to any classifier type but are common across all the classifier types.

1. ``'embedder'`` classifier type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _dnns_sequence_classification_models_embedder:

This classifier type includes neural models that are based on either an embedding lookup table or a deep contextualized embedder, along with a pooling operation on top of those embeddings before applying a classification layer.
For the former type of embeddings, embedding lookup table is created depending upon the individual tokens found in the training data, with tokens derived based on a chosen tokenization strategy-- word, sub-word, or character tokenization (see :ref:`Tokenization Choices <tokenization_choices>` section below for more details).
The lookup table by default is randomly initialized but can instead be initialized to a pretrained checkpoint (such as `GloVe <https://nlp.stanford.edu/projects/glove/>`_) when using the word tokenization strategy.
On the other hand, a deep contextualized embedder is a pretrained embedder such as :wiki_api:`BERT <BERT_(language_model)>`, which consists of its own tokenization strategy and neural embedding process.
In any case, all the underlying weights can be tuned to the training data provided, or can be skipped when using pretrained ones.

.. note::

   Specify the embedding choice using the param ``embedder_type``. Set it to ``None``, ``'glove'`` or ``'bert'`` to use with desired embeddings-- based on randomly initialized embedding lookup table, based on lookup table initialized with GloVe pretrained embeddings or a BERT-like transformers architecture based deep contextualized embedder, respectively.

Following are the different optional params that are configurable along with the chosen choice of ``embedder_type`` param.
See :ref:`Common Configurable Params <common_configurable_params>` section for list of additional configurable params that are common across classifiers.

1.1 Based on Embedding Lookup Table (``embedder_type``: ``None``)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration Key                       | Description                                                                                                                                                                                                                                      |
+=========================================+==================================================================================================================================================================================================================================================+
| ``emb_dim``                             | Number of dimensions for each token's embedding.                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``256``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``tokenizer_type``                      | The choice of tokenization strategy to extract tokens from the training data. See :ref:`Tokenization Choices <tokenization_choices>` section below for more details.                                                                             |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'whitespace-tokenizer'``                                                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: See `<tokenization_choices>`_                                                                                                                                                                                                           |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``add_terminals``                       | If set to ``True``, terminal tokens-- START_TEXT and END_TEXT -- are added at the beginning and ending for each input before applying any padding.                                                                                               |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``False``                                                                                                                                                                                                                               |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``update_embeddings``                   | If set to ``False``, the weights of embedding table or the deep contextualized embedder will not be updated during back-propogation of gradients. This boolean key is only valid when using a pretrained embedder type.                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``embedder_output_keep_prob``           | Keep probability for the dropout layer placed on top of embeddings. Dropout helps in regularization and reduces over-fitting.                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``embedder_output_pooling_type``        | Specifies the manner in which a query's token-wise embeddings are to be collated into a single embedding before passing through classification layer.                                                                                            |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'mean'``                                                                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``'first'``, ``'last'``, ``'max'``, ``'mean'``, ``'mean_sqrt'``                                                                                                                                                                         |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``output_keep_prob``                    | Keep probability for the dropout layer placed on top of classifier's penultimate layer (i.e the layer before logits are computed). Dropout helps in regularization and reduces over-fitting.                                                     |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Below is a minimalistic example of a classifier configuration for classifier based on embedding lookup table:

.. code-block:: python

   {
    'model_type': 'text',
    'train_label_set': 'train.*\.txt',
    'test_label_set': 'test.*\.txt',
    'model_settings': {'classifier_type': 'embedder'},
    'params': {
        'embedder_type': None,
        'emb_dim': 256,
    },
   }


1.2 Based on Pretrained Embedding Lookup Table (``embedder_type``: ``glove``)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration Key                       | Description                                                                                                                                                                                                                                      |
+=========================================+==================================================================================================================================================================================================================================================+
| ``token_dimension``                     | Specifies the dimension of the `GloVe-6B <https://nlp.stanford.edu/projects/glove/>`_ pretrained word vectors. This key is only valid when using ``embedder_type`` as ``'glove'``.                                                               |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``300``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``50``, ``100``, ``200``, ``300``                                                                                                                                                                                                       |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``token_pretrained_embedding_filepath`` | Specifies a local file path for pretrained embedding file. This key is only valid when using ``embedder_type`` as ``'glove'``.                                                                                                                   |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: Union[str, None]                                                                                                                                                                                                                           |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``None``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: File path to a valid GloVe-style embeddings file                                                                                                                                                                                        |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``add_terminals``                       | If set to ``True``, terminal tokens-- START_TEXT and END_TEXT -- are added at the beginning and ending for each input before applying any padding.                                                                                               |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``False``                                                                                                                                                                                                                               |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``update_embeddings``                   | If set to ``False``, the weights of embedding table or the deep contextualized embedder will not be updated during back-propogation of gradients. This boolean key is only valid when using a pretrained embedder type.                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``embedder_output_keep_prob``           | Keep probability for the dropout layer placed on top of embeddings. Dropout helps in regularization and reduces over-fitting.                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``embedder_output_pooling_type``        | Specifies the manner in which a query's token-wise embeddings are to be collated into a single embedding before passing through classification layer.                                                                                            |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'mean'``                                                                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``'first'``, ``'last'``, ``'max'``, ``'mean'``, ``'mean_sqrt'``                                                                                                                                                                         |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``output_keep_prob``                    | Keep probability for the dropout layer placed on top of classifier's penultimate layer (i.e the layer before logits are computed). Dropout helps in regularization and reduces over-fitting.                                                     |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Below is a minimalistic example of a classifier configuration for classifier based on pretrained-initialized embedding lookup table:

.. code-block:: python

   {
    'model_type': 'text',
    'train_label_set': 'train.*\.txt',
    'test_label_set': 'test.*\.txt',
    'model_settings': {'classifier_type': 'embedder'},
    'params': {
        'embedder_type': 'glove',
        'update_embeddings': True,
    },
   }

1.3 Based on Deep Contextualized Embeddings (``embedder_type``: ``bert``)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration Key                       | Description                                                                                                                                                                                                                                      |
+=========================================+==================================================================================================================================================================================================================================================+
| ``pretrained_model_name_or_path``       | Specifies a pretrained checkpoint's name or a valid file path to load a bert-like embedder. This key is only valid when using ``embedder_type`` as ``'bert'``.                                                                                   |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'bert-base-uncased'``                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any valid name from `Huggingface Models Hub <https://huggingface.co/models>`_ or a valid folder path where the model's weights as well as its tokenizer's resources are present.                                                        |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``update_embeddings``                   | If set to ``False``, the weights of embedding table or the deep contextualized embedder will not be updated during back-propogation of gradients. This boolean key is only valid when using a pretrained embedder type.                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``embedder_output_keep_prob``           | Keep probability for the dropout layer placed on top of embeddings. Dropout helps in regularization and reduces over-fitting.                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``embedder_output_pooling_type``        | Specifies the manner in which a query's token-wise embeddings are to be collated into a single embedding before passing through classification layer.                                                                                            |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'mean'``                                                                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``'first'``, ``'last'``, ``'max'``, ``'mean'``, ``'mean_sqrt'``                                                                                                                                                                         |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``output_keep_prob``                    | Keep probability for the dropout layer placed on top of classifier's penultimate layer (i.e the layer before logits are computed). Dropout helps in regularization and reduces over-fitting.                                                     |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Below is a minimalistic example of a classifier configuration for classifier based on BERT embedder:

.. code-block:: python

   {
    'model_type': 'text',
    'train_label_set': 'train.*\.txt',
    'test_label_set': 'test.*\.txt',
    'model_settings': {'classifier_type': 'embedder'},
    'params': {
        'embedder_type': 'bert',
        'pretrained_model_name_or_path': 'distilbert-base-uncased',
        'update_embeddings': True,
    },
   }

2. ``'cnn'`` classifier type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _dnns_sequence_classification_models_cnn:

This is CNN classifier. Add content here!

Following are the different optional params that are configurable with the ``'cnn'`` classifier type.
See :ref:`Common Configurable Params <common_configurable_params>` section for list of additional configurable params that are common across classifiers.

+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration Key                       | Description                                                                                                                                                                                                                                      |
+=========================================+==================================================================================================================================================================================================================================================+
| ``embedder_type``                       | The choice of embeddings to be used. Specifying ``None`` randomly initializes an embeddings lookup table whereas specifying ``'glove'`` initializes the table with pretrained GloVe embeddings.                                                  |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: Union[str, None]                                                                                                                                                                                                                           |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: None                                                                                                                                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``None``, ``'glove'``                                                                                                                                                                                                                   |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``emb_dim``                             | Number of dimensions for each token's embedding. This key is only valid when not using a pretrained embedder.                                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``256``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``tokenizer_type``                      | The choice of tokenization strategy to extract tokens from the training data. See :ref:`Tokenization Choices <tokenization_choices>` section below for more details.                                                                             |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'whitespace-tokenizer'``                                                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: See `<tokenization_choices>`_                                                                                                                                                                                                           |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``add_terminals``                       | If set to ``True``, terminal tokens-- START_TEXT and END_TEXT -- are added at the beginning and ending for each input before applying any padding.                                                                                               |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``False``                                                                                                                                                                                                                               |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``update_embeddings``                   | If set to ``False``, the weights of embedding table or the deep contextualized embedder will not be updated during back-propogation of gradients. This boolean key is only valid when using a pretrained embedder type.                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``embedder_output_keep_prob``           | Keep probability for the dropout layer placed on top of embeddings. Dropout helps in regularization and reduces over-fitting.                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``output_keep_prob``                    | Keep probability for the dropout layer placed on top of classifier's penultimate layer (i.e the layer before logits are computed). Dropout helps in regularization and reduces over-fitting.                                                     |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``window_sizes``                        | The lengths of 1D CNN kernels to be used for convolution on top of embeddings.                                                                                                                                                                   |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: List[int]                                                                                                                                                                                                                                  |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``[3,4,5]``                                                                                                                                                                                                                             |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A list of positive integers                                                                                                                                                                                                             |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``number_of_windows``                   | The number of kernels per each specified length of 1D CNN kernels.                                                                                                                                                                               |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: List[int]                                                                                                                                                                                                                                  |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ```[100,100,100]``                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A list of positive integers; same length as ``window_sizes``                                                                                                                                                                            |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Below is a minimalistic example of a classifier configuration for classifier based on CNNs:

.. code-block:: python

   {
    'model_type': 'text',
    'train_label_set': 'train.*\.txt',
    'test_label_set': 'test.*\.txt',
    'model_settings': {'classifier_type': 'cnn'},
    'params': {
        'embedder_type': 'glove',
        'window_sizes': [3,4,5],
        'number_of_windows': [100,100,100],
    },
   }

3. ``'lstm'`` classifier type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _dnns_sequence_classification_models_lstm:

This is LSTM classifier. Add content here!

Following are the different optional params that are configurable with the ``'cnn'`` classifier type.
See :ref:`Common Configurable Params <common_configurable_params>` section for list of additional configurable params that are common across classifiers.

+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration Key                       | Description                                                                                                                                                                                                                                      |
+=========================================+==================================================================================================================================================================================================================================================+
| ``embedder_type``                       | The choice of embeddings to be used. Specifying ``None`` randomly initializes an embeddings lookup table whereas specifying ``'glove'`` initializes the table with pretrained GloVe embeddings.                                                  |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: Union[str, None]                                                                                                                                                                                                                           |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: None                                                                                                                                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``None``, ``'glove'``                                                                                                                                                                                                                   |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``emb_dim``                             | Number of dimensions for each token's embedding. This key is only valid when not using a pretrained embedder.                                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``256``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``tokenizer_type``                      | The choice of tokenization strategy to extract tokens from the training data. See :ref:`Tokenization Choices <tokenization_choices>` section below for more details.                                                                             |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'whitespace-tokenizer'``                                                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: See `<tokenization_choices>`_                                                                                                                                                                                                           |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``add_terminals``                       | If set to ``True``, terminal tokens-- START_TEXT and END_TEXT -- are added at the beginning and ending for each input before applying any padding.                                                                                               |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``False``                                                                                                                                                                                                                               |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``update_embeddings``                   | If set to ``False``, the weights of embedding table or the deep contextualized embedder will not be updated during back-propogation of gradients. This boolean key is only valid when using a pretrained embedder type.                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``embedder_output_keep_prob``           | Keep probability for the dropout layer placed on top of embeddings. Dropout helps in regularization and reduces over-fitting.                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``output_keep_prob``                    | Keep probability for the dropout layer placed on top of classifier's penultimate layer (i.e the layer before logits are computed). Dropout helps in regularization and reduces over-fitting.                                                     |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lstm_hidden_dim``                     | Number of states per each LSTM layer.                                                                                                                                                                                                            |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``128``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lstm_num_layers``                     | The number of LSTM layers that are to be stacked sequentially.                                                                                                                                                                                   |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``2``                                                                                                                                                                                                                                   |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lstm_keep_prob``                      | Keep probability for the nodes that constitute the outputs of each LSTM layer except the last LSTM layer.                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lstm_bidirectional``                  | If ``True``, the LSTM layers will be bidirectional.                                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lstm_output_pooling_type``            | Specifies the manner in which a query's token-wise embeddings are to be collated into a single embedding before passing through classification layer.                                                                                            |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'last'``                                                                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``'first'``, ``'last'``, ``'max'``, ``'mean'``, ``'mean_sqrt'``                                                                                                                                                                         |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Below is a minimalistic example of a classifier configuration for classifier based on LSTMs:

.. code-block:: python

   {
    'model_type': 'text',
    'train_label_set': 'train.*\.txt',
    'test_label_set': 'test.*\.txt',
    'model_settings': {'classifier_type': 'cnn'},
    'params': {
        'embedder_type': 'glove',
        'lstm_hidden_dim': 128,
        'lstm_bidirectional': True,
    },
   }

Entity recognition (Token Classification)
-----------------------------------------

.. _dnns_token_classification:



Common Configurable Params
--------------------------

.. _common_configurable_params:

+---------------------------------+------------------------------------------------------------------------------------------------------------------+
| Parameter name                  | Description                                                                                                      |
+=================================+==================================================================================================================+
| ``device``                      | Name of the device on which torch tensors will be allocated. The ``'cuda'`` choice is to be specified only when  |
|                                 | building models in a `GPU <https://en.wikipedia.org/wiki/Graphics_processing_unit>`_ environment.                |
|                                 |                                                                                                                  |
|                                 | Type: str                                                                                                        |
|                                 |                                                                                                                  |
|                                 | Default: ``'cuda'`` if torch.cuda.is_available() else ``'cpu'``                                                  |
|                                 |                                                                                                                  |
|                                 | Choices: ``'cuda'``, ``'cpu'``                                                                                   |
+---------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``number_of_epochs``            | The total number of complete iterations of the training data to feed into the network. In each iteration, the    |
|                                 | data is shuffled to break any prior sequence patterns.                                                           |
|                                 |                                                                                                                  |
|                                 | Type: int                                                                                                        |
|                                 |                                                                                                                  |
|                                 | Default: ``100``                                                                                                 |
|                                 |                                                                                                                  |
|                                 | Choices: Any positive integer                                                                                    |
+---------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``patience``                    | The number of epochs to wait without any improvement on the validation metric before terminating training.       |
|                                 |                                                                                                                  |
|                                 | Type: int                                                                                                        |
|                                 |                                                                                                                  |
|                                 | Default: ``7``                                                                                                   |
|                                 |                                                                                                                  |
|                                 | Choices: Any positive integer                                                                                    |
+---------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``batch_size``                  | Size of each batch of training data to feed into the network (which uses mini-batch learning).                   |
|                                 |                                                                                                                  |
|                                 |                                                                                                                  |
|                                 | Type: int                                                                                                        |
|                                 |                                                                                                                  |
|                                 | Default: ``32``                                                                                                  |
|                                 |                                                                                                                  |
|                                 | Choices: Any positive integer                                                                                    |
+---------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``gradient_accumulation_steps`` | Number of consecutive mini-batches for which gradients will be averaged and accumulated before updating the      |
|                                 | weights of the network.                                                                                          |
|                                 |                                                                                                                  |
|                                 | Type: int                                                                                                        |
|                                 |                                                                                                                  |
|                                 | Default: ``1``                                                                                                   |
|                                 |                                                                                                                  |
|                                 | Choices: Any positive integer                                                                                    |
+---------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``max_grad_norm``               | Maximum norm to which the accumulated gradients' norm is to be clipped.                                          |
|                                 |                                                                                                                  |
|                                 | Type: Union[float, None]                                                                                         |
|                                 |                                                                                                                  |
|                                 | Default: ``None``                                                                                                |
|                                 |                                                                                                                  |
|                                 | Choices: Any positive float, ``None``                                                                            |
+---------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``optimizer``                   | Optimizer to use to minimize the network's stochastic objective function.                                        |
|                                 |                                                                                                                  |
|                                 | Type: str                                                                                                        |
|                                 |                                                                                                                  |
|                                 | Default: ``'Adam'``                                                                                              |
|                                 |                                                                                                                  |
|                                 | Choices: A valid name from `Pytorch optimizers <https://pytorch.org/docs/stable/optim.html#algorithms>`_         |
+---------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``learning_rate``               | Parameter to control the size of weight and bias changes of the training algorithm as it learns.                 |
|                                 | `This <https://en.wikibooks.org/wiki/Artificial_Neural_Networks/Error-Correction_Learning>`_ article explains    |
|                                 | Learning Rate in technical terms.                                                                                |
|                                 |                                                                                                                  |
|                                 | Type: float                                                                                                      |
|                                 |                                                                                                                  |
|                                 | Default: ``0.001``                                                                                               |
|                                 |                                                                                                                  |
|                                 | Choices: Any positive float                                                                                      |
+---------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``validation_metric``           | The metric used to track model improvements on the validation data split.                                        |
|                                 |                                                                                                                  |
|                                 | Type: str                                                                                                        |
|                                 |                                                                                                                  |
|                                 | Default: ``'accuracy'``                                                                                          |
|                                 |                                                                                                                  |
|                                 | Choices: ``'accuracy'``, ``'f1'``                                                                                |
+---------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``dev_split_ratio``             | The fraction of samples in the training data that are to be used for validation; sampled randomly.               |
|                                 |                                                                                                                  |
|                                 | Type: float                                                                                                      |
|                                 |                                                                                                                  |
|                                 | Default: ``0.2``                                                                                                 |
|                                 |                                                                                                                  |
|                                 | Choices: A float between 0 and 1                                                                                 |
+---------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``padding_length``              | The maximum number of tokens (words, sub-words, or characters) allowed in a query. If a query has                |
|                                 | more tokens than ``padding_length``, the surplus words are discarded. If specified as ``None``, the              |
|                                 | ``padding_length`` of a mini-batch is simply the maximum length of inputs to that mini-batch.                    |
|                                 |                                                                                                                  |
|                                 | Type: Union[int, None]                                                                                           |
|                                 |                                                                                                                  |
|                                 | Default: ``None``                                                                                                |
|                                 |                                                                                                                  |
|                                 | Choices: Any positive integer, ``None``                                                                          |
+---------------------------------+------------------------------------------------------------------------------------------------------------------+

Tokenization Choices
---------------------

.. _tokenization_choices:

List of tokenization choices will be added here.
