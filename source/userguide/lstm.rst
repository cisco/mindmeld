Using LSTM for Entity Recognition
=================================

Entity recognition is the one task within the NLP pipeline where deep learning models are among the available classification models. In particular, MindMeld provides a `Bi-Directional Long Short-Term Memory (LSTM) Network <https://en.wikipedia.org/wiki/Bidirectional_recurrent_neural_networks>`_, which has been shown to perform well on sequence labeling tasks such as entity recognition. The model is implemented in `TensorFlow <https://www.tensorflow.org/get_started/>`_.


LSTM network overview
^^^^^^^^^^^^^^^^^^^^^

The MindMeld Bi-Directional LSTM network

 - encodes words as pre-trained word embeddings using Stanford's `GloVe representation <https://nlp.stanford.edu/projects/glove/>`_
 - encodes characters using a convolutional network trained on the training data
 - concatenates the word and character embeddings together and feeds them into the bi-directional LSTM
 - couples the forget and input gates of the LSTM using a peephole connection, to improve overall accuracies on downstream NLP tasks
 - feeds the output of the LSTM into a `linear chain Conditional Random Field <https://en.wikipedia.org/wiki/Conditional_random_field>`_ (CRF) or `Softmax layer <https://en.wikipedia.org/wiki/Softmax_function>`_  which labels the target word as a particular entity

The diagram below describes the architecture of a typical Bi-Directional LSTM network.

.. figure:: /images/lstm_architecture_fix.png
   :scale: 50 %
   :align: center
   :alt: LSTM architecture diagram

   Courtesy: Guillaume Genthial

This design has these possible advantages:

- Deep neural networks (DNNs) outperform traditional machine learning models on training sets with about 1,000 or more queries, according to many research papers.
- DNNs require less feature engineering work than traditional machine learning models, because they use only two input features (word embeddings and gazetteers) compared to several hundred (n-grams, system entities, and so on).
- On GPU-enabled devices, the network can achieve training time comparable to some of the traditional models in MindMeld.

The possible disadvantages are:

- Performance may be no better than traditional machine learning models for training sets of about 1,000 queries or fewer.
- Training time on CPU-only machines is a lot slower than for traditional machine learning models.
- No automated hyperparameter tuning methods like :sk_api:`sklearn.model_selection.GridSearchCV <sklearn.model_selection.GridSearchCV.html>` are available for LSTMs.

LSTM parameter settings
^^^^^^^^^^^^^^^^^^^^^^^

Parameter tuning for an LSTM is more complex than for traditional machine learning models. A good starting point for understanding this subject is Andrej Karpathy's `course notes <https://cs231n.github.io/neural-networks-3/#baby>`_ from the Convolutional Neural Networks for Visual Recognition course at Stanford University.


``'params'`` (:class:`dict`)
  |

  A dictionary of values to be used for model hyperparameters during training.

+-----------------------------------------+------------------------------------------------------------------------------------------------+
| Parameter name                          | Description                                                                                    |
+=========================================+================================================================================================+
| ``padding_length``                      | The sequence model treats this as the maximum number of words in a query.                      |
|                                         | If a query has more words than ``padding_length``, the surplus words are discarded.            |
|                                         |                                                                                                |
|                                         | Typically set to the maximum word length of query expected both at train and predict time.     |
|                                         |                                                                                                |
|                                         | Default: ``20``                                                                                |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'padding_length': 20}``                                                                     |
|                                         |  - a query can have a maximum of twenty words                                                  |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``batch_size``                          | Size of each batch of training data to feed into the network (which uses mini-batch learning). |
|                                         |                                                                                                |
|                                         | Default: ``20``                                                                                |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'batch_size': 20}``                                                                         |
|                                         |  - feed twenty training queries to the network for each learning step                          |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``display_epoch``                       | The network displays training accuracy statistics at this interval, measured in epochs.        |
|                                         |                                                                                                |
|                                         | Default: ``5``                                                                                 |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'display_epoch': 5}``                                                                       |
|                                         |  - display accuracy statistics every five epochs                                               |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``number_of_epochs``                    | Total number of complete iterations of the training data to feed into the network.             |
|                                         | In each iteration, the data is shuffled to break any prior sequence patterns.                  |
|                                         |                                                                                                |
|                                         | Default: ``20``                                                                                |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'number_of_epochs': 20}``                                                                   |
|                                         |  - iterate through the training data twenty times                                              |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``optimizer``                           | Optimizer to use to minimize the network's stochastic objective function.                      |
|                                         |                                                                                                |
|                                         | Default: ``'adam'``                                                                            |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'optimizer': 'adam'}``                                                                      |
|                                         |  - use the Adam optimizer to minimize the objective function                                   |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``learning_rate``                       | Parameter to control the size of weight and bias changes                                       |
|                                         | of the training algorithm as it learns.                                                        |
|                                         |                                                                                                |
|                                         | `This <https://en.wikibooks.org/wiki/Artificial_Neural_Networks/Error-Correction_Learning>`_   |
|                                         | article explains Learning Rate in technical terms.                                             |
|                                         |                                                                                                |
|                                         | Default: ``0.005``                                                                             |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'learning_rate': 0.005}``                                                                   |
|                                         |  - set learning rate to 0.005                                                                  |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``dense_keep_prob``                     | In the context of the ''dropout'' technique (a regularization method to prevent overfitting),  |
|                                         | keep probability specifies the proportion of nodes to "keep"—that is, to exempt from dropout   |
|                                         | during the network's learning phase.                                                           |
|                                         |                                                                                                |
|                                         | The ``dense_keep_prob`` parameter sets the keep probability of the nodes                       |
|                                         | in the dense network layer that connects the output of the LSTM layer                          |
|                                         | to the nodes that predict the named entities.                                                  |
|                                         |                                                                                                |
|                                         | Default: ``0.5``                                                                               |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'dense_keep_prob': 0.5}``                                                                   |
|                                         |  - 50% of the nodes in the dense layer will not be turned off by dropout                       |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``lstm_input_keep_prob``                | Keep probability for the nodes that constitute the inputs to the LSTM cell.                    |
|                                         |                                                                                                |
|                                         | Default: ``0.5``                                                                               |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'lstm_input_keep_prob': 0.5}``                                                              |
|                                         |  - 50% of the nodes that are inputs to the LSTM cell will not be turned off by dropout         |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``lstm_output_keep_prob``               | Keep probability for the nodes that constitute the outputs of the LSTM cell.                   |
|                                         |                                                                                                |
|                                         | Default: ``0.5``                                                                               |
|                                         |                                                                                                |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'lstm_output_keep_prob': 0.5}``                                                             |
|                                         |  - 50% of the nodes that are outputs of the LSTM cell will not be turned off by dropout        |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``token_lstm_hidden_state_dimension``   | Number of states per LSTM cell.                                                                |
|                                         |                                                                                                |
|                                         | Default: ``300``                                                                               |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'token_lstm_hidden_state_dimension': 300}``                                                 |
|                                         |  - an LSTM cell will have 300 states                                                           |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``token_embedding_dimension``           | Number of dimensions for word embeddings.                                                      |
|                                         |                                                                                                |
|                                         | Allowed values: [50, 100, 200, 300].                                                           |
|                                         |                                                                                                |
|                                         | Default: ``300``                                                                               |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'token_embedding_dimension': 300}``                                                         |
|                                         |  - each word embedding will have 300 dimensions                                                |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``gaz_encoding_dimension``              | Number of nodes to connect to the gazetteer encodings in a fully-connected network.            |
|                                         |                                                                                                |
|                                         | Default: ``100``                                                                               |
|                                         |                                                                                                |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'gaz_encoding_dimension': 100}``                                                            |
|                                         |  - 100 nodes will be connected to the gazetteer encodings in a fully-connected network         |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``max_char_per_word``                   | The sequence model treats this as the maximum number of characters in a word.                  |
|                                         | If a word has more characters than ``max_char_per_word``, the surplus characters are discarded.|
|                                         |                                                                                                |
|                                         | Usually set to the size of the longest word in the training and test sets.                     |
|                                         |                                                                                                |
|                                         | Default: ``20``                                                                                |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'max_char_per_word': 20}``                                                                  |
|                                         |  - a word can have a maximum of twenty characters                                              |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``use_crf_layer``                       | If set to ``True``, use a linear chain Conditional Random Field layer for the final layer,     |
|                                         | which predicts sequence tags.                                                                  |
|                                         |                                                                                                |
|                                         | If set to ``False``, use a softmax layer to predict sequence tags.                             |
|                                         |                                                                                                |
|                                         | Default: ``False``                                                                             |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'use_crf_layer': True}``                                                                    |
|                                         |  - use the CRF layer                                                                           |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``use_character_embeddings``            | If set to ``True``, use the character embedding trained on the training data                   |
|                                         | using a convolutional network.                                                                 |
|                                         |                                                                                                |
|                                         | If set to ``False``, do not use character embeddings.                                          |
|                                         |                                                                                                |
|                                         | Note: Using character embedding significantly increases training time                          |
|                                         | compared to vanilla word embeddings only.                                                      |
|                                         |                                                                                                |
|                                         | Default: ``False``                                                                             |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'use_character_embeddings': True}``                                                         |
|                                         |  - use character embeddings                                                                    |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``char_window_sizes``                   | List of window sizes for convolutions that the network should use                              |
|                                         | to build the character embeddings.                                                             |
|                                         | Usually in decreasing numerical order.                                                         |
|                                         |                                                                                                |
|                                         | Note: This parameter is needed only if ``use_character_embeddings`` is set to ``True``.        |
|                                         |                                                                                                |
|                                         | Default: ``[5]``                                                                               |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'char_window_sizes': [5, 3]}``                                                              |
|                                         |  - first, use a convolution of size 5                                                          |
|                                         |  - next, feed the output of that convolution through a convolution of size 3                   |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``character_embedding_dimension``       | Initial dimension of each character before it is fed into the convolutional network.           |
|                                         |                                                                                                |
|                                         | Note: This parameter is needed only if ``use_character_embeddings`` is set to ``True``.        |
|                                         |                                                                                                |
|                                         | Default: ``10``                                                                                |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'character_embedding_dimension': 10}``                                                      |
|                                         |  - initialize the dimension of each character to ten                                           |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``word_level_character_embedding_size`` | The final dimension of each character after it is transformed                                  |
|                                         | by the convolutional network.                                                                  |
|                                         |                                                                                                |
|                                         | Usually greater than ``character_embedding_dimension`` since it encodes                        |
|                                         | more information about orthography and semantics.                                              |
|                                         |                                                                                                |
|                                         | Note: This parameter is needed only if ``use_character_embeddings`` is set to ``True``.        |
|                                         |                                                                                                |
|                                         | Default: ``40``                                                                                |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'word_level_character_embedding_size': 40}``                                                |
|                                         |  - each character should have dimension of forty, after convolutional network training         |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
