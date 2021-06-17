Active Learning in MindMeld
===========================

Building new conversational applications, adding new domains and intents for existing applications or improving performance based on user experience, all of these benefit from the addition of more informative data to the trained models. Data generation resources and especially user logs for the application are highly beneficial in capturing new trends and growing diversity in terms of the phrasing of queries. While adding both old and new data to the trained models can improve system performance in terms of accuracy, it reduces scalability by increasing the computation and time costs.

Active Learning is an approach to continuously and iteratively learn the best data for training the models while preserving good accuracies and maintaining best-case tradeoffs with respect to computational power and time. MindMeld provides an inbuilt functionality to select the must-have queries from existing data or additional logs and datasets to get the best out of your conversational applications.


Quick Start
-----------
This section is a quick reference on the command-line usage of the active learning Strategy Tuning (``tune``) and Query Selection (``select``) commands. These are detailed in the next sections.

Strategy tuning

.. code-block:: console

    mindmeld active_learning --train --app-path '<PATH>/app_name/' --output_folder '<PATH>'

Query selection

.. code-block:: console

    mindmeld active_learning --select --app-path "<PATH>/app_name/" --output_folder '<PATH>' --unlabeled_logs_path "<PATH>/logs.txt"

.. note::
    
    Running these commands without defining a custom active learning configuration in ``config.py`` would result in the use of a default configuration. The custom configuration settings and MindMeld's default active learning configuration are explained in the next section.

.. note::
    
    Accuracy and query selection results can be found in the directory ``output_folder/results`` in files ``accuracies.json`` and ``selected_queries.json`` respectively. Plots for the tuning results are saved in ``output_folder/plots``.


Defining the AL config
----------------------
The first step towards customizing the active learning pipeline to your application is configuration adjustment. This section details different components in the configuration and provides a default configuration setup which is applied if no active learning config is defined in in ``config.py``.

1. **output_folder** - Path to folder to save the output of the active learning pipeline.

2. **pre_tuning**

+------------------------+--------+----------------------------------------------------------------------------------+
| Configuration Key      | Type   | Definition and Examples                                                          |
+========================+========+==================================================================================+
| train_pattern          | str    | Regex pattern to match train files. Example for all train files ".*train.*.txt"  |
+------------------------+--------+----------------------------------------------------------------------------------+
| test_pattern           | str    | Regex pattern to match train files. Example for all test files ".*test.*.txt"    |
+------------------------+--------+----------------------------------------------------------------------------------+
| train_seed_pct         | float  | Percentage of training data to use as the initial seed                           |
+------------------------+--------+----------------------------------------------------------------------------------+


3. **tuning**

+------------------------+------------+----------------------------------------------------------------------------------+
| Configuration Key      | Type       | Definition and Examples                                                          |
+========================+============+==================================================================================+
| n_classifiers          | int        | Number of classifier instances to be used by a subset of strategies              |
+------------------------+------------+----------------------------------------------------------------------------------+
| n_epochs               | int        | Number of turns strategy tuning is run on the complete log data                  |
+------------------------+------------+----------------------------------------------------------------------------------+
| batch_size             | int        | Number of queries to sample at each iteration from the logs                      |
+------------------------+------------+----------------------------------------------------------------------------------+
| tuning_level           | str        | Selecting classifiers to use for tuning, options: "domain", "intent" or "joint"  |
+------------------------+------------+----------------------------------------------------------------------------------+
| tuning_strategies      | List[str]  | List of strategies (heuristics) to use for tuning                                |
+------------------------+------------+----------------------------------------------------------------------------------+

4. **tuning_output**

+------------------------+------------+----------------------------------------------------------------------------------+
| Configuration Key      | Type       | Definition and Examples                                                          |
+========================+============+==================================================================================+
| save_sampled_queries   | bool       | Whether to save the queries sampled at each iteration                            |
+------------------------+------------+----------------------------------------------------------------------------------+
| aggregate_statistic    | str        | Metric to record aggregate classifier performance data in the output and plots,  |
|                        |            | options: "accuracy", "f1_weighted", "f1_macro", "f1_micro"                       |
+------------------------+------------+----------------------------------------------------------------------------------+
| class_level_statistic  | str        | Metric to record class/tuning level performance data in the output and plots,    |
|                        |            | options: "f_beta", "percision", "recall"                                         |
+------------------------+------------+----------------------------------------------------------------------------------+

5. **query_selection**

+------------------------+------------+----------------------------------------------------------------------------------+
| Configuration Key      | Type       | Definition and Examples                                                          |
+========================+============+==================================================================================+
| selection_strategy     | str        | Strategy (heuristic) to use for log selection                                    |
+------------------------+------------+----------------------------------------------------------------------------------+
| log_usage_pct          | float      | Percentage of the log data to use for selection                                  |
+------------------------+------------+----------------------------------------------------------------------------------+
| labeled_logs_pattern   | str        | Regex pattern to match log files if already labeled. For example, ".*log.*.txt"  |
+------------------------+------------+----------------------------------------------------------------------------------+
| unlabeled_logs_path    | str        | Path to text file containing unlabeled queries from user logs or other resources |
+------------------------+------------+----------------------------------------------------------------------------------+


If there is no configuration defined in the ``config.py`` file or if fields are missing in the custom configuration, the relevant missing information is obtained from the following default configuration:

.. code-block:: python

    DEFAULT_ACTIVE_LEARNING_CONFIG = {
        "output_folder": None,
        "pre_tuning": {
            "train_pattern": ".*train.*.txt",
            "test_pattern": ".*test.*.txt",
            "train_seed_pct": 0.20,
        },
        "tuning": {
            "n_classifiers": 3,
            "n_epochs": 5,
            "batch_size": 100,
            "tuning_level": "domain",
            "tuning_strategies": [
                "LeastConfidenceSampling",
                "MarginSampling",
                "EntropySampling",
                "RandomSampling",
                "DisagreementSampling",
                "EnsembleSampling",
                "KLDivergenceSampling",
            ],
        },
        "tuning_output": {
            "save_sampled_queries": True,
            "aggregate_statistic": "accuracy",
            "class_level_statistic": "f_beta",
        },
        "query_selection": {
            "selection_strategy": "EntropySampling",
            "log_usage_pct": 1.00,
            "labeled_logs_pattern": None,
            "unlabeled_logs_path": "logs.txt",
        },
    }

Before diving deeper into strategy tuning and query selection, let's take a look at the different strategies and tuning levels. These hyperparameters are studied at the strategy tuning level with the best ones chosen for query selection.

Strategies
^^^^^^^^^^
The tuning step allows the application to run 7 possible strategies and choose the best performing one. Each strategy is a sampling function that samples the worst performing queries from the latest batch iteration of the training data. The assessment of worst performance comes from the classifiers' confidence in the predictions for that query. All heuristics use this information differently as described next.

+---------------------------+-----------------------------------------------------------------------------------------------+
| Strategy                  | How does it work?                                                                             |
+===========================+===============================================================================================+
| Random Sampling           | Samples the next set of queries at random                                                     |
+---------------------------+-----------------------------------------------------------------------------------------------+
| Least Confidence Sampling | From the available queries in the batch, this sampling strategy samples queries with the      |
|                           | lowest max confidence score across any class, i.e., queries that are the least confident about|
|                           | any of the available classes in the classifier.                                               |
+---------------------------+-----------------------------------------------------------------------------------------------+
| Entropy Sampling          | Calculates the entropy score of the classifier confidences per query. Samples the ones with   |
|                           | highest entropy.                                                                              |
+---------------------------+-----------------------------------------------------------------------------------------------+
| Margin Sampling           | From the available queries in the batch, this sampling strategy samples queries with the      |
|                           | that have the lowest confidence score difference between the top two class confidence scores  |
|                           | for the query. This difference is referred to as the "margin".                                |
+---------------------------+-----------------------------------------------------------------------------------------------+
| Disagreement Sampling     | Across n runs of the classifier, this sampling strategy calculates an agreement score for     |
|                           | every query (% of classifiers that voted for the most frequent class). The queries are then   |
|                           | ranked from lowest classifier agreement to highest and the sampled in order.                  |
+---------------------------+-----------------------------------------------------------------------------------------------+
| KL Divergence Sampling    | Across n runs of the classifier, this sampling strategy calculates the KL divergence between  |
|                           | average confidence distribution across all classifiers for a given class and the confidence   |
|                           | distribution for a given query for that class. Queries with higher divergence are sampled.    |
+---------------------------+-----------------------------------------------------------------------------------------------+
| Ensemble Sampling         | Combines ranks from all the above heuristics and samples in order.                            |
+---------------------------+-----------------------------------------------------------------------------------------------+

Tuning Levels
^^^^^^^^^^^^^

Tuning level defines the level at which the classifier evaluation is done and for which the confidence scores or model probabilites are passed onto the strategies defined above. One of three strategies can be employed - domain, intent or joint.

1. For the domain level, the domain classifier is run and the probability scores for the domain classifier are passed to the strategies.
2. For the intent level, the intent classifier scores (vectors) across all domains are compressed down to a single vector and passed on to the strategies.
3. For the joint level, a dot product is taken across the domain and intent level predictions and the combined representation of confidences/probability scores is provided to the strategies.


Strategy Tuning
---------------
The strategy tuning phase in the active learning pipeline is useful in choosing the best tuning strategies for your application. According to the config value of ``train_seed_pct``, the pipeline picks that percentage of training data (identified as filed that match the ``train_pattern`` regex in the config file) as seed data. Next, the classifiers are trained on this seed data and evaluated on the test data (files matching ``test_pattern``). Iteratively, data equivalent to the ``batch_size`` is added from the remaining training data to the initial seed data and the classifier models retrained. This selection is made according to different tuning strategies which are discussed later. The test or evaluation dataset remains static. In this process the train set keeps growing till all training data has been consumed by the model and the final iteration of classifier training has been trained on all possible training data. The accuracy, query selection and tuning strategy performance results are saved along with performance plots tracking the classifier performance in tandem with the increasing training data. This process is repeated for ``n_epochs`` (as defined in the config) to obtain average active learning performance. Results obtained at the end can be used to quantitatively and visually choose the best tuning approach for your application.


The following command can be used to run tuning using the settings defined in the application configuration:

.. code-block:: console

    mindmeld active_learning --tune --app-path '<PATH>/app_name/' --output_folder '<PATH>'

Flags for application path and output folder are required and overwrite the default configuration settings for active learning. In addition to the aforementioned required flags, the following optional flags can be used - tuning_level, batch_size, n_epochs, train_seed_pct, and plot. These are described in detail in the AL config section.

At the end of the tuning process, results are stored in the ``output_folder``. The ``accuracy.json`` file in the directory ``output_folder/results`` consist of strategy performance on the application's test/evaluation data for every iteration and epoch. ``selected_queries.json`` consists of the same information but instead of evaluation performance, this file records the queries selected at that iteration. the ``output_folder/plots`` directory consists of this quantitative information in a  visual format. The plots record performance of all chosen strategies across data iterations and gives a sense of which strategy is best suited for your application. The same information can be gauged from these results and plots about the best ``tuning_level`` for your application.


Query Selection
---------------

Once the best performing strategy and level is known through tuning, the same set of hyperparameters can be carried over to the query selection step. Here, the active learning pipeline picks the best subset of queries from the logs that can be added to the training files to give the maximum performance boost in terms of accuracy.

The following command can be used to run query selection using the settings defined in the application configuration if the log file or the log files' pattern has been specified in the config:

.. code-block:: console

    mindmeld active_learning --select --app-path '<PATH>/app_name/' --output_folder '<PATH>'


Alternatively, path to unlabeled logs (``unlabeled_logs_path``) can be provided as a flag. 

.. code-block:: console

    mindmeld active_learning --select --app-path "<PATH>/app_name/" --output_folder '<PATH>' --unlabeled_logs_path "<PATH>/logs.txt"


Also, if your log data is labelled and included in your MindMeld application you can specify the pattern for your log data using the following flag:

.. code-block:: console 

    mindmeld active_learning --select --app-path '<PATH>/app_name/' --output_folder '<PATH>' --labeled_logs_pattern ".*log.*.txt"

Optional flags that can be used for selection include: ``batch_size``, ``log_usage_pct``, ``strategy``.


.. note::

    For labelled logs pattern, ensure that your log pattern (``labeled_logs_pattern``) does not overlap with your train pattern (``train_pattern``).
