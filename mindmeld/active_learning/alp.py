from copy import deepcopy

import logging

import math
from .data_loading import DataBucketFactory
from .output_manager import OutputManager
from .plot_manager import PlotManager
from .classifiers import MindMeldClassifier
from .heuristics import HeuristicsFactory
from ..constants import STRATEGY_ABRIDGED

logger = logging.getLogger(__name__)


class ActiveLearningPipeline:  # pylint: disable=R0902
    """Class that executes the training and selection process for the Active Learning Pipeline"""

    def __init__(  # pylint: disable=R0913
        self,
        app_path: str,
        train_pattern: str,
        test_pattern: str,
        load: bool,
        save: bool,
        train_seed_pct: float,
        n_classifiers: int,
        n_epochs: int,
        batch_size: int,
        training_strategies: list,
        training_level: str,
        log_selection_strategy: str,
        save_sampled_queries: bool,
        early_stopping_window: int,
        log_usage_pct: float,
        labeled_logs_pattern: str,
        unlabeled_logs_path: str,
        output_folder: str,
    ):
        """
        Args:
            app_path (str): Path to MindMeld application
            train_pattern (str): Regex pattern to match train files. For example, ".*train.*.txt"
            test_pattern (str): Regex pattern to match test files. For example, ".*test.*.txt"
            load (bool): Whether to load pickled queries from a local folder
            save (bool): Whether to save queries as a local pickle file
            train_seed_pct (float): Percentage of training data to use as the initial seed
            n_classifiers (int): Number of classifiers to be used by a subset of heuristics
            n_epochs (int): Number of epochs to run training
            batch_size (int): Number of queries to select at each iteration
            training_level (str): The hierarchy level to train ("domain" or "intent")
            training_strategies (List[str]): List of strategies to use for training
            log_selection_strategy (str): Single strategy to use for log selection
            save_sampled_queries (bool): Whether to save the queries sampled at each iteration
            early_stopping_window (int): If the drops for n iterations, terminate training early
            log_usage_pct (float): Percentage of the log data to use for selection
            labeled_logs_pattern (str): Pattern to obtain logs already labeled in a MindMeld app
            unlabeled_logs_path (str): Path a logs text file with unlabeled queries
        """
        self.app_path = app_path
        self.train_pattern = train_pattern
        self.test_pattern = test_pattern
        self.load = load
        self.save = save
        self.train_seed_pct = train_seed_pct
        self.n_classifiers = n_classifiers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.training_level = training_level
        self.training_strategies = training_strategies
        self.log_selection_strategy = log_selection_strategy
        self.save_sampled_queries = save_sampled_queries
        self.early_stopping_window = early_stopping_window
        self.log_usage_pct = log_usage_pct
        self.labeled_logs_pattern = labeled_logs_pattern
        self.unlabeled_logs_path = unlabeled_logs_path

        self.output_manager = self.get_output_manager(output_folder)
        self.mindmeld_classifier = self.get_classifier()
        self.init_data_bucket = None
        self.data_bucket = None

    def get_output_manager(self, output_folder):
        return OutputManager(
            active_learning_params=deepcopy(self.__dict__), output_folder=output_folder
        )

    def get_classifier(self):
        return MindMeldClassifier(
            app_path=self.app_path,
            training_level=self.training_level,
            n_classifiers=self.n_classifiers,
        )

    def train(self):
        """Loads the initial data bucket and then trains on every strategy."""
        logger.info("Creating Training Data Bucket")
        self.output_manager.create_experiment_folder(self.training_strategies)
        self.init_data_bucket = DataBucketFactory.get_data_bucket_for_training(
            self.app_path,
            self.load,
            self.save,
            self.train_pattern,
            self.test_pattern,
            self.train_seed_pct,
        )
        logger.info("Starting Training")
        for strategy in self.training_strategies:
            self._train_strategy(strategy)

    def _train_strategy(
        self,
        strategy: str,
        selection_mode: bool = None,
    ):
        """Helper function to traing a single strategy.

        Args:
            strategy (str): Single strategy to train
            selection_mode (bool): If in selection mode, accuracies will not be recorded
                and the run will terminate after the first iteration
        """
        early_stop = False
        for epoch in range(self.n_epochs):
            del self.data_bucket
            self.data_bucket = deepcopy(self.init_data_bucket)
            num_iterations = math.ceil(
                len(self.data_bucket.unsampled_queries) / self.batch_size
            )
            iteration = 0
            while iteration <= num_iterations:
                logger.info(
                    "Strategy: %s. Epoch: %s. Iteration %s.", strategy, epoch, iteration
                )
                logger.info(
                    "Sampled Queries: %s", len(self.data_bucket.sampled_queries)
                )
                logger.info(
                    "Remaining (Unsampled) Queries: %s",
                    len(self.data_bucket.unsampled_queries),
                )
                (
                    eval_stats,
                    preds_single,
                    preds_multi,
                    domain_indices,
                ) = self.mindmeld_classifier.train(
                    data_bucket=self.data_bucket,
                    return_preds_multi=STRATEGY_ABRIDGED[strategy]
                    in ["ds", "ens", "kld"],
                )
                if not selection_mode:
                    self.output_manager.update_accuracies_json(
                        strategy, epoch, iteration, eval_stats
                    )
                    if self.save_sampled_queries:
                        self.output_manager.update_selected_queries_json(
                            strategy,
                            epoch,
                            iteration,
                            self.data_bucket.newly_sampled_queries,
                        )
                    if self.early_stopping_window > 0:
                        early_stop = self.output_manager.check_early_stopping(
                            strategy, self.early_stopping_window
                        )
                num_unsampled = len(self.data_bucket.unsampled_queries)
                if num_unsampled > 0:
                    sampling_size = (
                        self.batch_size
                        if num_unsampled > self.batch_size
                        else num_unsampled
                    )
                    heuristic = HeuristicsFactory.get_heuristic(strategy, sampling_size)
                    (
                        self.data_bucket.newly_sampled_queries,
                        self.data_bucket.sampled_queries,
                        self.data_bucket.unsampled_queries,
                        _,
                    ) = heuristic.sample(
                        sampled=self.data_bucket.sampled_queries,
                        unsampled=self.data_bucket.unsampled_queries,
                        preds_single=preds_single,
                        preds_multi=preds_multi,
                        domain_indices=domain_indices,
                    )
                if selection_mode or early_stop:
                    return
                iteration += 1

    def select_queries_to_label(self):
        """Selects the next batch of queries to label from a set of log queries."""
        logger.info("Loading Queries for Active Learning.")
        self.init_data_bucket = DataBucketFactory.get_data_bucket_for_selection(
            self.app_path,
            self.load,
            self.save,
            self.train_pattern,
            self.test_pattern,
            self.unlabeled_logs_path,
            self.labeled_logs_pattern,
            self.log_usage_pct,
        )
        logger.info("Starting Selection.")
        self._train_strategy(strategy=self.log_selection_strategy, selection_mode=True)
        self.output_manager.write_log_selected_queries_json(
            strategy=self.log_selection_strategy,
            queries=self.data_bucket.newly_sampled_queries,
        )

    def plot(self):
        """Creates the generated folder and its subfolders if they do not already exist."""
        plot_manager = PlotManager(self.output_manager.experiment_folder)
        plot_manager.generate_plots()


# TODO: This function is temporary. Replace with better data validation
def flatten_active_learning_config(original_config):
    """Create a flattened config to use as params.

    Args:
        original_config (dict): The original input config dictionary

    Returns:
        flattened_config (dict): Flattened config
    """
    flattened_config = {}
    for key in original_config:
        if not isinstance(original_config[key], dict):
            flattened_config[key] = original_config[key]
        else:
            flattened_config.update(original_config[key])
    return flattened_config
