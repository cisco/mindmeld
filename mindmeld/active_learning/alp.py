from copy import deepcopy
import logging
import math

from .data_loading import DataBucketFactory
from .results_manager import ResultsManager
from .plot_manager import PlotManager
from .classifiers import MindMeldALClassifier
from .heuristics import HeuristicsFactory

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
        selection_strategy: str,
        save_sampled_queries: bool,
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
            selection_strategy (str): Single strategy to use for log selection
            save_sampled_queries (bool): Whether to save the queries sampled at each iteration
            log_usage_pct (float): Percentage of the log data to use for selection
            labeled_logs_pattern (str): Pattern to obtain logs already labeled in a MindMeld app
            unlabeled_logs_path (str): Path a logs text file with unlabeled queries
            output_folder (str): Folder to store active learning results.
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
        self.selection_strategy = selection_strategy
        self.save_sampled_queries = save_sampled_queries
        self.log_usage_pct = log_usage_pct
        self.labeled_logs_pattern = labeled_logs_pattern
        self.unlabeled_logs_path = unlabeled_logs_path
        self.output_folder = output_folder

        self.results_manager = ResultsManager(output_folder)
        self.mindmeld_al_classifier = self._get_mindmeld_al_classifier()
        self.init_train_data_bucket = None
        self.data_bucket = None

    def _get_mindmeld_al_classifier(self):
        """ Creates an instance of a MindMeld Active Learning Classifier. """
        return MindMeldALClassifier(
            self.app_path, self.training_level, self.n_classifiers
        )

    @property
    def num_iterations(self) -> int:
        """
        Returns:
            num_iterations (int): Number of iterations needed for training.
        """
        return 1 + math.ceil(
            len(self.init_train_data_bucket.unsampled_queries) / self.batch_size
        )

    def _as_dict(self):
        """ Custom dictionary method used to save key experiment params. """
        return {
            "app_path": self.app_path,
            "train_pattern": self.train_pattern,
            "test_pattern": self.test_pattern,
            "load": self.load,
            "save": self.save,
            "train_seed_pct": self.train_seed_pct,
            "n_classifiers": self.n_classifiers,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "training_level": self.training_level,
            "training_strategies": self.training_strategies,
            "selection_strategy": self.selection_strategy,
            "save_sampled_queries": self.save_sampled_queries,
            "log_usage_pct": self.log_usage_pct,
            "labeled_logs_pattern": self.labeled_logs_pattern,
            "unlabled_logs_path": self.unlabeled_logs_path,
            "output_folder": self.output_folder,
        }

    def train(self):
        """Loads the initial data bucket and then trains on every strategy."""
        logger.info("Creating output folder and saving params.")
        self.results_manager.create_experiment_folder(
            active_learning_params=self._as_dict(),
            training_strategies=self.training_strategies,
        )
        logger.info("Creating training data bucket.")
        self.init_train_data_bucket = DataBucketFactory.get_data_bucket_for_training(
            self.app_path,
            self.load,
            self.save,
            self.training_level,
            self.train_pattern,
            self.test_pattern,
            self.train_seed_pct,
        )
        logger.info("Starting training.")
        self._train_all_strategies()

    def select(self):
        """Selects the next batch of queries to label from a set of log queries."""
        logger.info("Loading queries for active learning.")
        self.init_train_data_bucket = DataBucketFactory.get_data_bucket_for_selection(
            self.app_path,
            self.load,
            self.save,
            self.training_level,
            self.train_pattern,
            self.test_pattern,
            self.unlabeled_logs_path,
            self.labeled_logs_pattern,
            self.log_usage_pct,
        )
        logger.info("Starting selection.")
        self._train_single_strategy(
            strategy=self.selection_strategy, training_mode=False
        )
        self.results_manager.write_log_selected_queries_json(
            strategy=self.selection_strategy,
            queries=self.data_bucket.newly_sampled_queries,
        )

    def plot(self):
        """Creates the generated folder and its subfolders if they do not already exist."""
        plot_manager = PlotManager(self.results_manager.experiment_folder)
        plot_manager.generate_plots()

    def _train_all_strategies(self):
        """ Train with all active learning strategies."""
        for strategy in self.training_strategies:
            self._train_single_strategy(strategy)

    def _train_single_strategy(self, strategy: str, training_mode: bool = True):
        """Helper function to traing a single strategy.

        Args:
            strategy (str): Single strategy to train
            training_mode (bool): If True, accuracies will be recorded. If False, accuracies
                will not be recorded and run will terminate after first iteration.
        """
        heuristic = HeuristicsFactory.get_heuristic(strategy)
        for epoch in range(self.n_epochs):
            self._reset_data_bucket()
            for iteration in range(self.num_iterations):
                self._log_training_status(strategy, epoch, iteration)
                # Run training and obtain probability distributions for each query
                (
                    eval_stats,
                    confidences_2d,
                    confidences_3d,
                    confidence_segments,
                ) = self.mindmeld_al_classifier.train(self.data_bucket, heuristic)

                if training_mode:
                    self._save_training_data(strategy, epoch, iteration, eval_stats)

                num_unsampled = len(self.data_bucket.unsampled_queries)
                if num_unsampled > 0:
                    self.data_bucket.sample_and_update(
                        sampling_size=self._get_sampling_size(num_unsampled),
                        confidences_2d=confidences_2d,
                        confidences_3d=confidences_3d,
                        heuristic=heuristic,
                        confidence_segments=confidence_segments,
                    )
                # Terminate on the first iteration if in selection mode.
                if not training_mode:
                    return

    def _reset_data_bucket(self):
        """ Reset the DataBucket to the initial DataBucket after every epoch."""
        self.data_bucket = deepcopy(self.init_train_data_bucket)

    def _log_training_status(self, strategy, epoch, iteration):
        logger.info("Strategy: %s. Epoch: %s. Iter: %s.", strategy, epoch, iteration)
        logger.info("Sampled Elements: %s", len(self.data_bucket.sampled_queries))
        logger.info("Remaining Elements: %s", len(self.data_bucket.unsampled_queries))

    def _save_training_data(self, strategy, epoch, iteration, eval_stats):
        """ Save training data if in training_mode. """
        self.results_manager.update_accuracies_json(
            strategy, epoch, iteration, eval_stats
        )
        if self.save_sampled_queries:
            self.results_manager.update_selected_queries_json(
                strategy,
                epoch,
                iteration,
                self.data_bucket.newly_sampled_queries,
            )

    def _get_sampling_size(self, num_unsampled) -> int:
        """Calculate the number of elements to sample based on the batch_size and remaining
        number of elements in the pipeline.
        Returns:
            sampling_size (int): Number of elements to sample in the next iteration.
        """
        return self.batch_size if num_unsampled > self.batch_size else num_unsampled


class ActiveLearningPipelineFactory:
    """Creates an ActiveLearningPipeline instance from values in a config."""

    @staticmethod
    def create_from_config(config):
        """Creates an augmentor instance using the provided configuration
        Args:
            config (dict): A model configuration.
        Returns:
            ActiveLearningPipeline: An ActiveLearningPipeline class

        Raises:
            ValueError: When model configuration is invalid or required key is missing
        """
        return ActiveLearningPipeline(
            app_path=config.get("app_path"),
            train_pattern=config.get("pre_training").get("train_pattern"),
            test_pattern=config.get("pre_training").get("test_pattern"),
            load=config.get("pre_training").get("load"),
            save=config.get("pre_training").get("save"),
            train_seed_pct=config.get("pre_training").get("train_seed_pct"),
            n_classifiers=config.get("training").get("n_classifiers"),
            n_epochs=config.get("training").get("n_epochs"),
            batch_size=config.get("training").get("batch_size"),
            training_strategies=config.get("training").get("training_strategies"),
            training_level=config.get("training").get("training_level"),
            selection_strategy=config.get("selection").get("selection_strategy"),
            save_sampled_queries=config.get("training_output").get(
                "save_sampled_queries"
            ),
            log_usage_pct=config.get("selection").get("log_usage_pct"),
            labeled_logs_pattern=config.get("selection").get("labeled_logs_pattern"),
            unlabeled_logs_path=config.get("selection").get("unlabeled_logs_path"),
            output_folder=config.get("output_folder"),
        )
