import os
import json
import datetime
from typing import Dict, List

from mindmeld.markup import dump_query

from ..constants import STRATEGY_ABRIDGED


# File Creation Methods
def create_dir_if_absent(base_path: str):
    """Create a directory if one doesn't already exist at the given path.
    Args:
        base_path (str): Path to create directory."""
    try:
        os.makedirs(base_path)
    except FileExistsError:
        pass


def create_sub_dirs_if_absent(base_path: str, sub_dirs: List):
    """Create a subdirectories if they don't already exist at the given path.
    Args:
        base_path (str): Root directory of the give sub directories.
        sub_dirs (List): List of subdirectories from the base path.
    """
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(base_path, sub_dir)
        create_dir_if_absent(sub_dir_path)


# Get JSON File paths
def get_accuracies_json_path(experiment_dir_path) -> str:
    """
    Args:
        experiment_dir_path (str): Path to the experiment's directory.
    Returns:
        accuracies_json_path (str): Path to the experiment's queries.json file.
    """
    return os.path.join(*[experiment_dir_path, "results", "accuracies.json"])


def get_queries_json_path(experiment_dir_path) -> str:
    """
    Args:
        experiment_dir_path (str): Path to the experiment's directory.
    Returns:
        queries_json_path (str): Path to the experiment's queries.json file.
    """
    return os.path.join(*[experiment_dir_path, "results", "queries.json"])


def get_log_selected_queries_json_path(experiment_dir_path) -> str:
    """
    Args:
        experiment_dir_path (str): Path to the experiment's directory.
    Returns:
        queries_json_path (str): Path to the experiment's queries.json file.
    """
    return os.path.join(*[experiment_dir_path, "results", "log_selected_queries.json"])


class OutputManager:
    """Handles the initialization of generated folder and its contents. Keeps record of experiment
        results."""

    def __init__(
        self,
        active_learning_params: Dict,
        selection_strategies: List[str],
        save_accuracy_results=True,
        save_sampled_queries=False,
        early_stopping_window: int = None,
    ):
        """
        Args:
            config_dict (Dict): Dictionary representation of the config to store.
            selection_strategies (list): List of strategies used for the experiment.
            save_accuracy_results (bool): Whether to save accuracy metrics.
            save_sampled_queries (bool): Whether to save queries sampled at every batch.
            early_stopping_window (int): Number of iterations to stop after if accuracy keeps
                decreasing.
        """
        self.active_learning_params = active_learning_params
        self.selection_strategies = selection_strategies
        self.save_accuracy_results = save_accuracy_results
        self.save_sampled_queries = save_sampled_queries
        self.early_stopping_window = early_stopping_window
        self.experiment_dir_path = self._get_experiment_dir_path()
        OutputManager.create_generated_dir()
        self.create_current_experiment_dir()

    # Directory Initialization
    @staticmethod
    def create_generated_dir():
        """Creates the generated folder and its subfolders if they do not already exist."""
        create_sub_dirs_if_absent(
            base_path="generated", sub_dirs=["saved_queries", "experiments"]
        )

    def create_current_experiment_dir(self):
        """Creates the current experiment folder."""
        create_sub_dirs_if_absent(
            base_path=self.experiment_dir_path, sub_dirs=["results", "plots"]
        )
        self.create_json_files()
        self.create_saved_config_json()

    def create_json_files(self):
        """Creates accuracies.json and queries.json in the current experiment folder."""
        accuracies_json_path = get_accuracies_json_path(self.experiment_dir_path)
        queries_json_path = get_queries_json_path(self.experiment_dir_path)
        for json_path in [accuracies_json_path, queries_json_path]:
            with open(json_path, "w") as outfile:
                json.dump({}, outfile)
                outfile.close()

    def create_saved_config_json(self):
        """Creates a config.json to store in the experiment folder. config.json contains
        configuration parameters."""
        config_path = os.path.join(
            self.experiment_dir_path, "active_learning_params.json"
        )
        with open(config_path, "w") as outfile:
            json.dump(self.active_learning_params, outfile, indent=4)
            outfile.close()

    # Getters
    def _get_experiment_dir_path(self) -> str:
        """
        Returns:
            experiment_dir_path (str): Creates the path of the current experiment folder.
        """
        return os.path.join(
            *["generated", "experiments", self._get_experiment_dir_name()]
        )

    def _get_experiment_dir_name(self) -> str:
        """
        Returns:
            experiment_dir_name (str): Creates the name of the current experiment folder
                based on the current timestamp.
        """
        strategies = "_".join(self._get_strategies_abridged())
        now = datetime.datetime.now()
        return f"{now.month}-{now.day}_{now.hour}:{now.minute}_{strategies}"

    def _get_strategies_abridged(self) -> List:
        """
        Returns:
            strategies (list): Sorted list of strategies in abridged form.
        """
        strategies = [
            STRATEGY_ABRIDGED[strategy] for strategy in self.selection_strategies
        ]
        strategies.sort()
        return strategies

    # JSON Update Methods
    def write_log_selected_queries(
        self, strategy: str, epoch: int, iteration: int, sampled_queries_batch: List,
    ):
        """Update accuracies.json and queries.json if specified to do so in the config.
        Args:
            strategy (str): Current training strategy.
            epoch (int): Current epoch.
            iteration (int): Current iteration.
            sampled_queries_batch (List): List of queries sampled for the current iteration.
        """
        self.create_log_selected_queries_json()
        OutputManager._update_json(
            json_path=get_log_selected_queries_json_path(self.experiment_dir_path),
            strategy=strategy,
            epoch=epoch,
            iteration=iteration,
            data=OutputManager.queries_to_dict(sampled_queries_batch),
        )

    def create_log_selected_queries_json(self):
        log_selected_queries_json_path = get_log_selected_queries_json_path(
            self.experiment_dir_path
        )
        with open(log_selected_queries_json_path, "w") as outfile:
            json.dump({}, outfile)
            outfile.close()

    def update(
        self,
        strategy: str,
        epoch: int,
        iteration: int,
        classifier_output: Dict,
        sampled_queries_batch: List,
    ):
        """Update accuracies.json and queries.json if specified to do so in the config.
        Args:
            strategy (str): Current training strategy.
            epoch (int): Current epoch.
            iteration (int): Current iteration.
            classifier_output (Dict): Accuracy data to save to accuracies.json.
            sampled_queries_batch (List): List of queries sampled for the current iteration.
        """
        if self.save_sampled_queries:
            OutputManager._update_json(
                json_path=get_queries_json_path(self.experiment_dir_path),
                strategy=strategy,
                epoch=epoch,
                iteration=iteration,
                data=OutputManager.queries_to_dict(sampled_queries_batch),
            )
        if self.save_accuracy_results:
            OutputManager._update_json(
                json_path=get_accuracies_json_path(self.experiment_dir_path),
                strategy=strategy,
                epoch=epoch,
                iteration=iteration,
                data=classifier_output,
            )
            if self.early_stopping_window:
                return self._check_early_stopping(
                    json_path=get_accuracies_json_path(self.experiment_dir_path),
                    strategy=strategy,
                    epoch=epoch,
                    iteration=iteration,
                )

    @staticmethod
    def _update_json(json_path: str, strategy: str, epoch: int, iteration: int, data):
        """Helper method to update json files.
        Args:
            strategy (str): Current training strategy.
            epoch (int): Current epoch.
            iteration (int): Current iteration.
            data (Dict or List): Data to store for current strategy, epoch, and iteration.
        """
        with open(json_path, "r") as infile:
            json_data = json.load(infile)
            infile.close()
        if strategy not in json_data:
            json_data[strategy] = {}
        if str(epoch) not in json_data[strategy]:
            json_data[strategy][str(epoch)] = {}
        json_data[strategy][str(epoch)][str(iteration)] = data
        with open(json_path, "w") as outfile:
            json.dump(json_data, outfile, indent=4)
            outfile.close()

    def _check_early_stopping(
        self, json_path: str, strategy: str, epoch: int, iteration: int
    ):
        """Helper method to update json files.
        Args:
            strategy (str): Current training strategy.
            epoch (int): Current epoch.
            iteration (int): Current iteration.
        """
        with open(json_path, "r") as infile:
            json_data = json.load(infile)
            infile.close()
        if strategy in json_data and str(epoch) in json_data[strategy]:
            iteration_scores = [
                json_data[strategy][str(epoch)][str(i)]["accuracies"]["overall"]
                for i in range(int(iteration) + 1)
            ]
            if (
                len(iteration_scores) >= self.early_stopping_window
                and self.early_stopping_window > 0
            ):
                ref_score = iteration_scores[-1 * self.early_stopping_window]
                highest_score = max(
                    iteration_scores[-(self.early_stopping_window + 1) :]
                )
                if ref_score > highest_score:
                    print(
                        f"""Early stopping. Early stopping window {self.early_stopping_window}.
                    Reference score at the window start: {ref_score} is greater than the highest
                    after window start: {highest_score}."""
                    )
                    return True
        return False

    # Utility Method
    @staticmethod
    def queries_to_dict(queries: List) -> List:
        """Convert a list of ProcessedQueries into a list dictionaries.
        Args:
            queries (List): List of ProcessedQuery objects
        Returns:
            dict_queries (List): List of queries represented as a dict with the keys
                "text", "domain", and "intent".
        """
        return [
            {"text": dump_query(query), "domain": query.domain, "intent": query.intent}
            for query in queries
        ]
