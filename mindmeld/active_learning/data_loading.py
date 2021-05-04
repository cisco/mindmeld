from typing import Dict, List
import pickle
import logging
from os.path import normpath, basename

from .heuristics import Heuristic, stratified_random_sample

from ..auto_annotator import BootstrapAnnotator
from ..components._config import DEFAULT_AUTO_ANNOTATOR_CONFIG
from ..constants import TRAIN_LEVEL_DOMAIN, TRAIN_LEVEL_INTENT, AL_MAX_LOG_USAGE_PCT
from ..core import ProcessedQuery
from ..markup import read_query_file
from ..path import AL_QUERIES_CACHE_PATH
from ..query_factory import QueryFactory
from ..resource_loader import ResourceLoader


logger = logging.getLogger(__name__)


class LabelMap:
    """Class that handles label encoding and mapping."""

    def __init__(self, query_tree: Dict):
        """
        Args:
            query_tree (list): query_tree (dict): Nested Dictionary containing queries.
                Has the format: {"domain":{"intent":[Query List]}}.
        """
        domain_to_intents = LabelMap.get_domain_to_intents(query_tree)

        self.domain2id = LabelMap._get_domain_mappings(domain_to_intents)
        self.id2domain = LabelMap._reverse_dict(self.domain2id)
        self.intent2id = LabelMap._get_intent_mappings(domain_to_intents)
        self.id2intent = LabelMap._reverse_nested_dict(self.intent2id)

    @staticmethod
    def get_domain_to_intents(query_tree: Dict) -> Dict:
        """
        Args:
            query_tree (list): query_tree (dict): Nested Dictionary containing queries.
                Has the format: {"domain":{"intent":[Query List]}}

        Returns:
            domain_to_intents (dict): Dict mapping domains to a list of intents.
        """
        domain_to_intents = {}
        for domain in query_tree.keys():
            domain_to_intents[domain] = list(query_tree[domain].keys())
        return domain_to_intents

    @staticmethod
    def _get_domain_mappings(domain_to_intents: Dict) -> Dict:
        """Creates a dictionary that maps domains to encoded ids.

        Args:
            domain_to_intents (dict): Dict mapping domains to a list of intents.

        Returns:
            domain2id (dict): dict with domain to id mappings.
        """
        domain2id = {}
        domains = list(domain_to_intents.keys())
        for index, domain in enumerate(domains):
            domain2id[domain] = index
        return domain2id

    @staticmethod
    def _get_intent_mappings(domain_to_intents: Dict) -> Dict:
        """Creates a dictionary that maps intents to encoded ids.

        Args:
            domain_to_intents (dict): Dict mapping domains to a list of intents.

        Returns:
            intent2id (dict): dict with intent to id mappings.
        """
        intent2id = {}
        for domain in domain_to_intents:
            intent_labels = {}
            for index, intent in enumerate(domain_to_intents[domain]):
                intent_labels[intent] = index
            intent2id[domain] = intent_labels
        return intent2id

    @staticmethod
    def _reverse_dict(dictionary: Dict[str, int]):
        """
        Returns:
            reversed_dict (dict): Reversed dictionary.
        """
        reversed_dict = {v: k for k, v in dictionary.items()}
        return reversed_dict

    @staticmethod
    def _reverse_nested_dict(dictionary: Dict[str, Dict[str, int]]):
        """
        Returns:
            reversed_dict (dict): Reversed dictionary.
        """
        reversed_dict = {}

        for parent_key in dictionary:
            reversed_dict[parent_key] = LabelMap._reverse_dict(dictionary[parent_key])
        return reversed_dict

    @staticmethod
    def get_label_map(app_path, file_pattern):
        """Creates a label map.

        Args:
            app_path (str): Path to MindMeld application
            file_pattern (str): Regex pattern to match text files. (".*train.*.txt")

        Returns:
            label_map (LabelMap): A label map.
        """
        query_tree = QueryLoader.get_query_tree(app_path, file_pattern)
        return LabelMap(query_tree)


class QueryLoader:
    """Class that handles creating a query tree given an app path and file pattern."""

    def __init__(
        self,
        app_path: str,
        training_level: str,
        file_pattern: str,
        load: bool,
        save: bool,
    ):
        """
        Args:
            app_path (str): Path to MindMeld application
            training_level (str): The hierarchy level to train ("domain" or "intent")
            file_pattern (str): Regex pattern to match text files. (".*train.*.txt")
            load (bool): Whether to load the list of queries
            save (bool): Whether to save the list of queries
        """
        self.app_path = app_path
        self.training_level = training_level
        self.file_pattern = file_pattern
        self.load = load
        self.save = save

    @staticmethod
    def get_query_tree(app_path, file_pattern) -> Dict:
        """Creates a list of paths to the relevant text files.

        Args:
            app_path (str): Path to MindMeld application
            file_pattern (str): Regex pattern to match text files. (".*train.*.txt")

        Returns:
            query_tree (dict): Nested Dictionary {"domain":{"intent":[Query List]}}
        """
        resource_loader = ResourceLoader(
            app_path=app_path, query_factory=QueryFactory.create_query_factory(app_path)
        )
        query_tree = resource_loader.get_labeled_queries(label_set=file_pattern)
        return query_tree

    def get_queries_from_query_tree(self) -> List:
        """Iterates through the data path list and creates a list of
        ProcessedQuery objects for training.

        Returns:
            queries List[ProcessedQuery]: A list of processed queries
        """
        query_tree = QueryLoader.get_query_tree(self.app_path, self.file_pattern)
        queries = []
        for domain in query_tree:
            for intent in query_tree[domain]:
                queries.extend(query_tree[domain][intent])
        if len(queries) == 0:
            raise AssertionError(
                f"No queries found for file pattern: {self.file_pattern}."
                "Ensure that the correct file pattern has been used."
            )
        return queries

    @property
    def app_name(self):
        """
        Returns:
            app_name (str): Name of the MindMeld app.
        """
        return basename(normpath(self.app_path))

    @property
    def cache_file_path(self):
        """
        Returns:
            save_file_path (str): Path to save/load a query list pickle file
        """
        return AL_QUERIES_CACHE_PATH.format(
            app_path=self.app_path, file_name=f"al_cache_{self.file_pattern}.pickle"
        )

    def load_queries(self):
        """ Loads a list of queries from a pickle file. """
        logger.info("Loading queries with the file pattern: %s", self.file_pattern)
        try:
            with open(self.cache_file_path, "rb") as handle:
                query_list = pickle.load(handle)
            return query_list
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Could not find saved data for the file pattern {self.file_pattern}. "
                "If this is the first run, set load to False in the config."
            ) from error

    def save_queries(self, queries):
        """ Saves a list of queries to a pickle file. """
        logger.info("Saving queries with the file pattern: %s", self.file_pattern)
        with open(self.cache_file_path, "wb") as handle:
            pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @property
    def queries(self):
        queries = (
            self.load_queries() if self.load else self.get_queries_from_query_tree()
        )
        if self.save:
            self.save_queries(queries)
        return queries

    @property
    def class_labels(self):
        return QueryLoader.get_class_labels(self.training_level, self.queries)

    @staticmethod
    def get_class_labels(training_level: str, queries: List) -> List[str]:
        """Creates a class label for a set of queries. These labels are used to split
            queries by type. Labels follow the format of "domain" or "domain|intent".
            For example, "date|get_date".

        Args:
            training_level (str): The hierarchy level to train ("domain" or "intent")
            queries (List[ProcessedQuery]): List of query objects.
        Returns:
            class_labels (List[str]): list of labels for classification task.
        """
        if training_level == TRAIN_LEVEL_DOMAIN:
            return [f"{q.domain}" for q in queries]
        elif training_level == TRAIN_LEVEL_INTENT:
            return [f"{q.domain}|{q.intent}" for q in queries]
        else:
            raise ValueError(
                f"Invalid label_type {training_level}. Must be '{TRAIN_LEVEL_DOMAIN}'"
                f" or '{TRAIN_LEVEL_INTENT}'"
            )


class LogQueriesLoader:
    def __init__(self, app_path: str, training_level: str, log_file_path: str):
        """This class loads data as processed queries from a specified log file.
        Args:
            app_path (str): Path to the MindMeld application.
            training_level (str): The hierarchy level to train ("domain" or "intent")
            log_file_path (str): Path to the log file with log queries.
        """
        self.app_path = app_path
        self.training_level = training_level
        self.log_file_path = log_file_path

    @staticmethod
    def filter_raw_text_queries(log_queries_iter) -> List[str]:
        """Removes duplicates in the text queries.

        Args:
            log_queries_iter (generator): Log queries generator.
        Returns:
            filtered_text_queries (List[str]): a List of filtered text queries.
        """
        return list(set(q for q in log_queries_iter))

    def convert_text_queries_to_processed(
        self, text_queries: List[str]
    ) -> List[ProcessedQuery]:
        """Converts text queries to processed queries using an annotator.

        Args:
            text_queries (List[str]): a List of text queries.
        Returns:
            queries (List[ProcessedQuery]): List of processed queries.
        """
        logger.info("Loading a Bootstrap Annotator to process log queries.")
        annotator_params = DEFAULT_AUTO_ANNOTATOR_CONFIG
        annotator_params["app_path"] = self.app_path
        bootstrap_annotator = BootstrapAnnotator(**annotator_params)
        return bootstrap_annotator.text_queries_to_processed_queries(
            text_queries=text_queries
        )

    @property
    def queries(self):
        log_queries_iter = read_query_file(self.log_file_path)
        filtered_text_queries = LogQueriesLoader.filter_raw_text_queries(
            log_queries_iter
        )
        return self.convert_text_queries_to_processed(filtered_text_queries)

    @property
    def class_labels(self):
        return QueryLoader.get_class_labels(self.training_level, self.queries)


class DataBucket:
    """Class to hold data throughout the Active Learning training pipeline.
    Responsible for data conversion, filtration, and storage.
    """

    def __init__(
        self,
        label_map: LabelMap,
        test_queries: List[ProcessedQuery],
        unsampled_queries: List[ProcessedQuery],
        sampled_queries: List[ProcessedQuery],
        newly_sampled_queries: List[ProcessedQuery] = None,
    ):
        self.label_map = label_map
        self.test_queries = test_queries
        self.unsampled_queries = unsampled_queries
        self.sampled_queries = sampled_queries
        self.newly_sampled_queries = newly_sampled_queries

    @staticmethod
    def filter_queries(queries: List, domain: str, intent: str = None):
        """Filter queries for training preperation.

        Args:
            queries (list): List of queries to filter
            domain (str): Domain of desired queries
            intent (str): Intent of desired queries, can be None.

        Returns:
            filtered_queries_indices (list): List of indices of filtered queries.
            filtered_queries (list): List of filtered queries.
        """
        filtered_queries = []
        filtered_queries_indices = []
        for index, query in enumerate(queries):
            if query.domain == domain:
                # Add queries if not filtering at the intent level. Otherwise, add intent match.
                if not intent or (intent and query.intent == intent):
                    filtered_queries_indices.append(index)
                    filtered_queries.append(query)
        return filtered_queries_indices, filtered_queries

    def sample_and_update(
        self,
        sampling_size: int,
        confidences_2d: List[List[float]],
        confidences_3d: List[List[List[float]]],
        heuristic: Heuristic,
        confidence_segments: Dict = None,
    ):
        ranked_indices = (
            heuristic.rank_3d(confidences_3d, confidence_segments)
            if confidences_3d
            else heuristic.rank_2d(confidences_2d)
        )
        newly_sampled_indices = ranked_indices[:sampling_size]
        remaining_unsampled_indices = ranked_indices[sampling_size:]
        self.newly_sampled_queries = [
            self.unsampled_queries[i] for i in newly_sampled_indices
        ]
        self.sampled_queries += self.newly_sampled_queries
        self.unsampled_queries = [
            self.unsampled_queries[i] for i in remaining_unsampled_indices
        ]


class DataBucketFactory:
    """Class to generate the initial data for experimentation. (Seed Queries, Remaining Queries,
    and Test Queries). Loads/Saves data loaders, handles initial sampling and data split based
    on configuation details.
    """

    @staticmethod
    def get_data_bucket_for_training(
        app_path: str,
        load: bool,
        save: bool,
        training_level: str,
        train_pattern: str,
        test_pattern: str,
        train_seed_pct: float,
    ):
        """Creates a DataBucket to be used for training.

        Args:
            app_path (str): Path to MindMeld application
            load (bool): Whether to load pickled queries from a local folder
            save (bool): Whether to save queries as a local pickle file
            training_level (str): The hierarchy level to train ("domain" or "intent")
            train_pattern (str): Regex pattern to match train files. (".*train.*.txt")
            test_pattern (str): Regex pattern to match test files. (".*test.*.txt")
            train_seed_pct (float): Percentage of training data to use as the initial seed

        Returns:
            train_data_bucket (DataBucket): DataBucket for training
        """
        train_query_loader = QueryLoader(
            app_path, training_level, train_pattern, load, save
        )
        train_queries = train_query_loader.queries
        ranked_indices = stratified_random_sample(train_query_loader.class_labels)
        sampling_size = int(train_seed_pct * len(train_queries))
        sampled_queries = [train_queries[i] for i in ranked_indices[:sampling_size]]
        unsampled_queries = [train_queries[i] for i in ranked_indices[sampling_size:]]
        return DataBucket(
            label_map=LabelMap.get_label_map(app_path, train_pattern),
            test_queries=QueryLoader(
                app_path, training_level, test_pattern, load, save
            ).queries,
            unsampled_queries=unsampled_queries,
            sampled_queries=sampled_queries,
            newly_sampled_queries=sampled_queries,
        )

    @staticmethod
    def get_data_bucket_for_selection(
        app_path: str,
        load: bool,
        save: bool,
        training_level: str,
        train_pattern: str,
        test_pattern: str,
        unlabeled_logs_path: str,
        labeled_logs_pattern: str = None,
        log_usage_pct: float = AL_MAX_LOG_USAGE_PCT,
    ):
        """Creates a DataBucket to be used for log selection.

        Args:
            app_path (str): Path to MindMeld application
            load (bool): Whether to load pickled queries from a local folder
            save (bool): Whether to save queries as a local pickle file
            training_level (str): The hierarchy level to train ("domain" or "intent")
            train_pattern (str): Regex pattern to match train files. For example, ".*train.*.txt"
            test_pattern (str): Regex pattern to match test files. For example, ".*test.*.txt"
            unlabeled_logs_path (str): Path a logs text file with unlabeled queries
            labeled_logs_pattern (str): Pattern to obtain logs already labeled within a MindMeld app
            log_usage_pct (float): Percentage of the log data to use for selection

        Returns:
            selection_data_bucket (DataBucket): DataBucket for log selection
        """
        log_query_loader = (
            QueryLoader(app_path, training_level, labeled_logs_pattern, load, save)
            if labeled_logs_pattern
            else LogQueriesLoader(app_path, training_level, unlabeled_logs_path)
        )
        log_queries = log_query_loader.queries
        if log_usage_pct < AL_MAX_LOG_USAGE_PCT:
            sampling_size = int(log_usage_pct * len(log_queries))
            ranked_indices = stratified_random_sample(
                labels=log_query_loader.class_labels
            )
            log_queries = [log_queries[i] for i in ranked_indices[:sampling_size]]

        return DataBucket(
            label_map=LabelMap.get_label_map(app_path, train_pattern),
            test_queries=QueryLoader(
                app_path, training_level, test_pattern, load, save
            ).queries,
            unsampled_queries=log_queries,
            sampled_queries=QueryLoader(
                app_path, training_level, train_pattern, load, save
            ).queries,
        )
