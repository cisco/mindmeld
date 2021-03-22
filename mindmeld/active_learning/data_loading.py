from typing import Dict, List
import pickle
import logging
import os
from os.path import normpath, basename
from copy import deepcopy

from .heuristics import StrategicRandomSampling

from ..constants import SAVED_QUERIES_PATH
from ..auto_annotator import BootstrapAnnotator
from ..resource_loader import ResourceLoader
from ..query_factory import QueryFactory
from ..components._config import DEFAULT_AUTO_ANNOTATOR_CONFIG

logger = logging.getLogger(__name__)


class LabelMap:
    """Class that handles label encoding and mapping."""

    def __init__(self, query_tree: Dict):
        """
        Args:
            query_tree (list): query_tree (dict): Nested Dictionary {"domain":{"intent":[Query List]}}
        """
        domain_to_intents = self.get_domain_to_intents(query_tree)

        self.domain2id = self._get_domain_mappings(domain_to_intents)
        self.id2domain = LabelMap._reverse_dict(self.domain2id)
        self.intent2id = self._get_intent_mappings(domain_to_intents)
        self.id2intent = LabelMap._reverse_nested_dict(self.intent2id)

    def get_domain_to_intents(self, query_tree: Dict) -> Dict:
        """
        Args:
            query_tree (list): query_tree (dict): Nested Dictionary {"domain":{"intent":[Query List]}}

        Returns:
            domain_to_intents (dict): Dict mapping domains to a list of intents.
        """
        domain_to_intents = {}
        for domain in query_tree.keys():
            domain_to_intents[domain] = list(query_tree[domain].keys())
        return domain_to_intents

    def _get_domain_mappings(self, domain_to_intents: Dict) -> Dict:
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

    def _get_intent_mappings(self, domain_to_intents: Dict) -> Dict:
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
            file_pattern (str): Regex pattern to match text files. For example, ".*train.*.txt"
        
        Returns:
            label_map (LabelMap): A label map.
        """
        query_tree = QueryLoader.get_query_tree(app_path, file_pattern)
        return LabelMap(query_tree)


class QueryLoader:
    """Class that handles creating a query tree given an app path and file pattern."""

    def __init__(self, app_path, file_pattern, load, save):
        """
        Args:
            app_path (str): Path to MindMeld application
            file_pattern (str): Regex pattern to match text files. For example, ".*train.*.txt"
            load (bool): Whether to load the list of queries
            save (bool): Whether to save the list of queries
        """
        self.app_path = app_path
        self.file_pattern = file_pattern
        self.load = load
        self.save = save

    @staticmethod
    def get_query_tree(app_path, file_pattern) -> Dict:
        """Creates a list of paths to the relevant text files.

        Args:
            app_path (str): Path to MindMeld application
            file_pattern (str): Regex pattern to match text files. For example, ".*train.*.txt"

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
    def save_file_path(self):
        """
        Returns:
            save_file_path (str): Path to save/load a query list pickle file
        """
        return os.path.join(
            SAVED_QUERIES_PATH, f"{self.app_name}_{self.file_pattern}.pickle"
        )

    def load_queries(self):
        """ Loads a list of queries from a pickle file. """
        print(f"Loading queries with the file pattern {self.file_pattern}.")
        try:
            with open(self.save_file_path, "rb") as handle:
                query_list = pickle.load(handle)
                handle.close()
            return query_list
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Could not find saved data for the file pattern {self.file_pattern}. "
                "If this is the first run, set load to False in the config."
            ) from error

    def save_queries(self, queries):
        """ Saves a list of queries to a pickle file. """
        print(f"Saving queries with the file pattern {self.file_pattern}.")
        with open(self.save_file_path, "wb") as handle:
            pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

    @property
    def queries(self):
        queries = (
            self.load_queries() if self.load else self.get_queries_from_query_tree()
        )
        if self.save:
            self.save_queries(queries)
        return queries


class DataBucketFactory:
    """Class to generate the initial data for experimentation. (Seed Queries, Remaining Queries, and Test Queries)
    Loads/Saves data loaders, handles initial sampling and data split based on configuation details.
    """

    @staticmethod
    def get_data_bucket_for_training(
        app_path, load, save, train_pattern, test_pattern, init_train_seed_pct
    ):
        """ Creates a DataBucket to be used for training.
        
        Args:
            app_path (str): Path to MindMeld application
            load (bool): Whether to load pickled queries from a local folder
            save (bool): Whether to save queries as a local pickle file
            train_pattern (str): Regex pattern to match train files. For example, ".*train.*.txt"
            test_pattern (str): Regex pattern to match test files. For example, ".*test.*.txt"
            init_train_seed_pct (float): Percentage of training data to use as the initial seed
        
        Returns:
            train_data_bucket (DataBucket): DataBucket for training
        """

        train_queries = QueryLoader(app_path, train_pattern, load, save).queries
        sample_size = int(init_train_seed_pct * len(train_queries))
        (
            newly_sampled_queries,
            sampled_queries,
            unsampled_queries,
            _,
        ) = StrategicRandomSampling(sample_size).sample(unsampled=train_queries)

        return DataBucket(
            label_map=LabelMap.get_label_map(app_path, train_pattern),
            test_queries=QueryLoader(app_path, test_pattern, load, save).queries,
            unsampled_queries=unsampled_queries,
            sampled_queries=sampled_queries,
            newly_sampled_queries=newly_sampled_queries,
        )

    @staticmethod
    def get_data_bucket_for_selection(
        app_path,
        load,
        save,
        train_pattern,
        test_pattern,
        unlabeled_logs_path,
        labeled_logs_pattern=None,
        log_usage_pct=1.0,
    ):
        """ Creates a DataBucket to be used for log selection.
        
        Args:
            app_path (str): Path to MindMeld application
            load (bool): Whether to load pickled queries from a local folder
            save (bool): Whether to save queries as a local pickle file
            train_pattern (str): Regex pattern to match train files. For example, ".*train.*.txt"
            test_pattern (str): Regex pattern to match test files. For example, ".*test.*.txt"
            unlabeled_logs_path (str): Path a logs text file with unlabeled queries
            labeled_logs_pattern (str): Pattern to obtain logs already labeled and dispersed within the MindMeld app
            log_usage_pct (float): Percentage of the log data to use for selection

        Returns:
            selection_data_bucket (DataBucket): DataBucket for log selection
        """
        log_queries = (
            QueryLoader(app_path, labeled_logs_pattern, load, save).queries
            if labeled_logs_pattern
            else LogQueriesLoader(app_path, unlabeled_logs_path).queries
        )
        if log_usage_pct < 1.0:
            sample_size = int(log_usage_pct * len(log_queries))
            log_queries, _, _, _ = StrategicRandomSampling(sample_size).sample(
                unsampled=log_queries
            )
        return DataBucket(
            label_map=LabelMap.get_label_map(app_path, train_pattern),
            test_queries=QueryLoader(app_path, test_pattern, load, save).queries,
            unsampled_queries=log_queries,
            sampled_queries=QueryLoader(app_path, train_pattern, load, save).queries,
        )


class DataBucket:
    """Class to hold data throughout the Active Learning training pipeline.
    Responsible for data conversion, filtration, and storage.
    """

    def __init__(
        self,
        label_map,
        test_queries,
        unsampled_queries,
        sampled_queries,
        newly_sampled_queries=None,
    ):

        self.label_map = label_map
        self.test_queries = test_queries
        self.unsampled_queries = unsampled_queries
        self.sampled_queries = sampled_queries
        self.newly_sampled_queries = (
            newly_sampled_queries
            if newly_sampled_queries
            else deepcopy(sampled_queries)
        )

    def convert_queries_to_tuples(self, queries: List, class_type: str = "domain"):
        """Converts MindMeld queries into queries in a tuple format.

        Args:
            queries (list): List of queries to convert
            class_type (str): Queries can map to "domain" or "intent" labels

        Returns:
            tuple_queries (list): List of queries in a tuple format. [(Text, Class)...]
        """
        tuple_queries = []

        if class_type == "domain":
            for query in queries:
                encoded_domain = self.label_map.domain2id[query.domain]
                tuple_queries.append((query.query.text, encoded_domain))
        elif class_type == "intent":
            for query in queries:
                encoded_intent = self.label_map.intent2id[query.domain][query.intent]
                tuple_queries.append((query.query.text, encoded_intent))
        else:
            raise ValueError(
                f"Class type {class_type} is invalid. Must be 'domain' or 'intent'."
            )
        return tuple_queries

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
        filtered_queries_indices = []
        filtered_queries = []
        for index, query in enumerate(queries):
            if query.domain == domain:
                if not intent or (intent and query.intent == intent):
                    filtered_queries_indices.append(index)
                    filtered_queries.append(query)
        return filtered_queries_indices, filtered_queries


class LogQueriesLoader:
    def __init__(self, app_path: str, log_file_path: str):
        """This class loads data as processed queries from a specified log file.
        Args:
            app_path (str): Path to the MindMeld application.
            log_file_path (str): Path to the log file with log queries.
        """
        assert os.path.isfile(log_file_path), f"{log_file_path} is not a valid file"
        self.log_file_path = log_file_path
        self.app_path = app_path

    def get_raw_text_queries(self):
        """ Reads in the data from the log file.

        Returns:
            text_queries (List[str]): a List of text queries.
        """
        return open(self.log_file_path, "r").read().split("\n")

    def filter_raw_text_queries(self, text_queries):
        """ Removes duplicates in the text queries.

        Args:
            text_queries (List[str]): a List of text queries.
        
        Returns:
            filtered_text_queries (List[str]): a List of filtered text queries.
        """
        return list(set(text_queries))

    def convert_text_queries_to_processed(self, text_queries):
        """ Converts text queries to processed queries using an annotator.
        
        Args:
            text_queries (List[str]): a List of text queries.
        
        Returns:
            queries (List[ProcessedQuery]): List of processed queries.
        """ 
        print("Loading an Annotator")
        annotator_params = DEFAULT_AUTO_ANNOTATOR_CONFIG
        annotator_params["app_path"] = self.app_path
        bootstrap_annotator = BootstrapAnnotator(**annotator_params)
        return bootstrap_annotator.convert_text_queries_to_processed(
            text_queries=text_queries
        )

    @property
    def queries(self):
        raw_text_queries = self.get_raw_text_queries()
        filtered_text_queries = self.filter_raw_text_queries(raw_text_queries)
        return self.convert_text_queries_to_processed(filtered_text_queries)
