# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Cisco Systems, Inc. and others.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module saves the results from the Active Learning Pipeline.
"""

import os
import json
import datetime
import logging
from typing import Dict, List

from mindmeld.markup import dump_query
from ..path import (
    AL_PARAMS_PATH,
    AL_RESULTS_FOLDER,
    AL_PLOTS_FOLDER,
    AL_ACCURACIES_PATH,
    AL_SELECTED_QUERIES_PATH,
)
from ..constants import STRATEGY_ABRIDGED

logger = logging.getLogger(__name__)


class ResultsManager:
    """Handles the initialization of generated folder and its contents. Keeps record of experiment
    results."""

    def __init__(
        self,
        output_folder: str,
    ):
        """
        Args:
            output_folder (str): Directory to create an experiment folder or save log queries.
        """
        self.output_folder = output_folder
        self.experiment_folder_name = None

    def set_experiment_folder_name(self, selection_strategies) -> str:
        """
        Args:
            selection_strategies (list): List of strategies used for the experiment.
        Returns:
            experiment_folder_name (str): Creates the name of the current experiment folder
                based on the current timestamp.
        """
        strategies = "_".join(
            STRATEGY_ABRIDGED[s] for s in selection_strategies if s in STRATEGY_ABRIDGED
        )
        now = datetime.datetime.now()
        self.experiment_folder_name = (
            f"{now.year}-{now.month}-{now.day}_{now.hour}:{now.minute}_{strategies}"
        )

    @property
    def experiment_folder(self):
        """
        Returns:
            experiment_folder (str): Path to the Active Learning experiment folder.
        """
        return os.path.join(self.output_folder, self.experiment_folder_name)

    def create_experiment_folder(
        self, active_learning_params: Dict, tuning_strategies: List
    ):
        """Creates the active learning experiment folder.
        Args:
            active_learning_params (Dict): Dictionary representation of the params to store.
            tuning_strategies (list): List of strategies used for the experiment.
        """
        self.set_experiment_folder_name(tuning_strategies)
        os.makedirs(self.experiment_folder, exist_ok=True)
        self.dump_json(AL_PARAMS_PATH, active_learning_params)
        self.create_folder(AL_RESULTS_FOLDER)
        self.create_folder(AL_PLOTS_FOLDER)

    def create_folder(self, unformatted_path):
        """Creates a folder given an unformatted path.
        Args:
            unformatted_path (str): Unformatted path to JSON file.
        """
        os.makedirs(self.format_path(unformatted_path), exist_ok=True)

    def format_path(self, unformatted_path):
        """
        Args:
            unformatted_path (str): Unformatted path to JSON file.
        Returns:
            formatted_path (str): Path formatted with the experiment folder.
        """
        return unformatted_path.format(experiment_folder=self.experiment_folder)

    def load_json(self, unformatted_path: str):
        """Load JSON data from file. If the JSON file doesn't exist, an empty json file is created.
        Args:
            unformatted_path (str): Unformatted path to JSON file.
        Returns:
            json_data (Dict): Loaded JSON data.
        """
        formatted_path = unformatted_path.format(
            experiment_folder=self.experiment_folder
        )
        if not os.path.isfile(formatted_path):
            self.dump_json(formatted_path, data={})
        with open(formatted_path, "r") as infile:
            json_data = json.load(infile)
        return json_data

    def dump_json(self, unformatted_path: str, data: Dict):
        """Dump data to a JSON file.
        Args:
            unformatted_path (str): Unformatted path to JSON file.
            data (Dict): Data to dump.
        """
        formatted_path = self.format_path(unformatted_path)
        with open(formatted_path, "w") as outfile:
            json.dump(data, outfile, indent=4)

    def update_json(
        self, unformatted_path: str, strategy: str, epoch: int, iteration: int, data
    ):
        """Helper method to update json files.
        Args:
            unformatted_path (str): Unformatted path to JSON file.
            strategy (str): Current training strategy.
            epoch (int): Current epoch.
            iteration (int): Current iteration.
            data (Dict or List): Data to store for current strategy, epoch, and iteration.
        """
        json_data = self.load_json(unformatted_path)
        json_data[strategy] = json_data.get(strategy, {})
        json_data[strategy][str(epoch)] = json_data[strategy].get(
            str(epoch), {str(epoch): {}}
        )
        json_data[strategy][str(epoch)][str(iteration)] = data
        self.dump_json(unformatted_path, json_data)

    def update_accuracies_json(
        self, strategy: str, epoch: int, iteration: int, eval_stats
    ):
        """Update accuracies.json with iteration metrics"""
        self.update_json(AL_ACCURACIES_PATH, strategy, epoch, iteration, eval_stats)

    def update_selected_queries_json(
        self, strategy: str, epoch: int, iteration: int, queries
    ):
        """Update accuracies.json with iteration metrics"""
        query_dicts = ResultsManager.queries_to_dict(queries)
        self.update_json(
            AL_SELECTED_QUERIES_PATH, strategy, epoch, iteration, query_dicts
        )

    def write_log_selected_queries_json(self, strategy: str, queries):
        """Update accuracies.json with iteration metrics"""
        query_dicts = ResultsManager.queries_to_dict(queries)
        log_selected_queries_path = os.path.join(
            self.output_folder, "log_selected_queries.json"
        )
        data = {"strategy": strategy, "selected_queries": query_dicts}
        with open(log_selected_queries_path, "w") as outfile:
            json.dump(data, outfile, indent=4)
        logger.info("Selected Log Queries saved at: %s", log_selected_queries_path)

    @staticmethod
    def queries_to_dict(queries: List) -> List:
        """Convert a list of ProcessedQueries into a list dictionaries.
        Args:
            queries (List): List of ProcessedQuery objects
        Returns:
            query_dicts (List): List of queries represented as a dict with the keys
                "unannotated_text", "annotated_text", "domain", and "intent".
        """
        return [
            {
                "unannotated_text": query.query.text,
                "annotated_text": dump_query(query),
                "domain": query.domain,
                "intent": query.intent,
            }
            for query in queries
        ]
