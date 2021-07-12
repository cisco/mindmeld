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
This module plots the results from the Active Learning Pipeline.
"""

import os
import json
import logging
from collections import Counter
from typing import Dict, List
import numpy as np
from ..components._util import _is_module_available, _get_module_or_attr
from .classifiers import MindMeldALClassifier
from ..path import (
    AL_ACCURACIES_PATH,
    AL_SELECTED_QUERIES_PATH,
)


logger = logging.getLogger(__name__)

FIRST_EPOCH = "0"
FIRST_ITERATION = "0"


class MissingDataError(Exception):
    pass


class PlotManager:
    """Handles plotting. Plots supported:
    At the domain, intent and entity level:
        1. Plot Single Epoch (Compares Strategies)
        2. Plot Average Across Epochs (Compares Strategies)
        3. Plot All Epochs (Compares Epochs)
    At the domain and intent level:
        1. Plot Stacked Bar (Compares Selection Distributions across Iterations)
    """

    def __init__(
        self,
        experiment_dir_path: str,
        aggregate_statistic: str,
        class_level_statistic: str,
    ):
        """
        Args:
            experiment_dir_path (str): Path to the experiment directory.
            aggregate_statistic (str): Aggregate statistic to record.
                (Options: "accuracy", "f1_weighted", "f1_macro", "f1_micro".)
            class_level_statistic (str): Class_level statistic to record.
                (Options: "f_beta", "percision", "recall")
        """
        self.experiment_dir_path = experiment_dir_path
        self.aggregate_statistic = MindMeldALClassifier._validate_aggregate_statistic(
            aggregate_statistic
        )
        self.class_level_statistic = (
            MindMeldALClassifier._validate_class_level_statistic(class_level_statistic)
        )
        self.accuracies_data = self.get_accuracies_json_data()
        self.queries_data = self.get_queries_json_data()

        if not _is_module_available("matplotlib"):
            raise ModuleNotFoundError(
                "Library not found: 'matplotlib'. Run 'pip install mindmeld[active_learning]' to install."
            )

        self.plt = _get_module_or_attr("matplotlib.pyplot")

    # Get JSON Data
    def get_accuracies_json_data(self) -> Dict:
        """Loads accuracies.json from the experiment directory path.
        Returns:
            data (dict): Data loaded from accuracies.json.
        """
        accuracies_json_path = AL_ACCURACIES_PATH.format(
            experiment_folder=self.experiment_dir_path
        )
        with open(accuracies_json_path, "r") as infile:
            return json.load(infile)

    def get_queries_json_data(self) -> Dict:
        """Loads selected_queries.json from the experiment directory path.
        selected_queries.json stores the queries selected by active learning
        at each iteration.
        Returns:
            data (dict): Data loaded from selected_queries.json.
        """
        selected_queries_json_path = AL_SELECTED_QUERIES_PATH.format(
            experiment_folder=self.experiment_dir_path
        )
        with open(selected_queries_json_path, "r") as infile:
            return json.load(infile)

    def queries_json_data_has_data(self) -> Dict:
        """Checks whether queries.json is empty {}.
        Returns:
            has_data (bool): Whether queries.json has data.
        """
        return self.get_queries_json_data() != {}

    # Plotting Meta Functions
    def generate_plots(self):
        """Entry point for generating plots as per the specifications provided in the config."""
        logger.info("Starting to generate plots")
        plot_functions = [
            "plot_single_epoch",
            "plot_avg_across_epochs",
            "plot_all_epochs",
        ]
        for plot_function in plot_functions:
            logger.info("Plotting: %s", plot_function)
            plot_function = getattr(PlotManager, plot_function)
            self.plotting_wrapper(plot_function)
        if self.queries_json_data_has_data():
            logger.info("Plotting: plot_stacked_bar")
            self.plot_stacked_bar()

    def plotting_wrapper(
        self,
        function,
        plot_domain: bool = True,
        plot_intents: bool = True,
        plot_entities: bool = False,
    ):
        """Plotting wrapper functions for plots that use data from accuracies.json
        Args:
            function (generator): plotting function.
            plot_domain (bool): Whether to generate plots at the domain level.
            plot_intents (bool): Whether to generate plots at the intent level.
            plot_entities (bool): Whether to generate plots at the entity level.
        """
        self._check_first_epoch_and_iter_exist()
        if not plot_domain:
            return
        function(self, y_keys=["overall"])
        for domain in self.get_domain_list():
            function(self, y_keys=[domain, "overall"])
            if not plot_intents:
                continue
            for intent in self.get_intent_list(domain):
                function(
                    self,
                    y_keys=[domain, intent, "overall"],
                    use_aggregate_statistic=False,
                )
                if not plot_entities:
                    continue
                y_keys = [domain, intent, "entities", "overall"]
                function(self, y_keys=y_keys)
                for entity in self.get_entity_list(domain, intent):
                    y_keys = [domain, intent, "entities", entity]
                    function(self, y_keys=y_keys)

    # Helper Methods
    @staticmethod
    def get_nested(data_dict: Dict, selected_keys: List):
        """Filter data from a nested dictionary selecting from a set of keys.
        Args:
            data_dict (dict): Dictionary containing data to filter
            selected_keys (list): List of keys used to filter the given dictionary.

        Returns:
            data_dict (dict): Dictionary containing the filtered nested data.
        """
        for selected_key in selected_keys:
            data_dict = data_dict[selected_key]
        return data_dict

    @staticmethod
    def get_across_iterations(epoch_dict: Dict, selected_keys: List):
        """Filter data across all iterations in a single epoch as specified by a set of keys.
        Args:
            epoch_dict (dict): Dict containing accuracies across iterations for a single epoch.
            selected_keys (list): List of keys used to filter the given dictionary.
        Return:
            data (list): List of the selected data across iterations.
        """
        return [
            PlotManager.get_nested(epoch_dict[str(i)], selected_keys)
            for i in range(len(epoch_dict))
        ]

    @property
    def strategies(self) -> List:
        """
        Returns:
            strategies (list): List of selection strategies for the given experiment.
        """
        strategies = list(self.accuracies_data)
        if len(strategies) == 0:
            raise MissingDataError("Did not find data in accuracies.json.")
        return strategies

    def _check_first_epoch_and_iter_exist(self):
        """Check whether data for the first iteration in the first epoch exists.
        Data from the first epoch and iteration is used to determine the domains, intents,
        and entities included in training.
        Raises:
            MissingDataError: Throws an error if the anticipated data is not found.
        """
        first_strategy = self.strategies[0]
        if FIRST_EPOCH not in self.accuracies_data[first_strategy]:
            raise MissingDataError("Did not find data for the first epoch.")
        if FIRST_ITERATION not in self.accuracies_data[first_strategy][FIRST_EPOCH]:
            raise MissingDataError(
                "Did not find data for the first iteration in the first epoch."
            )

    def get_domain_list(self) -> List:
        """Method to get a list of domains included in training from the first epoch and iteration.
        Returns:
            domain_list (list): List of domains for the current experiment.
        """
        first_strategy = self.strategies[0]
        domain_list = list(
            self.accuracies_data[first_strategy][FIRST_EPOCH][FIRST_ITERATION][
                "accuracies"
            ].keys()
        )
        # The 'overall' score across domains is removed as it is not a domain
        domain_list.remove("overall")
        return domain_list

    def get_intent_list(self, domain: str) -> List:
        """Method to get a list of intents included in training from the first epoch and iteration.
        Args:
            domain (str): The domain to get retrieve intents for.
        Returns:
            intent_list (list): List of intent for a given domain in the current experiment.
        """
        first_strategy = self.strategies[0]
        intent_list = list(
            self.accuracies_data[first_strategy][FIRST_EPOCH][FIRST_ITERATION][
                "accuracies"
            ][domain].keys()
        )
        # The 'overall' score across intents is removed as it is not an intent
        intent_list.remove("overall")
        return intent_list

    def get_entity_list(self, domain: str, intent: str) -> List:
        """Method to get a list of entities included in training from the first epoch and iteration.
        Args:
            domain (str): The domain containing the intent to retreive entities from.
            intent (str): The intent to retreive entities from.
        Returns:
            entity_list (list): List of entities in the given intent.
        """
        first_strategy = self.strategies[0]
        entity_list = list(
            self.accuracies_data[first_strategy][FIRST_EPOCH][FIRST_ITERATION][
                "accuracies"
            ][domain][intent]["entities"].keys()
        )
        # The 'overall' score across entities is removed as it is not an entity
        entity_list.remove("overall")
        return entity_list

    def create_plot_dir(self, y_keys: List):
        """Creates folders to support the generated path if they do not already exist.
        Args:
            y_keys (list): Keys to access the data from a epoch to be used as y values for plotting.
        """
        path_list = [self.experiment_dir_path, "plots"] + y_keys
        os.makedirs(os.path.join(*path_list), exist_ok=True)

    def get_img_path(self, y_keys: List, file_name: str):
        """
        Args:
            y_keys (list): Keys to access the data from a epoch to be used as y values for plotting.
            file_name (str): Name of the file to save.
        Returns:
            img_path (str): Path of the image to be saved.
        """
        path_list = [self.experiment_dir_path, "plots"] + y_keys + [f"{file_name}.png"]
        return os.path.join(*path_list)

    # Plotting Functions
    def plot_single_epoch(
        self,
        y_keys: List,
        epoch: int = 0,
        display: bool = False,
        save: bool = True,
        use_aggregate_statistic: bool = True,
    ):
        """Plot accuracies across a single epoch for each strategy.
        Args:
            y_keys (list): Keys to access the data from a epoch to be used as y values for plotting.
            epoch (int): The epoch to plot.
            display (bool): Whether to show the plot.
            save (bool): Whether to save the plot.
            use_aggregate_statistic (bool): If True, the aggregate_statistic will be used as the
                label for the y_axis. If False, the class_level_statistic will be used.
        """
        self.create_plot_dir(y_keys)
        for strategy in self.strategies:
            epoch_dict = self.accuracies_data[strategy][str(epoch)]
            x_values = PlotManager.get_across_iterations(epoch_dict, ["num_sampled"])
            y_values = PlotManager.get_across_iterations(
                epoch_dict, ["accuracies"] + y_keys
            )
            self.plt.plot(x_values, y_values)

        self.plt.xlabel("Number of selected queries")
        y_label = (
            self.aggregate_statistic
            if use_aggregate_statistic
            else self.class_level_statistic
        )
        self.plt.ylabel(y_label.capitalize())
        title = f"Epoch_{epoch}_Results_({'-'.join(y_keys)})"
        self.plt.title(title)
        self.plt.legend(self.strategies, loc="lower right")
        self.plt.grid()
        self.plt.tight_layout()
        fig = self.plt.gcf()
        if display:
            self.plt.show()
        if save:
            fig.savefig(self.get_img_path(y_keys, title))
            self.plt.clf()

    def plot_avg_across_epochs(
        self,
        y_keys: List,
        display: bool = False,
        save: bool = True,
        use_aggregate_statistic: bool = True,
    ):
        """Plot average accuracies across all epochs for each strategy.
        Args:
            y_keys (list): Keys to access the data from a epoch to be used as y values for plotting.
            display (bool): Whether to show the plot.
            save (bool): Whether to save the plot.
            use_aggregate_statistic (bool): If True, the aggregate_statistic will be used as the
                label for the y_axis. If False, the class_level_statistic will be used.
        """
        self.create_plot_dir(y_keys)
        for strategy in self.strategies:
            n_epochs = len(self.accuracies_data[strategy])
            all_y_values = []
            for epoch in range(n_epochs):
                epoch_dict = self.accuracies_data[strategy][str(epoch)]
                x_values = PlotManager.get_across_iterations(
                    epoch_dict, ["num_sampled"]
                )
                y_values = PlotManager.get_across_iterations(
                    epoch_dict, ["accuracies"] + y_keys
                )
                all_y_values.append(y_values)
            all_y_values = np.array(all_y_values)
            y_avg_values = all_y_values.mean(axis=0)
            self.plt.plot(x_values, y_avg_values)

        self.plt.xlabel("Number of selected queries")
        y_label = (
            self.aggregate_statistic
            if use_aggregate_statistic
            else self.class_level_statistic
        )
        self.plt.ylabel(y_label.capitalize())
        title = f"Avg_Across_Epochs_({'-'.join(y_keys)})"
        self.plt.title(title)
        self.plt.legend(self.strategies, loc="lower right")
        self.plt.grid()
        self.plt.tight_layout()
        fig = self.plt.gcf()
        if display:
            self.plt.show()
        if save:
            fig.savefig(self.get_img_path(y_keys, title))
            self.plt.clf()

    def plot_all_epochs(
        self,
        y_keys: List,
        display: bool = False,
        save: bool = True,
        use_aggregate_statistic: bool = True,
    ):
        """Plot all epochs. Creates a plot for each strategy.
        Args:
            y_keys (list): Keys to access the data from a epoch to be used as y values for plotting.
            display (bool): Whether to show the plot.
            save (bool): Whether to save the plot.
            use_aggregate_statistic (bool): If True, the aggregate_statistic will be used as the
                label for the y_axis. If False, the class_level_statistic will be used.
        """
        self.create_plot_dir(y_keys)
        for strategy in self.strategies:
            n_epochs = len(self.accuracies_data[strategy].keys())
            all_y_values = []
            for epoch in range(n_epochs):
                epoch_dict = self.accuracies_data[strategy][str(epoch)]
                x_values = PlotManager.get_across_iterations(
                    epoch_dict, ["num_sampled"]
                )
                y_values = PlotManager.get_across_iterations(
                    epoch_dict, ["accuracies"] + y_keys
                )
                self.plt.plot(x_values, y_values)
                all_y_values.append(y_values)
            max_len = max([len(i) for i in all_y_values])
            for y_values in all_y_values:
                y_values = np.pad(
                    y_values,
                    (0, max_len - len(y_values)),
                    "constant",
                    constant_values=(0, y_values[-1]),
                )
            y_avg_values = np.array(all_y_values).mean(axis=0)
            self.plt.plot(x_values, y_avg_values)

            self.plt.xlabel("Number of selected queries")
            y_label = (
                self.aggregate_statistic
                if use_aggregate_statistic
                else self.class_level_statistic
            )
            self.plt.ylabel(y_label.capitalize())
            title = f"{strategy}_All_Epochs_({'-'.join(y_keys)})"
            self.plt.title(title)
            self.plt.legend(
                ["epoch " + str(epoch) for epoch in range(n_epochs)] + ["avg"],
                loc="lower right",
            )
            self.plt.grid()
            self.plt.tight_layout()
            fig = self.plt.gcf()
            if display:
                self.plt.show()
            if save:
                fig.savefig(self.get_img_path(y_keys, title))
                self.plt.clf()

    @staticmethod
    def get_unique_labels(all_counters: List) -> List:
        """
        Args:
            all_counters (list): List of Counters, each counter represents a single iteration.
        Returns:
            unique_labels (list): A List of unique and sorted keys across all_counters.
        """
        unique_labels = []
        for counter in all_counters:
            unique_labels.extend(list(counter.keys()))
        unique_labels = list(set(unique_labels))
        unique_labels.sort()
        return unique_labels

    @staticmethod
    def get_label_set_counter(all_counters: List, unique_labels: List) -> Dict:
        """
        Args:
            all_counters (list): List of Counters, each counter represents a single iteration.
            unique_labels (list): A List of unique and sorted keys across all_counters.
        Returns:
            label_set_counter (dict): Each unique label is mapped to a list, the value at each
                index in the list corresponds to the count of the label in that iteration.
        """
        label_set_counter = {k: np.zeros(len(all_counters)) for k in unique_labels}
        for iteration, counter in enumerate(all_counters):
            for label in counter:
                label_set_counter[label][iteration] += counter[label]
        return label_set_counter

    # pylint: disable=W0613
    def plot_stacked_bar(
        self,
        epoch: int = 0,
        plot_domains: bool = True,
        plot_intents: bool = True,
        **kwargs,
    ):
        """Plots a stacked bar graph of selection distributions across iterations for an epoch.
        Args:
            epoch (int): The epoch to plot.
            plot_domain (bool): Whether to generate plots at the domain level.
            plot_intents (bool): Whether to generate plots at the intent level.
        """
        for strategy in self.queries_data:
            epoch_dict = self.queries_data[strategy][str(epoch)]
            domain_counters, intent_counters = [], []
            for iteration in epoch_dict:
                query_list = epoch_dict[iteration]
                if plot_domains:
                    domains = [q["domain"] for q in query_list]
                    domain_counters.append(Counter(domains))
                if plot_intents:
                    intents = [f"{q['intent']}|{q['domain']}" for q in query_list]
                    intent_counters.append(Counter(intents))

            for level, all_counters in zip(
                ["domain", "intent"], [domain_counters, intent_counters]
            ):
                if all_counters:
                    unique_labels = PlotManager.get_unique_labels(all_counters)
                    label_set_counter = PlotManager.get_label_set_counter(
                        all_counters, unique_labels
                    )
                    self._plot_stacked_bar(
                        num_iters=len(all_counters),
                        label_set_counter=label_set_counter,
                        strategy=strategy,
                        intent_level=(level == "intent"),
                    )

    def _plot_stacked_bar(
        self,
        num_iters: int,
        label_set_counter: Dict,
        strategy: str,
        intent_level: bool,
        display: bool = False,
        save: bool = True,
    ):
        """Helper function to plot a stacked bar graph.
        Args:
            num_iters (int): Number of iterations in the given epoch.
            label_set_counter (dict): Each unique label is mapped to a list, the value at the
                index in the list corresponds to the count of the label in that iteration.
            strategy (str): Selection strategy.
            intent_level (bool): Whether the plot is for intent or domain level distributions.
            display (bool): Whether to show the plot.
            save (bool): Whether to save the plot.
        """
        fig, ax = self.plt.subplots()
        total_bottom = np.zeros(num_iters)
        iterations = [str(i) for i in range(num_iters)]
        for label in label_set_counter:
            ax.bar(
                iterations, label_set_counter[label], bottom=total_bottom, label=label
            )
            total_bottom = np.add(total_bottom, label_set_counter[label])

        ax.set_ylabel("Counts")
        ax.set_xlabel("Iteration")
        level = "Intent" if intent_level else "Domain"
        title = f"{strategy}_{level}_Selection_Distribution_Per_Iteration"
        ax.set_title(title)
        self.plt.tight_layout()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 2, box.height])
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        if display:
            self.plt.show()
        if save:
            img_dir_path = os.path.join(*[self.experiment_dir_path, "plots", "overall"])
            os.makedirs(img_dir_path, exist_ok=True)
            img_path = os.path.join(img_dir_path, f"{title}.png")
            fig.savefig(img_path, bbox_inches="tight")
            self.plt.clf()
