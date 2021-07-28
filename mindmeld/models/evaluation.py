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

"""This module contains base classes for models defined in the models subpackage."""
import logging
from collections import namedtuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support as score

from .helpers import (
    ENTITIES_LABEL_TYPE,
    entity_seqs_equal,
    get_label_encoder,
)
from .taggers.taggers import (
    BoundaryCounts,
    get_boundary_counts,
)

logger = logging.getLogger(__name__)


class EvaluatedExample(
    namedtuple(
        "EvaluatedExample", ["example", "expected", "predicted", "probas", "label_type"]
    )
):
    """Represents the evaluation of a single example

    Attributes:
        example: The example being evaluated
        expected: The expected label for the example
        predicted: The predicted label for the example
        proba (dict): Maps labels to their predicted probabilities
        label_type (str): One of CLASS_LABEL_TYPE or ENTITIES_LABEL_TYPE
    """

    @property
    def is_correct(self):
        # For entities compare just the type, span and text for each entity.
        if self.label_type == ENTITIES_LABEL_TYPE:
            return entity_seqs_equal(self.expected, self.predicted)
        # For other label_types compare the full objects
        else:
            return self.expected == self.predicted


class RawResults:
    """Represents the raw results of a set of evaluated examples. Useful for generating
    stats and graphs.

    Attributes:
        predicted (list): A list of predictions. For sequences this is a list of lists, and for
                          standard classifieris this is a 1d array. All classes are in their numeric
                          representations for ease of use with evaluation libraries and graphing.
        expected (list): Same as predicted but contains the true or gold values.
        text_labels (list): A list of all the text label values, the index of the text label in
                             this array is the numeric label
        predicted_flat (list): (Optional): For sequence models this is a flattened list of all
                                predicted tags (1d array)
        expected_flat (list): (Optional): For sequence models this is a flattened list of all gold
                              tags
    """

    def __init__(
        self, predicted, expected, text_labels, predicted_flat=None, expected_flat=None
    ):
        self.predicted = predicted
        self.expected = expected
        self.text_labels = text_labels
        self.predicted_flat = predicted_flat
        self.expected_flat = expected_flat


class ModelEvaluation(namedtuple("ModelEvaluation", ["config", "results"])):
    """Represents the evaluation of a model at a specific configuration
    using a collection of examples and labels.

    Attributes:
        config (ModelConfig): The model config used during evaluation.
        results (list of EvaluatedExample): A list of the evaluated examples.
    """

    def __init__(self, config, results):
        del results
        self.label_encoder = get_label_encoder(config)

    def get_accuracy(self):
        """The accuracy represents the share of examples whose predicted labels
        exactly matched their expected labels.

        Returns:
            float: The accuracy of the model.
        """
        num_examples = len(self.results)
        num_correct = len([e for e in self.results if e.is_correct])
        return float(num_correct) / float(num_examples)

    def __repr__(self):
        num_examples = len(self.results)
        num_correct = len(list(self.correct_results()))
        accuracy = self.get_accuracy()
        msg = "<{} score: {:.2%}, {} of {} example{} correct>"
        return msg.format(
            self.__class__.__name__,
            accuracy,
            num_correct,
            num_examples,
            "" if num_examples == 1 else "s",
        )

    def correct_results(self):
        """
        Returns:
            iterable: Collection of the examples which were correct
        """
        for result in self.results:
            if result.is_correct:
                yield result

    def incorrect_results(self):
        """
        Returns:
            iterable: Collection of the examples which were incorrect
        """
        for result in self.results:
            if not result.is_correct:
                yield result

    def get_stats(self):
        """
        Returns a structured stats object for evaluation.

        Returns:
            dict: Structured dict containing evaluation statistics. Contains precision, \
                  recall, f scores, support, etc.
        """
        raise NotImplementedError

    def print_stats(self):
        """
        Prints a useful stats table for evaluation.

        Returns:
            dict: Structured dict containing evaluation statistics. Contains precision, \
                  recall, f scores, support, etc.
        """
        raise NotImplementedError

    def raw_results(self):
        """
        Exposes raw vectors of expected and predicted for data scientists to use for any additional
        evaluation metrics or to generate graphs of their choice.

        Returns:
            (tuple): tuple containing:

                * NamedTuple: RawResults named tuple containing
                * expected: vector of predicted classes (numeric value)
                * predicted: vector of gold classes (numeric value)
                * text_labels: a list of all the text label values, the index of the text label in
                * this array is the numeric label
        """
        raise NotImplementedError

    @staticmethod
    def _update_raw_result(label, text_labels, vec):
        """
        Helper method for updating the text to numeric label vectors

        Returns:
            (tuple): tuple containing:

                * text_labels: The updated text_labels array
                * vec: The updated label vector with the given label appended
        """
        if label not in text_labels:
            text_labels.append(label)
        vec.append(text_labels.index(label))
        return text_labels, vec

    def _get_common_stats(self, raw_expected, raw_predicted, text_labels):
        """
        Prints a useful stats table and returns a structured stats object for evaluation.

        Returns:
            dict: Structured dict containing evaluation statistics. Contains precision, \
                  recall, f scores, support, etc.
        """
        labels = range(len(text_labels))

        confusion_stats = self._get_confusion_matrix_and_counts(
            y_true=raw_expected, y_pred=raw_predicted
        )
        stats_overall = self._get_overall_stats(
            y_true=raw_expected, y_pred=raw_predicted, labels=labels
        )
        counts_overall = confusion_stats["counts_overall"]
        stats_overall["tp"] = counts_overall.tp
        stats_overall["tn"] = counts_overall.tn
        stats_overall["fp"] = counts_overall.fp
        stats_overall["fn"] = counts_overall.fn

        class_stats = self._get_class_stats(
            y_true=raw_expected, y_pred=raw_predicted, labels=labels
        )
        counts_by_class = confusion_stats["counts_by_class"]

        class_stats["tp"] = counts_by_class.tp
        class_stats["tn"] = counts_by_class.tn
        class_stats["fp"] = counts_by_class.fp
        class_stats["fn"] = counts_by_class.fn

        return {
            "stats_overall": stats_overall,
            "class_labels": text_labels,
            "class_stats": class_stats,
            "confusion_matrix": confusion_stats["confusion_matrix"],
        }

    @staticmethod
    def _get_class_stats(y_true, y_pred, labels):
        """
        Method for getting some basic statistics by class.

        Returns:
            dict: A structured dictionary containing precision, recall, f_beta, and support \
                  vectors (1 x number of classes)
        """
        precision, recall, f_beta, support = score(
            y_true=y_true, y_pred=y_pred, labels=labels
        )

        stats = {
            "precision": precision,
            "recall": recall,
            "f_beta": f_beta,
            "support": support,
        }
        return stats

    @staticmethod
    def _get_overall_stats(y_true, y_pred, labels):
        """
        Method for getting some overall statistics.

        Returns:
            dict: A structured dictionary containing scalar values for f1 scores and overall \
                  accuracy.
        """
        f1_weighted = f1_score(
            y_true=y_true, y_pred=y_pred, labels=labels, average="weighted"
        )
        f1_macro = f1_score(
            y_true=y_true, y_pred=y_pred, labels=labels, average="macro"
        )
        f1_micro = f1_score(
            y_true=y_true, y_pred=y_pred, labels=labels, average="micro"
        )
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)

        stats_overall = {
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "accuracy": accuracy,
        }
        return stats_overall

    @staticmethod
    def _get_confusion_matrix_and_counts(y_true, y_pred):
        """
        Generates the confusion matrix where each element Cij is the number of observations known to
        be in group i predicted to be in group j

        Returns:
            dict: Contains 2d array of the confusion matrix, and an array of tp, tn, fp, fn values
        """
        confusion_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
        tp_arr, tn_arr, fp_arr, fn_arr = [], [], [], []

        num_classes = len(confusion_mat)
        for class_index in range(num_classes):
            # tp is C_classindex, classindex
            tp = confusion_mat[class_index][class_index]
            tp_arr.append(tp)

            # tn is the sum of Cij where i or j are not class_index
            mask = np.ones((num_classes, num_classes))
            mask[:, class_index] = 0
            mask[class_index, :] = 0
            tn = np.sum(mask * confusion_mat)
            tn_arr.append(tn)

            # fp is the sum of Cij where j is class_index but i is not
            mask = np.zeros((num_classes, num_classes))
            mask[:, class_index] = 1
            mask[class_index, class_index] = 0
            fp = np.sum(mask * confusion_mat)
            fp_arr.append(fp)

            # fn is the sum of Cij where i is class_index but j is not
            mask = np.zeros((num_classes, num_classes))
            mask[class_index, :] = 1
            mask[class_index, class_index] = 0
            fn = np.sum(mask * confusion_mat)
            fn_arr.append(fn)

        Counts = namedtuple("Counts", ["tp", "tn", "fp", "fn"])
        return {
            "confusion_matrix": confusion_mat,
            "counts_by_class": Counts(tp_arr, tn_arr, fp_arr, fn_arr),
            "counts_overall": Counts(
                sum(tp_arr), sum(tn_arr), sum(fp_arr), sum(fn_arr)
            ),
        }

    def _print_class_stats_table(self, stats, text_labels, title="Statistics by class"):
        """
        Helper for printing a human readable table for class statistics

        Returns:
            None
        """
        title_format = "{:>20}" + "{:>12}" * (len(stats))
        common_stats = [
            "f_beta",
            "precision",
            "recall",
            "support",
            "tp",
            "tn",
            "fp",
            "fn",
        ]
        stat_row_format = (
            "{:>20}"
            + "{:>12.3f}" * 3
            + "{:>12.0f}" * 5
            + "{:>12.3f}" * (len(stats) - len(common_stats))
        )
        table_titles = common_stats + [
            stat for stat in stats.keys() if stat not in common_stats
        ]
        print(title + ": \n")
        print(title_format.format("class", *table_titles))
        for label_index, label in enumerate(text_labels):
            row = []
            for stat in table_titles:
                row.append(stats[stat][label_index])
            print(stat_row_format.format(self._truncate_label(label, 18), *row))
        print("\n\n")

    def _print_class_matrix(self, matrix, text_labels):
        """
        Helper for printing a human readable class by class table for displaying
        a confusion matrix

        Returns:
            None
        """
        # Doesn't print if there isn't enough space to display the full matrix.
        if len(text_labels) > 10:
            print(
                "Not printing confusion matrix since it is too large. The full matrix is still"
                " included in the dictionary returned from get_stats()."
            )
            return
        labels = range(len(text_labels))
        title_format = "{:>15}" * (len(labels) + 1)
        stat_row_format = "{:>15}" * (len(labels) + 1)
        table_titles = [
            self._truncate_label(text_labels[label], 10) for label in labels
        ]
        print("Confusion matrix: \n")
        print(title_format.format("", *table_titles))
        for label_index, label in enumerate(text_labels):
            print(
                stat_row_format.format(
                    self._truncate_label(label, 10), *matrix[label_index]
                )
            )
        print("\n\n")

    @staticmethod
    def _print_overall_stats_table(stats_overall, title="Overall statistics"):
        """
        Helper for printing a human readable table for overall statistics

        Returns:
            None
        """
        title_format = "{:>12}" * (len(stats_overall))
        common_stats = ["accuracy", "f1_weighted", "tp", "tn", "fp", "fn"]
        stat_row_format = (
            "{:>12.3f}" * 2
            + "{:>12.0f}" * 4
            + "{:>12.3f}" * (len(stats_overall) - len(common_stats))
        )
        table_titles = common_stats + [
            stat for stat in stats_overall.keys() if stat not in common_stats
        ]
        print(title + ": \n")
        print(title_format.format(*table_titles))
        row = []
        for stat in table_titles:
            row.append(stats_overall[stat])
        print(stat_row_format.format(*row))
        print("\n\n")

    @staticmethod
    def _truncate_label(label, max_len):
        return (label[:max_len] + "..") if len(label) > max_len else label


class StandardModelEvaluation(ModelEvaluation):
    def raw_results(self):
        """Returns the raw results of the model evaluation"""
        text_labels = []
        predicted, expected = [], []

        for result in self.results:
            text_labels, predicted = self._update_raw_result(
                result.predicted, text_labels, predicted
            )
            text_labels, expected = self._update_raw_result(
                result.expected, text_labels, expected
            )

        return RawResults(
            predicted=predicted, expected=expected, text_labels=text_labels
        )

    def get_stats(self):
        """Prints model evaluation stats in a table to stdout"""
        raw_results = self.raw_results()
        stats = self._get_common_stats(
            raw_results.expected, raw_results.predicted, raw_results.text_labels
        )
        # Note can add any stats specific to the standard model to any of the tables here

        return stats

    def print_stats(self):
        """Prints model evaluation stats to stdout"""
        raw_results = self.raw_results()
        stats = self.get_stats()

        self._print_overall_stats_table(stats["stats_overall"])
        self._print_class_stats_table(stats["class_stats"], raw_results.text_labels)
        self._print_class_matrix(stats["confusion_matrix"], raw_results.text_labels)


class SequenceModelEvaluation(ModelEvaluation):
    def __init__(self, config, results):
        self._tag_scheme = config.model_settings.get("tag_scheme", "IOB").upper()
        super().__init__(config, results)

    def raw_results(self):
        """Returns the raw results of the model evaluation"""
        text_labels = []
        predicted, expected = [], []
        predicted_flat, expected_flat = [], []

        for result in self.results:
            raw_predicted = self.label_encoder.encode(
                [result.predicted], examples=[result.example]
            )[0]
            raw_expected = self.label_encoder.encode(
                [result.expected], examples=[result.example]
            )[0]

            vec = []
            for entity in raw_predicted:
                text_labels, vec = self._update_raw_result(entity, text_labels, vec)
            predicted.append(vec)
            predicted_flat.extend(vec)
            vec = []
            for entity in raw_expected:
                text_labels, vec = self._update_raw_result(entity, text_labels, vec)
            expected.append(vec)
            expected_flat.extend(vec)
        return RawResults(
            predicted=predicted,
            expected=expected,
            text_labels=text_labels,
            predicted_flat=predicted_flat,
            expected_flat=expected_flat,
        )

    def _get_sequence_stats(self):
        """
        TODO: Generate additional sequence level stats
        """
        sequence_accuracy = self.get_accuracy()
        return {"sequence_accuracy": sequence_accuracy}

    @staticmethod
    def _print_sequence_stats_table(sequence_stats):
        """
        Helper for printing a human readable table for sequence statistics

        Returns:
            None
        """
        title_format = "{:>18}" * (len(sequence_stats))
        table_titles = ["sequence_accuracy"]
        stat_row_format = "{:>18.3f}" * (len(sequence_stats))
        print("Sequence-level statistics: \n")
        print(title_format.format(*table_titles))
        row = []
        for stat in table_titles:
            row.append(sequence_stats[stat])
        print(stat_row_format.format(*row))
        print("\n\n")

    def get_stats(self):
        """Prints model evaluation stats in a table to stdout"""
        raw_results = self.raw_results()
        stats = self._get_common_stats(
            raw_results.expected_flat,
            raw_results.predicted_flat,
            raw_results.text_labels,
        )
        sequence_stats = self._get_sequence_stats()
        stats["sequence_stats"] = sequence_stats

        # Note: can add any stats specific to the sequence model to any of the tables here
        return stats

    def print_stats(self):
        """Prints model evaluation stats to stdout"""
        raw_results = self.raw_results()
        stats = self.get_stats()

        self._print_overall_stats_table(
            stats["stats_overall"], "Overall tag-level statistics"
        )
        self._print_class_stats_table(
            stats["class_stats"],
            raw_results.text_labels,
            "Tag-level statistics by class",
        )
        self._print_class_matrix(stats["confusion_matrix"], raw_results.text_labels)
        self._print_sequence_stats_table(stats["sequence_stats"])


class EntityModelEvaluation(SequenceModelEvaluation):
    """Generates some statistics specific to entity recognition"""

    def _get_entity_boundary_stats(self):
        """
        Calculate le, be, lbe, tp, tn, fp, fn as defined here:
        https://nlpers.blogspot.com/2006/08/doing-named-entity-recognition-dont.html
        """
        boundary_counts = BoundaryCounts()
        raw_results = self.raw_results()
        for expected_sequence, predicted_sequence in zip(
            raw_results.expected, raw_results.predicted
        ):
            expected_seq_labels = [
                raw_results.text_labels[i] for i in expected_sequence
            ]
            predicted_seq_labels = [
                raw_results.text_labels[i] for i in predicted_sequence
            ]
            boundary_counts = get_boundary_counts(
                expected_seq_labels, predicted_seq_labels, boundary_counts
            )
        return boundary_counts.to_dict()

    @staticmethod
    def _print_boundary_stats(boundary_counts):
        title_format = "{:>12}" * (len(boundary_counts))
        table_titles = boundary_counts.keys()
        stat_row_format = "{:>12}" * (len(boundary_counts))
        print("Segment-level statistics: \n")
        print(title_format.format(*table_titles))
        row = []
        for stat in table_titles:
            row.append(boundary_counts[stat])
        print(stat_row_format.format(*row))
        print("\n\n")

    def get_stats(self):
        stats = super().get_stats()
        if self._tag_scheme == "IOB":
            boundary_stats = self._get_entity_boundary_stats()
            stats["boundary_stats"] = boundary_stats
        return stats

    def print_stats(self):
        raw_results = self.raw_results()
        stats = self.get_stats()

        self._print_overall_stats_table(
            stats["stats_overall"], "Overall tag-level statistics"
        )
        self._print_class_stats_table(
            stats["class_stats"],
            raw_results.text_labels,
            "Tag-level statistics by class",
        )
        self._print_class_matrix(stats["confusion_matrix"], raw_results.text_labels)
        if self._tag_scheme == "IOB":
            self._print_boundary_stats(stats["boundary_stats"])
        self._print_sequence_stats_table(stats["sequence_stats"])
