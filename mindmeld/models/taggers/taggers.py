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
This module contains all code required to perform sequence tagging.
"""
import copy
import logging

from ...core import (
    TEXT_FORM_NORMALIZED,
    TEXT_FORM_RAW,
    QueryEntity,
    Span,
    _sort_by_lowest_time_grain,
)
from ...markup import MarkupError
from ...system_entity_recognizer import SystemEntityResolutionError
from ..helpers import ENABLE_STEMMING, get_feature_extractor

logger = logging.getLogger(__name__)


START_TAG = "START"
B_TAG = "B"
I_TAG = "I"
O_TAG = "O"
E_TAG = "E"
S_TAG = "S"
# End tag is used by the evaluation algorithm to mark the end of query. This
# differs from the E tag which is used to denote the end of an entity in IOBES tagging.
END_TAG = "END"


class Tagger:
    """A class for all sequence tagger models implemented in house.
    It is importent to follow this interface exactly when implementing a new model so that your
    model is configured and trained as expected in the MindMeld pipeline. Note that this follows
    the sklearn estimator interface so that GridSearchCV can be used on our sequence models.
    """

    def __init__(self, **parameters):
        """To be consistent with the sklearn interface, __init__ and set_params should have the same
        effect. We do all parameter setting and validation in set_params which is called from here.

        Args:
            **parameters: Arbitrary keyword arguments. The keys are model parameter names and the
                          values are what they should be set to
        Returns:
            self
        """
        self.set_params(**parameters)

    def __getstate__(self):
        """Returns the information needed pickle an instance of this class. By default, pickling removes
        attributes with names starting with underscores.
        """
        attributes = self.__dict__.copy()
        return attributes

    def fit(self, X, y):
        """Trains the model. X and y are the format of what is returned by extract_features. There is no
        restriction on their type or content. X should be the fully processed data with extracted
        features that are ready to be used to train the model. y should be a list of classes as
        encoded by the label_encoder

        Args:
            X (list): Generally a list of feature vectors, one for each training example
            y (list): A list of classification labels (encoded by the label_encoder, NOT MindMeld
                      entity objects)
        Returns:
            self
        """
        raise NotImplementedError

    def predict(self, X, dynamic_resource=None):
        """Predicts the labels from a feature matrix X. Again X is the format of what is returned by
        extract_features.

        Args:
            X (list): A list of feature vectors, one for each example
        Returns:
            (list of classification labels): a list of predicted labels (in an encoded format)
        """
        raise NotImplementedError

    def get_params(self, deep=True):
        """Gets a dictionary of all of the current model parameters and their values

        Args:
            deep (bool): Not used, needed for sklearn compatibility
        Returns:
            (dict): A dictionary of the model parameter names as keys and their set values
        """
        raise NotImplementedError

    def set_params(self, **parameters):
        """Sets the model parameters. Defaults should be set for all parameters such that a model
        is initialized with reasonable default parameters if none are explicitly passed in.

        Args:
            **parameters: Arbitrary keyword arguments. The keys are model parameter names and the
                          values are what they should be set to
        Returns:
            self
        """
        raise NotImplementedError

    def setup_model(self, config):
        """"Not implemented."""
        raise NotImplementedError

    def extract_features(self, examples, config, resources):
        """Extracts all features from a list of MindMeld examples. Processes the data and returns the
        features in the format that is expected as an input to fit(). Note that the MindMeld config
        and resources are passed in each time to make the underlying model implementation stateless.

        Args:
            examples (list of mindmeld.core.Query): A list of queries to extract features for
            config (ModelConfig): The ModelConfig which may contain information used for feature
                                  extraction
            resources (dict): Resources which may be used for this model's feature extraction

        Returns:
            (tuple): tuple containing:

                * (list of feature vectors): X
                * (list of labels): y
                * (list of groups): A list of groups to be used for splitting with \
                    sklearn GridSearchCV
        """
        raise NotImplementedError

    def extract_and_predict(self, examples, config, resources):
        """Does both feature extraction and prediction. Often necessary for sequence models when the
        prediction of the previous example is used as a feature for the next example. If this is
        not the case, extract is simply called before predict here. Note that the MindMeld config
        and resources are passed in each time to make the underlying model implementation stateless.

        Args:
            examples (list of mindmeld.core.Query): A list of queries to extract features for and
                                                       predict
            config (ModelConfig): The ModelConfig which may contain information used for feature
                                  extraction
            resources (dict): Resources which may be used for this model's feature extraction

        Returns:
            (list of classification labels): A list of predicted labels (in encoded format)
        """
        X, _, _ = self.extract_features(examples, config, resources)
        y = self.predict(X)
        return y

    def predict_proba(self, examples, config, resources):
        """
        Args:
            examples (list of mindmeld.core.Query): A list of queries to extract features for and
                                                       predict
            config (ModelConfig): The ModelConfig which may contain information used for feature
                                  extraction
            resources (dict): Resources which may be used for this model's feature extraction

        Returns:
            (list of lists): A list of predicted labels (in encoded format) and confidence scores
        """
        X, _, _ = self.extract_features(examples, config, resources)
        return self._predict_proba(X)

    @staticmethod
    def _predict_proba(X):
        del X
        pass

    @staticmethod
    def dump(model_path, config):
        """
        Since traditional SKLearn models are easily serializable, we can
        use JobLib to serialize them. So we alter the context object to make
        this explicit.

        Args:
            model_path (str): The path to dump the model to

        Returns:
            config (dict): The altered config object
        """
        del model_path
        config["serializable"] = True
        return config

    @staticmethod
    def unload():
        pass

    @staticmethod
    def load(model_path):
        """
        Load the model state to memory. This is a no-op since we do not
        have to do anything special to load default serializable models
        for SKLearn.

        Args:
            model_path (str): The path to dump the model to
        """
        del model_path
        pass


def get_tags_from_entities(query, entities, scheme="IOB"):
    """Get joint app and system IOB tags from a query's entities.

    Args:
        query (Query): A query instance.
        entities (List of QueryEntity): A list of queries found in the query

    Returns:
        (list of str): The tags for each token in the query. A tag has four \
            parts separated by '|'. The first two are the IOB status for \
            app entities followed by the type of app entity or \
            '' if the IOB status is 'O'. The last two are like the first two, \
            but for system entities.
    """
    try:
        iobs, types = _get_tags_from_entities(query, entities, scheme)
    except IndexError as e:
        raise MarkupError("Invalid entities {} in '{}'".format(entities, query)) from e
    tags = ["|".join(args) for args in zip(iobs, types)]
    return tags


def _get_tags_from_entities(query, entities, scheme="IOB"):
    normalized_tokens = query.normalized_tokens
    iobs = [O_TAG for _ in normalized_tokens]
    types = ["" for _ in normalized_tokens]

    # tag I and type for all tag schemes
    for entity in entities:

        for i in entity.normalized_token_span:
            iobs[i] = I_TAG
            types[i] = entity.entity.type

    # Replace I with B/E/S when appropriate
    if scheme in ("IOB", "IOBES"):
        for entity in entities:
            iobs[entity.normalized_token_span.start] = B_TAG
    if scheme == "IOBES":
        for entity in entities:
            if len(entity.normalized_token_span) == 1:
                iobs[entity.normalized_token_span.end] = S_TAG
            else:
                iobs[entity.normalized_token_span.end] = E_TAG

    return iobs, types


def get_entities_from_tags(query, tags, system_entity_recognizer):
    """From a set of joint IOB tags, parse the app and system entities.

    This performs the reverse operation of get_tags_from_entities.

    Args:
        query (Query): Any query instance.
        tags (list of str): Joint app and system tags, like those
            created by get_tags_from_entities.
        system_entity_recognizer (SystemEntityRecognizer)

    Returns:
        (list of QueryEntity) The tuple containing the list of entities.
    """
    normalized_tokens = query.normalized_tokens

    entities = []

    def _is_system_entity(entity_type):
        if entity_type.split("_")[0] == "sys":
            return True
        return False

    def _append_entity(token_start, entity_type, tokens):
        prefix = " ".join(normalized_tokens[:token_start])
        # If there is a prefix, we have to add one for the whitespace
        start = len(prefix) + 1 if len(prefix) else 0
        end = start - 1 + len(" ".join(tokens))

        norm_span = Span(start, end)
        entity = QueryEntity.from_query(
            query, normalized_span=norm_span, entity_type=entity_type
        )
        entities.append(entity)
        logger.debug("Appended %s.", entity)

    def _append_system_entity(token_start, token_end, entity_type):
        msg = "Looking for '%s' between %s and %s."
        logger.debug(msg, entity_type, token_start, token_end)
        prefix = " ".join(normalized_tokens[:token_start])
        # If there is a prefix, we have to add one for the whitespace
        start = len(prefix) + 1 if len(prefix) else 0
        end = start - 1 + len(" ".join(normalized_tokens[token_start:token_end]))

        norm_span = Span(start, end)

        span = query.transform_span(norm_span, TEXT_FORM_NORMALIZED, TEXT_FORM_RAW)

        try:
            entity = system_entity_recognizer.resolve_system_entity(
                query, entity_type, span
            )
            entities.append(entity)
            logger.debug("Appended system entity %s.", entity)
        except SystemEntityResolutionError:
            msg = "Found no matching system entity {}-{}, {!r}".format(
                token_start, token_end, entity_type
            )
            logger.debug(msg)

    entity_tokens = []
    entity_start = None
    prev_ent_type = ""

    for tag_idx, tag in enumerate(tags):
        iob, ent_type = tag.split("|")

        # Close entity and reset if the tag indicates a new entity
        if entity_start is not None and (
            iob in (O_TAG, B_TAG, S_TAG) or ent_type != prev_ent_type
        ):
            logger.debug("Entity closed at prev")
            if _is_system_entity(prev_ent_type):
                _append_system_entity(entity_start, tag_idx, prev_ent_type)
            else:
                _append_entity(entity_start, prev_ent_type, entity_tokens)
            entity_start = None
            prev_ent_type = ""
            entity_tokens = []

        # Check if an entity has started
        if iob in (B_TAG, S_TAG) or ent_type not in ("", prev_ent_type):
            entity_start = tag_idx
            if _is_system_entity(ent_type):
                # During predict time, we construct sys_candidates for the input query.
                # These candidates are "global" sys_candidates, in that the entire query
                # is sent to Duckling to extract sys_candidates and not just a span range
                # within the query. When we append system entities for a given token,
                # we pick among candidates with start_span equivalent to the token's tag_idx.

                picked_by_existing_system_entity_candidates = False

                sys_entities = query.get_system_entity_candidates(ent_type)
                if ent_type == "sys_time":
                    sys_entities = _sort_by_lowest_time_grain(sys_entities)

                for sys_candidate in sys_entities:
                    start_span = sys_candidate.normalized_token_span.start
                    end_span = sys_candidate.normalized_token_span.end

                    if start_span == tag_idx and tag_idx <= end_span:
                        # We currently don't prioritize any sys_candidate if there are
                        # multiple candidates that meet this conditional.
                        entity_start = sys_candidate.normalized_token_span.start
                    picked_by_existing_system_entity_candidates = True

                if not picked_by_existing_system_entity_candidates:
                    entity_start = tag_idx

        # Append the current token to the current entity, if applicable.
        if (
            iob != O_TAG
            and entity_start is not None
            and not _is_system_entity(ent_type)
        ):
            entity_tokens.append(normalized_tokens[tag_idx])

        # Close the entity if the tag indicates it closed
        if entity_start is not None and iob in (E_TAG, S_TAG):
            logger.debug("Entity closed here")
            if _is_system_entity(ent_type):
                _append_system_entity(entity_start, tag_idx + 1, ent_type)
            else:
                _append_entity(entity_start, ent_type, entity_tokens)
            entity_start = None
            ent_type = ""
            entity_tokens = []

        prev_ent_type = ent_type

    # Handle entities that end with the end of the query
    if entity_start is not None:
        logger.debug("Entity closed at end")
        if _is_system_entity(prev_ent_type):
            _append_system_entity(entity_start, len(tags), prev_ent_type)
        else:
            _append_entity(entity_start, prev_ent_type, entity_tokens)
    else:
        logger.debug("Entity did not end: %s.", entity_start)

    return tuple(entities)


# Methods for tag evaluation


class BoundaryCounts:
    """This class stores the counts of the boundary evaluation metrics.

    Attributes:
        le (int): Label error count. This is when the span is the same but the entity
                  label is incorrect
        be (int): Boundary error count. This is when the entity type is correct but the span is
                  incorrect
        lbe (int): Label boundary error count. This is when both the entity type and span are
                   incorrect, but there was an entity predicted
        tp (int): True positive count. When an entity was correctly predicted
        tn (int): True negative count. Count of times it was correctly predicted that there is no
                  entity
        fp (int): False positive count. When an entity was predicted but one shouldn't have been
        fn (int): False negative count. When an entity was not predicted where one should have been
    """

    def __init__(self):
        """Initializes the object with all counts set to 0"""
        self.le = 0
        self.be = 0
        self.lbe = 0
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def to_dict(self):
        """Converts the object to a dictionary"""
        return {
            "le": self.le,
            "be": self.be,
            "lbe": self.lbe,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
        }


def _get_tag_label(token):
    """Splits a token into its tag and label."""
    (
        tag,
        label,
    ) = token.split("|")
    return tag, label


def _contains_O(entity):
    """Returns true if there is an O tag in the list of tokens we are considering
    as an entity"""
    for token in entity:
        if token[0] == O_TAG:
            return True
    return False


def _all_O(entity):
    """Returns true if all of the tokens we are considering as an entity contain O tags"""
    for token in entity:
        if token[0] != O_TAG:
            return False
    return True


def _is_boundary_error(pred_entity, exp_entity):
    """Returns true if the predicted and expected entity form a boundary error"""
    trimmed_pred_entity = [token[1] for token in pred_entity if token[0] == B_TAG]
    trimmed_exp_entity = [token[1] for token in exp_entity if token[0] == B_TAG]
    return trimmed_pred_entity == trimmed_exp_entity


def _new_tag(last_entity, curr_tag):
    """Returns true if the current tag is different than the tag of the last entity"""
    if len(last_entity) < 1 or not curr_tag:
        return False
    elif (
        last_entity[-1][0] == I_TAG or last_entity[-1][0] == B_TAG
    ) and curr_tag == B_TAG:
        return True
    else:
        return False


def _determine_count_type(last_pred_entity, last_exp_entity, boundary_counts):
    """Determines which of TP, FP, FN, LE, LBE, or BE the last predicted and expected entity
    are and updates the boundary counts accordingly.
    """
    # TP if both are the same
    if last_pred_entity == last_exp_entity:
        boundary_counts.tp += 1
    # FP if entity predicted but not expected
    elif _all_O(last_exp_entity):
        boundary_counts.fp += 1
    # FN if entity expected but not predicted
    elif _all_O(last_pred_entity):
        boundary_counts.fn += 1
    # LE if wrong entity predicted
    elif not _contains_O(last_pred_entity) and not _contains_O(last_exp_entity):
        boundary_counts.le += 1
    # BE and LBE
    elif _contains_O(last_pred_entity) or _contains_O(last_exp_entity):
        if _is_boundary_error(last_pred_entity, last_exp_entity):
            boundary_counts.be += 1
        else:
            boundary_counts.lbe += 1
    return boundary_counts


def get_boundary_counts(expected_sequence, predicted_sequence, boundary_counts):
    """Gets the boundary counts for the expected and predicted sequence of entities."""
    # Initialize values
    in_coding_region = False
    start = True
    last_pred_entity = []
    last_exp_entity = []
    end_token = END_TAG + "|"

    # Iterate through the tokens in the sequence
    for predicted_token, expected_token in zip(
        predicted_sequence + [end_token], expected_sequence + [end_token]
    ):
        predicted_tag, predicted_label = _get_tag_label(predicted_token)
        expected_tag, expected_label = _get_tag_label(expected_token)

        # If we are exiting a coding region, determine the boundary count
        if predicted_tag == expected_tag == O_TAG:
            if in_coding_region:
                boundary_counts = _determine_count_type(
                    last_pred_entity, last_exp_entity, boundary_counts
                )
                in_coding_region = False
                last_pred_entity = []
                last_exp_entity = []

        # If we are entering a new coding region (with a new tag), determine the boundary count and
        # reset entity history
        elif _new_tag(last_pred_entity, predicted_tag) or _new_tag(
            last_exp_entity, expected_tag
        ):
            if in_coding_region:
                boundary_counts = _determine_count_type(
                    last_pred_entity, last_exp_entity, boundary_counts
                )
                last_pred_entity = [(predicted_tag, predicted_label)]
                last_exp_entity = [(expected_tag, expected_label)]

        # If we are at the end of the sequence, determine the boundary count for the last section
        elif predicted_tag == expected_tag == END_TAG and not start:
            if in_coding_region:
                boundary_counts = _determine_count_type(
                    last_pred_entity, last_exp_entity, boundary_counts
                )
            else:
                boundary_counts.tn += 1

        else:
            # If going from a non coding to coding region, add a count for the true negative
            if not in_coding_region:
                if not start:
                    boundary_counts.tn += 1
                in_coding_region = True

            # If continuing in a coding region, append context to current entity
            last_pred_entity.append((predicted_tag, predicted_label))
            last_exp_entity.append((expected_tag, expected_label))

        start = False

    return boundary_counts


def extract_sequence_features(example, example_type, feature_config, resources):
    """Extracts feature dicts for each token in an example.

    Args:
        example (mindmeld.core.Query): a query
        example_type (str): The type of example
        feature_config (dict): The config for features
        resources (dict): Resources of this model

    Returns:
        (list of dict): features
    """
    feat_seq = []
    workspace_features = copy.deepcopy(feature_config)
    enable_stemming = workspace_features.pop(ENABLE_STEMMING, False)

    for name, kwargs in workspace_features.items():
        if callable(kwargs):
            # a feature extractor function was passed in directly
            feat_extractor = kwargs
        else:
            kwargs[ENABLE_STEMMING] = enable_stemming
            feat_extractor = get_feature_extractor(example_type, name)(**kwargs)

        update_feat_seq = feat_extractor(example, resources)
        if not feat_seq:
            feat_seq = update_feat_seq
        else:
            for idx, features in enumerate(update_feat_seq):
                feat_seq[idx].update(features)

    return feat_seq
