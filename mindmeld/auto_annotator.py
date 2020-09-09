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
from abc import ABC, abstractmethod
import re
import logging
import os
import importlib
from enum import Enum
from tqdm import tqdm
import spacy


from .resource_loader import ResourceLoader
from .components._config import get_auto_annotator_config
from .system_entity_recognizer import DucklingRecognizer
from .markup import load_query, dump_queries
from .core import Entity, Span, QueryEntity
from .query_factory import QueryFactory
from .exceptions import MarkupError
from .models.helpers import register_annotator
from .constants import SPACY_ANNOTATOR_SUPPORTED_ENTITIES

logger = logging.getLogger(__name__)

EN_CORE_WEB_SM = "en_core_web_sm"
EN_CORE_WEB_MD = "en_core_web_md"
EN_CORE_WEB_LG = "en_core_web_lg"


class AnnotatorAction(Enum):
    ANNOTATE = "annotate"
    UNANNOTATE = "unannotate"


class Annotator(ABC):
    """
    Abstract Annotator class that can be used to build a custom Annotation class.
    """

    def __init__(self, app_path, config=None, resource_loader=None):
        """ Initializes an annotator."""
        self.app_path = app_path
        self.config = get_auto_annotator_config(app_path=app_path, config=config)
        self._resource_loader = (
            resource_loader or ResourceLoader.create_resource_loader(app_path)
        )
        self.annotate_file_entities_map = self._get_file_entities_map(action="annotate")

    def _get_file_entities_map(self, action="annotate"):
        """ Creates a dictionary that maps file paths to entities given
        regex rules defined in the config.

        Args:
            action (str): Can be "annotate" or "unannotate". Used as a key
                to access a list of regex rules in the config dictionary.

        Returns:
            file_entities_map (dict): A dictionary that maps file paths in an
                App to a list of entities.
        """
        all_file_paths = self._resource_loader.get_all_file_paths()
        file_entities_map = {path: [] for path in all_file_paths}

        if action == AnnotatorAction.ANNOTATE.value:
            rules = self.config[AnnotatorAction.ANNOTATE.value]
        elif action == AnnotatorAction.UNANNOTATE.value:
            rules = self.config[AnnotatorAction.UNANNOTATE.value]

        for rule in rules:
            pattern = Annotator._get_pattern(rule)
            filtered_paths = self._resource_loader.filter_file_paths(
                file_pattern=pattern, file_paths=all_file_paths
            )
            for path in filtered_paths:
                entities = self._get_entities(rule)
                file_entities_map[path] = entities
        return file_entities_map

    @staticmethod
    def _get_pattern(rule):
        """ Convert a rule represented as a dictionary with the keys "domains", "intents",
        "entities" into a regex pattern.

        Args:
            rule (dict): Annotation/Unannotation rule.

        Returns:
            pattern (str): Regex pattern specifying allowed file paths.
        """
        pattern = []
        for x in ["domains", "intents", "files"]:
            processed_segment = Annotator._process_segment(rule[x])
            pattern.append(processed_segment)
        pattern = "/".join(pattern)
        return ".*/" + pattern

    @staticmethod
    def _process_segment(segment):
        """ Process an individual segment from a rule dictionary.

        Args:
            segment (str): Section of a rule dictionary ("domains", "intents", "entities").

        Returns:
            segment (str): Cleaned section of the rule dictionary.
        """
        segment = re.sub("[()]", "", segment)
        segment = segment.replace(".*", ".+")
        segment = segment.replace("*", ".+")
        segment = segment.split("|")
        segment = "|".join([x.strip() for x in segment])
        segment = "(" + segment + ")" if "|" in segment else segment
        return segment

    def _get_entities(self, rule):
        """ Process the entities specified in a rule dictionary. Check if they are valid
        for the given annotator.

        Args:
            rule (dict): Annotation/Unannotation rule with an "entities" key.

        Returns:
            valid_entities (list): List of valid entities specified in the rule.
        """
        if rule["entities"].strip() == "*":
            return ["*"]
        entities = re.sub("[()]", "", rule["entities"]).split("|")
        valid_entities = []
        for entity in entities:
            entity = entity.strip()
            if self.valid_entity_check(entity):
                valid_entities.append(entity)
            else:
                logger.warning("%s is not a valid entity. Skipping entity.", entity)
        return valid_entities

    @abstractmethod
    def valid_entity_check(self, entity):
        """ Determine if an entity type is valid.

        Args:
            entity (str): Name of entity to annotate.

        Returns:
            bool: Whether entity is valid.
        """
        return True

    def annotate(self):
        """ Annotate data based on configurations in the config.py file.
        """
        file_entities_map = self.annotate_file_entities_map
        self._modify_queries(file_entities_map, action="annotate")

    def unannotate(self):
        """ Unannotate data based on configurations in the config.py file.
        """
        if not self.config["unannotate"]:
            logger.warning("'unannotate' is None in the config. Nothing to unannotate.")
            return
        file_entities_map = self._get_file_entities_map(action="unannotate")
        self._modify_queries(file_entities_map, action="unannotate")

    def _modify_queries(self, file_entities_map, action):
        """ Iterates through App files and annotates or unannotates queries.

        Args:
            file_entities_map (dict): A dictionary that maps a file paths
                in an App to a list of entities.
        """
        query_factory = QueryFactory.create_query_factory(self.app_path)
        for path in file_entities_map:
            processed_queries = Annotator._get_processed_queries(
                file_path=path, query_factory=query_factory
            )
            tqdm_desc = "Processing " + path + ": "
            for processed_query in tqdm(processed_queries, ascii=True, desc=tqdm_desc):
                entity_types = file_entities_map[path]
                if action == AnnotatorAction.ANNOTATE.value:
                    self._annotate_query(
                        processed_query=processed_query, entity_types=entity_types
                    )
                elif action == AnnotatorAction.UNANNOTATE.value:
                    self._unannotate_query(
                        processed_query=processed_query, remove_entities=entity_types
                    )

            annotated_queries = list(dump_queries(processed_queries))
            with open(path, "w") as outfile:
                outfile.write("".join(annotated_queries))
                outfile.close()

    @staticmethod
    def _get_processed_queries(file_path, query_factory):
        """ Converts queries in a given path to processed queries.
        Skips and presents a warning if loading the query creates an error.

        Args:
            file_path (str): Path to file containing queries.
            query_factory (QueryFactory): Used to generate processed queries.

        Returns:
            processed_queries (list): List of processed queries from file.
        """
        with open(file_path) as infile:
            queries = infile.readlines()
        processed_queries = []
        for query in queries:
            try:
                processed_query = load_query(markup=query, query_factory=query_factory)
                processed_queries.append(processed_query)
            except (AssertionError, MarkupError):
                logger.warning("Skipping query. Error in processing: %s", query)
        return processed_queries

    def _annotate_query(self, processed_query, entity_types):
        """ Updates the entities of a processed query with newly
        annotated entities.

        Args:
            processed_query (ProcessedQuery): The processed query to update.
            entity_types (list): List of entities allowed for annotation.
        """
        current_entities = list(processed_query.entities)
        annotated_entities = self._get_annotated_entities(
            processed_query=processed_query, entity_types=entity_types
        )
        final_entities = self._resolve_conflicts(
            current_entities=current_entities, annotated_entities=annotated_entities
        )
        processed_query.entities = tuple(final_entities)

    def _get_annotated_entities(self, processed_query, entity_types=None):
        """ Creates a list of query entities after parsing the text of a
        processed query.

        Args:
            processed_query (ProcessedQuery): A processed query.
            entity_types (list): List of entities allowed for annotation.

        Returns:
            query_entities (list): List of query entities.
        """
        if len(entity_types) == 0:
            return []
        entity_types = None if entity_types == ["*"] else entity_types
        items = self.parse(
            sentence=processed_query.query.text, entity_types=entity_types
        )
        query_entities = [
            Annotator._item_to_query_entity(item, processed_query) for item in items
        ]
        return query_entities if len(query_entities) > 0 else []

    @staticmethod
    def _item_to_query_entity(item, processed_query):
        """ Converts an item returned from parse into a query entity.

        Args:
            item (dict): Dictionary representing an entity with the keys -
                "body", "start", "end", "value", "dim".
            processed_query (ProcessedQuery): The processed query that the
                entity is found in.

        Returns:
            query_entity (QueryEntity): The converted query entity.
        """
        span = Span(start=item["start"], end=item["end"] - 1)
        entity = Entity(text=item["body"], entity_type=item["dim"], value=item["value"])
        query_entity = QueryEntity.from_query(
            query=processed_query.query, span=span, entity=entity
        )
        return query_entity

    def _resolve_conflicts(self, current_entities, annotated_entities):
        """ Resolve overlaps between existing entities and newly annotad entities.

        Args:
            current_entities (list): List of existing query entities.
            annotated_entities (list): List of new query entities.

        Returns:
            final_entities (list): List of resolved query entities.
        """
        overwrite = self.config["overwrite"]
        base_entities = annotated_entities if overwrite else current_entities
        other_entities = current_entities if overwrite else annotated_entities

        additional_entities = []
        for o_entity in other_entities:
            no_overlaps = [
                Annotator._no_overlap(o_entity, b_entity) for b_entity in base_entities
            ]
            if all(no_overlaps):
                additional_entities.append(o_entity)
        return base_entities + additional_entities

    @staticmethod
    def _no_overlap(entity_one, entity_two):
        """ Returns True if two query entities do not overlap.
        """
        return (
            entity_one.span.start > entity_two.span.end
            or entity_two.span.start > entity_one.span.end
        )

    # pylint: disable=R0201
    def _unannotate_query(self, processed_query, remove_entities):
        """ Removes specified entities in a processed query. If all entities are being
        removed, this function will not remove entities that the annotator does not support
        unless it is explicitly specified to do so in the config with the param
        "unannotate_supported_entities_only" (boolean).

        Args:
            processed_query (ProcessedQuery): A processed query.
            remove_entities (list): List of entities to remove.
        """
        remove_supported_only = self.config["unannotate_supported_entities_only"]
        keep_entities = []
        for query_entity in processed_query.entities:
            if remove_entities == ["*"]:
                is_supported_entity = self.valid_entity_check(query_entity.entity.type)
                if remove_supported_only and not is_supported_entity:
                    keep_entities.append(query_entity)
            elif query_entity.entity.type not in remove_entities:
                keep_entities.append(query_entity)
        processed_query.entities = tuple(keep_entities)

    @abstractmethod
    def parse(self, sentence, **kwargs):
        """ Extract entities from a sentence. Detected entities should be
        represented as dictionaries with the following keys: "body", "start"
        (start index), "end" (end index), "value", "dim" (entity type).

        Args:
            sentence (str): Sentence to detect entities.

        Returns:
            entities (list): List of entity dictionaries.
        """
        raise NotImplementedError("Subclasses must implement this method")


class SpacyAnnotator(Annotator):
    """ Annotator class that uses spacy to generate annotations.
    """

    def __init__(self, app_path, config=None):
        super().__init__(app_path=app_path, config=config)

        self.model = self.config.get("spacy_model", EN_CORE_WEB_LG)
        self.nlp = SpacyAnnotator._load_model(self.model)
        self.duckling = DucklingRecognizer.get_instance()
        self.ANNOTATOR_TO_DUCKLING_ENTITY_MAPPINGS = {
            "money": "sys_amount-of-money",
            "cardinal": "sys_number",
            "ordinal": "sys_ordinal",
            "person": "sys_person",
            "percent": "sys_percent",
            "distance": "sys_distance",
            "quantity": "sys_weight",
        }

    @staticmethod
    def _load_model(model):
        """ Load Spacy English model. Download if needed.

        Args:
            model (str): Spacy model ("en_core_web_sm", "en_core_web_md", or
                "en_core_web_lg").

        Returns:
            nlp (spacy.lang.en.English): Spacy language model.
        """

        if model in [EN_CORE_WEB_SM, EN_CORE_WEB_MD, EN_CORE_WEB_LG]:
            logger.info("Loading Spacy model %s.", model)
            try:
                return spacy.load(model)
            except OSError:
                os.system("python -m spacy download " + model)
                language_module = importlib.import_module(model)
                return language_module.load()
        else:
            error_msg = (
                "Unknown Spacy model name: {!r}. Model must be {},"
                " {}, or {}".format(
                    model, EN_CORE_WEB_SM, EN_CORE_WEB_MD, EN_CORE_WEB_LG
                )
            )
            raise ValueError(error_msg)

    def valid_entity_check(self, entity):
        entity = entity.lower().strip()
        return entity in SPACY_ANNOTATOR_SUPPORTED_ENTITIES

    def parse(self, sentence, entity_types=None):
        """ Extracts entities from a sentence. Detected entities should are
        represented as dictionaries with the following keys: "body", "start"
        (start index), "end" (end index), "value", "dim" (entity type).

        Args:
            sentence (str): Sentence to detect entities.
            entity_types (list): List of entity types to parse. If None, all
                possible entity types will be parsed.

        Returns:
            entities (list): List of entity dictionaries.
        """
        doc = self.nlp(sentence)
        spacy_entities = [
            {
                "body": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "value": {"value": ent.text},
                "dim": ent.label_.lower(),
            }
            for ent in doc.ents
        ]

        entities = []
        for entity in spacy_entities:
            if entity["dim"] in ["time", "date"]:
                entity = self._resolve_time_date(entity, entity_types)
            elif entity["dim"] == "cardinal":
                entity = self._resolve_cardinal(entity)
            elif entity["dim"] == "money":
                entity = self._resolve_money(entity, sentence)
            elif entity["dim"] == "ordinal":
                entity = self._resolve_ordinal(entity)
            elif entity["dim"] == "quantity":
                entity = self._resolve_quantity(entity)
            elif entity["dim"] == "percent":
                entity = self._resolve_percent(entity)
            elif entity["dim"] == "person":
                entity = self._resolve_person(entity)
            else:
                entity["dim"] = "sys_" + entity["dim"].replace("_", "-")

            if entity:
                entities.append(entity)

        if entity_types:
            entities = [e for e in entities if e["dim"] in entity_types]

        return entities

    def _resolve_time_date(self, entity, entity_types=None):
        """ Resolves a time related entity. First looks for an exact match, then
        for the largest substring match. Order of priority is sys_duration, sys_interval,
        and sys_time.

        Args:
            entity (dict): A dictionary representing an entity.
            entity_types (list): List of entity types to parse. If None, all possible
                entity types will be parsed.

        Returns:
            entity (dict): A resolved entity dict or None if the entity isn't resolved.
        """
        candidates = self.duckling.get_candidates_for_text(entity["body"])

        if len(candidates) == 0:
            return

        time_entities = ["sys_duration", "sys_interval", "sys_time"]
        if entity_types:
            time_entities = [e for e in time_entities if e in entity_types]

        if SpacyAnnotator._resolve_time_exact_match(entity, candidates, time_entities):
            return entity
        elif SpacyAnnotator._resolve_largest_substring(
            entity, candidates, entity_types=time_entities, is_time_related=True
        ):
            return entity

    @staticmethod
    def _get_time_entity_type(candidate):
        """ Determine the "sys" type given a time-related Duckling candidate dictionary.

        Args:
            candidate (dict): A Duckling candidate.

        Returns:
            entity_type (str): Entity type. ("sys_duration", "sys_interval" or "sys_time")
        """
        if candidate["dim"] == "duration":
            return "sys_duration"
        if candidate["dim"] == "time":
            if candidate["value"]["type"] == "interval":
                return "sys_interval"
            else:
                return "sys_time"

    @staticmethod
    def _resolve_time_exact_match(entity, candidates, time_entities):
        """ Resolve a time-related entity given Duckling candidates on the first
        exact match.

        Args:
            entity (dict): A dictionary representing an entity.
            candidates (list): List of dictionary candidates returned by Duckling.parse().
            time_entities (list): List of allowed time-related entity types.

        Returns:
            entity (dict): A resolved entity dict or None if the entity isn't resolved.
        """
        for candidate in candidates:
            candidate_entity = SpacyAnnotator._get_time_entity_type(candidate)
            if (
                candidate_entity in time_entities
                and candidate["body"] == entity["body"]
            ):
                entity["dim"] = candidate_entity
                entity["value"] = candidate["value"]
                return entity

    @staticmethod
    def _resolve_largest_substring(entity, candidates, entity_types, is_time_related):
        """ Resolve an entity by the largest substring match given Duckling candidates.

        Args:
            entity (dict): A dictionary representing an entity.
            candidates (list): List of dictionary candidates returned by Duckling.parse().
            entity_types (list): List of entity types to check.
            is_time_related (bool): Whether the entity is related to time.

        Returns:
            entity (dict): A resolved entity dict or None if the entity isn't resolved.
        """
        largest_candidate = None
        resolved_entity_type = None
        for entity_type in entity_types:
            for candidate in candidates:
                if is_time_related:
                    candidate_entity = SpacyAnnotator._get_time_entity_type(candidate)
                else:
                    candidate_entity = candidate["dim"]

                if (
                    candidate_entity == entity_type
                    and candidate["body"] in entity["body"]
                    and (
                        largest_candidate is None
                        or len(candidate["body"]) > len(largest_candidate["body"])
                    )
                ):
                    largest_candidate = candidate
                    resolved_entity_type = entity_type

        if largest_candidate:
            entity["body"] = largest_candidate["body"]
            offset = entity["start"]
            entity["start"] = offset + largest_candidate["start"]
            entity["end"] = offset + largest_candidate["end"]
            entity["value"] = largest_candidate["value"]
            entity["dim"] = resolved_entity_type
            return entity

    def _resolve_cardinal(self, entity):
        return self._resolve_exact_match(entity)

    def _resolve_money(self, entity, sentence):
        # Update entity to include the $ symbol if it's left of the body text.
        if "$" in sentence:
            start = entity["start"]
            if (start == 1 and sentence[0] == "$") or (
                start > 1 and sentence[start - 2 : start] == " $"
            ):
                entity["start"] -= 1
                entity["body"] = sentence[entity["start"] : entity["end"]]

        return self._resolve_exact_match(entity)

    def _resolve_ordinal(self, entity):
        return self._resolve_exact_match(entity)

    def _resolve_exact_match(self, entity):
        """ Resolves an entity by exact match and corresponding type.

        Args:
            entity (dict): A dictionary representing an entity.

        Returns:
            entity (dict): A resolved entity dict or None if the entity isn't resolved.
        """
        entity["dim"] = self.ANNOTATOR_TO_DUCKLING_ENTITY_MAPPINGS[entity["dim"]]

        candidates = self.duckling.get_candidates_for_text(entity["body"])
        if len(candidates) == 0:
            return

        for candidate in candidates:
            if (
                candidate["entity_type"] == entity["dim"]
                and entity["body"] == candidate["body"]
            ):
                entity["value"] = candidate["value"]
                return entity

    def _resolve_quantity(self, entity):
        """ Resolves a quantity related entity. First looks for an exact match, then
        for the largest substring match. Order of priority is "sys_distance" then "sys_quantity".
        Unresolved entities are labelled as "sys_other-quantity"

        Args:
            entity (dict): A dictionary representing an entity.

        Returns:
            entity (dict): A resolved entity dict or None if the entity isn't resolved.
        """
        candidates = self.duckling.get_candidates_for_text(entity["body"])
        if len(candidates) == 0:
            entity["dim"] = "sys_other-quantity"
            return entity

        entity_types = ["distance", "quantity"]
        for entity_type in entity_types:
            for candidate in candidates:
                if (
                    candidate["dim"] == entity_type
                    and candidate["body"] == entity["body"]
                ):
                    entity["value"] = candidate["value"]
                    entity["dim"] = self.ANNOTATOR_TO_DUCKLING_ENTITY_MAPPINGS[
                        entity_type
                    ]
                    return entity

        if SpacyAnnotator._resolve_largest_substring(
            entity, candidates, entity_types=entity_types, is_time_related=False
        ):
            return entity
        else:
            entity["dim"] = "sys_other-quantity"
            return entity

    def _resolve_percent(self, entity):
        """ Resolves an entity related to percentage. Uses a heuristic of finding
        the largest candidate value and dividing by 100.

        Args:
            entity (dict): A dictionary representing an entity.

        Returns:
            entity (dict): A resolved entity dict or None if the entity isn't resolved.
        """
        entity["dim"] = self.ANNOTATOR_TO_DUCKLING_ENTITY_MAPPINGS[entity["dim"]]

        candidates = self.duckling.get_candidates_for_text(entity["body"])
        if len(candidates) == 0:
            return

        possible_values = []
        for candidate in candidates:
            if candidate["entity_type"] == "sys_number":
                possible_values.append(candidate["value"]["value"])
        entity["value"]["value"] = max(possible_values) / 100
        return entity

    def _resolve_person(self, entity):
        """ Resolves a person entity by unlabelling a possessive "'s" from the
        name if it exists.

        Args:
            entity (dict): A dictionary representing an entity.

        Returns:
            entity (dict): A resolved entity dict.
        """
        entity["dim"] = self.ANNOTATOR_TO_DUCKLING_ENTITY_MAPPINGS[entity["dim"]]

        if len(entity["body"]) >= 2 and entity["body"][-2:] == "'s":
            entity["value"] = {"value": entity["body"][:-2]}
            entity["body"] = entity["body"][:-2]
            entity["end"] -= 2
        return entity


register_annotator("SpacyAnnotator", SpacyAnnotator)
