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
import importlib
import logging
import os
import re
from abc import ABC, abstractmethod
from enum import Enum
import spacy
from tqdm import tqdm
from ._util import get_pattern
from .resource_loader import ResourceLoader
from .components._config import (
    ENGLISH_LANGUAGE_CODE,
    ENGLISH_US_LOCALE,
)
from .components.translators import NoOpTranslator, TranslatorFactory
from .system_entity_recognizer import (
    DucklingRecognizer,
    duckling_item_to_query_entity,
)
from .markup import load_query, dump_queries
from .core import Entity, Span, QueryEntity, _get_overlap
from .exceptions import MarkupError
from .models.helpers import register_annotator
from .constants import (
    SPACY_ANNOTATOR_WEB_LANGUAGES,
    SPACY_ANNOTATOR_SUPPORTED_LANGUAGES,
    SPACY_ANNOTATOR_MODEL_SIZES,
    DUCKLING_TO_SYS_ENTITY_MAPPINGS,
    ANNOTATOR_TO_SYS_ENTITY_MAPPINGS,
    SPACY_SYS_ENTITIES_NOT_IN_DUCKLING,
    CURRENCY_SYMBOLS,
)
from .components import NaturalLanguageProcessor
from .path import get_entity_types
from .query_factory import QueryFactory

logger = logging.getLogger(__name__)


class AnnotatorAction(Enum):
    ANNOTATE = "annotate"
    UNANNOTATE = "unannotate"


class Annotator(ABC):
    """
    Abstract Annotator class that can be used to build a custom Annotation class.
    """

    # pylint: disable=W0613
    def __init__(
        self,
        app_path,
        annotation_rules=None,
        language=ENGLISH_LANGUAGE_CODE,
        locale=ENGLISH_US_LOCALE,
        overwrite=False,
        unannotate_supported_entities_only=True,
        unannotation_rules=None,
        **kwargs,
    ):
        """Initializes an annotator.

        Args:
            app_path (str): The location of the MindMeld app.
            annotation_rules (list): List of Annotation rules.
            language (str, optional): Language as specified using a 639-1/2 code.
            locale (str, optional): The locale representing the ISO 639-1 language code and \
                ISO3166 alpha 2 country code separated by an underscore character.
            overwrite (bool): Whether to overwrite existing annotations with conflicting spans.
            unannotate_supported_entities_only (bool): Only allow removal of supported entities.
            unannotation_rules (list): List of Annotation rules.
        """
        self.app_path = app_path
        self.language = language
        self.locale = locale
        self.overwrite = overwrite
        self.annotation_rules = annotation_rules or []
        self.unannotate_supported_entities_only = unannotate_supported_entities_only
        self.unannotation_rules = unannotation_rules or []
        self._resource_loader = ResourceLoader.create_resource_loader(app_path)
        self.duckling = DucklingRecognizer.get_instance()

    def _get_file_entities_map(self, action: AnnotatorAction):
        """Creates a dictionary that maps file paths to entities given
        regex rules defined in the config.

        Args:
            action (AnnotatorAction): Can be "annotate" or "unannotate". Used as a key
                to access a list of regex rules in the config dictionary.

        Returns:
            file_entities_map (dict): A dictionary that maps file paths in an
                App to a list of entities.
        """
        all_file_paths = self._resource_loader.get_all_file_paths()
        file_entities_map = {path: [] for path in all_file_paths}

        if action == AnnotatorAction.ANNOTATE:
            rules = self.annotation_rules
        elif action == AnnotatorAction.UNANNOTATE:
            rules = self.unannotation_rules
        else:
            raise AssertionError(f"{action} is an invalid Annotator action.")

        for rule in rules:
            pattern = get_pattern(rule)
            compiled_pattern = re.compile(pattern)
            filtered_paths = self._resource_loader.filter_file_paths(
                compiled_pattern=compiled_pattern, file_paths=all_file_paths
            )
            for path in filtered_paths:
                entities = self._get_entities(rule)
                file_entities_map[path] = entities
        return file_entities_map

    def _get_entities(self, rule):
        """Process the entities specified in a rule dictionary. Check if they are valid
        for the given annotator.

        Args:
            rule (dict): Annotation/Unannotation rule with an "entities" key.

        Returns:
            valid_entities (list): List of valid entities specified in the rule.
        """
        if rule["entities"].strip() in ["*", ".*", ".+"]:
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

    @property
    @abstractmethod
    def supported_entity_types(self):
        """
        Returns:
            supported_entity_types (list): List of supported entity types.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def valid_entity_check(self, entity):
        """Determine if an entity type is valid.

        Args:
            entity (str): Name of entity to annotate.

        Returns:
            bool: Whether entity is valid.
        """
        entity = entity.lower().strip()
        return entity in self.supported_entity_types

    def annotate(self):
        """Annotate data."""
        if not self.annotation_rules:
            logger.warning(
                """'annotate' field is not configured or misconfigured in the `config.py`.
                 We can't find any file to annotate."""
            )
            return
        self._modify_queries(action=AnnotatorAction.ANNOTATE)

    def unannotate(self):
        """Unannotate data."""
        if not self.unannotate:
            logger.warning(
                """'unannotate' field is not configured or misconfigured in the `config.py`.
                 We can't find any file to unannotate."""
            )
            return
        self._modify_queries(action=AnnotatorAction.UNANNOTATE)

    def _modify_queries(self, action: AnnotatorAction):
        """Iterates through App files and annotates or unannotates queries.

        Args:
            action (AnnotatorAction): Can be "annotate" or "unannotate".
        """
        file_entities_map = self._get_file_entities_map(action=action)
        query_factory = QueryFactory.create_query_factory(self.app_path)
        path_list = [p for p in file_entities_map if file_entities_map[p]]
        for path in path_list:
            processed_queries = Annotator._get_processed_queries(
                file_path=path, query_factory=query_factory
            )
            tqdm_desc = "Processing " + path + ": "
            for processed_query in tqdm(processed_queries, ascii=True, desc=tqdm_desc):
                entity_types = file_entities_map[path]
                if action == AnnotatorAction.ANNOTATE:
                    self._annotate_query(
                        processed_query=processed_query, entity_types=entity_types,
                    )
                elif action == AnnotatorAction.UNANNOTATE:
                    self._unannotate_query(
                        processed_query=processed_query, remove_entities=entity_types,
                    )
            with open(path, "w") as outfile:
                outfile.write("".join(list(dump_queries(processed_queries))))
                outfile.close()

    @staticmethod
    def _get_processed_queries(file_path, query_factory):
        """Converts queries in a given path to processed queries.
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
        domain, intent = file_path.split(os.sep)[-3:-1]
        for query in queries:
            try:
                processed_query = load_query(
                    markup=query,
                    domain=domain,
                    intent=intent,
                    query_factory=query_factory,
                )
                processed_queries.append(processed_query)
            except (AssertionError, MarkupError):
                logger.warning("Skipping query. Error in processing: %s", query)
        return processed_queries

    def _annotate_query(self, processed_query, entity_types):
        """Updates the entities of a processed query with newly
        annotated entities.

        Args:
            processed_query (ProcessedQuery): The processed query to update.
            entity_types (list): List of entities allowed for annotation.
        """
        current_entities = list(processed_query.entities)
        annotated_entities = self._get_annotated_entities(
            processed_query=processed_query, entity_types=entity_types
        )
        final_entities = Annotator._resolve_conflicts(
            target_entities=annotated_entities if self.overwrite else current_entities,
            other_entities=current_entities if self.overwrite else annotated_entities,
        )
        processed_query.entities = tuple(final_entities)

    def _get_annotated_entities(self, processed_query, entity_types=None):
        """Creates a list of query entities after parsing the text of a
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
        return self.parse(
            sentence=processed_query.query.text,
            entity_types=entity_types,
            domain=processed_query.domain,
            intent=processed_query.intent,
        )

    @staticmethod
    def _item_to_query_entity(item, processed_query):
        """Converts an item returned from parse into a query entity.

        Args:
            item (dict): Dictionary representing an entity with the keys -
                "body", "start", "end", "value", "dim". ("role" is an optional attribute.)
            processed_query (ProcessedQuery): The processed query that the
                entity is found in.

        Returns:
            query_entity (QueryEntity): The converted query entity.
        """
        span = Span(start=item["start"], end=item["end"] - 1)
        role = item.get("role")
        entity = Entity(
            text=item["body"], entity_type=item["dim"], role=role, value=item["value"]
        )
        query_entity = QueryEntity.from_query(
            query=processed_query.query, span=span, entity=entity
        )
        return query_entity

    @staticmethod
    def _resolve_conflicts(target_entities, other_entities):
        """Resolve overlaps between existing entities and newly annotad entities.

        Args:
            target_entities (list): List of existing query entities.
            other_entities (list): List of new query entities.

        Returns:
            final_entities (list): List of resolved query entities.
        """
        additional_entities = []
        for o_entity in other_entities:
            no_overlaps = [
                not _get_overlap(o_entity.span, t_entity.span)
                for t_entity in target_entities
            ]
            if all(no_overlaps):
                additional_entities.append(o_entity)
        target_entities.extend(additional_entities)
        return target_entities

    # pylint: disable=R0201
    def _unannotate_query(self, processed_query, remove_entities):
        """Removes specified entities in a processed query. If all entities are being
        removed, this function will not remove entities that the annotator does not support
        unless it is explicitly specified to do so in the config with the param
        "unannotate_supported_entities_only" (bool).

        Args:
            processed_query (ProcessedQuery): A processed query.
            remove_entities (list): List of entities to remove.
        """
        keep_entities = []
        for query_entity in processed_query.entities:
            if remove_entities == ["*"]:
                is_supported_entity = self.valid_entity_check(query_entity.entity.type)
                if self.unannotate_supported_entities_only and not is_supported_entity:
                    keep_entities.append(query_entity)
            elif query_entity.entity.type not in remove_entities:
                keep_entities.append(query_entity)
        processed_query.entities = tuple(keep_entities)

    @abstractmethod
    def parse(self, sentence, **kwargs):
        """Extract entities from a sentence. Detected entities should be
        represented as dictionaries with the following keys: "body", "start"
        (start index), "end" (end index), "value", "dim" (entity type).

        Args:
            sentence (str): Sentence to detect entities.

        Returns:
            query_entities (list): List of QueryEntity objects.
        """
        raise NotImplementedError("Subclasses must implement this method")


class SpacyAnnotator(Annotator):
    """Annotator class that uses spacy to generate annotations.
    Depending on the language, supported entities can include: "sys_time", "sys_interval",
    "sys_duration", "sys_number", "sys_amount-of-money", "sys_distance", "sys_weight",
    "sys_ordinal", "sys_quantity", "sys_percent", "sys_org", "sys_loc", "sys_person",
    "sys_gpe", "sys_norp", "sys_fac", "sys_product", "sys_event", "sys_law", "sys_langauge",
    "sys_work-of-art", "sys_other-quantity".
    For more information on the supported entities for the Spacy Annotator check the MindMeld docs.
    """

    def __init__(self, *args, **kwargs):
        """Initializes a SpacyAnnotator.

        Args:
            app_path (str): The location of the MindMeld app.
            annotation_rules (list): List of Annotation rules.
            language (str, optional): Language as specified using a 639-1/2 code.
            locale (str, optional): The locale representing the ISO 639-1 language code and \
                ISO3166 alpha 2 country code separated by an underscore character.
            overwrite (bool): Whether to overwrite existing annotations with conflicting spans.
            spacy_model_size (str, optional): Size of the Spacy model to use. ("sm", "md", or "lg")
            unannotate_supported_entities_only (bool): Only allow removal of supported entities.
            unannotation_rules (list): List of Annotation rules.
        """
        super().__init__(*args, **kwargs)
        if self.language not in SPACY_ANNOTATOR_SUPPORTED_LANGUAGES:
            raise ValueError(
                "Spacy does not currently support: {!r}.".format(self.language)
            )
        self.spacy_model_size = kwargs.get("spacy_model_size", "lg")
        if self.spacy_model_size not in SPACY_ANNOTATOR_MODEL_SIZES:
            raise ValueError(
                "{!r} is not a valid model size. Select from: {!r}.".format(
                    self.language, " ".join(SPACY_ANNOTATOR_MODEL_SIZES)
                )
            )
        self.nlp = self._load_model()

    def _get_spacy_model_name(self):
        """Get the name of a Spacy Model.

        Returns:
            spacy_model_name (str): Name of the Spacy NER model
        """
        model_type = "web" if self.language in SPACY_ANNOTATOR_WEB_LANGUAGES else "news"
        return f"{self.language}_core_{model_type}_{self.spacy_model_size}"

    def _load_model(self):
        """Load Spacy English model. Download if needed.

        Args:
            model (str): Spacy model (Ex: "en_core_web_sm", "zh_core_web_md", etc.)

        Returns:
            nlp: Spacy language model. (Ex: "spacy.lang.es.Spanish")
        """
        model = self._get_spacy_model_name()
        logger.info("Loading Spacy model %s.", model)
        try:
            return spacy.load(model)
        except OSError:
            logger.warning("%s not found on disk. Downloading the model.", model)
            os.system("python -m spacy download " + model)
            try:
                language_module = importlib.import_module(model)
            except ModuleNotFoundError as e:
                raise ValueError("Unknown Spacy model name: {!r}.".format(model)) from e
            return language_module.load()

    @property
    def supported_entity_types(self):  # pylint: disable=W0236
        """This function generates a list of supported entities for the given language.
        These entities labels are mapped to MindMeld sys_entities.
        The "misc" spacy entity is skipped since the category too broad to be
        helpful in an application.

        Returns:
            supported_entity_types (list): List of supported entity types.
        """
        spacy_supported_entities = [e.lower() for e in self.nlp.get_pipe("ner").labels]
        supported_entities = set()
        for entity in spacy_supported_entities:
            if entity == "misc":
                continue
            if entity in ["time", "date", "datetime"]:
                supported_entities.update(["sys_time", "sys_duration", "sys_interval"])
            elif entity in ANNOTATOR_TO_SYS_ENTITY_MAPPINGS:
                supported_entities.add(ANNOTATOR_TO_SYS_ENTITY_MAPPINGS[entity])
            else:
                supported_entities.add(f"sys_{entity}")
        if "sys_weight" in supported_entities:
            supported_entities.update(["sys_distance", "sys_other-quantity"])
        supported_entities = self._remove_unresolvable_entities(supported_entities)
        return supported_entities

    def _remove_unresolvable_entities(self, entities):
        """Remove entities that need duckling to be resolved but are not
        supported by duckling for the given language.

        Args:
            entities (list): List of entities to filter.
        Returns:
            filtered_entities (list): Filtered entities.
        """
        filtered_entities = []
        for entity in entities:
            if entity not in SPACY_SYS_ENTITIES_NOT_IN_DUCKLING:
                if (
                    self.language in DUCKLING_TO_SYS_ENTITY_MAPPINGS
                    and entity in DUCKLING_TO_SYS_ENTITY_MAPPINGS[self.language]
                ):
                    filtered_entities.append(entity)
            else:
                filtered_entities.append(entity)
        return filtered_entities

    def parse(self, sentence, entity_types=None, **kwargs):
        """Extracts entities from a sentence. Detected entities should are
        represented as dictionaries with the following keys: "body", "start"
        (start index), "end" (end index), "value", "dim" (entity type).

        Args:
            sentence (str): Sentence to detect entities.
            entity_types (list): List of entity types to annotate. If None, all
                possible entity types will be annotated.

        Returns:
            query_entities (list): List of QueryEntity objects.
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

        entity_resolution_func_map = {
            "time": self._resolve_time_date,
            "date": self._resolve_time_date,
            "datetime": self._resolve_time_date,
            "cardinal": self._resolve_cardinal,
            "money": self._resolve_money,
            "ordinal": self._resolve_ordinal,
            "quantity": self._resolve_quantity,
            "percent": self._resolve_percent,
            "person": self._resolve_person,
        }

        entities = []
        for entity in spacy_entities:
            if entity["dim"] in ["per", "persName"]:
                entity["dim"] = "person"
            elif entity["dim"] == "misc":
                continue
            if entity["dim"] in entity_resolution_func_map:
                params = {"entity": entity}
                if entity["dim"] in ["time", "date", "datetime"]:
                    params["entity_types"] = entity_types
                elif entity["dim"] in ["money"]:
                    params["sentence"] = sentence
                entity = entity_resolution_func_map[entity["dim"]](**params)
            else:
                entity["dim"] = "sys_" + entity["dim"].replace("_", "-")

            if entity:
                entities.append(entity)

        if entity_types:
            entities = [e for e in entities if e["dim"] in entity_types]

        processed_query = load_query(
            sentence,
            query_factory=self._resource_loader.query_factory,
            domain=kwargs.get("domain"),
            intent=kwargs.get("intent"),
        )
        return [
            Annotator._item_to_query_entity(entity, processed_query)
            for entity in entities
        ]

    def _resolve_time_date(self, entity, entity_types=None):
        """Resolves a time related entity. First, an exact match is searched for. If
        not found, the largest substring match is searched for. If the span of the entity
        does not share the exact span match with duckling entities then it is likely that
        spacy has recognized an additional word in the span. For example, "nearly 15 minutes"
        doesn't have an exact match but the largest substring match correctly resolves for
        the substring "15 minutes". Order of priority for the time entities is sys_duration,
        sys_interval, and sys_time.

        Args:
            entity (dict): A dictionary representing an entity.
            entity_types (list): List of entity types to parse. If None, all possible
                entity types will be parsed.

        Returns:
            entity (dict): A resolved entity dict or None if the entity isn't resolved.
        """
        candidates = self.duckling.get_candidates_for_text(
            entity["body"], language=self.language, locale=self.locale
        )
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
        """Determine the "sys" type given a time-related Duckling candidate dictionary.

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
        """Resolve a time-related entity given Duckling candidates on the first
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
        """Resolve an entity by the largest substring match given Duckling candidates.

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
                    candidate_entity = candidate["entity_type"]

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
        if self._resolve_exact_match(entity):
            return entity
        candidates = self.duckling.get_candidates_for_text(
            entity["body"], language=self.language, locale=self.locale
        )
        if self._resolve_largest_substring(
            entity, candidates, entity_types=["sys_number"], is_time_related=False
        ):
            return entity

    def _resolve_money(self, entity, sentence):
        for symbol in CURRENCY_SYMBOLS:
            if symbol in sentence:
                start = entity["start"]
                if (start == 1 and sentence[0] == symbol) or (
                    start >= 2 and sentence[start - 2 : start] == " " + symbol
                ):
                    entity["start"] -= 1
                    entity["body"] = sentence[entity["start"] : entity["end"]]

        return self._resolve_exact_match(entity)

    def _resolve_ordinal(self, entity):
        return self._resolve_exact_match(entity)

    def _resolve_exact_match(self, entity):
        """Resolves an entity by exact match and corresponding type.

        Args:
            entity (dict): A dictionary representing an entity.

        Returns:
            entity (dict): A resolved entity dict or None if the entity isn't resolved.
        """
        entity["dim"] = ANNOTATOR_TO_SYS_ENTITY_MAPPINGS[entity["dim"]]

        candidates = self.duckling.get_candidates_for_text(
            entity["body"], language=self.language, locale=self.locale
        )

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
        """Resolves a quantity related entity. First looks for an exact match, then
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
                    entity["dim"] = ANNOTATOR_TO_SYS_ENTITY_MAPPINGS[entity_type]
                    return entity

        if SpacyAnnotator._resolve_largest_substring(
            entity, candidates, entity_types=entity_types, is_time_related=False
        ):
            return entity
        else:
            entity["dim"] = "sys_other-quantity"
            return entity

    def _resolve_percent(self, entity):
        """Resolves an entity related to percentage. Uses a heuristic of finding
        the largest candidate value and dividing by 100. If the candidate value is
        a float, the float value divided by 100 is immediately returned.

        Args:
            entity (dict): A dictionary representing an entity.

        Returns:
            entity (dict): A resolved entity dict or None if the entity isn't resolved.
        """
        entity["dim"] = ANNOTATOR_TO_SYS_ENTITY_MAPPINGS[entity["dim"]]

        candidates = self.duckling.get_candidates_for_text(
            entity["body"], language=self.language, locale=self.locale
        )

        if len(candidates) == 0:
            return
        possible_values = []
        for candidate in candidates:
            if candidate["entity_type"] == "sys_number":
                value = candidate["value"]["value"]
                if isinstance(value, float):
                    entity["value"]["value"] = value / 100
                    return entity
                else:
                    possible_values.append(value)
        entity["value"]["value"] = max(possible_values) / 100
        return entity

    def _resolve_person(self, entity):
        """Resolves a person entity by unlabelling a possessive "'s" from the
        name if it exists.

        Args:
            entity (dict): A dictionary representing an entity.

        Returns:
            entity (dict): A resolved entity dict.
        """
        entity["dim"] = ANNOTATOR_TO_SYS_ENTITY_MAPPINGS[entity["dim"]]

        if self._is_plural_entity(entity):
            entity["value"] = {"value": entity["body"][:-2]}
            entity["body"] = entity["body"][:-2]
            entity["end"] -= 2
        return entity

    def _is_plural_entity(self, entity):
        """Check if an entity is plural.

        Args:
            entity (dict): A dictionary representing an entity.

        Returns:
            is_plural (bool): Whether the entity is plural.
        """
        return (
            self.language == ENGLISH_LANGUAGE_CODE
            and len(entity["body"]) >= 2
            and entity["body"][-2:] == "'s"
        )


class BootstrapAnnotator(Annotator):
    """Bootstrap Annotator class used to generate annotations based on existing annotations."""

    def __init__(self, *args, **kwargs):
        """Initializes a BootstrapAnnotator.

        Args:
            app_path (str): The location of the MindMeld app.
            annotation_rules (list): List of Annotation rules.
            confidence_threshold (float): The minimum confidence value to accept a detected entity.
            language (str, optional): Language as specified using a 639-1/2 code.
            locale (str, optional): The locale representing the ISO 639-1 language code and \
                ISO3166 alpha 2 country code separated by an underscore character.
            overwrite (bool): Whether to overwrite existing annotations with conflicting spans.
            unannotate_supported_entities_only (bool): Only allow removal of supported entities.
            unannotation_rules (list): List of Annotation rules.
        """
        super().__init__(*args, **kwargs)
        self.confidence_threshold = kwargs.get("confidence_threshold", 0)
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            raise ValueError(
                "{!r} is not a valid confidence threshold. Select a value between 0 and 1.".format(
                    self.confidence_threshold
                )
            )
        logger.info("BootstrapAnnotator is loading %s.", self.app_path)
        self.nlp = NaturalLanguageProcessor(self.app_path)
        self.nlp.build()

    def parse(self, sentence, entity_types, domain: str, intent: str, **kwargs):
        """
        Args:
            sentence (str): Sentence to detect entities.
            entity_types (list): List of entity types to parse. If None, all
                    possible entity types will be parsed.
            domain (str): Allowed domain.
            intent (str): Allowed intent.

        Returns:
            query_entities (list): List of QueryEntity objects.
        """
        response = self.nlp.process(
            sentence, allowed_nlp_classes={domain: {intent: {}}}, verbose=True
        )
        entities = []
        for i, entity in enumerate(response["entities"]):
            if not entity_types or entity["type"] in entity_types:
                entity_confidence = response["confidences"]["entities"][i][
                    entity["type"]
                ]
                if entity_confidence >= self.confidence_threshold:
                    entities.append(
                        {
                            "body": entity["text"],
                            "start": entity["span"]["start"],
                            "end": entity["span"]["end"] + 1,
                            "dim": entity["type"],
                            "value": entity["value"],
                            "role": entity["role"],
                        }
                    )
        processed_query = load_query(
            sentence,
            query_factory=self._resource_loader.query_factory,
            domain=kwargs.get("domain"),
            intent=kwargs.get("intent"),
        )
        return [
            Annotator._item_to_query_entity(entity, processed_query)
            for entity in entities
        ]

    @property
    def supported_entity_types(self):  # pylint: disable=W0236
        """
        Returns:
            supported_entity_types (list): List of supported entity types.
        """
        return get_entity_types(self.app_path)

    def valid_entity_check(self, entity):
        """Determine if an entity type is valid.

        Args:
            entity (str): Name of entity to annotate.

        Returns:
            bool: Whether entity is valid.
        """
        entity = entity.lower().strip()
        return Entity.is_system_entity(entity) or entity in self.supported_entity_types


class NoTranslationDucklingAnnotator(Annotator):
    """The NoTranslationDucklingAnnotator detects entities by filtering non-English candidates
    from Duckling to a set containing the largest non-overlapping spans.

    Unlike the TranslationDucklingAnnotator, this annotator does not use a translation service.
    Unlike the MultiLingualAnnotator, this annotator does not use non-English Spacy NER models.
    """

    def __init__(self, *args, **kwargs):
        """Initializes a NoTranslationDucklingAnnotator.

        Args:
            app_path (str): The location of the MindMeld app.
            annotation_rules (list): List of Annotation rules.
            language (str, optional): Language as specified using a 639-1/2 code.
            locale (str, optional): The locale representing the ISO 639-1 language code and \
                ISO3166 alpha 2 country code separated by an underscore character.
            overwrite (bool): Whether to overwrite existing annotations with conflicting spans.
            unannotate_supported_entities_only (bool): Only allow removal of supported entities.
            unannotation_rules (list): List of Annotation rules.
        """
        super().__init__(*args, **kwargs)

    def parse(self, sentence, entity_types=None, **kwargs):
        """
        Args:
            sentence (str): Sentence to detect entities.
            entity_types (list): List of entity types to parse. If None, all
                    possible entity types will be parsed.
        Returns:
            query_entities (list): List of QueryEntity objects.
        """
        duckling_candidates = self.duckling.get_candidates_for_text(
            sentence,
            entity_types=entity_types,
            language=self.language,
            locale=self.locale,
        )
        filtered_candidates = NoTranslationDucklingAnnotator._filter_out_bad_duckling_candidates(
            duckling_candidates
        )
        spans = [
            Span(candidate["start"], candidate["end"] - 1)
            for candidate in filtered_candidates
        ]
        final_spans = NoTranslationDucklingAnnotator._get_largest_non_overlapping_candidates(
            spans
        )
        final_candidates = []
        for span in final_spans:
            for candidate in filtered_candidates:
                if span == Span(candidate["start"], candidate["end"] - 1):
                    final_candidates.append(candidate)
                    break
        if entity_types:
            final_candidates = [
                e for e in final_candidates if e["entity_type"] in entity_types
            ]
        query = self._resource_loader.query_factory.create_query(sentence)
        return [
            duckling_item_to_query_entity(query, candidate)
            for candidate in final_candidates
        ]

    @property
    def supported_entity_types(self):  # pylint: disable=W0236
        """
        Returns:
            supported_entity_types (list): List of supported entity types.
        """
        return DUCKLING_TO_SYS_ENTITY_MAPPINGS[self.language]

    @staticmethod
    def _get_largest_non_overlapping_candidates(spans):
        """Finds the set of the largest non-overlapping candidates.

        Args:
            spans (list): List of tuples representing candidate spans (start_index, end_index + 1).
        Returns:
            selected_spans (list): List of the largest non-overlapping spans.
        """
        spans.sort(reverse=True)
        selected_spans = []
        for span in spans:
            has_overlaps = [
                span.has_overlap(selected_span) for selected_span in selected_spans
            ]
            if not any(has_overlaps):
                selected_spans.append(span)
        return selected_spans

    @staticmethod
    def _filter_out_bad_duckling_candidates(candidates):
        """Pipeline function to filter initial list of duckling candidates using heuristics.

        Args:
            candidates (list): List of duckling candidates
        Returns:
            filtered_candidates (list): List of filtered duckling candidates.
        """
        filtered_candidates = NoTranslationDucklingAnnotator._remove_unresolved_sys_amount_of_money(
            candidates
        )
        return filtered_candidates

    @staticmethod
    def _remove_unresolved_sys_amount_of_money(candidates):
        """Do not label candidate entities that are sys_amount-of-money but
        do not have an "unknown" unit type.
        """
        return [
            candidate
            for candidate in candidates
            if not (
                candidate["dim"] == "amount-of-money"
                and candidate["value"].get("unit") == "unknown"
            )
        ]


class TranslationDucklingAnnotator(Annotator):
    """ The TranslationDucklingAnnotator detects entities in non-English sentences using
    a translation service and Duckling by following these steps:
        1. The non-English sentence is translated to English.
        2. Spacy detects entities in the translated English sentence.
        3. Duckling detects non-English entities in the non-English sentence.
        4. A heuristic in parse() is used to match and filer the non-English entities
        against the English entities.
        5. The final set of filtered non-English entities are returned.
    Unlike the NoTranslationDucklingAnnotator, this annotator uses a translation service.
    Unlike the MultiLingualAnnotator, this annotator does not use non-English Spacy NER models.
    """

    def __init__(self, *args, **kwargs):
        """Initializes a TranslationDucklingAnnotator.

        Args:
            app_path (str): The location of the MindMeld app.
            annotation_rules (list): List of Annotation rules.
            en_annotator (SpacyAnnotator): A Spacy Annotator with language set to English ("en").
            translator (str): A translator to use such as 'GoogleTranslator' or 'NoOpTranslator'.
            language (str, optional): Language as specified using a 639-1/2 code.
            locale (str, optional): The locale representing the ISO 639-1 language code and \
                ISO3166 alpha 2 country code separated by an underscore character.
            overwrite (bool): Whether to overwrite existing annotations with conflicting spans.
            unannotate_supported_entities_only (bool): Only allow removal of supported entities.
            unannotation_rules (list): List of Annotation rules.
        """
        super().__init__(*args, **kwargs)
        assert (
            self.language != ENGLISH_LANGUAGE_CODE
        ), "The 'language' for a TranslationDucklingAnnotator cannot be set to English."
        translator = kwargs.get("translator")
        if not translator:
            raise AssertionError("'translator' cannot be None.")
        elif translator == NoOpTranslator.__name__:
            raise AssertionError(
                "The 'translator' for a TranslationDucklingAnnotator cannot "
                f"be set to {NoOpTranslator.__name__}."
            )
        self.translator = TranslatorFactory().get_translator(translator)
        self.en_annotator = kwargs.get("en_annotator") or SpacyAnnotator(
            app_path=self.app_path,
            language=ENGLISH_LANGUAGE_CODE,
            locale=ENGLISH_US_LOCALE,
        )

    def parse(self, sentence, entity_types=None, **kwargs):
        """ Implements a heuristic to match English entities detected by Spacy on the
        translated non-English sentence against the non-English entities detected by
        Duckling on the non-English sentence.

        Args:
            sentence (str): Sentence to detect entities.
            entity_types (list): List of entity types to parse. If None, all
                    possible entity types will be parsed.
        Returns:
            query_entities (list): List of QueryEntity objects.
        """
        candidates = self.en_annotator.duckling.get_candidates_for_text(
            sentence,
            entity_types=entity_types,
            language=self.language,
            locale=self.locale,
        )
        en_sentence = self.translator.translate(  # pylint: disable=E1128
            sentence, target_language=ENGLISH_LANGUAGE_CODE
        )
        en_entities = self.en_annotator.parse(en_sentence, entity_types=entity_types)
        final_candidates = []
        for entity in en_entities:
            value_matched_candidates = []
            for candidate in candidates:
                # Skip the candidate if the type does not match
                if entity.entity.type != candidate["entity_type"]:
                    continue
                # Store the candidate if there is a value match
                if entity.entity.value == candidate["value"]:
                    value_matched_candidates.append(candidate)
                # Skip the the translation-match check if value-match candidates exist
                if value_matched_candidates:
                    continue
                # Check if the translated entity text matches candidate entity text
                if (
                    self.translator.translate(
                        entity.entity.text, target_language=self.language
                    )
                    == candidate["body"]
                ):
                    final_candidates.append(candidate)
                    break
            # Select the largest of the candidates with a value match
            if value_matched_candidates:
                final_candidates.append(
                    max(value_matched_candidates, key=lambda x: len(x["body"]))
                )
        if entity_types:
            final_candidates = [
                e for e in final_candidates if e["entity_type"] in entity_types
            ]
        query = self._resource_loader.query_factory.create_query(sentence)
        return [
            duckling_item_to_query_entity(query, candidate)
            for candidate in final_candidates
        ]

    @property
    def supported_entity_types(self):  # pylint: disable=W0236
        """
        Returns:
            supported_entity_types (list): List of supported entity types.
        """
        supported_entity_types = set(
            self.en_annotator.supported_entity_types
        ).intersection(DUCKLING_TO_SYS_ENTITY_MAPPINGS[self.language])
        return list(supported_entity_types)


class MultiLingualAnnotator(Annotator):
    """ The MultiLingualAnnotator detects entities in English and non-English sentences.

    1. If the 'language' is English, this annotator solely uses the Spacy's English NER model to
        detect entities.
    2. If the 'language' is not English, this annotator will detect entities using both Spacy
        non-English NER models and a Duckling-based Annotator.
        A. The TranslationDucklingAnnotator will be used if a 'translator' service is available
        (E.g. "GoogleTranslator"). Non-English duckling candidates are matched to English
        entities detected by Spacy's English NER model.
        B. The NoTranslationDucklingAnnotator will be used if a 'translator' service is not
        available. The set of Non-English duckling candidates with the largest non-overlapping
        spans is selected.
    """

    def __init__(self, *args, **kwargs):
        """Initializes a TranslationDucklingAnnotator.

        Args:
            app_path (str): The location of the MindMeld app.
            annotation_rules (list): List of Annotation rules.
            en_annotator (SpacyAnnotator): A Spacy Annotator with language set to English ("en").
            translator (str): A translator to use such as 'GoogleTranslator' or 'NoOpTranslator'.
            language (str, optional): Language as specified using a 639-1/2 code.
            locale (str, optional): The locale representing the ISO 639-1 language code and \
                ISO3166 alpha 2 country code separated by an underscore character.
            overwrite (bool): Whether to overwrite existing annotations with conflicting spans.
            unannotate_supported_entities_only (bool): Only allow removal of supported entities.
            unannotation_rules (list): List of Annotation rules.
        """
        super().__init__(*args, **kwargs)
        self.translator = kwargs.get("translator", NoOpTranslator.__name__)
        self.en_annotator = SpacyAnnotator(
            app_path=self.app_path,
            language=ENGLISH_LANGUAGE_CODE,
            locale=ENGLISH_US_LOCALE,
        )
        if self.language != ENGLISH_LANGUAGE_CODE:
            self.duckling_annotator = self._get_duckling_annotator()
            self.non_en_annotator = SpacyAnnotator(
                app_path=self.app_path, language=self.language, locale=self.locale,
            )

    def _get_duckling_annotator(self):
        if self.translator != NoOpTranslator.__name__:
            return TranslationDucklingAnnotator(
                app_path=self.app_path,
                language=self.language,
                locale=self.locale,
                en_annotator=self.en_annotator,
                translator=self.translator,
            )
        return NoTranslationDucklingAnnotator(
            app_path=self.app_path, language=self.language, locale=self.locale,
        )

    def parse(self, sentence, entity_types=None, **kwargs):
        """
        Args:
            sentence (str): Sentence to detect entities.
            entity_types (list): List of entity types to parse. If None, all
                possible entity types will be parsed.
        Returns:
            query_entities (list): List of QueryEntity objects.
        """
        if self.language == ENGLISH_LANGUAGE_CODE:
            return self.en_annotator.parse(sentence, entity_types=entity_types)
        non_en_spacy_entities = self.non_en_annotator.parse(
            sentence, entity_types=entity_types
        )
        duckling_entities = self.duckling_annotator.parse(
            sentence, entity_types=entity_types
        )
        merged_entities = Annotator._resolve_conflicts(
            non_en_spacy_entities, duckling_entities
        )
        return merged_entities

    @property
    def supported_entity_types(self):  # pylint: disable=W0236
        """
        Returns:
            supported_entity_types (list): List of supported entity types.
        """
        if self.language == ENGLISH_LANGUAGE_CODE:
            return self.en_annotator.supported_entity_types
        supported_entities = set(self.non_en_annotator.supported_entity_types)
        if self.language in DUCKLING_TO_SYS_ENTITY_MAPPINGS:
            supported_entities.update(self.duckling_annotator.supported_entity_types)
        return supported_entities


def register_all_annotators():
    register_annotator("SpacyAnnotator", SpacyAnnotator)
    register_annotator("BootstrapAnnotator", BootstrapAnnotator)
    register_annotator("MultiLingualAnnotator", MultiLingualAnnotator)
