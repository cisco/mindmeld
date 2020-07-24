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
from os import walk
import re
import logging
import spacy

from .resource_loader import ResourceLoader
from .components._config import get_sys_annotator_config
from .system_entity_recognizer import DucklingRecognizer
from .markup import load_query_file, dump_queries
from .core import Entity, Span, QueryEntity


logger = logging.getLogger(__name__)

class Annotator(ABC):
    """
    Abstract Annotator class that can be used to build a Custom Annotation class.
    """
    
    def __init__(self, app_path, config=None, resource_loader=None, **kwargs):
        """Initializes an annotator."""
        self.app_path = app_path
        self.config = get_sys_annotator_config(app_path=app_path,config=config)
        self._resource_loader = (
            resource_loader or ResourceLoader.create_resource_loader(app_path)
        )
        self.file_entities_map = self._get_file_entities_map()
        
    def _get_file_entities_map(self):
        all_file_paths = self._resource_loader.get_all_file_paths()
        file_entities_map = {path:[] for path in all_file_paths}

        # TODO: CHANGE TO UPDATE BY MOST SPECIFIC RULE INSTEAD OF OVERWRITE
        for rule in self.config["annotate"]:
            pattern = self._get_pattern(rule) 
            filtered_paths = self._resource_loader.filter_file_paths(
                file_pattern=pattern, file_paths=all_file_paths 
            )
            for path in filtered_paths:
                entities = self._get_entities(rule)
                file_entities_map[path] = entities

        return file_entities_map

    def _get_pattern(self, rule):
        pattern = "/".join(rule.split("/")[:-1])
        pattern = pattern.replace("*", ".+")
        return ".*/" + pattern

    def _get_entities(self, rule):
        entities = rule.split("/")[-1]
        entities = re.sub('[()]',"", entities).split("|")
        # TODO: ADD CHECK FOR VALID ENTITY
        return entities
    
    def annotate(self):
        """ Annotate data based on configurations in the config.py file.
        """
        for path in self.file_entities_map:
            processed_queries = load_query_file(
                file_path=path, app_path=self.app_path
            )
            for processed_query in processed_queries:
                entity_types = self.file_entities_map[path]
                self._annotate_query(
                    processed_query=processed_query, entity_types=entity_types
                )
            annotated_queries = [query for query in dump_queries(processed_queries)]
            with open(path, "w") as outfile:
                outfile.write("\n".join(annotated_queries))
                outfile.close()

    def _annotate_query(self, processed_query, entity_types):
        current_entities = list(processed_query.entities)
        annotated_entities = self._get_annotated_entities(
            processed_query=processed_query, entity_types=entity_types
        )
        final_entities = self._resolve_conflicts(
            current_entities=current_entities, annotated_entities=annotated_entities
        )
        processed_query.entities = tuple(final_entities)

    def _get_annotated_entities(self, processed_query, entity_types=None):
        if len(entity_types) == 0:
            return []
        entity_types = None if entity_types == ["*"] else entity_types
        items = self.parse(
            sentence = processed_query.query.text, entity_types=entity_types
        )
        query_entities = [self._item_to_query_entity(item, processed_query) for item in items]
        return query_entities if len(query_entities) > 0 else []

    def _item_to_query_entity(self, item, processed_query):
        span = Span(
            start=item["start"], end=item["end"] - 1
        )
        entity = Entity(
            text=item["body"], entity_type=item["dim"], value=item["value"]
        )
        query_entity = QueryEntity.from_query(
            query=processed_query.query, span=span, entity=entity
        )
        return query_entity

    def _resolve_conflicts(self, current_entities, annotated_entities):
        final = []
        while(len(current_entities) > 0 or len(annotated_entities) > 0):
            if not current_entities:
                return final + annotated_entities
            if not annotated_entities:
                return final + current_entities

            curr_entity = current_entities[0]
            annot_entity = annotated_entities[0]    

            if curr_entity.span.end < annot_entity.span.start:
                final.append(curr_entity)
                current_entities.pop(0)
            elif annot_entity.span.end < curr_entity.span.start:
                final.append(annot_entity)
                annotated_entities.pop(0)
            else:
                overwrite = self.config["overwrite"]
                entity = annot_entity if overwrite else curr_entity
                final.append(entity)
                annotated_entities.pop(0)
                current_entities.pop(0)
        return final

    @abstractmethod
    def parse(self, sentence, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

class SpacyAnnotator(Annotator):
    """ Annotator class that uses spacy to generate annotations.
    """
    def __init__(self, app_path, config=None, model="en_core_web_lg", **kwargs):
        super().__init__(app_path=app_path, config=config, **kwargs)
        logger.info("Loading spacy model %s.", model)
        self.nlp = spacy.load(model)
        self.model = model
        self.duckling = DucklingRecognizer.get_instance()
        self.SYS_MAPPINGS = {
                        "money": "sys_amount-of-money",
                        "cardinal": "sys_number",
                        "ordinal": "sys_ordinal",
                        "person": "sys_person",
                        "percent": "sys_percent",
                        "distance": "sys_distance",
                        "quantity": "sys_weight"
                    }

    def parse(
        self,
        sentence,
        entity_types=None
    ):
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
                entity = self._resolve_money(entity)
            elif entity["dim"] == "ordinal":
                entity = self._resolve_ordinal(entity)
            elif entity["dim"] == "quantity":
                entity = self._resolve_quantity(entity)
            elif entity["dim"] == "percent":
                entity = self._resolve_percent(entity)
            elif entity["dim"] == "person":
                entity = self._resolve_person(entity)
            else:
                entity["dim"] = "sys_" + entity["dim"]
            
            if entity:
                entities.append(entity)

        if entity_types:
            entities = [e for e in entities if e["dim"] in entity_types]
        
        return entities

    def _resolve_time_date(self, entity, entity_types=None):
        """ Heuristic is to assign value if there is an exact body match. Order of priority
        is duration, interval, time."""
        candidates = self.duckling.get_candidates_for_text(entity["body"])

        if len(candidates) == 0:
            entity["dim"] = "spacy_time"
            return entity
        
        time_entities = ["sys_duration", "sys_interval", "sys_time"]
        if entity_types:
            time_entities = [e for e in time_entities if e in entity_types]
        
        if self._resolve_time_exact_match(entity, candidates, time_entities):
            return entity
        elif self._resolve_time_largest_substring(entity, candidates, time_entities):
            return entity
        else:
            print("spacy_time", entity["body"])
            entity["dim"] = "spacy_time"
            return entity
        
    def _get_time_entity_type(self, candidate):
        if candidate["dim"] == "duration":
            return "sys_duration"
        if candidate["dim"] == "time":
            if candidate["value"]["type"] == "interval":
                return "sys_interval"
            else:
                return "sys_time"
            
    def _resolve_time_exact_match(self, entity, candidates, time_entities):
        for candidate in candidates:
            candidate_entity = self._get_time_entity_type(candidate)
            if ( 
                candidate_entity in time_entities and
                candidate["body"] == entity["body"]
            ):
                entity["dim"] = candidate_entity
                entity["value"] = candidate["value"]
                return entity
    
    def _resolve_time_largest_substring(self, entity, candidates, time_entities):
        for time_entity in time_entities:
            largest_candidate = None
            for candidate in candidates:
                candidate_entity = self._get_time_entity_type(candidate)
                if ( 
                    candidate_entity == time_entity and
                    candidate["body"] in entity["body"] and
                    (
                        not largest_candidate or
                        len(candidate["body"]) > len(largest_candidate["body"])
                    )                                                                   
                ):
                    largest_candidate = candidate
            if largest_candidate:
                entity["body"] = largest_candidate["body"]
                offset = entity["start"]
                entity["start"] = offset + largest_candidate["start"]
                entity["end"] = offset + largest_candidate["end"]
                entity["value"] = largest_candidate["value"]
                entity["dim"] = time_entity
                return entity
            
    def _resolve_cardinal(self, entity):
        return self._resolve_exact_match(entity)
    
    def _resolve_money(self, entity):
        # TODO: Check if a '$' is infront of the token
        return self._resolve_exact_match(entity)
        
    def _resolve_ordinal(self, entity):
        return self._resolve_exact_match(entity)
            
    def _resolve_exact_match(self, entity):
        entity["dim"] = self.SYS_MAPPINGS[entity["dim"]]
        
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
        candidates = self.duckling.get_candidates_for_text(entity["body"])
        if len(candidates) == 0:
            entity["dim"] = "spacy_quantity"
            return entity

        for entity_type in ["distance", "quantity"]:
            for candidate in candidates:
                if (
                    candidate["dim"] == entity_type and
                    candidate["body"] == entity["body"]
                ):
                    entity["value"] = candidate["value"]
                    entity["dim"] = self.SYS_MAPPINGS[entity_type]
                    return entity

        print("spacy_quantity", entity["body"])
        entity["dim"] = "spacy_quantity"
        return entity
    
    def _resolve_percent(self, entity):
        entity["dim"] = self.SYS_MAPPINGS[entity["dim"]]
        
        candidates = self.duckling.get_candidates_for_text(entity["body"])
        if len(candidates) == 0:
            return

        possible_values = []
        for candidate in candidates:
            if candidate["entity_type"] == "sys_number":
                possible_values.append(candidate["value"]["value"])
        entity['value']['value'] = max(possible_values)/100
        return entity

    
    def _resolve_person(self, entity):
        entity["dim"] = self.SYS_MAPPINGS[entity["dim"]]
        
        if len(entity["body"]) >= 2 and entity["body"][-2:] == "'s":
            entity["value"] = entity["body"][:-2]
            entity["body"] = entity["body"][:-2]
            entity["end"] -= 2
        return entity
