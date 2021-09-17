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

"""This module contains the data augmentation processes for MindMeld."""

import logging
import re
import string
import random
import warnings

from abc import ABC, abstractmethod
from tqdm import tqdm

from ._util import get_pattern, read_path_queries, write_to_file
from .components._util import _is_module_available, _get_module_or_attr
from .models.helpers import register_augmentor, AUGMENTATION_MAP
from .markup import load_query, dump_query
from .core import Entity, Span, QueryEntity, ProcessedQuery, _get_overlap
from .query_factory import QueryFactory

logger = logging.getLogger(__name__)

# pylint: disable=R0201

SUPPORTED_LANGUAGE_CODES = ["en", "es", "fr", "it", "pt", "ro"]
EOS_TOKEN = '</s>'

class UnsupportedLanguageError(Exception):
    pass


class AugmentorFactory:
    """Creates an Augmentor object.

    Attributes:
        config (dict): A model configuration.
        language (str): Language for data augmentation.
        resource_loader (object): Resource Loader object for the application.
    """
    def __init__(self, config, language, resource_loader):
        self.config = config
        self.language = language
        self.resource_loader = resource_loader

    def create_augmentor(self):
        """Creates an augmentor instance using the provided configuration

        Returns:
            Augmentor: An Augmentor class

        Raises:
            ValueError: When model configuration is invalid or required key is missing
        """
        if "augmentor_class" not in self.config:
            raise KeyError(
                "Missing required argument in AUGMENTATION_CONFIG: 'augmentor_class'"
            )
        try:
            # Validate configuration input
            app_path = self.config.get("app_path")
            batch_size = self.config.get("batch_size", 8)
            paths = self.config.get(
                "paths",
                [
                    {
                        "domains": ".*",
                        "intents": ".*",
                        "files": ".*",
                    }
                ],
            )
            path_suffix = self.config.get("path_suffix", "-augment.txt")
            retain_entities = self.config.get("retain_entities", False)
            register_all_augmentors()
            return AUGMENTATION_MAP[self.config["augmentor_class"]](
                app_path=app_path,
                batch_size=batch_size,
                language=self.language,
                retain_entities=retain_entities,
                paths=paths,
                path_suffix=path_suffix,
                resource_loader=self.resource_loader,
            )
        except KeyError as e:
            msg = "Invalid model configuration: Unknown model type {!r}"
            raise ValueError(msg.format(self.config["augmentor_class"])) from e


class Augmentor(ABC):
    """
    Abstract Augmentor class.
    """

    def __init__(self, language, paths, path_suffix, resource_loader):
        """Initializes an augmentor.

        Args:
            language (str): The language code for paraphrasing
            paths (list): Path rules for fetching relevant files to Paraphrase.
            path_suffix (str): Suffix to be added to new augmented files.
            resource_loader (object): Resource Loader object for the application.
        """
        self.language_code = language
        self.files_to_augment = paths
        self.path_suffix = path_suffix
        self._resource_loader = resource_loader
        self._check_dependencies()
        self._check_language_support()

    def _check_dependencies(self):
        """Checks module dependencies."""
        if not _is_module_available("torch"):
            raise ModuleNotFoundError(
                "Library not found: 'torch'. Run 'pip install mindmeld[augment]' to install."
            )

        if not _is_module_available("transformers"):
            raise ModuleNotFoundError(
                "Library not found: 'transformers'. Run 'pip install mindmeld[augment]' to install."
            )

    def _check_language_support(self):
        """Checks if language is currently supported for augmentation."""
        if self.language_code not in SUPPORTED_LANGUAGE_CODES:
            raise UnsupportedLanguageError(
                f"'{self.language_code}' is not supported yet. "
                "English (en), French (fr), and Italian (it), Portuguese (pt), Romanian (ro) "
                " and Spanish (es) are currently supported."
            )

    def augment(self, **kwargs):
        """Augments queries given initial queries in application."""
        filtered_paths = self._get_files(path_rules=self.files_to_augment)

        for path in tqdm(filtered_paths):
            queries = read_path_queries(path)
            # To-Do: Use generator to write files incrementally.
            augmented_queries = self.augment_queries(queries, **kwargs)
            write_to_file(path, augmented_queries, suffix=self.path_suffix)

    @abstractmethod
    def augment_queries(self, queries):
        """Generates augmented data given application queries.

        Args:
            queries (list): List of queries.

        Return:
            augmented_queries (list): List of augmented queries.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _validate_generated_query(self, query):
        """Validates whether augmented query has atleast one alphanumeric character

        Args:
            query (str): Generated query to be validated.
        """
        pattern = re.compile("^.*[a-zA-Z0-9].*$")
        return pattern.search(query) and True

    def _get_files(self, path_rules=None):
        """Fetches relevant files given the path rules specified in the config.

        Args:
            path_rules (list): Path rules for fetching relevant files.

        Return:
            filtered_paths (list): List of file paths to be augmeted.
        """
        all_file_paths = self._resource_loader.get_all_file_paths()

        if not path_rules:
            logger.warning(
                """'paths' field is not configured or misconfigured in the `config.py`.
                 Can't find files to augment."""
            )
            return []

        filtered_paths = []

        for rule in path_rules:
            pattern = get_pattern(rule)
            compiled_pattern = re.compile(pattern)
            filtered_paths.extend(
                self._resource_loader.filter_file_paths(
                    compiled_pattern=compiled_pattern, file_paths=all_file_paths
                )
            )
        return filtered_paths


class EnglishParaphraser(Augmentor):
    """Paraphraser class for generating English paraphrases."""

    def __init__(self, app_path, batch_size, language, retain_entities, paths, path_suffix, resource_loader):
        """Initializes an English paraphraser.

        Args:
            batch_size (int): Batch size for batch processing.
            language (str): The language code for paraphrasing.
            paths (list): Path rules for fetching relevant files to Paraphrase.
            path_suffix (str): Suffix to be added to new augmented files.
            resource_loader (object): Resource Loader object for the application.
        """
        super().__init__(
            language=language,
            paths=paths,
            path_suffix=path_suffix,
            resource_loader=resource_loader,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.query_factory = QueryFactory.create_query_factory(app_path=app_path, duckling=True)
        PegasusTokenizer = _get_module_or_attr("transformers", "PegasusTokenizer")
        PegasusForConditionalGeneration = _get_module_or_attr(
            "transformers", "PegasusForConditionalGeneration"
        )
        self.retain_entities = retain_entities
        if self.retain_entities:
            #model_name = "mindmeld/pegasus_paraphrase_entities"
            model_name = "/Users/vembar/repo/mindmeld/final_model"
        else:
            model_name = "tuner007/pegasus_paraphrase"
        torch_device = (
            "cuda" if _get_module_or_attr("torch.cuda", "is_available")() else "cpu"
        )
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

        # Update default params with user model config
        self.batch_size = batch_size

        self.default_paraphraser_model_params = {
            "max_length": 60,
            "num_beams": 10,
            "num_return_sequences": 10,
            "temperature": 1.5,
        }

        self.default_tokenizer_params = {
            "truncation": True,
            "padding": "longest",
            "max_length": 60,
        }

    def _prepare_inputs(self, queries):
        model_inputs = []
        processed_entities = []
        for query in queries:
            processed_query = load_query(query)
            if self.retain_entities:
                text = [processed_query.query.text.strip().lower(), EOS_TOKEN]
                for entity in processed_query.entities:
                    text.append(entity.text.lower())
                processed_entities.append(processed_query)
                model_inputs.append(' '.join(text))
            else:
                model_inputs.append(processed_query.query.text.strip())
        return model_inputs, processed_entities

    def _replace_with_random_entity(self, paraphrase_text, entity_matches):
        new_paraphrase_text = []
        entity_matches.sort(key=lambda x: x[1].start, reverse=False)
        running_start = 0
        previous_end = 0
        replaced_spans_entities = []
        for (entity, span) in entity_matches:
            if not running_start:
                running_start = span.start
                new_paraphrase_text.append(paraphrase_text[:running_start])
            else:
                running_start += (span.start - previous_end)
                new_paraphrase_text.append(paraphrase_text[previous_end:span.start])
            gaz = self._resource_loader.get_gazetteer(entity.type)['entities']
            if gaz:
                random_entity = random.sample(gaz, 1)[0]
                new_span = Span(start=running_start, end=running_start + len(random_entity) - 1)
                new_entity = Entity(text=random_entity, entity_type=entity.type, role=None, value=None)
            else:
                new_entity = entity
                new_span = Span(start=running_start, end=running_start + len(entity.text) - 1)
            running_start += len(new_entity.text)
            replaced_spans_entities.append([new_entity, new_span])
            new_paraphrase_text.append(new_entity.text)
            previous_end = span.end + 1
        new_paraphrase_text.append(paraphrase_text[previous_end:])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            processed_query = self.query_factory.create_query(''.join(new_paraphrase_text))
        final_entities = []
        for (entity, span) in replaced_spans_entities:
            query_entity = QueryEntity.from_query(query=processed_query, span=span, entity=entity)
            final_entities.append(query_entity)
        return ProcessedQuery(query=processed_query, entities=tuple(final_entities))

    def _annotate_entities(self, paraphrases, processed_queries):
        valid_paraphrases = []
        for i in range(len(processed_queries)):
            entities = sorted(list(processed_queries[i].entities), key=lambda x: len(x.text), reverse=True)
            queries = paraphrases[(i*10) : (i*10) + 10]
            for query in queries:
                all_matches = []
                for entity in entities:
                    found_matches = re.finditer(entity.text.lower(), query)
                    for match in found_matches:
                        matched_span = Span(start=match.start(0), end=match.end(0)-1)
                        matched_entity = Entity(text=match.group(0),
                                                entity_type=entity.entity.type,
                                                role=None, value=None)
                        no_overlaps = [not _get_overlap(m[1], matched_span) for m in all_matches]
                        if all(no_overlaps):
                            all_matches.append((matched_entity, matched_span))
                if len(all_matches) == len(entities):
                    processed_paraphrase = self._replace_with_random_entity(query, all_matches)
                    valid_paraphrases.append(dump_query(processed_paraphrase))
        return valid_paraphrases

    @staticmethod
    def _normalize_paraphrases(queries):
        without_puncts = [s.lower().translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
                          for s in queries]
        queries = [' '.join(s.split()) for s in without_puncts]
        return queries

    def _generate_paraphrases(self, queries):
        """Generates paraphrase responses for given query.

        Args:
            queries (list(str)): List of application queries.

        Return:
            paraphrases (list(str)): List of paraphrased queries.
        """
        all_generated_queries = []

        for pos in range(0, len(queries), self.batch_size):
            tokenizer_input, processed_input_queries = self._prepare_inputs(queries[pos : pos + self.batch_size])
            batch = self.tokenizer.prepare_seq2seq_batch(
                tokenizer_input,
                **self.default_tokenizer_params,
            )
            generated = self.model.generate(
                **batch,
                **self.default_paraphraser_model_params,
            )
            decoded_queries = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            if self.retain_entities:
                decoded_queries = self._normalize_paraphrases(decoded_queries)
                decoded_queries = self._annotate_entities(decoded_queries, processed_input_queries)
            all_generated_queries.extend(decoded_queries)
        return all_generated_queries

    def augment_queries(self, queries, **kwargs):
        augmented_queries = list(
            set(
                p.lower()
                for p in self._generate_paraphrases(queries, **kwargs)
                if self._validate_generated_query(p)
            )
        )
        return augmented_queries


class MultiLingualParaphraser(Augmentor):
    """Paraphraser class for generating paraphrases based on language code of the app
    (currently supports: French, Italian, Portuguese, Romanian and Spanish).
    """

    def __init__(self, batch_size, language, paths, path_suffix, resource_loader):
        """Initializes a multi-lingual paraphraser.

        Args:
            batch_size (int): Batch size for batch processing.
            language (str): The language code for paraphrasing.
            paths (list): Path rules for fetching relevant files to Paraphrase.
            path_suffix (str): Suffix to be added to new augmented files.
            resource_loader (object): Resource Loader object for the application.
        """
        super().__init__(
            language=language,
            paths=paths,
            path_suffix=path_suffix,
            resource_loader=resource_loader,
        )

        self.torch_device = (
            "cuda" if _get_module_or_attr("torch.cuda", "is_available")() else "cpu"
        )

        MarianTokenizer = _get_module_or_attr("transformers", "MarianTokenizer")
        MarianMTModel = _get_module_or_attr("transformers", "MarianMTModel")

        en_model_name = "Helsinki-NLP/opus-mt-ROMANCE-en"
        self.en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
        self.en_model = MarianMTModel.from_pretrained(en_model_name)
        self.en_model.to(self.torch_device)

        target_model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
        self.target_tokenizer = MarianTokenizer.from_pretrained(target_model_name)
        self.target_model = MarianMTModel.from_pretrained(target_model_name).to(
            self.torch_device
        )

        # Update default params with user model config
        self.batch_size = batch_size

        self.default_forward_params = {
            "max_length": 60,
            "num_beams": 5,
            "num_return_sequences": 5,
            "temperature": 1.0,
            "top_k": 0,
        }

        self.default_reverse_params = {
            "max_length": 60,
            "num_beams": 3,
            "num_return_sequences": 3,
            "temperature": 1.0,
            "top_k": 0,
        }

    def _translate(self, *, queries, model, tokenizer, **kwargs):
        """The core translation step for forward and reverse translation.

        Args:
            template (lambda func): Structure input text to model.
            queries (list(str)): List of input queries.
            model: Machine translation model (en-ROMANCE or ROMANCE-en).
            tokenizer: Language tokenizer for input query text.
        """
        all_translated_queries = []
        for pos in range(0, len(queries), self.batch_size):
            encoded = tokenizer.prepare_seq2seq_batch(
                queries[pos : pos + self.batch_size], return_tensors="pt"
            )
            for key in encoded:
                encoded[key] = encoded[key].to(self.torch_device)

            translated = model.generate(**encoded, **kwargs)
            translated_queries = tokenizer.batch_decode(
                translated, skip_special_tokens=True
            )
            all_translated_queries.extend(translated_queries)
        return all_translated_queries

    def augment_queries(self, queries):
        translated_queries = self._translate(
            queries=queries,
            model=self.en_model,
            tokenizer=self.en_tokenizer,
            **self.default_forward_params,
        )

        def template(text):
            return f">>{self.language_code}<< {text}"

        translated_queries = [template(query) for query in set(translated_queries)]

        reverse_translated_queries = self._translate(
            queries=translated_queries,
            model=self.target_model,
            tokenizer=self.target_tokenizer,
            **self.default_reverse_params,
        )
        augmented_queries = list(
            set(
                p.lower()
                for p in reverse_translated_queries
                if self._validate_generated_query(p)
            )
        )

        return augmented_queries


def register_all_augmentors():
    register_augmentor("EnglishParaphraser", EnglishParaphraser)
    register_augmentor("MultiLingualParaphraser", MultiLingualParaphraser)
