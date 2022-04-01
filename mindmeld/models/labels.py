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

"""This module contains classes related to encoding labels for models defined in the models
subpackage."""

import logging

from .helpers import register_label
from .taggers.taggers import (
    get_entities_from_tags,
    get_tags_from_entities,
)
from ..system_entity_recognizer import SystemEntityRecognizer

logger = logging.getLogger(__name__)


class LabelEncoder:
    """The label encoder is responsible for converting between rich label
    objects such as a ProcessedQuery and basic formats a model can interpret.

    A MindMeld model uses its label encoder at fit time to encode labels into a
    form it can deal with, and at predict time to decode predictions into
    objects
    """

    def __init__(self, config):
        """Initializes an encoder

        Args:
            config (ModelConfig): The model
        """
        self.config = config

    @staticmethod
    def encode(labels, **kwargs):
        """Transforms a list of label objects into a vector of classes.


        Args:
            labels (list): A list of labels to encode
        """
        del kwargs
        return labels

    @staticmethod
    def decode(classes, **kwargs):
        """Decodes a vector of classes into a list of labels

        Args:
            classes (list): A list of classes

        Returns:
            list: The decoded labels
        """
        del kwargs
        return classes


class EntityLabelEncoder(LabelEncoder):
    def __init__(self, config):
        """Initializes an encoder

        Args:
            config (ModelConfig): The model configuration
        """
        self.config = config

    def _get_tag_scheme(self):
        return self.config.model_settings.get("tag_scheme", "IOB").upper()

    def encode(self, labels, **kwargs):
        """Gets a list of joint app and system IOB tags from each query's entities.

        Args:
            labels (list): A list of labels associated with each query
            kwargs (dict): A dict containing atleast the "examples" key, which is a
                list of queries to process

        Returns:
            list: A list of list of joint app and system IOB tags from each
                query's entities
        """
        examples = kwargs["examples"]
        scheme = self._get_tag_scheme()
        # Here each label is a list of entities for the corresponding example
        all_tags = []
        for idx, label in enumerate(labels):
            all_tags.append(get_tags_from_entities(examples[idx], label, scheme))
        return all_tags

    def decode(self, tags_by_example, **kwargs):
        """Decodes the labels from the tags passed in for each query

        Args:
            tags_by_example (list): A list of tags per query
            kwargs (dict): A dict containing at least the "examples" key, which is a
                list of queries to process

        Returns:
            list: A list of decoded labels per query
        """
        examples = kwargs["examples"]
        labels = [
            get_entities_from_tags(examples[idx], tags, SystemEntityRecognizer.get_instance())
            for idx, tags in enumerate(tags_by_example)
        ]
        return labels


register_label("class", LabelEncoder)
register_label("entities", EntityLabelEncoder)
