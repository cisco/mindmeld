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

"""This module contains the abstract Coverter class used to convert other software's
projects into MindMeld projects"""

import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Converter(ABC):
    """Abstract class that is used to instantiate concrete converter classes.
    The class contains the necessary functions to convert other software's projects
    into MindMeld projects."""

    def __init__(self):
        pass

    @abstractmethod
    def convert_project(self):
        """Converts project into MindMeld project."""
        pass

    @abstractmethod
    def create_mindmeld_directory(self):
        """Creates key MindMeld folders for project."""
        pass

    @abstractmethod
    def create_mindmeld_training_data(self):
        """Converts traning data from other software into MindMeld training data."""
        pass

    @abstractmethod
    def create_mindmeld_init(self):
        """Creates MindMeld __init__.py file."""
        pass

    @staticmethod
    def create_directory(directory):
        """Creates folder at specified location.

        Args:
            directory: Location where folder should be created.
        """
        if not os.path.isdir(directory):
            try:
                os.mkdir(directory)
            except OSError:
                logger.error("Cannot create directory at %s", directory)

    @staticmethod
    def create_main(mindmeld_project_directory, main_file_loc):
        """Creates __main__.py file for MindMeld project.

        Args:
            mindmeld_project_directory: Location of MindMeld directory.
            main_file_loc: Location where default __main__.py is stored.
        """
        MINDMELD_MODEL_MAIN_FILE_NAME = "/template_main.txt"
        with open(main_file_loc + MINDMELD_MODEL_MAIN_FILE_NAME, "r") as f:
            string = f.read()
        with open(mindmeld_project_directory + "/__main__.py", "w") as f:
            f.write(string)
