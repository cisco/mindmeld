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
projects into mindmeld projects"""

from abc import ABC, abstractmethod
import sys, os, re, yaml

class Converter(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def convert_project(self):
        pass

    @abstractmethod
    def create_mindmeld_directory(self):
        pass

    @abstractmethod
    def create_training_data(self):
        pass

    @abstractmethod
    def create_main(self):
        pass
    
    @abstractmethod
    def create_init(self):
        pass

    @abstractmethod
    def create_config(self):
        pass
    
    @staticmethod
    def create_directory(directory):
        if not os.path.isdir(directory):
            try:
                os.mkdir(directory)
            except OSError as e:
                print("Cannot create directory at %s" % directory)
                print(e)