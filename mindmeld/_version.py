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

"""Defines mindmeld version information"""
from __future__ import absolute_import, unicode_literals
import logging
import warnings

import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict

from .exceptions import MindMeldVersionWarning, MindMeldVersionError

current = '4.1.0'

logger = logging.getLogger(__name__)


def _get_mm_version():
    return current


def validate_mindmeld_version(app_path, raise_exception=False):
    """ Validate the application's mindmeld requirement
    """
    requirements = app_path + "/requirements.txt"
    try:
        with open(requirements) as f:
            lines = f.readlines()
    except (OSError, IOError):
        logger.warning('requirements.txt is missing at %s.', app_path)
        return
    mm_req = None
    for item in pkg_resources.parse_requirements(lines):
        if item.name == 'mindmeld':
            mm_req = item
    if not mm_req:
        logger.warning('mindmeld is not in requirements.txt.')
        return
    if len(mm_req.specifier) == 0:
        logger.warning('mindmeld version is not specified in requirements.txt.')
        return
    mm_version = _get_mm_version()
    mm_req = [mm_req.name + str(mm_req.specifier)]
    try:
        pkg_resources.require(mm_req)
    except (DistributionNotFound, VersionConflict) as error:
        error_msg = 'Current mindmeld ({version}) does not satisfy ' \
                    '{condition} in pip requirements caused by ' \
                    '{cause}'.format(version=mm_version, condition=mm_req[0], cause=str(error))
        if raise_exception:
            raise MindMeldVersionError(error_msg)
        else:
            warnings.warn(error_msg, category=MindMeldVersionWarning)
            return
    logger.debug("mindmeld version %s satisfies app's requirements.txt.", mm_version)
