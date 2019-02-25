# -*- coding: utf-8 -*-
"""Defines mmworkbench version information"""
from __future__ import absolute_import, unicode_literals
import logging
import warnings

import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict

from .exceptions import WorkbenchVersionWarning, WorkbenchVersionError

current = '4.0.0rc3'

logger = logging.getLogger(__name__)


def _get_wb_version():
    import mmworkbench as wb
    return wb.__version__


def validate_workbench_version(app_path, raise_exception=False):
    """ Validate the application's mmworkbench requirement
    """
    requirements = app_path + "/requirements.txt"
    try:
        with open(requirements) as f:
            lines = f.readlines()
    except (OSError, IOError):
        logger.warning('requirements.txt is missing at {app_path}.'.format(app_path=app_path))
        return
    wb_req = None
    for item in pkg_resources.parse_requirements(lines):
        if item.name == 'mmworkbench':
            wb_req = item
    if not wb_req:
        logger.warning('mmworkbench is not in requirements.txt.')
        return
    if len(wb_req.specifier) == 0:
        logger.warning('mmworkbench version is not specified in requirements.txt.')
        return
    wb_version = _get_wb_version()
    wb_req = [wb_req.name + str(wb_req.specifier)]
    try:
        pkg_resources.require(wb_req)
    except (DistributionNotFound, VersionConflict) as error:
        error_msg = 'Current mmworkbench ({version}) does not satisfy ' \
                    '{condition} in pip requirements caused by ' \
                    '{cause}'.format(version=wb_version, condition=wb_req[0], cause=str(error))
        if raise_exception:
            raise WorkbenchVersionError(error_msg)
        else:
            warnings.warn(error_msg, category=WorkbenchVersionWarning)
            return
    logger.debug("mmworkbench version {version} satisfies app's requirements.txt.".format(
        version=wb_version))
