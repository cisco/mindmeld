# -*- coding: utf-8 -*-
"""Defines mmworkbench version information"""
from __future__ import absolute_import, unicode_literals
import os
import pip
import pkg_resources
import logging

from pkg_resources import DistributionNotFound, VersionConflict
from .exceptions import WorkbenchVersionError

current = '3.3.0rc4'

logger = logging.getLogger(__name__)


def _get_wb_version():
    import mmworkbench as wb
    return wb.__version__


def validate_workbench_version(app_path):
    """ Validate the application's mmworkbench requirement
    """
    requirements = app_path + "/requirements.txt"
    if not os.path.isfile(requirements):
        logger.warning('requirements.txt is missing at {app_path}.'.format(app_path=app_path))
        return
    wb_req = None
    for item in pip.req.parse_requirements(requirements, session='wb_session'):
        if item.name == 'mmworkbench':
            wb_req = item
    if not wb_req:
        logger.warning('mmworkbench is not in requirements.txt.')
        return
    if not (wb_req.req and len(wb_req.req.specifier) > 0):
        logger.warning('mmworkbench version is not specified in requirements.txt.')
        return
    wb_version = _get_wb_version()
    wb_req = [wb_req.name + str(wb_req.req.specifier)]
    try:
        pkg_resources.require(wb_req)
    except (DistributionNotFound, VersionConflict) as error:
        error_msg = 'Current mworkbench ({version}) does not satisfy ' \
                    '{condition} in pip requirements caused by ' \
                    '{cause}'.format(version=wb_version, condition=wb_req[0], cause=str(error))

        raise WorkbenchVersionError(error_msg)
    logger.debug("mmworkbench version {version} satisfies app's requirements.txt.".format(
        version=wb_version))
