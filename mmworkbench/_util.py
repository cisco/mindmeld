"""A module containing various utility functions for Workbench.
These are capabilities that do not have an obvious home within the existing
project structure.
"""

from __future__ import unicode_literals
from builtins import object

from datetime import datetime
from email.utils import parsedate
import logging
import os
import shutil

try:
    from urllib.request import urlretrieve
    from urllib.parse import urljoin
except ImportError:
    from urllib import urlretrieve
    from urlparse import urljoin

import requests

from . import path


logger = logging.getLogger(__name__)


BLUEPRINT_S3_URL_BASE = 'https://s3-us-west-2.amazonaws.com/mindmeld/workbench-data/blueprints/'
BLUEPRINT_APP_ARCHIVE = 'app.tar.gz'
BLUEPRINT_KB_ARCHIVE = 'kb.tar.gz'
BLUEPRINTS = {
    'quick_start': {},
    'food_ordering': {}
}


class Blueprint(object):

    def __call__(self, name, app_path=None):
        if name not in BLUEPRINTS:
            raise ValueError('Unknown blueprint name : {!r}'.format(name))
        app_path = self.setup_app(name, app_path)
        # self.setup_kb(name)  # need to implement this

        return app_path

    @classmethod
    def setup_app(cls, name, app_path=None):
        """Setups up the app folder for the specified blueprint.

        Args:
            name (str): The name of the blueprint

        Raises:
            ValueError: When an unknown blueprint is specified
        """
        if name not in BLUEPRINTS:
            raise ValueError('Unknown blueprint name : {!r}'.format(name))

        app_path = app_path or os.path.join(os.getcwd(), name)
        app_path = os.path.abspath(app_path)

        local_archive = cls._fetch_archive(name, 'app')
        shutil.unpack_archive(local_archive, app_path)
        return app_path

    @staticmethod
    def _fetch_archive(name, archive_type):
        cache_dir = path.get_cached_blueprint_path(name)
        try:
            os.makedirs(cache_dir)
        except FileExistsError:
            # dir already exists -- no worries
            pass

        filename = {'app': BLUEPRINT_APP_ARCHIVE, 'kb': BLUEPRINT_KB_ARCHIVE}.get(archive_type)
        local_archive = os.path.join(cache_dir, filename)
        blueprint_dir = urljoin(BLUEPRINT_S3_URL_BASE, name + '/')

        remote_archive = urljoin(blueprint_dir, filename)
        req = requests.head(remote_archive)
        remote_modified = datetime(*parsedate(req.headers.get('last-modified'))[:6])
        try:
        except FileNotFoundError:
            local_modified = datetime.min

        if remote_modified < local_modified:
            logger.info('Using cached %r %r', name, archive_type)
        else:
            logger.info('Fetching %r from %r', archive_type, remote_archive)
            urlretrieve(remote_archive, local_archive)
        return local_archive


blueprint = Blueprint()  # pylint: disable=locally-disabled,invalid-name




def configure_logs(**kwargs):
    """Helper method for easily configuring logs from the python shell.
    Args:
        level (TYPE, optional): A logging level recognized by python's logging module.
    """
    import sys
    level = kwargs.get('level', logging.INFO)
    log_format = kwargs.get('format', '%(message)s')
    logging.basicConfig(stream=sys.stdout, format=log_format)
    package_logger = logging.getLogger(__package__)
    package_logger.setLevel(level)
