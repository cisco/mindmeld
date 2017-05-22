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


_BLUEPRINT_S3_URL_TEMPLATE = ('https://s3-us-west-2.amazonaws.com/mindmeld/workbench-data/'
                              'blueprints/')

_BLUEPRINTS = {
    'quick_start': {},
    'food_ordering': {}
}


class _Blueprint(object):
    def __call__(self, name, config=None):
        config = config or {}
        self.setup_app(name, config)
        self.setup_kb(name, config)

    @staticmethod
    def setup_kb(name, config):
        """Sets up the knowledge base for the specified blueprint.

        Args:
            name (str): The name of the blueprint

        Raises:
            ValueError: When an unknown blueprint is specified
        """
        # TODO: implement
        pass

    @staticmethod
    def setup_app(name, config):
        """Setups up the app folder for the specified blueprint.

        Args:
            name (str): The name of the blueprint

        Raises:
            ValueError: When an unknown blueprint is specified
        """
        if name not in _BLUEPRINTS:
            raise ValueError('Unknown blueprint name : {!r}'.format(name))

        cache_dir = path.get_cached_blueprint_path(name)
        try:
            os.makedirs(cache_dir)
        except FileExistsError:
            # dir already exists -- no worries
            pass

        filename = 'app.tar.gz'
        local_tarball = os.path.join(cache_dir, filename)
        app_dir = urljoin(_BLUEPRINT_S3_URL_TEMPLATE, name + '/')

        remote_tarball = urljoin(app_dir, filename)
        req = requests.head(remote_tarball)
        remote_modified = datetime(*parsedate(req.headers.get('last-modified'))[:6])
        try:
            local_modified = datetime.fromtimestamp(os.path.getmtime(local_tarball))
        except FileNotFoundError:
            local_modified = datetime.min

        if remote_modified < local_modified:
            logger.info('Using cached %r blueprint', name)
        else:
            logger.info('Fetching blueprint from %r', remote_tarball)
            urlretrieve(remote_tarball, local_tarball)

        shutil.unpack_archive(local_tarball, name)


blueprint = _Blueprint()  # pylint: disable=locally-disabled,invalid-name


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
