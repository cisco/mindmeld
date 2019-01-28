# -*- coding: utf-8 -*-
"""A module containing various utility functions for Workbench.
These are capabilities that do not have an obvious home within the existing
project structure.
"""
import datetime
from email.utils import parsedate
import logging
import os
import shutil
import tarfile

from dateutil import tz
import py
import requests
from requests.auth import HTTPBasicAuth

from . import path
from .components import QuestionAnswerer
from .exceptions import AuthNotFoundError, KnowledgeBaseConnectionError
from .components._config import get_app_namespace
from .constants import DEVCENTER_URL


logger = logging.getLogger(__name__)

CONFIG_FILE_NAME = 'mmworkbench.cfg'
BLUEPRINT_URL = '{mindmeld_url}/bp/{blueprint}/{filename}'

BLUEPRINT_APP_ARCHIVE = 'app.tar.gz'
BLUEPRINT_KB_ARCHIVE = 'kb.tar.gz'
BLUEPRINTS = {
    'kwik_e_mart': {},
    'food_ordering': {},
    'home_assistant': {'kb': False},
    'template': {'kb': False},
    'video_discovery': {}
}


class Blueprint:
    """This is a callable class used to set up a blueprint app.

    The MindMeld website hosts the blueprints in a directory structure like this:

       - https://mindmeld.com/blueprints/
       |-food_ordering
       |  |-app.tar.gz
       |  |-kb.tar.gz
       |-quickstart
          |-app.tar.gz
          |-kb.tar.gz


    Within each blueprint directory `app.tar.gz` contains the workbench
    application data (`app.py`, `domains`, `entities`, etc.) and `kb.tar.gz`
    contains the a JSON file for each index in the knowledge base.

    The blueprint method will check the MindMeld website for when the blueprint
    files were last updated and compare that with any files in a local
    blueprint cache at ~/.mmworkbench/blueprints. If the cache is out of date,
    the updated archive is downloaded. The archive is then extracted into a
    directory named for the blueprint.
    """
    def __call__(self, name, app_path=None, es_host=None, skip_kb=False):
        """
        Args:
            name (str): The name of the blueprint
            app_path (str, optional): The path to the app
            es_host (str, optional): The hostname of the elasticsearch cluster
                for the knowledge base. If no value is passed, value will be
                read from the environment.
            skip_kb (bool, optional): If True, the blueprint knowledge base
                will not be set up.

        Returns:
            str: The path where the blueprint was created

        Raises:
            ValueError: When an unknown blueprint is specified.
        """
        if name not in BLUEPRINTS:
            raise ValueError('Unknown blueprint name: {!r}'.format(name))
        bp_config = BLUEPRINTS[name]

        app_path = self.setup_app(name, app_path)
        if bp_config.get('kb', True) and not skip_kb:
            self.setup_kb(name, app_path, es_host=es_host)
        return app_path

    @classmethod
    def setup_app(cls, name, app_path=None):
        """Setups up the app folder for the specified blueprint.

        Args:
            name (str): The name of the blueprint
            app_path (str, optional): The path to the app

        Raises:
            ValueError: When an unknown blueprint is specified
        """
        if name not in BLUEPRINTS:
            raise ValueError('Unknown blueprint name: {!r}'.format(name))

        app_path = app_path or os.path.join(os.getcwd(), name)
        app_path = os.path.abspath(app_path)

        local_archive = cls._fetch_archive(name, 'app')
        tarball = tarfile.open(local_archive)
        tarball.extractall(path=app_path)
        logger.info('Created %r app at %r', name, app_path)
        return app_path

    @classmethod
    def setup_kb(cls, name, app_path=None, es_host=None):
        """Sets up the knowledge base for the specified blueprint.

        Args:
            name (str): The name of the blueprint
            app_path (str, optional): The path to the app
            es_host (str, optional): The hostname of the elasticsearch cluster
                for the knowledge base. If no value is passed, value will be
                read from the environment.

        Raises:
            EnvironmentError: When no Elasticsearch host is specified directly
                or in the environment.
            ValueError: When an unknown blueprint is specified.
        """
        if name not in BLUEPRINTS:
            raise ValueError('Unknown blueprint name: {!r}'.format(name))

        app_path = app_path or os.path.join(os.getcwd(), name)
        app_path = os.path.abspath(app_path)
        app_namespace = get_app_namespace(app_path)
        es_host = es_host or os.environ.get('MM_ES_HOST', 'localhost')
        cache_dir = path.get_cached_blueprint_path(name)
        try:
            local_archive = cls._fetch_archive(name, 'kb')
        except ValueError:
            logger.warning('No knowledge base to set up.')
            return

        kb_dir = os.path.join(cache_dir, 'kb')
        tarball = tarfile.open(local_archive)
        tarball.extractall(path=kb_dir)

        _, _, index_files = next(os.walk(kb_dir))

        for index in index_files:
            index_name, _ = os.path.splitext(index)
            data_file = os.path.join(kb_dir, index)

            try:
                QuestionAnswerer.load_kb(app_namespace, index_name, data_file, es_host)
            except KnowledgeBaseConnectionError as ex:
                logger.error('Cannot set up knowledge base. Unable to connect to Elasticsearch '
                             'instance at %r. Confirm it is running or specify an alternate '
                             'instance with the MM_ES_HOST environment variable', es_host)

                raise ex

        logger.info('Created %r knowledge base at %r', name, es_host)

    @staticmethod
    def _fetch_archive(name, archive_type):
        """Fetches a blueprint archive from S3.

        Args:
            name (str): The name of the blueprint.
            archive_type (str): The type or the archive. Can be 'app' or 'kb'.

        Returns:
            str: The path of the local archive after it is downloaded.

        Raises:
            EnvironmentError: When AWS credentials are not available
        """
        cache_dir = path.get_cached_blueprint_path(name)
        try:
            os.makedirs(cache_dir)
        except (OSError, IOError):
            # dir already exists -- no worries
            pass

        filename = {'app': BLUEPRINT_APP_ARCHIVE, 'kb': BLUEPRINT_KB_ARCHIVE}.get(archive_type)

        local_archive = os.path.join(cache_dir, filename)

        config = load_global_configuration()
        mindmeld_url = config.get('mindmeld_url', DEVCENTER_URL)
        token = config.get('token', None)
        if token:
            username = 'token'
            password = token
        else:
            username = config['username']
            password = config['password']

        remote_url = BLUEPRINT_URL.format(mindmeld_url=mindmeld_url, blueprint=name,
                                          filename=filename)
        auth = HTTPBasicAuth(username, password)

        res = requests.head(remote_url, auth=auth)
        if res.status_code == 401:
            # authentication error
            msg = ('Invalid MindMeld credentials. Cannot download blueprint. Please confirm '
                   'they are correct and try again.')
            logger.error(msg)
            raise EnvironmentError(msg)
        if res.status_code != 200:
            # Unknown error
            msg = 'Unknown error fetching {} archive from {!r}'.format(archive_type, remote_url)
            logger.warning(msg)
            raise ValueError('Unknown error fetching archive')
        remote_modified = datetime.datetime(*parsedate(res.headers.get('last-modified'))[:6],
                                            tzinfo=tz.tzutc())
        try:
            local_modified = datetime.datetime.fromtimestamp(os.path.getmtime(local_archive),
                                                             tz.tzlocal())
        except (OSError, IOError):
            # File doesn't exist, use minimum possible time
            local_modified = datetime.datetime(datetime.MINYEAR, 1, 1, tzinfo=tz.tzutc())

        if remote_modified < local_modified:
            logger.info('Using cached %r %s archive', name, archive_type)
        else:
            logger.info('Fetching %s archive from %r', archive_type, remote_url)
            res = requests.get(remote_url, stream=True, auth=auth)
            if res.status_code == 200:
                with open(local_archive, 'wb') as file_pointer:
                    res.raw.decode_content = True
                    shutil.copyfileobj(res.raw, file_pointer)
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


def load_global_configuration():
    """Loads the global configuration file (~/.mmworkbench/config)

    Returns:
        dict: An object containing configuration values.
    """
    def _filter_bad_keys(config):
        return {key: config[key] for key in config if key is not None}

    config = {
        'mindmeld_url': os.environ.get('MM_URL', None),
        'username': os.environ.get('MM_USERNAME', None),
        'password': os.environ.get('MM_PASSWORD', None),
        'token': os.environ.get('MM_TOKEN', None)
    }
    if config['username'] or config['token']:
        return _filter_bad_keys(config)

    try:
        logging.info('loading auth from mmworkbench config file.')
        config_file = path.get_user_config_path()
        iniconfig = py.iniconfig.IniConfig(config_file)
        config = {
            'mindmeld_url': iniconfig.get('mmworkbench', 'mindmeld_url'),
            'username': iniconfig.get('mmworkbench', 'username'),
            'password': iniconfig.get('mmworkbench', 'password'),
            'token': iniconfig.get('mmworkbench', 'token')
        }
        return _filter_bad_keys(config)
    except OSError:
        raise AuthNotFoundError(
            'Cannot load auth from either the environment or the config file.')


def load_configuration():
    """Loads a configuration file (mmworkbench.cfg) for the current app. The
    file is located by searching first in the current directory, and in parent
    directories.
    """
    config_file = _find_config_file()
    if config_file:
        logger.debug("Using config file at '{}'".format(config_file))
        # Do the thing
        iniconfig = py.iniconfig.IniConfig(config_file)
        config = {}
        config['app_name'] = iniconfig.get('mmworkbench', 'app_name')
        config['app_path'] = iniconfig.get('mmworkbench', 'app_path')
        config['use_quarry'] = iniconfig.get('mmworkbench', 'use_quarry')
        config['input_method'] = iniconfig.get('mmworkbench', 'input_method')
        # resolve path if necessary
        if config['app_path'] and not os.path.isabs(config['app_path']):
            config_dir = os.path.dirname(config_file)
            config['app_path'] = os.path.abspath(os.path.join(config_dir, config['app_path']))
        return config
    else:
        logger.debug('No config file was found.')


def _find_config_file():
    prev_dir = None
    current_dir = os.getcwd()

    while prev_dir != current_dir:
        config_file = os.path.join(current_dir, CONFIG_FILE_NAME)
        if os.path.isfile(config_file):
            # found a config file!
            return config_file

        # go up one directory
        prev_dir = current_dir
        current_dir = os.path.abspath(os.path.join(current_dir, '..'))

    return None
