"""A module containing various utility functions for Workbench.
These are capabilities that do not have an obvious home within the existing
project structure.
"""

from __future__ import unicode_literals
from builtins import object

import datetime
from email.utils import parsedate
import logging
import os
import shutil

import boto3
import botocore
from dateutil import tz


from . import path
from .components import QuestionAnswerer


logger = logging.getLogger(__name__)

BLUEPRINT_S3_BUCKET = 'mindmeld'
BLUEPRINT_S3_KEY_BASE = 'workbench-data/blueprints'
BLUEPRINT_APP_ARCHIVE = 'app.tar.gz'
BLUEPRINT_KB_ARCHIVE = 'kb.tar.gz'
BLUEPRINTS = {
    'quick_start': {},
    'food_ordering': {}
}


class Blueprint(object):
    """This is a callable class used to set up a blueprint app.

    In S3 the directory structure looks like this:

       - s3://mindmeld/workbench-data/blueprints/
       |-food_ordering
       |  |-app.tar.gz
       |  |-kb.tar.gz
       |-quick_start
          |-app.tar.gz
          |-kb.tar.gz


    Within each blueprint dir `app.tar.gz` contains the workbench application
    data (`app.py`, `domains`, `entities`, etc.) and `kb.tar.gz`

    The blueprint method will check S3 for when the blueprint files were last
    updated and compare that with any files in a local blueprint cache at
    ~/.mmworkbench/blueprints. If the cache is out of date, the updated archive
    is downloaded. The archive is then extracted into a directory named for the
    blueprint.
    """
    def __call__(self, name, app_path=None, es_host=None):
        """

        Args:
            name (str): The name of the blueprint
            app_path (str, optional): The path to the app
            es_host (str, optional): The hostname of the elasticsearch cluster
                for the knowledge base. If no value is passed, value will be
                read from the environment.

        Returns:
            str: The path where the blueprint was created

        Raises:
            ValueError: When an unknown blueprint is specified.
        """
        if name not in BLUEPRINTS:
            raise ValueError('Unknown blueprint name: {!r}'.format(name))
        app_path = self.setup_app(name, app_path)
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
        shutil.unpack_archive(local_archive, app_path)
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
        _, app_name = os.path.split(app_path)

        if not es_host:
            try:
                es_host = os.environ['MM_ES_HOST']
            except KeyError:
                raise EnvironmentError('Cannot set up knowledge base. No Elasticsearch host was '
                                       'specified. Specify it with the es_host keyword argument '
                                       'or the MM_ES_HOST environment variable')

        cache_dir = path.get_cached_blueprint_path(name)
        local_archive = cls._fetch_archive(name, 'kb')
        kb_dir = os.path.join(cache_dir, 'kb')
        shutil.unpack_archive(local_archive, kb_dir)

        _, _, index_files = next(os.walk(kb_dir))

        for index in index_files:
            index_name, _ = os.path.splitext(index)
            data_file = os.path.join(kb_dir, index)
            QuestionAnswerer.load_index(app_name, index_name, data_file, es_host)

    @staticmethod
    def _fetch_archive(name, archive_type):
        cache_dir = path.get_cached_blueprint_path(name)
        try:
            os.makedirs(cache_dir)
        except IOError:
            # dir already exists -- no worries
            pass

        filename = {'app': BLUEPRINT_APP_ARCHIVE, 'kb': BLUEPRINT_KB_ARCHIVE}.get(archive_type)

        local_archive = os.path.join(cache_dir, filename)

        s3 = boto3.resource('s3')
        object_key = '/'.join((BLUEPRINT_S3_KEY_BASE, name, filename))

        try:
            object_summary = s3.ObjectSummary(BLUEPRINT_S3_BUCKET, object_key)
            remote_modified = object_summary.last_modified
        except botocore.exceptions.NoCredentialsError as ex:
            msg = 'Unable to locate AWS credentials. Cannot download blueprint.'
            logger.error(msg)
            raise EnvironmentError(msg)
        except botocore.exceptions.ClientError as ex:
            if ex.response['Error']['Code'] == "404":
                logger.error('Unable to locate the requested blueprint.')

            raise

        try:
            local_modified = datetime.datetime.fromtimestamp(os.path.getmtime(local_archive), tz.tzlocal())
        except IOError:
            # Minimum possible time
            local_modified = datetime.datetime(datetime.MINYEAR, 1, 1, tzinfo=datetime.timezone.utc)


        if remote_modified < local_modified:
            logger.info('Using cached %r %s archive', name, archive_type)
        else:
            logger.info('Fetching %s archive from %r', archive_type, object_key)
            s3.Bucket(BLUEPRINT_S3_BUCKET).download_file(object_key, local_archive)
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
