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

"""A module containing various utility functions for MindMeld.
These are capabilities that do not have an obvious home within the existing
project structure.
"""
import datetime
import logging
import os
import shutil
import sys
import tarfile
from email.utils import parsedate

import py
import requests
from dateutil import tz

from . import path
from .components import QuestionAnswerer
from .components._config import get_app_namespace
from .constants import BLUEPRINTS_URL
from .exceptions import ElasticsearchKnowledgeBaseConnectionError

logger = logging.getLogger(__name__)

CONFIG_FILE_NAME = "mindmeld.cfg"
BLUEPRINT_URL = "{mindmeld_url}/{blueprint}/{filename}"

BLUEPRINT_APP_ARCHIVE = "app.tar.gz"
BLUEPRINT_KB_ARCHIVE = "kb.tar.gz"
BLUEPRINTS = {
    "kwik_e_mart": {},
    "food_ordering": {},
    "home_assistant": {"kb": False},
    "template": {"kb": False},
    "video_discovery": {},
    "hr_assistant": {},
    "banking_assistant": {},
    "screening_app": {},
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


    Within each blueprint directory `app.tar.gz` contains the MindMeld
    application data (`app.py`, `domains`, `entities`, etc.) and `kb.tar.gz`
    contains the a JSON file for each index in the knowledge base.

    The blueprint method will check the MindMeld website for when the blueprint
    files were last updated and compare that with any files in a local
    blueprint cache at ~/.mindmeld/blueprints. If the cache is out of date,
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
            raise ValueError("Unknown blueprint name: {!r}".format(name))
        bp_config = BLUEPRINTS[name]

        app_path = self.setup_app(name, app_path)
        if bp_config.get("kb", True) and not skip_kb:
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
            raise ValueError("Unknown blueprint name: {!r}".format(name))

        app_path = app_path or os.path.join(os.getcwd(), name)
        app_path = os.path.abspath(app_path)

        local_archive = cls._fetch_archive(name, "app")
        tarball = tarfile.open(local_archive)
        tarball.extractall(path=app_path)
        logger.info("Created %r app at %r", name, app_path)
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
            raise ValueError("Unknown blueprint name: {!r}".format(name))

        app_path = app_path or os.path.join(os.getcwd(), name)
        app_path = os.path.abspath(app_path)
        app_namespace = get_app_namespace(app_path)
        es_host = es_host or os.environ.get("MM_ES_HOST", "localhost")
        cache_dir = path.get_cached_blueprint_path(name)
        try:
            local_archive = cls._fetch_archive(name, "kb")
        except ValueError:
            logger.warning("No knowledge base to set up.")
            return

        kb_dir = os.path.join(cache_dir, "kb")
        tarball = tarfile.open(local_archive)
        tarball.extractall(path=kb_dir)

        _, _, index_files = next(os.walk(kb_dir))

        data_dir = os.path.join(app_path, "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        for index in index_files:
            index_name, _ = os.path.splitext(index)
            data_file = os.path.join(kb_dir, index)

            # Copy the index files into the app's data directory
            shutil.copy2(data_file, data_dir)

            try:
                QuestionAnswerer.load_kb(app_namespace, index_name, data_file, es_host)
            except ElasticsearchKnowledgeBaseConnectionError as ex:
                logger.error(
                    "Cannot set up knowledge base. Unable to connect to Elasticsearch "
                    "instance at %r. Confirm it is running or specify an alternate "
                    "instance with the MM_ES_HOST environment variable",
                    es_host,
                )

                raise ex

        logger.info("Created %r knowledge base at %r", name, es_host)

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

        filename = {"app": BLUEPRINT_APP_ARCHIVE, "kb": BLUEPRINT_KB_ARCHIVE}.get(
            archive_type
        )

        local_archive = os.path.join(cache_dir, filename)
        remote_url = BLUEPRINT_URL.format(
            mindmeld_url=BLUEPRINTS_URL, blueprint=name, filename=filename
        )

        res = requests.head(remote_url)
        if res.status_code == 401:
            # authentication error
            msg = (
                "Invalid MindMeld credentials. Cannot download blueprint. Please confirm "
                "they are correct and try again."
            )
            logger.error(msg)
            raise EnvironmentError(msg)
        if res.status_code != 200:
            # Unknown error
            msg = "Unknown error fetching {} archive from {!r}".format(
                archive_type, remote_url
            )
            logger.warning(msg)
            raise ValueError("Unknown error fetching archive")
        remote_modified = datetime.datetime(
            *parsedate(res.headers.get("last-modified"))[:6], tzinfo=tz.tzutc()
        )
        try:
            local_modified = datetime.datetime.fromtimestamp(
                os.path.getmtime(local_archive), tz.tzlocal()
            )
        except (OSError, IOError):
            # File doesn't exist, use minimum possible time
            local_modified = datetime.datetime(
                datetime.MINYEAR, 1, 1, tzinfo=tz.tzutc()
            )

        if remote_modified < local_modified:
            logger.info("Using cached %r %s archive", name, archive_type)
        else:
            logger.info("Fetching %s archive from %r", archive_type, remote_url)
            res = requests.get(remote_url, stream=True)
            if res.status_code == 200:
                with open(local_archive, "wb") as file_pointer:
                    res.raw.decode_content = True
                    shutil.copyfileobj(res.raw, file_pointer)
        return local_archive


blueprint = Blueprint()  # pylint: disable=invalid-name


def configure_logs(**kwargs):
    """Helper method for easily configuring logs from the python shell.

    Args:
        level (TYPE, optional): A logging level recognized by python's logging module.
    """
    level = kwargs.get("level", logging.INFO)
    log_format = kwargs.get("format", "%(message)s")
    logging.basicConfig(stream=sys.stdout, format=log_format)
    package_logger = logging.getLogger(__package__)
    package_logger.setLevel(level)


def load_configuration():
    """Loads a configuration file (mindmeld.cfg) for the current app. The
    file is located by searching first in the current directory, and in parent
    directories.
    """
    config_file = _find_config_file()
    if config_file:
        logger.debug("Using config file at '%s'", config_file)
        # Do the thing
        iniconfig = py.iniconfig.IniConfig(config_file)  # pylint: disable=no-member
        config = {}
        config["app_name"] = iniconfig.get("mindmeld", "app_name")
        config["app_path"] = iniconfig.get("mindmeld", "app_path")
        config["use_quarry"] = iniconfig.get("mindmeld", "use_quarry")
        config["input_method"] = iniconfig.get("mindmeld", "input_method")
        # resolve path if necessary
        if config["app_path"] and not os.path.isabs(config["app_path"]):
            config_dir = os.path.dirname(config_file)
            config["app_path"] = os.path.abspath(
                os.path.join(config_dir, config["app_path"])
            )
        return config
    else:
        logger.debug("No config file was found.")


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
        current_dir = os.path.abspath(os.path.join(current_dir, ".."))

    return None


def get_pattern(rule):
    """Convert a rule represented as a dictionary with the keys "domains", "intents",
    "files" into a regex pattern.

    Args:
        rule (dict): An annotation or augmentation rule.

    Returns:
        pattern (str): Regex pattern specifying allowed file paths.
    """
    pattern = [rule[x] for x in ["domains", "intents", "files"]]
    return ".*/" + "/".join(pattern)


def read_path_queries(filepath):
    """Reads queries from given file path.

        Args:
            filepath (str): File path to read from.

        Returns:
            queries (list): List of queries.
    """
    with open(filepath, "r") as f:
        queries = f.readlines()
    return queries


def write_to_file(filepath, queries, suffix):
    """Writes queries to a new file in the path with given suffix.

    Args:
        filepath (str): File path to the original file.
        queries (list): List of queries to be written to file.
    """
    write_path = filepath.rstrip(".txt") + suffix

    with open(write_path, "w") as outfile:
        for query in queries:
            outfile.write(query.rstrip() + "\n")
