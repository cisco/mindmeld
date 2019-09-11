#! /usr/bin/env python
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

import asyncio
import json
import logging
import os
import stat
import signal
import shutil
import subprocess
import sys
import time
import warnings
import datetime

import math
import hashlib
import click
import click_log
import distro
import requests

from tqdm import tqdm
from . import markup, path
from .components import Conversation, QuestionAnswerer
from .converter import RasaConverter, DialogflowConverter
from .exceptions import (KnowledgeBaseConnectionError, KnowledgeBaseError, MindMeldError)
from .path import QUERY_CACHE_PATH, QUERY_CACHE_TMP_PATH, MODEL_CACHE_PATH
from ._version import current as __version__
from ._util import blueprint
from .constants import DEVCENTER_URL


logger = logging.getLogger(__name__)

click.disable_unicode_literals_warning = True

CONTEXT_SETTINGS = {
    'help_option_names': ['-h', '--help'],
    'auto_envvar_prefix': 'MM'
}


def _version_msg():
    """Returns the MindMeld version, location and Python powering it."""
    python_version = sys.version[:3]
    location = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    message = 'MindMeld %(version)s from {} (Python {})'
    return message.format(location, python_version)


#
# App only Commands
#

@click.group()
def _app_cli(ctx):
    """Command line interface for MindMeld apps."""

    # configure logger settings for dependent libraries
    urllib3_logger = logging.getLogger('urllib3')
    urllib3_logger.setLevel(logging.ERROR)
    es_logger = logging.getLogger('elasticsearch')
    es_logger.setLevel(logging.ERROR)

    if ctx.obj is None:
        ctx.obj = {}


@_app_cli.command('run', context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option('-P', '--port', type=int, default=7150)
@click.option('--no-debug', is_flag=True,
              help='starts the service with debug mode turned off')
@click.option('-r', '--reloader', is_flag=True,
              help='starts the service with the reloader enabled')
def run_server(ctx, port, no_debug, reloader):
    """Starts the MindMeld service."""
    app = ctx.obj.get('app')
    if app is None:
        raise ValueError("No app was given. Run 'python app.py run' from your app folder.")

    # make sure num parser is running
    ctx.invoke(num_parser, start=True)

    app.run(port=port, debug=not no_debug, host='0.0.0.0', threaded=True, use_reloader=reloader)


@_app_cli.command('converse', context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option('--context', help='JSON object to be used as the context')
def converse(ctx, context):
    """Starts a conversation with the app."""

    try:
        app = ctx.obj.get('app')
        if isinstance(context, str):
            context = json.loads(context)
        if app is None:
            raise ValueError("No app was given. Run 'python app.py converse' from your app"
                             " folder.")

        # make sure num parser is running
        ctx.invoke(num_parser, start=True)

        if app.async_mode:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(_converse_async(app, context))
            return

        convo = Conversation(app=app, context=context)

        while True:
            message = click.prompt('You')
            responses = convo.say(message)

            for index, response in enumerate(responses):
                prefix = 'App: ' if index == 0 else '...  '
                click.secho(prefix + response, fg='blue', bg='white')
    except MindMeldError as ex:
        logger.error(ex.message)
        ctx.exit(1)


async def _converse_async(app, context):
    convo = Conversation(app=app, context=context)
    while True:
        message = click.prompt('You')
        responses = await convo.say(message)

        for index, response in enumerate(responses):
            prefix = 'App: ' if index == 0 else '...  '
            click.secho(prefix + response, fg='blue', bg='white')


@_app_cli.command('build', context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option('-i', '--incremental', is_flag=True, default=False,
              help='only build models with changed training data or configuration')
def build(ctx, incremental):
    """Builds the app with default config."""
    try:
        app = ctx.obj.get('app')
        if app is None:
            raise ValueError("No app was given. Run 'python app.py build' from your app folder.")

        # make sure num parser is running
        ctx.invoke(num_parser, start=True)

        app.lazy_init()
        nlp = app.app_manager.nlp
        nlp.build(incremental=incremental)
        nlp.dump()
    except MindMeldError as ex:
        logger.error(ex.message)
        ctx.exit(1)
    except RuntimeError as ex:
        logger.error(ex)
        ctx.exit(1)


@_app_cli.command('evaluate', context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option('-v', '--verbose', is_flag=True,
              help='Print the full metrics instead of just accuracy.')
def evaluate(ctx, verbose):
    """Evaluates the app with default config."""
    try:
        app = ctx.obj.get('app')
        if app is None:
            raise ValueError("No app was given. Run 'python app.py evaluate' from your app folder.")

        # make sure num parser is running
        ctx.invoke(num_parser, start=True)

        app.lazy_init()
        nlp = app.app_manager.nlp
        try:
            nlp.load()
        except MindMeldError:
            logger.error("You must build the app before running evaluate. "
                         "Try 'python app.py build'.")
            ctx.exit(1)
        nlp.evaluate(verbose)
    except MindMeldError as ex:
        logger.error(ex.message)
        ctx.exit(1)
    except RuntimeError as ex:
        logger.error(ex)
        ctx.exit(1)


@_app_cli.command('predict', context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option('-o', '--output', required=False,
              help='Send output to file rather than standard out')
@click.option('-c', '--confidence', is_flag=True,
              help='Show confidence scores for each prediction')
@click.option('-D', '--no_domain', is_flag=True,
              help='Suppress predicted domain column')
@click.option('-I', '--no_intent', is_flag=True,
              help='Suppress predicted intent column')
@click.option('-E', '--no_entity', is_flag=True,
              help='Suppress predicted entity annotations')
@click.option('-R', '--no_role', is_flag=True,
              help='Suppress predicted role annotations')
@click.option('-G', '--no_group', is_flag=True,
              help='Suppress predicted group annotations')
@click.argument('input', required=True)
def predict(ctx, input_file, output, confidence, no_domain, no_intent, no_entity, no_role,
            no_group):
    """Runs predictions on a given query file"""
    app = ctx.obj.get('app')
    if app is None:
        raise ValueError("No app was given. Run 'python app.py predict' from your app folder.")

    ctx.invoke(num_parser, start=True)

    app.lazy_init()
    nlp = app.app_manager.nlp
    try:
        nlp.load()
    except MindMeldError:
        logger.error("You must build the app before running predict. "
                     "Try 'python app.py build'.")
        ctx.exit(1)

    markup.bootstrap_query_file(input_file, output, nlp, confidence=confidence,
                                no_domain=no_domain, no_intent=no_intent,
                                no_entity=no_entity, no_role=no_role, no_group=no_group)


@_app_cli.command('clean', context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option('-q', '--query-cache', is_flag=True, required=False, help='Clean only query cache')
@click.option('-m', '--model-cache', is_flag=True, required=False, help='Clean only model cache')
@click.option('-d', '--days', type=int, default=7,
              help='Clear model cache older than the specified days')
def clean(ctx, query_cache, model_cache, days):
    """Deletes all built data, undoing `build`."""
    app = ctx.obj.get('app')
    if app is None:
        raise ValueError("No app was given. Run 'python app.py clean' from your app folder.")
    if query_cache:
        try:
            main_cache_location = QUERY_CACHE_PATH.format(app_path=app.app_path)
            tmp_cache_location = QUERY_CACHE_TMP_PATH.format(app_path=app.app_path)

            if os.path.exists(main_cache_location):
                os.remove(main_cache_location)

            if os.path.exists(tmp_cache_location):
                os.remove(tmp_cache_location)

            logger.info('Query cache deleted')
        except FileNotFoundError:
            logger.info('No query cache to delete')
        return

    if model_cache:
        model_cache_path = MODEL_CACHE_PATH.format(app_path=app.app_path)

        if not os.path.exists(model_cache_path):
            logger.warning("Model cache directory doesn't exist")
            return

        if days:
            for ts_folder in os.listdir(model_cache_path):
                full_path = os.path.join(model_cache_path, ts_folder)

                if not os.path.isdir(full_path):
                    logger.warning(
                        'Expected timestamped folder. Ignoring the file %s.', full_path)
                    continue

                try:
                    current_ts = datetime.datetime.fromtimestamp(time.time())
                    folder_ts = datetime.datetime.strptime(ts_folder, markup.TIME_FORMAT)
                    diff_days = current_ts - folder_ts
                    if diff_days.days > days:
                        shutil.rmtree(full_path)
                        logger.info('Removed cached ts folder: %s', full_path)
                except ValueError:
                    logger.warning('Folder %s is not named as a proper timestamp. Ignoring it.',
                                   full_path)
        else:
            try:
                shutil.rmtree(model_cache_path)
                logger.info('Model cache data deleted')
            except FileNotFoundError:
                logger.info('No model cache to delete')
        return

    gen_path = path.get_generated_data_folder(app.app_path)
    try:
        shutil.rmtree(gen_path)
        logger.info('Generated data deleted')
    except FileNotFoundError:
        logger.info('No generated data to delete')

#
# Shared commands
#


@click.group()
def shared_cli():
    """Commands for MindMeld module and apps"""
    pass


@shared_cli.command('load-kb', context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option('-n', '--es-host', required=False)
@click.argument('app_namespace', required=True)
@click.argument('index_name', required=True)
@click.argument('data_file', required=True)
def load_index(ctx, es_host, app_namespace, index_name, data_file):
    """Loads data into a question answerer index."""

    try:
        QuestionAnswerer.load_kb(app_namespace, index_name, data_file, es_host)
    except (KnowledgeBaseConnectionError, KnowledgeBaseError) as ex:
        logger.error(ex.message)
        ctx.exit(1)


def _find_duckling_os_executable():
    """Returns the correct duckling path for this OS."""
    os_platform_name = '-'.join(distro.linux_distribution(
        full_distribution_name=False)).lower()
    for os_key in path.DUCKLING_OS_MAPPINGS:
        if os_key in os_platform_name:
            return path.DUCKLING_OS_MAPPINGS[os_key]


@shared_cli.command('num-parse', context_settings=CONTEXT_SETTINGS)
@click.option('--start/--stop', default=True, help='Start or stop numerical parser')
@click.option('-p', '--port', required=False, default='7151')
def num_parser(start, port):
    """Starts or stops the local numerical parser service."""
    if start:
        pid = _get_duckling_pid()

        if pid:
            # if duckling is already running, leave it be
            logger.info('Numerical parser running, PID %s', pid[0])
            return

        # We redirect all the output of starting the process to /dev/null and all errors
        # to stdout.
        exec_path = _find_duckling_os_executable()

        if not exec_path:
            logger.warning('OS is incompatible with duckling executable. '
                           'Use docker to install duckling.')
            return

        # Download the binary from the cloud if the binary does not already exist OR
        # the binary is out of date.
        if os.path.exists(exec_path):
            hash_digest = hashlib.md5(open(exec_path, 'rb').read()).hexdigest()
            if hash_digest != path.DUCKLING_PATH_TO_MD5_MAPPINGS[exec_path]:
                os.remove(exec_path)

        if not os.path.exists(exec_path):
            url = os.path.join(os.path.join(DEVCENTER_URL, 'binaries'), os.path.basename(exec_path))
            logger.info('Could not find %s binary file, downloading from %s', exec_path, url)
            r = requests.get(url, stream=True)

            # Total size in bytes.
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024

            with open(exec_path, 'wb') as f:
                for data in tqdm(r.iter_content(block_size),
                                 total=math.ceil(total_size // block_size),
                                 unit='KB',
                                 unit_scale=True):
                    f.write(data)
                    f.flush()

        # make the file executable
        st = os.stat(exec_path)
        os.chmod(exec_path, st.st_mode | stat.S_IEXEC)

        # run duckling
        duckling_service = subprocess.Popen([exec_path, '--port', port], stderr=subprocess.STDOUT)

        # duckling takes some time to start so sleep for a bit
        for _ in range(50):
            if duckling_service.pid:
                logger.info('Starting numerical parsing service, PID %s', duckling_service.pid)
                return
            time.sleep(0.1)
    else:
        for pid in _get_duckling_pid():
            os.kill(int(pid), signal.SIGKILL)
            logger.info('Stopping numerical parsing service, PID %s', pid)


def _get_duckling_pid():
    pid = []
    for line in os.popen('ps ax | grep duckling | grep -v grep'):
        pid.append(line.split()[0])
    return pid


#
# Module only Commands
#

@click.group()
def module_cli():
    """Commands for MindMeld module only"""
    pass


@module_cli.command('blueprint', context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option('-n', '--es-host')
@click.option('--skip-kb', is_flag=True, help="Skip setting up the knowledge base")
@click.argument('blueprint_name', required=True)
@click.argument('app_path', required=False)
def setup_blueprint(ctx, es_host, skip_kb, blueprint_name, app_path):
    """Sets up a blueprint application."""
    try:
        blueprint(blueprint_name, app_path, es_host=es_host, skip_kb=skip_kb)
    except ValueError as ex:
        logger.error(ex)
        ctx.exit(1)
    except (KnowledgeBaseConnectionError, KnowledgeBaseError) as ex:
        logger.error(ex.message)
        ctx.exit(1)


@module_cli.command('convert', context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option('-d', '--df', is_flag=True, help="Convert a Dialogflow project")
@click.option('-r', '--rs', is_flag=True, help="Convert a Rasa project")
@click.argument('project_path', required=True, type=click.Path(exists=True))
@click.argument('mindmeld_path', required=False)
def convert(ctx, df, rs, project_path, mindmeld_path=None):
    """Converts a Rasa or DialogueFlow project to a MindMeld project"""
    if df:
        framework = 'Dialogflow'
    elif rs:
        framework = 'Rasa'
    else:
        logger.warning("Please specify the project's platform Rasa/Dialogflow.")
        ctx.exit(1)

    try:
        project_path = os.path.abspath(project_path)
        mindmeld_path = os.path.abspath(mindmeld_path or "converted_app")
        converter_cls = {'Rasa': RasaConverter, 'Dialogflow': DialogflowConverter}
        converter_cls = converter_cls[framework]
        converter = converter_cls(project_path, mindmeld_path)
        converter.convert_project()
        msg = "Successfully converted {framework} project at {project_path} to"\
              " MindMeld project at {mindmeld_path}."
        msg = msg.format(framework=framework, project_path=project_path,
                         mindmeld_path=mindmeld_path)
        logger.info(msg)
    except IOError as e:
        logger.error(e)
        ctx.exit(1)


#
# Command collections
#

@click.command(cls=click.CommandCollection, context_settings=CONTEXT_SETTINGS,
               sources=[module_cli, shared_cli])
@click.version_option(__version__, '-V', '--version', message=_version_msg())
@click.pass_context
@click_log.simple_verbosity_option()
@click_log.init(__package__)
def cli(ctx):
    """Command line interface for MindMeld."""

    # configure logger settings for dependent libraries
    urllib3_logger = logging.getLogger('urllib3')
    urllib3_logger.setLevel(logging.ERROR)
    es_logger = logging.getLogger('elasticsearch')
    es_logger.setLevel(logging.ERROR)
    warnings.filterwarnings("module", category=DeprecationWarning,
                            module="sklearn.preprocessing.label")
    if ctx.obj is None:
        ctx.obj = {}


@click.command(cls=click.CommandCollection, context_settings=CONTEXT_SETTINGS,
               sources=[_app_cli, shared_cli])
@click.version_option(__version__, '-V', '--version', message=_version_msg())
@click.pass_context
@click_log.simple_verbosity_option()
@click_log.init(__package__)
def app_cli(ctx):
    """Command line interface for MindMeld apps."""

    # configure logger settings for dependent libraries
    urllib3_logger = logging.getLogger('urllib3')
    urllib3_logger.setLevel(logging.ERROR)
    es_logger = logging.getLogger('elasticsearch')
    es_logger.setLevel(logging.ERROR)
    warnings.filterwarnings("module", category=DeprecationWarning,
                            module="sklearn.preprocessing.label")

    if ctx.obj is None:
        ctx.obj = {}


if __name__ == '__main__':
    cli({})
