#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals
from builtins import str

import errno
import json
import logging
import os
import signal
import shutil
import subprocess
import sys
import time

import click
import click_log

from . import path
from .components import Conversation, QuestionAnswerer
from .exceptions import (FileNotFoundError, KnowledgeBaseConnectionError,
                         KnowledgeBaseError, WorkbenchError, AuthNotFoundError)

from ._util import blueprint
from ._version import current as __version__


logger = logging.getLogger(__name__)

click.disable_unicode_literals_warning = True

CONTEXT_SETTINGS = {
    'help_option_names': ['-h', '--help'],
    'auto_envvar_prefix': 'MM'
}


def version_msg():
    """Returns the Workbench version, location and Python powering it."""
    python_version = sys.version[:3]
    location = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    message = 'Workbench %(version)s from {} (Python {})'
    return message.format(location, python_version)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__, '-V', '--version', message=version_msg())
@click.pass_context
@click_log.simple_verbosity_option()
@click_log.init(__package__)
def cli(ctx):
    """Command line interface for mmworkbench."""

    # configure logger settings for dependent libraries
    urllib3_logger = logging.getLogger('urllib3')
    urllib3_logger.setLevel(logging.ERROR)
    es_logger = logging.getLogger('elasticsearch')
    es_logger.setLevel(logging.ERROR)

    if ctx.obj is None:
        ctx.obj = {}


@cli.command('run', context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option('-P', '--port', type=int, default=7150)
@click.option('--no-debug', is_flag=True,
              help='starts the service with debug mode turned off')
@click.option('-r', '--reloader', is_flag=True,
              help='starts the service with the reloader enabled')
def run_server(ctx, port, no_debug, reloader):
    """Starts the workbench service."""
    app = ctx.obj.get('app')
    if app is None:
        raise ValueError("No app was given. Run 'python app.py run' from your app folder.")

    ctx.invoke(num_parser, start=True)
    app.run(port=port, debug=not no_debug, host='0.0.0.0', threaded=True, use_reloader=reloader)


@cli.command('converse', context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option('--session', help='JSON object to be used as the session')
def converse(ctx, session):
    """Starts a conversation with the app."""
    try:
        app = ctx.obj.get('app')
        if isinstance(session, str):
            session = json.loads(session)
        if app is None:
            raise ValueError("No app was given. Run 'python app.py converse' from your app"
                             " folder.")

        ctx.invoke(num_parser, start=True)

        convo = Conversation(app=app, session=session)

        while True:
            message = click.prompt('You')
            responses = convo.say(message)

            for index, response in enumerate(responses):
                prefix = 'App: ' if index == 0 else '...  '
                click.secho(prefix + response, fg='blue', bg='white')
    except WorkbenchError as ex:
        logger.error(ex.message)
        ctx.exit(1)


@cli.command('build', context_settings=CONTEXT_SETTINGS)
@click.pass_context
def build(ctx):
    """Builds the app with default config."""
    try:
        app = ctx.obj.get('app')
        if app is None:
            raise ValueError("No app was given. Run 'python app.py build' from your app folder.")

        app.lazy_init()
        nlp = app.app_manager.nlp
        nlp.build()
        nlp.dump()
    except WorkbenchError as ex:
        logger.error(ex.message)
        ctx.exit(1)
    except RuntimeError as ex:
        logger.error(ex)
        ctx.exit(1)


@cli.command('clean', context_settings=CONTEXT_SETTINGS)
@click.pass_context
def clean(ctx):
    """Deletes all built data, undoing `build`."""
    app = ctx.obj.get('app')
    if app is None:
        raise ValueError("No app was given. Run 'python app.py clean' from your app folder.")

    gen_path = path.get_generated_data_folder(app.app_path)
    try:
        shutil.rmtree(gen_path)
        logger.info('Generated data deleted')
    except FileNotFoundError:
        logger.info('No generated data to delete')


@cli.command('load-kb', context_settings=CONTEXT_SETTINGS)
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


@cli.command('num-parse', context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option('--start/--stop', default=True, help='Start or stop numerical parser')
def num_parser(ctx, start):
    """Starts or stops the numerical parser service."""
    if start:
        pid = _get_mallard_pid()

        if pid:
            # if mallard is already running, leave it be
            logger.info('Numerical parser running, PID %s', pid[0])
            return

        try:
            # We redirect all the output of starting the process to /dev/null and all errors
            # to stdout.
            with open(os.devnull, 'w') as dev_null:
                mallard_service = subprocess.Popen(['java', '-jar', path.MALLARD_JAR_PATH],
                                                   stdout=dev_null, stderr=subprocess.STDOUT)

            # mallard takes some time to start so sleep for a bit
            time.sleep(5)
            logger.info('Starting numerical parsing service, PID %s', mallard_service.pid)
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                logger.error('Java is not found; please verify that Java 8 is '
                             'installed and in your path.')
                ctx.exit(1)
            else:
                raise exc
    else:
        for pid in _get_mallard_pid():
            os.kill(int(pid), signal.SIGKILL)
            logger.info('Stopping numerical parsing service, PID %s', pid)


def _get_mallard_pid():
    _, filename = os.path.split(path.MALLARD_JAR_PATH)
    pid = []
    for line in os.popen('ps ax | grep %s | grep -v grep' % filename):
        pid.append(line.split()[0])
    return pid


@cli.command('blueprint', context_settings=CONTEXT_SETTINGS)
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
    except (AuthNotFoundError, KnowledgeBaseConnectionError, KnowledgeBaseError) as ex:
        logger.error(ex.message)
        ctx.exit(1)


if __name__ == '__main__':
    cli({})
