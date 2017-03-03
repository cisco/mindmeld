#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import errno
import logging
import os
import signal
import subprocess
import sys
import time

import click
import click_log

from . import __version__, Conversation

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
    """Starts the workbench service"""
    app = ctx.obj.get('app')
    if app is None:
        raise ValueError('No app was given')

    ctx.invoke(num_parser, start=True)
    app.run(port=port, debug=not no_debug, host='0.0.0.0', threaded=True, use_reloader=reloader)


@cli.command('converse', context_settings=CONTEXT_SETTINGS)
@click.pass_context
def converse(ctx):
    """Starts a conversation with the app"""
    app = ctx.obj.get('app')
    if app is None:
        raise ValueError('No app was given')

    ctx.invoke(num_parser, start=True)

    convo = Conversation(app=app)

    while True:
        message = click.prompt('You')
        responses = convo.say(message)

        for index, response in enumerate(responses):
            prefix = 'App: ' if index == 0 else ''
            click.secho(prefix + response, fg='blue', bg='white')


@cli.command('num-parse', context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option('--start/--stop', default=True, help='Start or stop numerical parser')
def num_parser(ctx, start):
    """Starts or stops the numerical parser service"""
    if start:
        pid = _get_mallard_pid()

        if len(pid) > 0:
            # if mallard is already running, leave it be
            logger.info('Numerical parser running, PID %s', pid[0])
            return

        pwd = os.path.dirname(os.path.abspath(__file__))
        mallard_path = os.path.join(pwd, 'mindmeld-mallard.jar')
        try:
            mallard_service = subprocess.Popen(['java', '-jar', mallard_path])
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
    pid = []
    for line in os.popen('ps ax | grep mindmeld-mallard.jar | grep -v grep'):
        pid.append(line.split()[0])
    return pid


if __name__ == '__main__':
    cli({})
