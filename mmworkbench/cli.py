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

from . import __version__

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

    start_num_parser(True)
    app.run(port=port, debug=not no_debug, host='0.0.0.0', threaded=True, use_reloader=reloader)


def _get_mallard_pid():
    pid = []
    for line in os.popen("ps ax | grep mindmeld-mallard.jar | grep -v grep"):
        pid.append(line.split()[0])
    return pid

# TODO: expose this as a command
def start_num_parser(start):
    """Simple command that starts or stops Mallard, the numerical parser service."""
    if start:
        pid = _get_mallard_pid()

        if len(pid) > 0:
            # if mallard is already running, leave it be
            click.echo("Numerical parser running, PID {0:s}".format(pid[0]))
            return

        pwd = os.path.dirname(os.path.abspath(__file__))
        mallard_path = os.path.join(pwd, 'mindmeld-mallard.jar')
        try:
            mallard_service = subprocess.Popen(["java", "-jar", mallard_path])
            # mallard takes some time to start so sleep for a bit
            time.sleep(5)
            click.echo("Starting numerical parsing service, PID %s" % mallard_service.pid)
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                click.echo("Java is not found; please verify that Java 8 is installed and in your"
                           " path variable.")
                exit(1)
            else:
                raise exc
    else:
        for pid in _get_mallard_pid():
            os.kill(int(pid), signal.SIGKILL)
            click.echo("Stopping numerical parsing service, PID %s" % pid)


if __name__ == "__main__":
    cli({})
