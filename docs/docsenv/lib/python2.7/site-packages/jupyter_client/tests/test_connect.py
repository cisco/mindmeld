"""Tests for kernel connection utilities"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

import json
import os

import nose.tools as nt

from traitlets.config import Config
from jupyter_core.application import JupyterApp
from ipython_genutils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from ipython_genutils.py3compat import str_to_bytes
from jupyter_client import connect, KernelClient
from jupyter_client.consoleapp import JupyterConsoleApp
from jupyter_client.session import Session


class DummyConsoleApp(JupyterApp, JupyterConsoleApp):
    def initialize(self, argv=[]):
        JupyterApp.initialize(self, argv=argv)
        self.init_connection_file()

sample_info = dict(ip='1.2.3.4', transport='ipc',
        shell_port=1, hb_port=2, iopub_port=3, stdin_port=4, control_port=5,
        key=b'abc123', signature_scheme='hmac-md5', kernel_name='python'
    )

sample_info_kn = dict(ip='1.2.3.4', transport='ipc',
        shell_port=1, hb_port=2, iopub_port=3, stdin_port=4, control_port=5,
        key=b'abc123', signature_scheme='hmac-md5', kernel_name='test'
    )

def test_write_connection_file():
    with TemporaryDirectory() as d:
        cf = os.path.join(d, 'kernel.json')
        connect.write_connection_file(cf, **sample_info)
        nt.assert_true(os.path.exists(cf))
        with open(cf, 'r') as f:
            info = json.load(f)
    info['key'] = str_to_bytes(info['key'])
    nt.assert_equal(info, sample_info)


def test_load_connection_file_session():
    """test load_connection_file() after """
    session = Session()
    app = DummyConsoleApp(session=Session())
    app.initialize(argv=[])
    session = app.session

    with TemporaryDirectory() as d:
        cf = os.path.join(d, 'kernel.json')
        connect.write_connection_file(cf, **sample_info)
        app.connection_file = cf
        app.load_connection_file()

    nt.assert_equal(session.key, sample_info['key'])
    nt.assert_equal(session.signature_scheme, sample_info['signature_scheme'])


def test_load_connection_file_session_with_kn():
    """test load_connection_file() after """
    session = Session()
    app = DummyConsoleApp(session=Session())
    app.initialize(argv=[])
    session = app.session

    with TemporaryDirectory() as d:
        cf = os.path.join(d, 'kernel.json')
        connect.write_connection_file(cf, **sample_info_kn)
        app.connection_file = cf
        app.load_connection_file()

    nt.assert_equal(session.key, sample_info_kn['key'])
    nt.assert_equal(session.signature_scheme, sample_info_kn['signature_scheme'])


def test_app_load_connection_file():
    """test `ipython console --existing` loads a connection file"""
    with TemporaryDirectory() as d:
        cf = os.path.join(d, 'kernel.json')
        connect.write_connection_file(cf, **sample_info)
        app = DummyConsoleApp(connection_file=cf)
        app.initialize(argv=[])

    for attr, expected in sample_info.items():
        if attr in ('key', 'signature_scheme'):
            continue
        value = getattr(app, attr)
        nt.assert_equal(value, expected, "app.%s = %s != %s" % (attr, value, expected))


def test_load_connection_info():
    client = KernelClient()
    info = {
        'control_port': 53702,
        'hb_port': 53705,
        'iopub_port': 53703,
        'ip': '0.0.0.0',
        'key': 'secret',
        'shell_port': 53700,
        'signature_scheme': 'hmac-sha256',
        'stdin_port': 53701,
        'transport': 'tcp',
    }
    client.load_connection_info(info)
    assert client.control_port == info['control_port']
    assert client.session.key.decode('ascii') == info['key']
    assert client.ip == info['ip']


def test_find_connection_file():
    cfg = Config()
    with TemporaryDirectory() as d:
        cfg.ProfileDir.location = d
        cf = 'kernel.json'
        app = DummyConsoleApp(config=cfg, connection_file=cf)
        app.initialize()

        security_dir = app.runtime_dir
        profile_cf = os.path.join(security_dir, cf)

        with open(profile_cf, 'w') as f:
            f.write("{}")

        for query in (
            'kernel.json',
            'kern*',
            '*ernel*',
            'k*',
            ):
            nt.assert_equal(connect.find_connection_file(query, path=security_dir), profile_cf)

        JupyterApp._instance = None

