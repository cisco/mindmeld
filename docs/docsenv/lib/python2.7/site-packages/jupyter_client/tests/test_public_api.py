"""Test the jupyter_client public API
"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

import nose.tools as nt

from jupyter_client import launcher, connect
import jupyter_client


def test_kms():
    for base in ("", "Multi"):
        KM = base + "KernelManager"
        nt.assert_in(KM, dir(jupyter_client))

def test_kcs():
    for base in ("", "Blocking"):
        KM = base + "KernelClient"
        nt.assert_in(KM, dir(jupyter_client))

def test_launcher():
    for name in launcher.__all__:
        nt.assert_in(name, dir(jupyter_client))

def test_connect():
    for name in connect.__all__:
        nt.assert_in(name, dir(jupyter_client))
