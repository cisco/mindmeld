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

"""Sphinx extension for conditionally exposing pages"""
from builtins import open

import os
import re
from sphinx import addnodes

docs_to_remove = []


def setup(app):
    app.ignore = []
    app.connect('builder-inited', builder_inited)
    app.connect('env-get-outdated', env_get_outdated)
    app.connect('doctree-read', doctree_read)


def builder_inited(app):
    for doc in app.env.found_docs:
        first_directive = None

        for suffix in app.env.config.source_suffix:
            filename = os.path.join(app.env.srcdir, doc + suffix)
            try:
                with open(filename, 'r') as f:
                    first_directive = f.readline() + f.readline()
            except OSError:
                # File not found, try next suffix
                continue
            if first_directive:
                m = re.match(r'^\.\. meta::\s+:scope: ([a-zA-Z0-9_-]+)', first_directive)
                if m and not app.tags.has(m.group(1)):
                    docs_to_remove.append(doc)

    app.env.found_docs.difference_update(docs_to_remove)


def env_get_outdated(app, env, added, changed, removed):
    added.difference_update(docs_to_remove)
    changed.difference_update(docs_to_remove)
    removed.update(docs_to_remove)
    return []


def doctree_read(app, doctree):
    for toctreenode in doctree.traverse(addnodes.toctree):
        for e in toctreenode['entries']:
            ref = str(e[1])
            if ref in docs_to_remove:
                toctreenode['entries'].remove(e)
