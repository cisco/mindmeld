# -*- coding: utf-8 -*-
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

