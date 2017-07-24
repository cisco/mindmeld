# -*- coding: utf-8 -*-
"""This module contains exceptions used by the mmworkbench package."""
from __future__ import absolute_import, unicode_literals
from builtins import super

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError
FileNotFoundError = FileNotFoundError


class WorkbenchError(Exception):

    def __init__(self, *args):
        super().__init__(*args)
        self.message = args[0] if len(args) > 0 else None


class BadWorkbenchRequestError(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        obj = dict(self.payload or ())
        obj['error'] = self.message
        return obj


class ClassifierLoadError(Exception):
    pass


class ProcessorError(Exception):
    """An exception which indicates an error with a processor."""
    pass


class ParserTimeout(Exception):
    """An exception for when parsing takes an unexpected length of time"""
    pass


class MarkupError(Exception):
    pass


class SystemEntityMarkupError(MarkupError):
    pass


class SystemEntityResolutionError(Exception):
    """An exception representing an error resolving a system entity"""
    pass


class KnowledgeBaseConnectionError(WorkbenchError):
    """An exception representing an issue connecting to a knowledge base"""
    def __init__(self, es_host):
        self.es_host = es_host
        if (not es_host) or (not es_host[0]):
            self.message = 'Unable to connect to elasticsearch for knowledgebase - please' \
                           ' verify your connection to local host.'
        else:
            es_host = [host['host'] for host in es_host]
            self.message = 'Unable to connect to elasticsearch for knowledgebase - please verify' \
                           ' your connection to: {hosts}.'.format(hosts=', '.join(es_host))


class EntityResolverConnectionError(WorkbenchError):
    """An exception representing an issue connecting to elasticsearch for entity resolver"""
    def __init__(self, es_host):
        self.es_host = es_host
        if (not es_host) or (not es_host[0]):
            self.message = 'Unable to connect to elasticsearch for entity resolution -' \
                           ' please verify your connection to local host.'
        else:
            es_host = [host['host'] for host in es_host]
            self.message = 'Unable to connect to elasticsearch for entity resolution - verify' \
                           ' your connection to: {hosts}.'.format(hosts=', '.join(es_host))
