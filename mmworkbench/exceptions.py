# -*- coding: utf-8 -*-
"""This module contains exceptions used by the mmworkbench package."""

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError
FileNotFoundError = FileNotFoundError


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


class MarkupError(Exception):
    pass


class SystemEntityMarkupError(MarkupError):
    pass


class SystemEntityResolutionError(Exception):
    """An exception representing an error resolving a system entity"""
    pass
