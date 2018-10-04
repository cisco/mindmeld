# -*- coding: utf-8 -*-
"""This module contains exceptions used by the mmworkbench package."""
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError
FileNotFoundError = FileNotFoundError


class WorkbenchVersionWarning(UserWarning):
    pass


class WorkbenchError(Exception):

    def __init__(self, *args):
        super().__init__(*args)
        self.message = args[0] if len(args) > 0 else None


class BadWorkbenchRequestError(WorkbenchError):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        obj = dict(self.payload or ())
        obj['error'] = self.message
        return obj


class EmbeddingDownloadError(WorkbenchError):
    pass


class AllowedNlpClassesKeyError(WorkbenchError):
    pass


class ClassifierLoadError(WorkbenchError):
    pass


class ProcessorError(WorkbenchError):
    """An exception which indicates an error with a processor."""
    pass


class ParserTimeout(WorkbenchError):
    """An exception for when parsing takes an unexpected length of time"""
    pass


class MarkupError(WorkbenchError):
    pass


class SystemEntityMarkupError(MarkupError):
    pass


class SystemEntityResolutionError(WorkbenchError):
    """An exception representing an error resolving a system entity"""
    pass


class KnowledgeBaseError(WorkbenchError):
    """An exception for unexpected error from knowledge base."""
    def __init__(self, message):

        super().__init__(message)


class KnowledgeBaseConnectionError(KnowledgeBaseError):
    """An exception for problem connecting to knowledge base."""
    def __init__(self, es_host):

        self.es_host = es_host
        if (not es_host) or (not es_host[0]):
            self.message = 'Unable to connect to Elasticsearch for knowledge base. Please' \
                           ' verify your connection to localhost.'
        else:
            es_host = [host['host'] for host in es_host]
            self.message = 'Unable to connect to knowledge base. Please verify' \
                           ' your connection to: {hosts}.'.format(hosts=', '.join(es_host))
        super().__init__(self.message)


class EntityResolverError(WorkbenchError):
    """An exception for unexpected error from entity resolver."""
    def __init__(self, message):

        super().__init__(message)


class EntityResolverConnectionError(EntityResolverError):
    """An exception for connection error to Elasticsearch for entity resolver"""
    def __init__(self, es_host):
        self.es_host = es_host
        if (not es_host) or (not es_host[0]):
            self.message = 'Unable to connect to Elasticsearch for entity resolution. ' \
                           'Please verify your connection to localhost.'
        else:
            es_host = [host['host'] for host in es_host]
            self.message = 'Unable to connect to Elasticsearch for entity resolution. ' \
                           'Please verify your connection to: {hosts}.'.format(
                            hosts=', '.join(es_host))


class AuthNotFoundError(WorkbenchError):
    pass


class WorkbenchVersionError(WorkbenchError):
    pass


class WorkbenchImportError(WorkbenchError):
    pass
