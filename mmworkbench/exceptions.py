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

"""This module contains exceptions used by the mmworkbench package."""


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
    pass


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
    pass


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
