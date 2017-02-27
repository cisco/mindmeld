# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from .maxent import MaxentRoleModel
from .text import TextModel
from .ner.memm import MemmModel
from .core import ModelConfig

CLASSIFIER_TYPE_MAXENT = 'maxent'
CLASSIFIER_TYPE_MEMM = 'memm'
CLASSIFIER_TYPE_TEXT = 'text'

__all__ = ['ModelConfig', 'CLASSIFIER_TYPE_MAXENT', 'MaxentRoleModel', 'CLASSIFIER_TYPE_MEMM',
           'MemmModel', 'CLASSIFIER_TYPE_TEXT', 'TextModel']
