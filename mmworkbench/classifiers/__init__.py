# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from .maxent import MaxentRoleClassifier
from .text_classifier import TextClassifier

CLASSIFIER_TYPE_MAXENT = 'maxent'
CLASSIFIER_TYPE_MEMM = 'memm'
CLASSIFIER_TYPE_TEXT = 'text'

__all__ = ['CLASSIFIER_TYPE_MAXENT', 'MaxentClassifier', 'CLASSIFIER_TYPE_MEMM', 'MemmClassifier',
           'CLASSIFIER_TYPE_TEXT', 'TextClassifier']
