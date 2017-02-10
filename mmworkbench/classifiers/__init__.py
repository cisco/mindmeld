# -*- coding: utf-8 -*-
from maxent import MaxentClassifier
from memm import MemmClassifier
from text_classifier import TextClassifier

CLASSIFIER_TYPE_MAXENT = 'maxent'
CLASSIFIER_TYPE_MEMM = 'memm'
CLASSIFIER_TYPE_TEXT = 'text'

__all__ = ['CLASSIFIER_TYPE_MAXENT', 'MaxentClassifier', 'CLASSIFIER_TYPE_MEMM', 'MemmClassifier',
           'CLASSIFIER_TYPE_TEXT', 'TextClassifier']
