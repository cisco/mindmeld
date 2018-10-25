# -*- coding: utf-8 -*-
try:
    # register test module for assertion rewrite
    import pytest
    pytest.register_assert_rewrite('mmworkbench.test')
except ImportError:
    pass  # no worries

from .app import Application
from .components import (Conversation, DialogueResponder,
                         NaturalLanguageProcessor, QuestionAnswerer)
from ._util import blueprint, configure_logs
from ._version import current

__all__ = ['blueprint', 'configure_logs', 'Application', 'Conversation', 'DialogueResponder',
           'NaturalLanguageProcessor', 'QuestionAnswerer']

__author__ = 'MindMeld, Inc.'
__email__ = 'contact@mindmeld.com'
__version__ = current
