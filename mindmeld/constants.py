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
import unicodedata

DEFAULT_TRAIN_SET_REGEX = r"train.*\.txt"
DEFAULT_TEST_SET_REGEX = r"test.*\.txt"
BLUEPRINTS_URL = "https://blueprints.mindmeld.com"
BINARIES_URL = "https://binaries.mindmeld.com"
DUCKLING_VERSION = "20200701"

SPACY_ANNOTATOR_SUPPORTED_ENTITIES = [
    "sys_time",
    "sys_interval",
    "sys_duration",
    "sys_number",
    "sys_amount-of-money",
    "sys_distance",
    "sys_weight",
    "sys_ordinal",
    "sys_quantity",
    "sys_percent",
    "sys_org",
    "sys_loc",
    "sys_person",
    "sys_gpe",
    "sys_norp",
    "sys_fac",
    "sys_product",
    "sys_event",
    "sys_law",
    "sys_langauge",
    "sys_work-of-art",
    "sys_other-quantity",
]


def _no_overlap(entity_one, entity_two):
    """ Returns True if two query entities do not overlap.
    Args:
        entity_one (QueryEntity): First entity.
        entity_two (QueryEntity): Second Entity.

    Returns:
        no_overlap (bool): True if no overlap.
    """
    return (
        entity_one.span.start > entity_two.span.end
        or entity_two.span.start > entity_one.span.end
    )


# fetches all currency symbols in unicode by iterating through the character set and
# selecting the currency symbols based on the unicode currency category 'Sc'
CURRENCY_SYMBOLS = u"".join(
    chr(i) for i in range(0xFFFF) if unicodedata.category(chr(i)) == "Sc"
)
