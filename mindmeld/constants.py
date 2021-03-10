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

SPACY_ANNOTATOR_WEB_LANGUAGES = ["en", "zh"]
SPACY_ANNOTATOR_NEWS_LANGUAGES = [
    "da",
    "nl",
    "fr",
    "de",
    "el",
    "it",
    "lt",
    "ja",
    "nb",
    "pl",
    "pt",
    "ro",
    "es",
]
SPACY_ANNOTATOR_SUPPORTED_LANGUAGES = (
    SPACY_ANNOTATOR_WEB_LANGUAGES + SPACY_ANNOTATOR_NEWS_LANGUAGES
)
SPACY_ANNOTATOR_MODEL_SIZES = ["sm", "md", "lg"]

UNANNOTATE_ALL_RULE = [
    {
        "domains": ".*",
        "intents": ".*",
        "files": ".*",
        "entities": ".*",
    }
]

ANNOTATOR_TO_SYS_ENTITY_MAPPINGS = {
    "money": "sys_amount-of-money",
    "cardinal": "sys_number",
    "numeric_value": "sys_number",
    "ordinal": "sys_ordinal",
    "geogName": "sys_loc",
    "person": "sys_person",
    "per": "sys_person",
    "persname": "sys_person",
    "placename": "sys_gpe",
    "percent": "sys_percent",
    "distance": "sys_distance",
    "quantity": "sys_weight",
    "organization": "sys_org",
    "orgname": "sys_org",
    "facility": "sys_fac",
}

SPACY_SYS_ENTITIES_NOT_IN_DUCKLING = [
    "sys_event",
    "sys_fac",
    "sys_gpe",
    "sys_language",
    "sys_law",
    "sys_loc",
    "sys_norp",
    "sys_org",
    "sys_other-quantity",
    "sys_person",
    "sys_product",
    "sys_weight",
    "sys_work_of_art",
]

# TODO: Create a script to retreive these automatically
DUCKLING_TO_SYS_ENTITY_MAPPINGS = {
    "af": ["sys_number"],
    "ar": [
        "sys_duration",
        "sys_number",
        "sys_ordinal",
        "sys_quantity",
        "sys_time",
        "sys_interval",
        "sys_amount-of-money",
    ],
    "bg": [
        "sys_distance",
        "sys_duration",
        "sys_number",
        "sys_ordinal",
        "sys_time",
        "sys_interval",
        "sys_amount-of-money",
    ],
    "bn": ["sys_number"],
    "cs": ["sys_distance", "sys_number"],
    "da": ["sys_duration", "sys_number", "sys_ordinal", "sys_time", "sys_interval"],
    "de": [
        "sys_distance",
        "sys_duration",
        "sys_number",
        "sys_ordinal",
        "sys_time",
        "sys_interval",
    ],
    "el": ["sys_duration", "sys_number", "sys_ordinal", "sys_time", "sys_interval"],
    "en": [
        "sys_distance",
        "sys_duration",
        "sys_number",
        "sys_ordinal",
        "sys_quantity",
        "sys_time",
        "sys_interval",
        "sys_amount-of-money",
    ],
    "es": [
        "sys_distance",
        "sys_duration",
        "sys_number",
        "sys_ordinal",
        "sys_quantity",
        "sys_time",
        "sys_interval",
        "sys_amount-of-money",
    ],
    "et": ["sys_number", "sys_ordinal"],
    "fa": ["sys_number"],
    "fi": ["sys_number"],
    "fr": [
        "sys_distance",
        "sys_duration",
        "sys_number",
        "sys_ordinal",
        "sys_quantity",
        "sys_time",
        "sys_interval",
        "sys_amount-of-money",
    ],
    "ga": [
        "sys_distance",
        "sys_duration",
        "sys_number",
        "sys_ordinal",
        "sys_time",
        "sys_interval",
        "sys_amount-of-money",
    ],
    "he": ["sys_duration", "sys_number", "sys_ordinal", "sys_time", "sys_interval"],
    "hi": ["sys_duration", "sys_number", "sys_ordinal"],
    "hr": [
        "sys_distance",
        "sys_duration",
        "sys_number",
        "sys_ordinal",
        "sys_quantity",
        "sys_time",
        "sys_interval",
        "sys_amount-of-money",
    ],
    "hu": ["sys_duration", "sys_number", "sys_ordinal", "sys_time", "sys_interval"],
    "id": ["sys_number", "sys_ordinal", "sys_amount-of-money"],
    "is": ["sys_number"],
    "it": [
        "sys_distance",
        "sys_duration",
        "sys_number",
        "sys_ordinal",
        "sys_time",
        "sys_interval",
    ],
    "ja": ["sys_duration", "sys_number", "sys_ordinal"],
    "ka": [
        "sys_duration",
        "sys_number",
        "sys_ordinal",
        "sys_time",
        "sys_interval",
        "sys_amount-of-money",
    ],
    "km": ["sys_distance", "sys_number", "sys_ordinal", "sys_quantity"],
    "kn": ["sys_number"],
    "ko": [
        "sys_distance",
        "sys_duration",
        "sys_number",
        "sys_ordinal",
        "sys_quantity",
        "sys_time",
        "sys_interval",
        "sys_amount-of-money",
    ],
    "lo": ["sys_number"],
    "ml": ["sys_number", "sys_ordinal"],
    "mn": ["sys_distance", "sys_duration", "sys_number", "sys_ordinal", "sys_quantity"],
    "my": ["sys_number"],
    "nb": [
        "sys_duration",
        "sys_number",
        "sys_ordinal",
        "sys_time",
        "sys_interval",
        "sys_amount-of-money",
    ],
    "ne": ["sys_number"],
    "nl": [
        "sys_distance",
        "sys_duration",
        "sys_number",
        "sys_ordinal",
        "sys_quantity",
        "sys_time",
        "sys_interval",
        "sys_amount-of-money",
    ],
    "pl": ["sys_duration", "sys_number", "sys_ordinal", "sys_time", "sys_interval"],
    "pt": [
        "sys_distance",
        "sys_duration",
        "sys_number",
        "sys_ordinal",
        "sys_quantity",
        "sys_time",
        "sys_interval",
        "sys_amount-of-money",
    ],
    "ro": [
        "sys_distance",
        "sys_duration",
        "sys_number",
        "sys_ordinal",
        "sys_quantity",
        "sys_time",
        "sys_interval",
        "sys_amount-of-money",
    ],
    "ru": [
        "sys_distance",
        "sys_duration",
        "sys_number",
        "sys_ordinal",
        "sys_quantity",
        "sys_amount-of-money",
    ],
    "sk": ["sys_number"],
    "sv": [
        "sys_duration",
        "sys_duration",
        "sys_number",
        "sys_ordinal",
        "sys_time",
        "sys_interval",
        "sys_amount-of-money",
    ],
    "sw": ["sys_number"],
    "ta": ["sys_number", "sys_ordinal"],
    "te": ["sys_number"],
    "th": ["sys_number"],
    "tr": ["sys_distance", "sys_duration", "sys_number", "sys_ordinal"],
    "uk": ["sys_duration", "sys_number", "sys_ordinal", "sys_time", "sys_interval"],
    "vi": [
        "sys_duration",
        "sys_number",
        "sys_ordinal",
        "sys_time",
        "sys_interval",
        "sys_amount-of-money",
    ],
    "zh": [
        "sys_distance",
        "sys_duration",
        "sys_number",
        "sys_ordinal",
        "sys_quantity",
        "sys_time",
        "sys_interval",
    ],
}


# fetches all currency symbols in unicode by iterating through the character set and
# selecting the currency symbols based on the unicode currency category 'Sc'
CURRENCY_SYMBOLS = u"".join(
    chr(i) for i in range(0xFFFF) if unicodedata.category(chr(i)) == "Sc"
)
