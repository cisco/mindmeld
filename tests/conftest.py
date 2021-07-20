# -*- coding: utf-8 -*-

"""
conftest
----------------------------------

Configurations for tests. Include shared fixtures here.
"""
# pylint: disable=locally-disabled,redefined-outer-name
import asyncio
import codecs
from distutils.util import strtobool
import os
import sys
import warnings

import pytest

from mindmeld.converter.rasa import RasaConverter
from mindmeld.converter.dialogflow import DialogflowConverter
from mindmeld.components import NaturalLanguageProcessor, QuestionAnswerer
from mindmeld.components._elasticsearch_helpers import create_es_client
from mindmeld.query_factory import QueryFactory
from mindmeld.resource_loader import ResourceLoader
from mindmeld.system_entity_recognizer import DucklingRecognizer
from mindmeld.text_preparation.preprocessor import Preprocessor
from mindmeld.text_preparation.stemmers import EnglishNLTKStemmer
from mindmeld.text_preparation.tokenizers import WhiteSpaceTokenizer
from mindmeld.text_preparation.normalizers import RegexNormalizer, Lowercase, ASCIIFold
from mindmeld.text_preparation.text_preparation_pipeline import TextPreparationPipeline

warnings.filterwarnings(
    "module", category=DeprecationWarning, module="sklearn.preprocessing.label"
)


APP_NAME = "kwik_e_mart"
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), APP_NAME)
FOOD_ORDERING_APP_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "food_ordering"
)
HOME_ASSISTANT_APP_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "home_assistant"
)
AENEID_FILE = "aeneid.txt"
AENEID_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), AENEID_FILE)
STORE_DATA_FILE_PATH = os.path.join(APP_PATH, "data/stores.json")

CONVERTER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'converter')
RASA_CONVERTER_PROJECT_PATH = os.path.join(CONVERTER_PATH, 'rasa_sample_project')
DIALOG_CONVERTER_PROJECT_PATH = os.path.join(CONVERTER_PATH, 'dialogflow_sample_project')
MINDMELD_RASA_CONVERTER_PROJECT_PATH = os.path.join(CONVERTER_PATH, 'mm_rasa_converted_project')
MINDMELD_DIALOG_CONVERTER_PROJECT_PATH = os.path.join(CONVERTER_PATH, 'mm_df_converted_project')


@pytest.fixture
def lstm_entity_config():
    return {
        "model_type": "tagger",
        "label_type": "entities",
        "model_settings": {
            "classifier_type": "lstm",
            "tag_scheme": "IOB",
            "feature_scaler": "max-abs",
        },
        "params": {"number_of_epochs": 1},
        "features": {
            "in-gaz-span-seq": {},
        },
    }


@pytest.fixture(scope="session")
def kwik_e_mart_app_path():
    return APP_PATH


@pytest.fixture(scope="session")
def food_ordering_app_path():
    return FOOD_ORDERING_APP_PATH


@pytest.fixture(scope="session")
def home_assistant_app_path():
    return HOME_ASSISTANT_APP_PATH


@pytest.fixture(scope="session")
def kwik_e_mart_nlp(kwik_e_mart_app_path):
    """Provides a built processor instance"""
    nlp = NaturalLanguageProcessor(app_path=kwik_e_mart_app_path)
    nlp.build()
    nlp.dump()
    return nlp


@pytest.fixture(scope="session")
def food_ordering_nlp(food_ordering_app_path):
    """Provides a built processor instance"""
    nlp = NaturalLanguageProcessor(app_path=food_ordering_app_path)
    nlp.build()
    nlp.dump()
    return nlp


@pytest.fixture
def home_assistant_nlp(home_assistant_app_path):
    """Provides a built processor instance"""
    nlp = NaturalLanguageProcessor(app_path=home_assistant_app_path)
    nlp.build()
    nlp.dump()
    return nlp


@pytest.fixture(scope="session")
def kwik_e_mart_app(kwik_e_mart_nlp):
    from .kwik_e_mart import app

    app.lazy_init(kwik_e_mart_nlp)
    return app


@pytest.fixture(scope="session")
def async_kwik_e_mart_app(kwik_e_mart_nlp):
    from .kwik_e_mart import app_async

    app = app_async.app
    app.lazy_init(kwik_e_mart_nlp)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(app.app_manager.load())
    return app


@pytest.fixture
def es_client():
    """An Elasticsearch client"""
    return create_es_client()


@pytest.fixture
def qa_kwik_e_mart(kwik_e_mart_app_path, es_client):
    QuestionAnswerer.load_kb(
        app_namespace="kwik_e_mart",
        index_name="stores",
        data_file=STORE_DATA_FILE_PATH,
    )
    qa = QuestionAnswerer(kwik_e_mart_app_path)
    return qa


@pytest.fixture(scope="session")
def rasa_converter():
    converter = RasaConverter(RASA_CONVERTER_PROJECT_PATH,
                              MINDMELD_RASA_CONVERTER_PROJECT_PATH)
    converter.convert_project()
    return converter


@pytest.fixture(scope="session")
def mindmeld_rasa_converter_app_path():
    return MINDMELD_RASA_CONVERTER_PROJECT_PATH


@pytest.fixture(scope="session")
def dialogflow_converter():
    return DialogflowConverter(DIALOG_CONVERTER_PROJECT_PATH,
                               MINDMELD_DIALOG_CONVERTER_PROJECT_PATH)


class GhostPreprocessor(Preprocessor):
    """A simple preprocessor that removes all instances of the word `ghost`"""

    def process(self, text):
        while "ghost" in text:
            text = text.replace("ghost", "")
        return text


@pytest.fixture
def tokenizer():
    """A tokenizer for normalizing text"""
    return WhiteSpaceTokenizer()


@pytest.fixture
def preprocessor():
    """A simple preprocessor"""
    return GhostPreprocessor()


@pytest.fixture
def stemmer():
    """The english stemmer"""
    return EnglishNLTKStemmer()


@pytest.fixture
def duckling():
    return DucklingRecognizer.get_instance()


@pytest.fixture
def regex_normalizer():
    regex_norm_rules = [
        {
            "description": "Remove apostrophe at the end of possessive form",
            "pattern": "^'(?=\S)|(?<=\S)'$",
            "replacement": ""
        },
        {
            "description": "Remove Apostrophes and Spaces Preceding or Proceding",
            "pattern": " '|' ",
            "replacement": "" 
        },
        {
            "description": "Remove Beginning Spaces",
            "pattern": "^\s+",
            "replacement": "" 
        },
        {
            "description": "Remove Trailing Spaces",
            "pattern": "\s+$",
            "replacement": "" 
        },
        {
            "description": "Remove All Spaces",
            "pattern": "\s+",
            "replacement": "" 
        },
        {
            "description": "Remove Underscores",
            "pattern": "_",
            "replacement": "" 
        },
        {
            "description": "Separate 's to a separate word'",
            "pattern": "(?<=[^\\s])'[sS]",
            "replacement": " 's" 
        }
    ]
    return RegexNormalizer(regex_norm_rules)


@pytest.fixture
def text_preparation_pipeline(preprocessor, tokenizer, stemmer, regex_normalizer):
    """The Text Preparation Pipeline Object"""
    return TextPreparationPipeline(
        preprocessors=[preprocessor],
        normalizers=[regex_normalizer, Lowercase(), ASCIIFold()],
        tokenizer=tokenizer,
        stemmer=stemmer
    )


@pytest.fixture
def query_factory(text_preparation_pipeline):
    """For creating queries"""
    return QueryFactory(
        text_preparation_pipeline=text_preparation_pipeline,
        system_entity_recognizer=None,
        duckling=True,
    )


@pytest.fixture
def resource_loader(query_factory):
    """A resource loader"""
    return ResourceLoader(APP_PATH, query_factory)


@pytest.fixture
def aeneid_path():
    return AENEID_PATH


@pytest.fixture
def aeneid_content(aeneid_path):
    with codecs.open(aeneid_path, mode="r", encoding="utf-8") as f:
        return f.read()


class FakeApp:
    def __init__(self, app_path):
        self.app_path = app_path


@pytest.fixture
def fake_app():
    return FakeApp("123")


def pytest_collection_modifyitems(config, items):
    use_extras = strtobool(os.environ.get("MM_EXTRAS", "false"))
    skip_markers = ["no_extras"] if use_extras else ["extras"]
    skip = pytest.mark.skip(
        reason=(
            "Skipping tests which require a clean mindmeld install"
            if use_extras
            else "Skipping tests which require mindmeld extras"
        )
    )

    py_version = sys.version_info

    # skip bert test for python 3.5 and below with extras
    if py_version.minor < 6 and use_extras:
        skip_markers.append("bert")

    try:
        from elasticsearch import Elasticsearch

        es = Elasticsearch()
        es_version = es.info()["version"]["number"]
        (major, _, _) = es_version.split(".")
        if int(major) < 7:
            skip_markers.append("es7")
    except ModuleNotFoundError:
        skip_markers.append("es7")

    for item in items:
        for marker in skip_markers:
            if marker in item.keywords:
                item.add_marker(skip)
