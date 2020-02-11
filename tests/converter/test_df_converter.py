import os
import shutil
import importlib

from mindmeld.components import NaturalLanguageProcessor
from mindmeld.test import TestConversation
from mindmeld.converter.dialogflow import DialogflowConverter


def test_df_converter():
    df_project_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dialogflow_sample_project"
    )

    # This is the dialogflow app converted to mindmeld app
    mm_df_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "mm_df_converted_project"
    )

    df_init = DialogflowConverter(df_project_path, mm_df_path)
    df_init.convert_project()

    mm_df_nlp = NaturalLanguageProcessor(app_path=mm_df_path)
    mm_df_nlp.build()

    # check to make sure the NLP object contains the correct hierarchy
    assert set(mm_df_nlp.domains.keys()) == {'app_specific', 'unrelated'}

    assert set(mm_df_nlp.domains['app_specific'].intents.keys()) == {
        'accountopen_en', 'accountbalancecheck_en',
        'accountearningcheck_context__earning_date_en',
        'transfermoney_no_en', 'accountbalancecheck_context__account_en',
        'transfermoney_yes_en', 'transfermoney_en', 'transferamountcheck_en',
        'paymentdue_date_en', 'accountspendingcheck_context__spending_date_en',
        'transferdatecheck_en', 'accountspendingcheck_en', 'transfersendercheck_en',
        'accountbalancecheck_context__balance_en', 'accountearningcheck_en'}

    assert set(mm_df_nlp.domains['unrelated'].intents.keys()) == {
        'default_welcome_intent_en', 'default_fallback_intent_en'}

    entities = set()
    for domain in mm_df_nlp.domains:
        for intent in mm_df_nlp.domains[domain].intents:
            for entity in mm_df_nlp.domains[domain].intents[intent].entities:
                entities.add(entity)

    for expected_entity in {'category_en', 'transfer_type_en', 'merchant_en', 'account_en'}:
        assert expected_entity in entities

    mm_df_app = importlib.import_module('mm_df_converted_project').app
    mm_df_app.lazy_init(mm_df_nlp)

    conv = TestConversation(app=mm_df_app)
    conv.process("what is my balance")
    conv.assert_text("Here's your latest balance:")
    conv.assert_domain("app_specific")
    conv.assert_intent("accountbalancecheck_en")
    conv.assert_frame({})

    conv.process("when is the due date")
    conv.assert_text("The due date is:")
    conv.assert_domain("app_specific")
    conv.assert_intent("paymentdue_date_en")
    conv.assert_frame({})

    conv.process("show my withdrawals")
    conv.assert_text("Here are your withdrawals:")
    conv.assert_domain("app_specific")
    conv.assert_intent("accountspendingcheck_en")
    conv.assert_frame({})

    # delete generated files
    shutil.rmtree(mm_df_path)
