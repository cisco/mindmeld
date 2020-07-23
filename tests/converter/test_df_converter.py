import os
import pytest
import shutil
import importlib

from mindmeld.components import NaturalLanguageProcessor
from mindmeld.test import ConversationTestHelper
from mindmeld.converter.dialogflow import DialogflowConverter


@pytest.mark.skip(reason="Test is taking too long to pass and feature is experimental")
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
    assert set(mm_df_nlp.domains.keys()) == {"app_specific", "unrelated"}

    assert set(mm_df_nlp.domains["app_specific"].intents.keys()) == {
        "accountopen_en",
        "accountbalancecheck_en",
        "accountearningcheck_context__earning_date_en",
        "transfermoney_no_en",
        "accountbalancecheck_context__account_en",
        "transfermoney_yes_en",
        "transfermoney_en",
        "transferamountcheck_en",
        "paymentdue_date_en",
        "accountspendingcheck_context__spending_date_en",
        "transferdatecheck_en",
        "accountspendingcheck_en",
        "transfersendercheck_en",
        "accountbalancecheck_context__balance_en",
        "accountearningcheck_en",
    }

    assert set(mm_df_nlp.domains["unrelated"].intents.keys()) == {
        "default_welcome_intent_en",
        "default_fallback_intent_en",
    }

    entities = set()
    for domain in mm_df_nlp.domains:
        for intent in mm_df_nlp.domains[domain].intents:
            for entity in mm_df_nlp.domains[domain].intents[intent].entities:
                entities.add(entity)

    for expected_entity in {
        "category_en",
        "transfer_type_en",
        "merchant_en",
        "account_en",
    }:
        assert expected_entity in entities

    mm_df_app = importlib.import_module("mm_df_converted_project").app
    mm_df_app.lazy_init(mm_df_nlp)

    conv = ConversationTestHelper(app=mm_df_app)
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

    conv.process("transfer money")
    conv.assert_text("Sure. Transfer from which account?")

    conv.process("checking account")
    conv.assert_text("To which account?")

    conv.process("transfer to savings account")
    conv.assert_text("And, how much do you want to transfer?")

    conv.process("transfer $200")
    conv.assert_text(
        "All right. So, you're transferring $200 from your checking to a savings. Is that right?"
    )

    conv.process("hello!")
    conv.assert_text(
        ["Hello, thanks for choosing ACME Bank.", "Hello. Welcome to ACME Bank."]
    )

    conv.process("I dont know what the laptop")
    conv.assert_text(
        [
            "Sorry, I didn’t get that.",
            "I'm afraid I don't understand.",
            "Sorry, say that again?",
            "Sorry, can you say that again?",
            "I didn't get that. Can you say it again?",
            "Sorry, could you say that again?",
            "Sorry, can you tell me again?",
            "Sorry, tell me one more time?",
            "Sorry, can you say that again?",
        ]
    )

    # delete generated files
    shutil.rmtree(mm_df_path)
