from mindmeld.test import TestConversation


def test_df_converter(mm_df_app):
    conv = TestConversation(app=mm_df_app)
    conv.process("what is my balance")
    conv.assert_text("Here's your latest balance:")
    conv.assert_domain("app_specific")
    conv.assert_intent("accountbalancecheck_en")
    conv.assert_frame({})
