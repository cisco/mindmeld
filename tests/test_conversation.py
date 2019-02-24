from mmworkbench.test import TestConversation


def test_conversation(kwik_e_mart_app_path):
    conv = TestConversation(app_path=kwik_e_mart_app_path)
    conv.process('hi')
    conv.assert_text('Hello. I can help you find store hours for your local Kwik-E-Mart.'
                     ' How can I help?')
    conv.assert_domain('store_info')
    conv.assert_intent('greet')
    conv.assert_frame({})
