def test_greeting(vi_kwik_e_mart_nlp):
    result = vi_kwik_e_mart_nlp.process("Xin chào")
    assert result["domain"] == "store_info"
    assert result["intent"] == "greet"
