def test_greeting(vi_kwik_e_mart_nlp):
    result = vi_kwik_e_mart_nlp.process("Xin chào")
    assert result["domain"] == "store_info"
    assert result["intent"] == "greet"

def test_get_store_hours(vi_kwik_e_mart_nlp):
	result = vi_kwik_e_mart_nlp.process("Central Plaza sẽ đóng cửa mấy giờ hôm nay?")
	assert result["domain"] == "store_info"
	assert result["intent"] == "get_store_hours"
