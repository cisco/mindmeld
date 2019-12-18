def test_greeting(vi_kwik_e_mart_nlp):
    result = vi_kwik_e_mart_nlp.process("Xin chào")
    assert result["domain"] == "store_info"
    assert result["intent"] == "greet"

def test_find_nearest_store(vi_kwik_e_mart_nlp):
	result = vi_kwik_e_mart_nlp.process("Cửa hàng nào gần nhất tôi vậy?")
	assert result["domain"] == "store_info"
	assert result["intent"] == "find_nearest_store"

def test_get_store_hours(vi_kwik_e_mart_nlp):
	result = vi_kwik_e_mart_nlp.process("Central Plaza sẽ đóng cửa mấy giờ hôm nay?")
	assert result["domain"] == "store_info"
	assert result["intent"] == "get_store_hours"

def test_get_help(vi_kwik_e_mart_nlp):
	result = vi_kwik_e_mart_nlp.process("Giúp tôi")
	assert result["domain"] == "store_info"
	assert result["intent"] == "help"
