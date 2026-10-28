from weatherformer.parser import parse_pending_call, remove_tool_calls


def test_pending_call():
    assert parse_pending_call(" [Get_weather(Abuja, Nigeria) ") == ("Abuja", "Nigeria")
    assert parse_pending_call(" [Get_weather(San Francisco, United States)") == ("San Francisco", "United States")


def test_no_pending_call():
    assert parse_pending_call("It is sunny ") is None
    assert parse_pending_call(" [Get_weather(Abuja) ") is None
    assert parse_pending_call(" [Get_weather(Abuja, Nigeria) -> 30°C, clear sky] ok") is None


def test_remove_tool_calls():
    text = "[Get_weather(Helsinki, Finland) -> 7.1°C, overcast, wind 24.5 km/h] Not a great day."
    assert remove_tool_calls(text) == "Not a great day."
