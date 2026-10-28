from fastapi.testclient import TestClient

from api import main
from weatherformer import agent


class FakeCompleter:
    async def complete(self, prompt, stop, max_tokens):
        if "->" in prompt:
            return " Yes, bring a jacket.", "\n"
        return " [Get_weather(Seoul, South Korea) ", "->"


def test_ask(monkeypatch):
    async def fake_weather(city, country, client):
        return "5.0°C, overcast, wind 10.0 km/h"

    monkeypatch.setattr(agent, "get_weather", fake_weather)
    with TestClient(main.app) as client:
        main.app.state.completer = FakeCompleter()
        response = client.post("/v1/ask", json={"question": "Jacket in Seoul, South Korea?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Yes, bring a jacket."
    assert body["tool_calls"][0]["city"] == "Seoul"


def test_ask_rejects_empty_question():
    with TestClient(main.app) as client:
        assert client.post("/v1/ask", json={"question": ""}).status_code == 422
