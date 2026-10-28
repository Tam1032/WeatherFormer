import httpx
import pytest

from weatherformer import agent


class FakeCompleter:
    """Replays what vLLM returns for one tool-using answer."""

    def __init__(self):
        self.prompts = []
        self.replies = [
            (" [Get_weather(Helsinki, Finland) ", "->"),
            (" Not a great day for the beach.", "\n"),
        ]

    async def complete(self, prompt, stop, max_tokens):
        self.prompts.append(prompt)
        return self.replies.pop(0)


@pytest.mark.asyncio
async def test_agent_fills_tool_call(monkeypatch):
    async def fake_weather(city, country, client):
        assert (city, country) == ("Helsinki", "Finland")
        return "7.1°C, overcast, wind 24.5 km/h"

    monkeypatch.setattr(agent, "get_weather", fake_weather)
    completer = FakeCompleter()
    async with httpx.AsyncClient() as http:
        result = await agent.run_agent("Beach day in Helsinki, Finland?", completer, http)

    assert completer.prompts[1].endswith("[Get_weather(Helsinki, Finland) -> 7.1°C, overcast, wind 24.5 km/h]")
    assert result.answer == "Not a great day for the beach."
    assert result.tool_calls[0].result == "7.1°C, overcast, wind 24.5 km/h"


@pytest.mark.asyncio
async def test_agent_marks_failed_tool(monkeypatch):
    async def failing_weather(city, country, client):
        return None

    monkeypatch.setattr(agent, "get_weather", failing_weather)
    completer = FakeCompleter()
    async with httpx.AsyncClient() as http:
        result = await agent.run_agent("Beach day in Helsinki, Finland?", completer, http)

    assert "-> unavailable]" in result.answer_with_trace
