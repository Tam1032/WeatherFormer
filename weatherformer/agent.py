import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from weatherformer.parser import parse_pending_call, remove_tool_calls
from weatherformer.tools import get_weather

# "->" pauses generation right after the model opens a tool call.
# "\n" ends the answer: training never taught the model to emit EOS,
# so without it the model keeps writing a new "Q: ..." line.
STOP = ["->", "\n"]


class Completer(Protocol):
    async def complete(self, prompt: str, stop: list[str], max_tokens: int) -> tuple[str, str | None]:
        """Return (generated_text, matched_stop_string or None)."""


class VLLMCompleter:
    def __init__(self, base_url: str, model: str, client: httpx.AsyncClient):
        self.url = f"{base_url}/v1/completions"
        self.model = model
        self.client = client

    async def complete(self, prompt: str, stop: list[str], max_tokens: int) -> tuple[str, str | None]:
        response = await self.client.post(self.url, json={
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stop": stop,
        })
        response.raise_for_status()
        choice = response.json()["choices"][0]
        return choice["text"], choice.get("stop_reason")


@dataclass
class ToolCall:
    city: str
    country: str
    result: str | None


@dataclass
class AgentResult:
    answer: str
    answer_with_trace: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_ms: float = 0.0


async def run_agent(question: str, completer: Completer, http: httpx.AsyncClient,
                    max_new_tokens: int = 256, max_tool_calls: int = 3) -> AgentResult:
    start = time.perf_counter()
    prompt = f"Q: {question}\nA:"
    output = ""
    calls: list[ToolCall] = []

    while True:
        text, stop_reason = await completer.complete(prompt + output, STOP, max_new_tokens)
        output += text
        if stop_reason != "->":
            break
        args = parse_pending_call(output)
        if args is None or len(calls) >= max_tool_calls:
            break
        result = await get_weather(args[0], args[1], http)
        calls.append(ToolCall(args[0], args[1], result))
        # "unavailable" instead of skipping, so the model does not invent numbers
        output = output.rstrip() + f" -> {result or 'unavailable'}]"

    trace = output.strip()
    return AgentResult(
        answer=remove_tool_calls(trace),
        answer_with_trace=trace,
        tool_calls=calls,
        latency_ms=(time.perf_counter() - start) * 1000,
    )
