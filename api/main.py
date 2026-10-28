import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from weatherformer.agent import VLLMCompleter, run_agent

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "weather")  # LoRA name given to --lora-modules


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
    app.state.completer = VLLMCompleter(VLLM_URL, MODEL_NAME, app.state.http)
    yield
    await app.state.http.aclose()


app = FastAPI(title="WeatherFormer API", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    max_new_tokens: int = Field(default=256, ge=16, le=512)


class ToolCallOut(BaseModel):
    city: str
    country: str
    result: str | None


class AskResponse(BaseModel):
    answer: str
    answer_with_trace: str
    tool_calls: list[ToolCallOut]
    latency_ms: float


@app.post("/v1/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    try:
        result = await run_agent(req.question, app.state.completer, app.state.http, req.max_new_tokens)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"LLM backend error: {exc}") from exc
    return AskResponse(
        answer=result.answer,
        answer_with_trace=result.answer_with_trace,
        tool_calls=[ToolCallOut(**vars(c)) for c in result.tool_calls],
        latency_ms=round(result.latency_ms, 1),
    )


@app.get("/health")
async def health() -> dict:
    try:
        response = await app.state.http.get(f"{VLLM_URL}/health", timeout=5)
        ready = response.status_code == 200
    except httpx.HTTPError:
        ready = False
    if not ready:
        raise HTTPException(status_code=503, detail="vLLM not ready")
    return {"status": "ok"}
