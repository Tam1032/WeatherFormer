"""Accuracy and latency for one serving configuration.

API mode (vLLM fp16 + LoRA, or AWQ, behind FastAPI):
    python scripts/benchmark.py --name vllm-lora --url http://localhost:8080 --concurrency 1 8 32
HF baseline (original 4-bit bitsandbytes loop, one request at a time):
    python scripts/benchmark.py --name hf-bnb --hf
"""
import argparse
import asyncio
import json
import re
import statistics
import time
from pathlib import Path

import httpx

CALL = re.compile(r"\[Get_weather\(([^,()]*),\s*([^()]*)\)")


def load_questions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_call(answer: str) -> tuple[str, str] | None:
    m = CALL.search(answer)
    return (m.group(1).strip().lower(), m.group(2).strip().lower()) if m else None


def score(rows: list[dict], answers: list[str]) -> dict:
    tool = [(r, parse_call(a)) for r, a in zip(rows, answers) if r["expect_call"]]
    no_tool = [parse_call(a) for r, a in zip(rows, answers) if not r["expect_call"]]
    return {
        "tool_call_recall": statistics.mean(call is not None for _, call in tool),
        "argument_accuracy": statistics.mean(call == (r["city"].lower(), r["country"].lower()) for r, call in tool),
        "false_call_rate": statistics.mean(call is not None for call in no_tool),
    }


def latency_stats(latencies: list[float], wall_s: float) -> dict:
    return {
        "p50_s": round(statistics.median(latencies), 3),
        "p95_s": round(statistics.quantiles(latencies, n=100)[94], 3),
        "throughput_rps": round(len(latencies) / wall_s, 3),
    }


async def run_api(url: str, rows: list[dict], concurrency: int):
    sem = asyncio.Semaphore(concurrency)
    answers, latencies = [""] * len(rows), [0.0] * len(rows)
    async with httpx.AsyncClient(timeout=300) as client:
        await client.post(f"{url}/v1/ask", json={"question": rows[0]["question"]})  # warm-up

        async def one(i: int, row: dict):
            async with sem:
                t0 = time.perf_counter()
                r = await client.post(f"{url}/v1/ask", json={"question": row["question"]})
                r.raise_for_status()
                latencies[i] = time.perf_counter() - t0
                answers[i] = r.json()["answer_with_trace"]

        t0 = time.perf_counter()
        await asyncio.gather(*(one(i, row) for i, row in enumerate(rows)))
        wall_s = time.perf_counter() - t0
    return answers, latencies, wall_s


def run_hf(rows: list[dict]):
    import torch

    from weatherformer.legacy_hf import generate_with_real_tools, load_model

    model, tokenizer = load_model()
    generate_with_real_tools(rows[0]["question"], model, tokenizer)  # warm-up
    answers, latencies = [], []
    t_start = time.perf_counter()
    for row in rows:
        t0 = time.perf_counter()
        answers.append(generate_with_real_tools(row["question"], model, tokenizer))
        latencies.append(time.perf_counter() - t0)
    wall_s = time.perf_counter() - t_start
    return answers, latencies, wall_s, round(torch.cuda.max_memory_allocated() / 1e9, 2)


def save(name: str, rows: list[dict], answers: list[str], result: dict):
    out = Path("results")
    out.mkdir(exist_ok=True)
    with open(out / f"answers-{name}.jsonl", "w", encoding="utf-8") as f:
        for row, answer in zip(rows, answers):
            f.write(json.dumps({**row, "answer": answer}, ensure_ascii=False) + "\n")
    with open(out / "benchmark.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--questions", default="eval/questions.jsonl")
    parser.add_argument("--url", help="FastAPI base URL")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1])
    parser.add_argument("--hf", action="store_true", help="run the original HF + bitsandbytes loop")
    args = parser.parse_args()
    rows = load_questions(args.questions)

    if args.hf:
        answers, latencies, wall_s, vram_gb = run_hf(rows)
        save(args.name, rows, answers, {"name": args.name, "concurrency": 1, "peak_vram_gb": vram_gb,
                                        **latency_stats(latencies, wall_s), **score(rows, answers)})
        return

    for concurrency in args.concurrency:
        answers, latencies, wall_s = asyncio.run(run_api(args.url, rows, concurrency))
        save(f"{args.name}-c{concurrency}", rows, answers, {"name": args.name, "concurrency": concurrency,
                                                           **latency_stats(latencies, wall_s), **score(rows, answers)})


if __name__ == "__main__":
    main()
