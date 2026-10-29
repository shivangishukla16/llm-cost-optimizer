import csv
import time
from pathlib import Path

import ollama

LOG_FILE = Path("results/llm_calls.csv")


def _log_call(row: dict) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def ask_model(prompt: str, model: str = "llama3.2:3b", strategy: str = "baseline") -> dict:
    start = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    latency_ms = (time.perf_counter() - start) * 1000

    result = {
        "strategy": strategy,
        "model": model,
        "prompt": prompt,
        "answer": response["message"]["content"],
        "input_tokens": response.get("prompt_eval_count", 0),
        "output_tokens": response.get("eval_count", 0),
        "latency_ms": round(latency_ms, 1),
    }
    _log_call(result)
    return result


if __name__ == "__main__":
    out = ask_model("Ek line mein bolo: API kaam kar raha hai?")
    print("Jawab:", out["answer"])
    print("Input tokens:", out["input_tokens"])
    print("Output tokens:", out["output_tokens"])
    print("Latency (ms):", out["latency_ms"])