import csv
from pathlib import Path

import numpy as np

from src.data import get_all_label_names
from src.classify import build_prompt, normalize
from src.llm import ask_model
from src.embed_shortlist import embed, label_to_text, make_shortlist

DATA = Path("data/hinglish_queries.csv")


def load_rows() -> list[dict]:
    with DATA.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run(model: str = "llama3.2:3b", k: int = 10, use_shortlist: bool = False):
    labels = sorted(get_all_label_names())
    valid = set(labels)
    rows = load_rows()

    bad = [r["label"] for r in rows if r["label"] not in valid]
    if bad:
        raise SystemExit(f"CSV mein galat labels hain: {bad}")

    label_matrix = embed([label_to_text(l) for l in labels], prefix="search_document: ")

    correct = 0
    total_tokens = 0
    total_latency = 0.0
    mode = f"embed_shortlist_k{k}" if use_shortlist else "baseline"

    for r in rows:
        if use_shortlist:
            qv = embed([r["text"]], prefix="search_query: ")[0]
            cands = make_shortlist(qv, label_matrix, labels, k)
        else:
            cands = labels

        prompt = build_prompt(r["text"], cands)
        out = ask_model(prompt, model=model, strategy=f"hinglish_{mode}")
        pred = normalize(out["answer"], valid)
        ok = pred == r["label"]
        correct += ok
        total_tokens += out["input_tokens"] + out["output_tokens"]
        total_latency += out["latency_ms"]
        print(f"[{'OK' if ok else 'X'}] truth={r['label']} | pred={pred}")

    n = len(rows)
    print(f"\nHinglish | mode={mode} | model={model}")
    print(f"Queries: {n}")
    print(f"Accuracy: {correct}/{n} = {correct / n:.0%}")
    print(f"Avg latency: {total_latency / n:.0f} ms")
    print(f"Total tokens: {total_tokens}")


if __name__ == "__main__":
    run()