import time

import numpy as np
import ollama

from src.data import load_banking77_sample, get_all_label_names
from src.classify import build_prompt, normalize
from src.llm import ask_model

EMBED_MODEL = "nomic-embed-text"


def embed(texts: list[str], prefix: str) -> np.ndarray:
    res = ollama.embed(model=EMBED_MODEL, input=[prefix + t for t in texts])
    v = np.array(res["embeddings"], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def label_to_text(label: str) -> str:
    return label.replace("_", " ")


def make_shortlist(query_vec: np.ndarray, label_matrix: np.ndarray, labels: list[str], k: int) -> list[str]:
    sims = label_matrix @ query_vec
    top = np.argsort(-sims)[:k]
    return [labels[i] for i in top]


def run(n: int = 50, model: str = "llama3.2:3b", k: int = 10):
    samples = load_banking77_sample(n)
    labels = sorted(get_all_label_names())
    valid = set(labels)

    label_matrix = embed([label_to_text(l) for l in labels], prefix="search_document: ")

    correct = 0
    in_shortlist = 0
    total_latency = 0.0
    total_tokens = 0
    total_embed_ms = 0.0

    for item in samples:
        t0 = time.perf_counter()
        query_vec = embed([item["text"]], prefix="search_query: ")[0]
        total_embed_ms += (time.perf_counter() - t0) * 1000

        shortlist = make_shortlist(query_vec, label_matrix, labels, k)
        hit = item["label"] in shortlist
        in_shortlist += hit

        prompt = build_prompt(item["text"], shortlist)
        out = ask_model(prompt, model=model, strategy=f"embed_shortlist_k{k}")
        predicted = normalize(out["answer"], valid)

        ok = predicted == normalize(item["label"], valid)
        correct += ok
        total_latency += out["latency_ms"]
        total_tokens += out["input_tokens"] + out["output_tokens"]
        print(f"[{'OK' if ok else 'X'}] truth={item['label']} | pred={predicted} | in_shortlist={hit}")

    print(f"\nEmbedding shortlist k={k}, model={model}")
    print(f"Recall (sahi label shortlist mein): {in_shortlist}/{n} = {in_shortlist / n:.0%}")
    print(f"Accuracy: {correct}/{n} = {correct / n:.0%}")
    print(f"Avg LLM latency: {total_latency / n:.0f} ms")
    print(f"Avg embed latency: {total_embed_ms / n:.0f} ms")
    print(f"Total LLM tokens: {total_tokens}")


if __name__ == "__main__":
    run(n=50, k=10)