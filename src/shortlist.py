import math
import re
from collections import Counter

from src.data import load_banking77_sample, get_all_label_names
from src.classify import build_prompt, normalize
from src.llm import ask_model

STOP = {
    "i", "my", "the", "a", "an", "is", "it", "to", "of", "do", "what", "how",
    "why", "if", "on", "in", "for", "and", "be", "not", "me", "im", "that",
    "this", "with", "from", "at", "can", "get", "there", "does", "so", "when",
    "was", "are", "have", "has", "its", "or", "just", "please", "help", "want",
    "need", "i'm", "you", "your", "they", "their", "been", "will", "would",
}


def tokens(text: str) -> list[str]:
    text = text.lower().replace("_", " ")
    return [w for w in re.findall(r"[a-z]+", text) if w not in STOP]


def build_idf(labels: list[str]) -> dict[str, float]:
    df = Counter()
    for label in labels:
        df.update(set(tokens(label)))
    n = len(labels)
    return {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}


def make_shortlist(query: str, labels: list[str], idf: dict[str, float], k: int = 10) -> list[str]:
    q = set(tokens(query))
    scored = []
    for label in labels:
        overlap = q & set(tokens(label))
        score = sum(idf.get(w, 0.0) for w in overlap)
        scored.append((score, label))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [label for _, label in scored[:k]]


def run(n: int = 50, model: str = "llama3.2:3b", k: int = 10):
    samples = load_banking77_sample(n)
    all_labels = sorted(get_all_label_names())
    valid = set(all_labels)
    idf = build_idf(all_labels)

    correct = 0
    in_shortlist = 0
    total_latency = 0.0
    total_tokens = 0

    for item in samples:
        shortlist = make_shortlist(item["text"], all_labels, idf, k=k)
        hit = item["label"] in shortlist
        in_shortlist += hit

        prompt = build_prompt(item["text"], shortlist)
        out = ask_model(prompt, model=model, strategy=f"shortlist_k{k}")
        predicted = normalize(out["answer"], valid)

        ok = predicted == normalize(item["label"], valid)
        correct += ok
        total_latency += out["latency_ms"]
        total_tokens += out["input_tokens"] + out["output_tokens"]
        print(f"[{'OK' if ok else 'X'}] truth={item['label']} | pred={predicted} | in_shortlist={hit}")

    print(f"\nShortlist k={k}, model={model}")
    print(f"Recall (sahi label shortlist mein): {in_shortlist}/{n} = {in_shortlist / n:.0%}")
    print(f"Accuracy: {correct}/{n} = {correct / n:.0%}")
    print(f"Avg latency: {total_latency / n:.0f} ms")
    print(f"Total tokens: {total_tokens}")


if __name__ == "__main__":
    run(n=50, k=10)