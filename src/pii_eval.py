import csv
from pathlib import Path

from src.data import get_all_label_names
from src.classify import build_prompt, normalize
from src.llm import ask_model
from src.pii_mask import mask_pii

DATA = Path("data/pii_queries.csv")


def load_rows() -> list[dict]:
    with DATA.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run(model: str = "llama3.2:3b"):
    labels = sorted(get_all_label_names())
    valid = set(labels)
    rows = load_rows()

    bad = [r["label"] for r in rows if r["label"] not in valid]
    if bad:
        raise SystemExit(f"CSV mein galat labels hain: {bad}")

    stats = {mode: {"correct": 0, "leaks": 0, "tokens": 0} for mode in ("raw", "masked")}

    for r in rows:
        truth = normalize(r["label"], valid)
        for mode in ("raw", "masked"):
            text = r["text"] if mode == "raw" else mask_pii(r["text"])[0]
            prompt = build_prompt(text, labels)
            out = ask_model(prompt, model=model, strategy=f"pii_{mode}")
            pred = normalize(out["answer"], valid)

            s = stats[mode]
            s["correct"] += pred == truth
            s["tokens"] += out["input_tokens"] + out["output_tokens"]
            # Leak: asli PII value LLM ko bheje gaye prompt mein hai ya nahi
            if r["pii_value"] and r["pii_value"] in prompt:
                s["leaks"] += 1

            ok = "OK" if pred == truth else "X"
            print(f"[{mode:6}] [{ok}] truth={truth} | pred={pred} | pii_sent={bool(r['pii_value'] and r['pii_value'] in prompt)}")

    total = len(rows)
    pii_rows = sum(1 for r in rows if r["pii_value"])
    print(f"\nModel: {model}")
    print(f"Queries: {total} (PII wali: {pii_rows})")
    for mode in ("raw", "masked"):
        s = stats[mode]
        print(f"\n[{mode}]")
        print(f"  Accuracy: {s['correct']}/{total} = {s['correct'] / total:.0%}")
        print(f"  PII LLM tak pahunchi: {s['leaks']}/{pii_rows}")
        print(f"  Total tokens: {s['tokens']}")


if __name__ == "__main__":
    run()