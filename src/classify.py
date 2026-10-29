import re

from src.data import load_banking77_sample, get_all_label_names
from src.llm import ask_model
from src.pii_mask import mask_pii


def build_prompt(query: str, labels: list[str]) -> str:
    label_list = ", ".join(labels)
    return (
        "Neeche ek banking customer ka sawaal hai. "
        "Possible labels mein se sabse sahi ek label chuno. "
        "Sirf label likho, koi aur text nahi.\n"
        f"Possible labels: {label_list}\n"
        f"Sawaal: {query}\n"
        "Label:"
    )


def normalize(text: str, valid_labels: set[str] | None = None) -> str:
    """Model ke jawab ya truth label se clean label nikalta hai."""
    text = text.strip().lower()
    text = re.sub(r"^(label|answer|intent)\s*:\s*", "", text)
    text = text.splitlines()[0] if text else text
    candidate = re.sub(r"[^a-z0-9_ ]", "", text).strip().replace(" ", "_")
    if valid_labels is None or candidate in valid_labels:
        return candidate
    for token in re.findall(r"[a-z0-9_]+", text):
        if token in valid_labels:
            return token
    return candidate


def run(n: int = 50, model: str = "llama3.2:3b", strategy: str = "baseline_classify"):
    samples = load_banking77_sample(n)
    all_labels = sorted(get_all_label_names())
    valid = set(all_labels)
    correct = 0
    total_latency = 0.0
    total_tokens = 0
    total_pii = 0

    for item in samples:
        # PII pehle mask hoga, prompt aur log dono mein masked text jayega
        masked_text, counts = mask_pii(item["text"])
        total_pii += sum(counts.values())

        prompt = build_prompt(masked_text, all_labels)
        out = ask_model(prompt, model=model, strategy=strategy)
        predicted = normalize(out["answer"], valid)
        ok = predicted == normalize(item["label"], valid)
        correct += ok
        total_latency += out["latency_ms"]
        total_tokens += out["input_tokens"] + out["output_tokens"]
        print(f"[{'OK' if ok else 'X'}] truth={item['label']} | pred={predicted}")

    print(f"\nModel: {model}")
    print(f"PII tokens masked: {total_pii}")
    print(f"Accuracy: {correct}/{n} = {correct / n:.0%}")
    print(f"Avg latency: {total_latency / n:.0f} ms")
    print(f"Total tokens: {total_tokens}")


if __name__ == "__main__":
    run(n=50)