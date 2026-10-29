from src.data import load_banking77_sample, get_all_label_names
from src.classify import build_prompt, normalize
from src.llm import ask_model


def is_valid_label(predicted: str, valid_labels: set[str]) -> bool:
    return predicted in valid_labels


def run_routed(n: int = 50, cheap: str = "llama3.2:3b", big: str = "llama3.1:8b"):
    samples = load_banking77_sample(n)
    all_labels = sorted(get_all_label_names())
    valid = set(all_labels)

    correct = 0
    total_latency = 0.0
    total_tokens = 0
    escalated = 0

    for item in samples:
        prompt = build_prompt(item["text"], all_labels)

        out = ask_model(prompt, model=cheap, strategy="routed_cheap")
        predicted = normalize(out["answer"], valid)
        latency = out["latency_ms"]
        tokens = out["input_tokens"] + out["output_tokens"]

        if not is_valid_label(predicted, valid):
            escalated += 1
            try:
                out = ask_model(prompt, model=big, strategy="routed_escalated")
                predicted = normalize(out["answer"], valid)
                latency += out["latency_ms"]
                tokens += out["input_tokens"] + out["output_tokens"]
            except Exception as e:
                print(f"  [escalation failed, cheap answer rakha: {e}]")

        ok = predicted == normalize(item["label"], valid)
        correct += ok
        total_latency += latency
        total_tokens += tokens
        print(f"[{'OK' if ok else 'X'}] truth={item['label']} | pred={predicted}")

    print(f"\nRouted (cheap={cheap}, big={big})")
    print(f"Escalated to big model: {escalated}/{n}")
    print(f"Accuracy: {correct}/{n} = {correct / n:.0%}")
    print(f"Avg latency: {total_latency / n:.0f} ms")
    print(f"Total tokens: {total_tokens}")


if __name__ == "__main__":
    run_routed(n=50)