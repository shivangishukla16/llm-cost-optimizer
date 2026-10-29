from src.data import load_banking77_sample, get_all_label_names
from src.classify import build_prompt, normalize
from src.llm import ask_model
from src.semantic_cache import SemanticCache

PARAPHRASES = [
    "mera card block ho gaya kya karu",
    "mera card block ho gaya hai, kya karna chahiye",
    "card band ho gaya hai, ab kya karoon",
    "transfer abhi tak pending hai",
    "mera paisa transfer nahi hua abhi tak",
    "transfer still pending hai kya karu",
]


def run(model: str = "llama3.2:3b", threshold: float = 0.90):
    labels = sorted(get_all_label_names())
    valid = set(labels)
    cache = SemanticCache(threshold)
    calls = 0
    hits = 0
    for text in PARAPHRASES:
        ans = cache.lookup(text)
        if ans is not None:
            hits += 1
            print(f"[HIT ] {text}")
            continue
        out = ask_model(build_prompt(text, labels), model=model, strategy="semantic_cache")
        calls += 1
        cache.store(text, out["answer"])
        print(f"[MISS] {text} -> {normalize(out['answer'], valid)}")
    print(f"\nLLM calls: {calls}/{len(PARAPHRASES)}, cache hits: {hits}")


if __name__ == "__main__":
    run()