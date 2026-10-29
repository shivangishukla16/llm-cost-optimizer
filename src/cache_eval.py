import random

from src.data import load_banking77_sample, get_all_label_names
from src.classify import build_prompt, normalize
from src.cache import cached_ask, clear_cache


def run(n_pool: int = 50, n_traffic: int = 200, model: str = "llama3.2:3b", seed: int = 42):
    pool = load_banking77_sample(n_pool)
    labels = sorted(get_all_label_names())
    valid = set(labels)

    # Repeated traffic: pool mein se random requests (same query dobara aayegi)
    traffic = random.Random(seed).choices(pool, k=n_traffic)

    clear_cache()

    hits = 0
    tokens_spent = 0
    tokens_without_cache = 0
    correct = 0

    for item in traffic:
        prompt = build_prompt(item["text"], labels)
        out = cached_ask(prompt, model=model, strategy="cache_eval")

        hits += out["cached"]
        tokens_spent += out["input_tokens"] + out["output_tokens"]
        tokens_without_cache += out["orig_tokens"]

        predicted = normalize(out["answer"], valid)
        correct += predicted == normalize(item["label"], valid)

    saved = tokens_without_cache - tokens_spent
    print(f"Model: {model}")
    print(f"Unique queries in pool: {n_pool}, total requests: {n_traffic}")
    print(f"Cache hits: {hits}/{n_traffic} = {hits / n_traffic:.0%}")
    print(f"Accuracy: {correct}/{n_traffic} = {correct / n_traffic:.0%}")
    print(f"Tokens without cache: {tokens_without_cache}")
    print(f"Tokens with cache: {tokens_spent}")
    print(f"Tokens saved: {saved} ({saved / tokens_without_cache:.0%})")


if __name__ == "__main__":
    run()