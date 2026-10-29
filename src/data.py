from datasets import load_dataset


def _load_test_split():
    return load_dataset("PolyAI/banking77", split="test")


def load_banking77_sample(n: int = 20, seed: int = 42):
    """Banking77 test set se n random queries aur unke intent labels deta hai."""
    ds = _load_test_split().shuffle(seed=seed).select(range(n))
    label_names = ds.features["label"].names
    return [
        {"text": row["text"], "label": label_names[row["label"]]}
        for row in ds
    ]


def get_all_label_names():
    """Banking77 ke saare 77 intent names deta hai."""
    return _load_test_split().features["label"].names


if __name__ == "__main__":
    for item in load_banking77_sample(5):
        print(item)