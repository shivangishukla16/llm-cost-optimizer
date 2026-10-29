import numpy as np

from src.embed_shortlist import embed

THRESHOLD = 0.90


class SemanticCache:
    def __init__(self, threshold: float = THRESHOLD):
        self.threshold = threshold
        self.vecs: list[np.ndarray] = []
        self.answers: list[str] = []

    def lookup(self, text: str) -> str | None:
        if not self.vecs:
            return None
        q = embed([text], prefix="search_query: ")[0]
        sims = np.array(self.vecs) @ q
        best = int(np.argmax(sims))
        if sims[best] >= self.threshold:
            return self.answers[best]
        return None

    def store(self, text: str, answer: str) -> None:
        v = embed([text], prefix="search_query: ")[0]
        self.vecs.append(v)
        self.answers.append(answer)