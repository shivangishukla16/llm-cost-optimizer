import hashlib
import json
import sqlite3
from pathlib import Path

from src.llm import ask_model

DB_PATH = Path("results/cache.sqlite")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
    return conn


def make_key(model: str, prompt: str) -> str:
    """Model aur prompt dono ka hash, taaki alag model ka jawab mix na ho."""
    return hashlib.sha256(f"{model}\n{prompt}".encode("utf-8")).hexdigest()


def clear_cache() -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM cache")
        conn.commit()
    finally:
        conn.close()


def cached_ask(prompt: str, model: str = "llama3.2:3b", strategy: str = "cached") -> dict:
    """Pehle cache dekhta hai. Hit ho to LLM call nahi hota, miss ho to call karke save karta hai.

    orig_tokens hamesha original call ke tokens hain, taaki cache na hota to kitna lagta, woh pata chale.
    """
    key = make_key(model, prompt)

    conn = _connect()
    try:
        row = conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        if row:
            out = json.loads(row[0])
            out["cached"] = True
            out["input_tokens"] = 0
            out["output_tokens"] = 0
            out["latency_ms"] = 0.0
            out["strategy"] = strategy
            return out

        out = ask_model(prompt, model=model, strategy=strategy)
        out["orig_tokens"] = out["input_tokens"] + out["output_tokens"]
        out["cached"] = False
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
            (key, json.dumps(out, ensure_ascii=False)),
        )
        conn.commit()
        return out
    finally:
        conn.close()