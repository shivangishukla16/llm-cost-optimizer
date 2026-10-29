from fastapi import FastAPI
from pydantic import BaseModel

from src.data import get_all_label_names
from src.classify import build_prompt, normalize
from src.llm import ask_model
from src.pii_mask import mask_pii

app = FastAPI(title="Banking intent classifier")
LABELS = sorted(get_all_label_names())
VALID = set(LABELS)


class Query(BaseModel):
    text: str
    model: str = "llama3.2:3b"


@app.post("/classify")
def classify(q: Query):
    masked, pii_counts = mask_pii(q.text)
    prompt = build_prompt(masked, LABELS)
    out = ask_model(prompt, model=q.model, strategy="api")
    return {
        "label": normalize(out["answer"], VALID),
        "pii_masked": pii_counts,
        "input_tokens": out["input_tokens"],
        "output_tokens": out["output_tokens"],
        "latency_ms": out["latency_ms"],
    }