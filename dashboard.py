import pandas as pd
import streamlit as st

st.set_page_config(page_title="LLM Cost Optimizer", layout="wide")
st.title("LLM Cost Optimizer: strategy-wise summary")

df = pd.read_csv("results/llm_calls.csv")
df["total_tokens"] = df["input_tokens"] + df["output_tokens"]
FINAL = ["baseline_3b_final", "baseline_8b_final", "shortlist_k10", "embed_shortlist_k10", "hinglish_baseline", "hinglish_embed_shortlist_k10", "cache_eval", "pii_masked"]
df = df[df["strategy"].isin(FINAL)]

summary = (
    df.groupby(["strategy", "model"])
    .agg(calls=("total_tokens", "size"),
         total_tokens=("total_tokens", "sum"),
         avg_latency_ms=("latency_ms", "mean"))
    .reset_index()
    .sort_values("total_tokens", ascending=False)
)

st.subheader("Calls aur tokens")
st.dataframe(summary, use_container_width=True)

st.subheader("Total tokens by strategy")
st.bar_chart(summary.set_index("strategy")["total_tokens"])

st.caption(
    "Note: CSV mein ek strategy ke kai runs (alag n, alag time) mix hain. "
    "Final numbers ke liye README ka table dekho."
)