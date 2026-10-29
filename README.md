# 🏦 LLM Cost Optimizer for Banking Intent Classification

💡 An experiment on reducing LLM token cost for banking customer intent classification, and measuring what each cost reduction does to accuracy. All models run locally through 🦙 Ollama. The project covers 🧭 prompt shortlisting, ⚡ exact-match caching, 🔐 PII masking for Indian banking identifiers, and evaluation on English (Banking77) and 🇮🇳 Hinglish queries.

## 🎯 Problem

Each classification request in a naive setup includes all 77 Banking77 intent labels in the prompt. For a short customer query, the label list accounts for most of the input tokens. This project asks three questions:

1. ✂️ Can a shortlist of candidate labels, selected before the LLM call, reduce tokens without hurting accuracy?
2. ⚡ How much does an exact-match cache save when customer queries repeat?
3. 🔐 Can sensitive identifiers (Aadhaar, PAN, account numbers, UPI IDs) be masked before reaching the model, and what does masking cost in accuracy?

## 📊 Summary of results

| Question | Result |
|---|---|
| ✂️ Token reduction with embedding shortlist (k=10) | About **71%** fewer prompt tokens (89,180 → 25,790 on 200 queries) |
| 🎯 Accuracy with shortlist and the 8B model | **66%** vs 60% baseline (200 queries). Within noise, so "same or slightly better" |
| 🇮🇳 Hinglish accuracy with shortlist and the 3B model | **56%** vs 40% baseline (62 queries) |
| 🔐 PII leakage to the model | **100%** without masking → **0%** with masking (19 queries) |
| ⚡ Cache token savings on repeated traffic | **76%** (simulated traffic) |

## 🔄 Pipeline

```
📝 Customer query
   │
   ▼
🔐 PII masking (regex)        → Aadhaar, PAN, IFSC, UPI, email, phone, account replaced with placeholders
   │
   ▼
🧭 Candidate shortlist        → top 10 labels by embedding similarity (nomic-embed-text)
   │
   ▼
🧠 Prompt + LLM (Ollama)      → llama3.2:3b or llama3.1:8b, temperature 0
   │
   ▼
🏷️ Label normalization        → maps model output to a valid Banking77 label
   │
   ▼
📒 Logging                    → tokens, latency, strategy saved to results/llm_calls.csv
```

⚡ An exact-match cache (SQLite, keyed by model and prompt hash) sits in front of the LLM call.

## 📚 Dataset

- 🏦 **Banking77 (English):** the official test split, 77 intents, loaded from Hugging Face (`PolyAI/banking77`). Evaluations use a fixed random sample with seed 42.
- 🇮🇳 **Hinglish pilot set:** 62 queries written by hand for this project, each mapped to a Banking77 label. Examples: `mera card block ho gaya hai kya karu` and `ATM se paise nikle nahi par cut gaye`. This is a pilot set and has not been reviewed by native speakers.
- 🔐 **PII test set:** 19 queries with synthetic Aadhaar, PAN, UPI, phone, account and email values, plus one control query with no PII.

## 🧪 Experiments and results

All English results use 200 queries, `temperature=0`, and the same label normalization. ⏱️ Latency varies between runs, so treat it as approximate.

### 1️⃣ English intent classification (Banking77, n=200)

| Strategy | Model | Recall | Accuracy | Avg LLM latency | Total LLM tokens |
|---|---|---|---|---|---|
| Baseline (77 labels) | llama3.2:3b | 100% | 48% (95/200) | 210 ms | 92,196 |
| Baseline (77 labels) | llama3.1:8b | 100% | 60% (121/200) | 889 ms | 89,180 |
| Word-overlap shortlist, k=10 | llama3.2:3b | 78% | 48% (96/200) | 270 ms | 27,602 |
| Embedding shortlist, k=10 | llama3.2:3b | 92% | 54% (107/200) | 194 ms | 28,839 |
| Embedding shortlist, k=10 | llama3.1:8b | 92% | 66% (133/200) | 630 ms | 25,790 |

🔎 **Recall** is the share of queries whose correct label appears in the shortlist. Accuracy cannot exceed recall, because the model cannot choose a label it was not shown.

✅ **Findings:**
- ✂️ Embedding shortlist reduced prompt tokens by about 71%.
- 🎯 Embedding retrieval gave better recall than word overlap (92% vs 78%).
- 🧠 Shortlisting did not improve the 3B model's accuracy on English, but the 8B model's accuracy was similar or slightly higher.
- 📏 With 200 queries, the 95% confidence interval is roughly ±7 points, so the 6-point gain for 8B is not statistically confirmed.

### 2️⃣ Hinglish pilot (62 queries)

| Strategy | Model | Accuracy | Avg LLM latency | Total LLM tokens |
|---|---|---|---|---|
| Baseline (77 labels) | llama3.2:3b | 40% (25/62) | 187 ms | 28,500 |
| Baseline (77 labels) | llama3.1:8b | 65% (40/62) | 726 ms | 27,568 |
| Embedding shortlist, k=10 | llama3.2:3b | 56% (35/62) | 211 ms | 8,896 |
| Embedding shortlist, k=10 | llama3.1:8b | 60% (37/62) | 650 ms | 7,952 |

✅ **Findings:**
- 🇮🇳 On Hinglish, the 3B model improved from 40% to 56% with the shortlist, and tokens dropped by about 68%. This effect was not visible on English.
- 🏆 The 8B baseline gave the highest accuracy (65%) but used about 3.5 times the tokens of the shortlist setups.
- 📏 With 62 queries, the noise is about ±8 points. Treat these as directional results from a pilot set.

### 3️⃣ Exact-match cache (simulated repeated traffic)

🔁 Banking77 test queries are unique, so a cache has no hits on them. To measure caching, the evaluation draws 200 requests from a pool of 50 unique queries, which mimics repeated customer questions.

| Metric | Value |
|---|---|
| Unique queries in pool | 50 |
| Total requests | 200 |
| Cache hits | 151 (76%) |
| Tokens without cache | 92,339 |
| Tokens with cache | 22,600 |
| Tokens saved | **69,739 (76%)** |

⚠️ The hit rate depends on how often real queries repeat. On unique queries, this cache gives 0% hits.

### 4️⃣ PII masking

🔐 Regex patterns cover Aadhaar, PAN, IFSC, UPI ID, email, phone (with optional +91) and bank account numbers. Matches are replaced with placeholders such as `[AADHAAR]` and `[PAN]`. Seven unit tests pass (`pytest tests`).

| Mode | Accuracy (19 queries) | PII values sent to the model | Total tokens |
|---|---|---|---|
| Raw (no masking) | 58% (11/19) | 18/18 (100%) ❌ | 8,836 |
| Masked | 47% (9/19) | 0/18 (0%) ✅ | 8,792 |

✅ **Findings:**
- 🛡️ Masking removed all PII from the prompt, and the control query was left unchanged.
- 🔢 Token counts were nearly identical.
- ⚠️ Accuracy dropped by 2 queries. Both were PAN or phone queries where the placeholder appeared to confuse the model. With 19 queries, this is not statistically conclusive, but it is a real cost to track.

### 5️⃣ Semantic cache (negative result) 🚫

🔍 An embedding-similarity cache was tested to catch paraphrased queries. Similarity scores with nomic-embed-text:

| Pair | Type | Score |
|---|---|---|
| transfer abhi tak pending hai / transfer still pending hai kya karu | ✅ Same meaning | 0.826 |
| mera card block ho gaya hai / card block ho gaya kya karu | ✅ Same meaning | 0.877 |
| mera card block karo / mera card unblock karo | ❌ Opposite meaning | 0.945 |
| transfer pending hai / transfer cancel karna hai | ❌ Different | 0.726 |
| card expire ho raha hai / card lost ho gaya hai | ❌ Different | 0.750 |

🚫 No single threshold separates the same-meaning pairs from the others. The "block" and "unblock" pair scores highest of all. In banking, a cache hit on the wrong action can cause a wrong response, so this project does not use a semantic cache. Exact-match caching is the safer choice. Semantic caching would need a negation-aware check or a reranker.

## 🧩 Optional components

### 🚀 FastAPI endpoint (`app.py`)

A `/classify` route runs PII masking, builds the prompt, and calls the model.

```bash
pip install fastapi uvicorn
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` and send a request such as `{"text": "mera Aadhaar 2345 6789 0123 hai, card block karo"}`. The response includes `pii_masked: {"AADHAAR": 1}`, which shows the Aadhaar was masked before reaching the model.

⚠️ **Limitation:** the model sometimes returns a label that is not in the Banking77 set (a test returned `card_block`). The endpoint currently returns such labels without rejecting them. A production version needs validation and a retry.

### 📈 Streamlit dashboard (`dashboard.py`)

Shows call counts, total tokens and average latency per strategy, read from `results/llm_calls.csv`.

```bash
streamlit run dashboard.py
```

🗂️ The log file contains several runs per strategy label, so the dashboard shows raw logs. Use the results tables above for final numbers.

## 🔗 Related work

Several tools already address LLM cost and routing:

- 🌐 **LiteLLM:** a multi-provider proxy with cost tracking and budgets.
- 🔀 **RouteLLM:** trained routers that choose between strong and weak models.
- 🧠 **GPTCache:** a semantic caching library.
- 📡 **Helicone, Langfuse, Portkey:** observability and gateway platforms.

This project does not replace these tools. It focuses on a banking intent-classification workload, measures the accuracy cost of shortlisting, evaluates Hinglish queries, and tests masking of Indian identifiers.

## ⚠️ Limitations

- 📏 **Sample size:** 200 English queries (±7 points) and 62 Hinglish queries (±8 points). Results are directional.
- ⏱️ **Latency:** varies between runs. Compare strategies using token counts, which are stable.
- 🔀 **Routing:** a router that escalates from 3B to 8B when the output is not a valid label did not improve accuracy (44% vs 48%), so it is not in the final results.
- 🏷️ **Labels:** some Banking77 labels overlap in meaning (for example `pending_transfer` and `transfer_timing`), and a few contain capital letters or `?`.
- 🇮🇳 **Hinglish and PII labels:** written by the author, not verified by native speakers or a second annotator.
- 🔢 **Token accounting:** embedding calls are not included in the LLM token totals. They take about 50 to 80 ms per query.
- 🔐 **PII:** regex matches formatted identifiers only. Names and addresses are not detected.
- ⚡ **Cache:** results come from simulated repeated traffic.

## 🗂️ Project structure

```
llm-cost-optimizer/
├── app.py                 # 🚀 FastAPI endpoint (optional)
├── dashboard.py           # 📈 Streamlit dashboard (optional)
├── data/
│   ├── hinglish_queries.csv
│   └── pii_queries.csv
├── src/
│   ├── data.py            # 🏦 Banking77 loading
│   ├── llm.py             # 🦙 Ollama calls and logging
│   ├── classify.py        # 🏷️ Baseline classification, label normalization
│   ├── shortlist.py       # 🔤 Word-overlap shortlist
│   ├── embed_shortlist.py # 🧭 Embedding shortlist
│   ├── router.py          # 🔀 3B to 8B escalation
│   ├── cache.py           # ⚡ Exact-match cache
│   ├── cache_eval.py      # 🔁 Cache simulation
│   ├── semantic_cache.py  # 🚫 Embedding cache (negative result)
│   ├── semantic_eval.py
│   ├── pii_mask.py        # 🔐 Regex PII masking
│   ├── pii_eval.py        # 🛡️ Raw vs masked evaluation
│   └── hinglish_eval.py   # 🇮🇳 Hinglish evaluation
├── tests/
│   └── test_pii_mask.py   # ✅ Unit tests
├── requirements.txt
└── README.md
```

## 🛠️ Setup and reproduction

🖥️ Tested on Windows 10/11, Python 3.13, Ollama 0.40.12, 16 GB RAM, with the Ollama models running on the local GPU.

**1️⃣ Clone and create a virtual environment**

```bash
git clone https://github.com/shivangishukla16/llm-cost-optimizer.git
cd llm-cost-optimizer
python -m venv venv
venv\Scripts\activate          # Windows PowerShell: .\venv\Scripts\Activate.ps1
```

**2️⃣ Install Python dependencies**

```bash
pip install -r requirements.txt
```

**3️⃣ Install Ollama and pull the models**

📥 Install Ollama from https://ollama.com, then run:

```bash
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

🔓 Ollama needs no API key. The models run locally.

**4️⃣ Run the unit tests**

```bash
python -m pytest tests -v
```

**5️⃣ Reproduce the experiments**

```bash
python -c "from src.classify import run; run(n=200, model='llama3.2:3b', strategy='baseline_3b_final')"
python -c "from src.classify import run; run(n=200, model='llama3.1:8b', strategy='baseline_8b_final')"
python -c "from src.shortlist import run; run(n=200, model='llama3.2:3b', k=10)"
python -c "from src.embed_shortlist import run; run(n=200, model='llama3.2:3b', k=10)"
python -c "from src.embed_shortlist import run; run(n=200, model='llama3.1:8b', k=10)"
python -c "from src.cache_eval import run; run()"
python -c "from src.pii_eval import run; run(model='llama3.2:3b')"
python -c "from src.hinglish_eval import run; run(model='llama3.2:3b', use_shortlist=False)"
python -c "from src.hinglish_eval import run; run(model='llama3.1:8b', use_shortlist=False)"
python -c "from src.hinglish_eval import run; run(model='llama3.2:3b', use_shortlist=True)"
python -c "from src.hinglish_eval import run; run(model='llama3.1:8b', use_shortlist=True)"
```

📒 Each run prints its summary and appends every model call to `results/llm_calls.csv`. LLM outputs can differ slightly between hardware and Ollama versions, so expect numbers close to, not identical to, those above.

**🩹 Troubleshooting**
- 💥 If Ollama crashes with a CUDA error when switching between the 3B and 8B models, restart Ollama and set `OLLAMA_MAX_LOADED_MODELS=1`.
- ℹ️ The Hugging Face "unauthenticated requests" warning is harmless. Set `HF_TOKEN` only if downloads are slow.

## 🔮 Next steps

- 🇮🇳 Add a Hinglish set reviewed by native speakers and expand it beyond 62 queries.
- 🛡️ Add label validation and retry to the API.
- 🧠 Test a negation-aware semantic cache, or a reranker on top of embedding retrieval.
- 🧾 Replace regex name and address detection with an NER tool such as Microsoft Presidio.
