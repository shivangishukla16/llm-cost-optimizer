\# LLM Cost Optimizer (Banking77)



Ek chhota experiment jo dekhta hai ki LLM calls ka token cost kaise kam kiya ja sakta hai, aur accuracy par kya asar padta hai. Sab models Ollama par local chalte hain (llama3.2:3b, llama3.1:8b), embeddings nomic-embed-text se.



\## Problem



Har classification request mein 77 intent labels ka prompt jaata hai, jo tokens ka bada hissa hai. Ye project dekhta hai ki shortlist, cache aur PII masking se cost kitna kam hota hai, aur accuracy par kya asar padta hai.



\## Results 1: English intent classification (Banking77 test set, n=200, temperature=0)



| Strategy | Model | Recall | Accuracy | Avg LLM latency | Total LLM tokens |

|---|---|---|---|---|---|

| Baseline (77 labels) | llama3.2:3b | 100% | 48% (95/200) | 210 ms | 92,196 |

| Baseline (77 labels) | llama3.1:8b | 100% | 60% (121/200) | 889 ms | 89,180 |

| Word-overlap shortlist, k=10 | llama3.2:3b | 78% | 48% (96/200) | 270 ms | 27,602 |

| Embedding shortlist, k=10 | llama3.2:3b | 92% | 54% (107/200) | 194 ms | 28,839 |

| Embedding shortlist, k=10 | llama3.1:8b | 92% | 66% (133/200) | 630 ms | 25,790 |



\*\*Main finding:\*\* Embedding shortlist se prompt tokens lagbhag 71% kam hue. Accuracy same ya thodi behtar rahi, lekin 200 samples par ye farak statistically pakka nahi hai.



\## Results 2: Hinglish pilot (62 hand-written queries)



| Strategy | Model | Accuracy | Avg LLM latency | Total LLM tokens |

|---|---|---|---|---|

| Baseline (77 labels) | llama3.2:3b | 40% (25/62) | 187 ms | 28,500 |

| Baseline (77 labels) | llama3.1:8b | 65% (40/62) | 726 ms | 27,568 |

| Embedding shortlist, k=10 | llama3.2:3b | 56% (35/62) | 211 ms | 8,896 |

| Embedding shortlist, k=10 | llama3.1:8b | 60% (37/62) | 650 ms | 7,952 |



\*\*Main finding:\*\* Hinglish par embedding shortlist ne 3b ki accuracy 40% se 56% kar di, aur tokens lagbhag 68% kam hue. English par ye effect nahi dikha tha. Ye 62 queries ka pilot hai, isliye ise final truth nahi, direction maana jaye.



\## Cache (simulated repeated traffic)



| Metric | Value |

|---|---|

| Unique queries in pool | 50 |

| Total requests | 200 |

| Cache hits | 151/200 (76%) |

| Tokens without cache | 92,339 |

| Tokens with cache | 22,600 |

| Tokens saved | 69,739 (76%) |



Ye hit rate ek simulated setup ka number hai. Banking77 test queries unique hain, to unpar hit rate 0% hoga.



\## PII masking



Regex se Aadhaar, PAN, IFSC, UPI, email, phone aur account number ko placeholder se replace kiya jata hai. Unit tests (7/7) pass hain.



| Mode | Accuracy (19 queries) | PII LLM tak pahunchi | Total tokens |

|---|---|---|---|

| Raw | 58% (11/19) | 18/18 (100%) | 8,836 |

| Masked | 47% (9/19) | 0/18 (0%) | 8,792 |



Masking ne PII leak 100% se 0% kiya. Accuracy par asar 2 query ka hai, jo chhota sample hai. Dono galat queries PAN/phone wale the, jahan placeholder ne model ko confuse kiya.



\## Related work



\- \*\*LiteLLM\*\*: multi-provider proxy, cost tracking aur budgets.

\- \*\*RouteLLM\*\*: trained model router.

\- \*\*GPTCache\*\*: semantic caching library.

\- \*\*Helicone / Langfuse / Portkey\*\*: observability aur gateway.



Ye project inhe replace nahi karta. Iska focus banking-domain par intent classification, shortlist ka trade-off, Hinglish queries, aur Indian PII masking par hai.



\## Limitations



\- English sample 200 queries ka hai, accuracy ka ±7 points tak noise ho sakta hai. Hinglish sample 62 queries ka hai, ±8 points.

\- Latency har run mein badalti hai, isliye ek hi run ke numbers fixed comparison nahi hain.

\- Routing (3b → 8b on invalid label) ne accuracy nahi badhayi (44% vs 48%), isliye ise final table mein nahi rakha.

\- Banking77 labels mein overlap hai (jaise `pending\_transfer` vs `transfer\_timing`), aur kuch labels mein `?` ya capital letters hain.

\- Hinglish set mera hand-written hai, aur labels meri samajh par based hain. Ye native speakers se verify nahi hua.

\- PII set mere banaye hue 19 queries ka hai, aur labels kuch subjective hain.

\- Embedding calls ke tokens LLM tokens mein shamil nahi hain.

\- Regex PII sirf format-based hai. Naam aur address nahi pakde jaate.

\- Cache numbers simulated traffic par hain, real traffic par nahi.



\## Next steps



\- Semantic cache (embedding similarity) add karna.

\- Presidio jaisa NER tool add karke naam/address masking test karna.

\- Hinglish set ko native speaker se review karwana aur bada karna.



\## Setup



```bash

pip install -r requirements.txt

ollama pull llama3.2:3b

ollama pull llama3.1:8b

ollama pull nomic-embed-text

python -m src.classify

python -m src.embed\_shortlist

python -m src.hinglish\_eval

```

