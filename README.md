# Claims Denial Early-Warning (pre-submission risk + fix notes)

> Personal project by Somya Goyal, built while applying to Optum. Not affiliated with or endorsed by Optum; uses public/synthetic data.

## Why this matters for Optum

**Problem:** a denied claim gets worked after the fact: someone appeals, corrects and resubmits it. Most denial causes (no prior authorisation, late filing, a coding mismatch, a duplicate, missing fields) are visible *before* the claim is sent.
**Who feels it:** revenue-cycle and billing teams on the provider side, and the payer operations teams that process the rework.
**What this shows:** a model that scores each claim's denial risk before submission. For every flagged claim it names the top drivers and attaches a short fix note grounded in a playbook. The output is a CSV worklist that drops straight into Power BI or Tableau.

Public context:
- Fierce Healthcare (2025), citing MDaudit's 2025 benchmark: average denied claim amounts rose 12% for inpatient and 14% for outpatient vs 2024. An Experian survey it cites found more than 41% of providers report denial rates above 10%.
- Optum's careers page for this role lists EDA, feature engineering, statistics, BI dashboards, SQL/Python, scikit-learn/XGBoost and basic GenAI/RAG. This repo uses each of those.

Impact (ESTIMATE, synthetic data): with the default cost assumptions in `CostModel`, the cost-based threshold gives an estimated net saving of about **1,490 per 1,000 claims** on the held-out quarter. The assumptions are $6 to review a flagged claim, $45 to rework a denial, and 60% of real denials prevented by a pre-submission review. Change the three numbers to match a real setting; the result moves with them.

## Results (real numbers from `python -m denial_ew.run`)

| Test set | AUC | Avg precision | Flagged | Precision @ threshold | Recall @ threshold |
|---|---|---|---|---|---|
| Same quarter, new sample (5,000 claims) | 0.833 | 0.575 | 19.5% | 0.52 | 0.63 |
| **Held-out next quarter** (new payer mix, one payer tightens prior auth; never trained on) | **0.831** | 0.588 | 17.1% | 0.55 | 0.58 |

- **Fix-note correctness:** for the 162 truly denied claims in the top-200 worklist, the first playbook entry in the note matched the true denial reason **92%** of the time.
- **Retrieval on hand-written, free-text questions:** 12 queries in billing-desk wording, written separately and never used for tuning (`data/heldout_queries.jsonl`). The TF-IDF retriever got **41.7% top-1**. It works when the query uses the playbook's words (the pipeline's own queries do) but fails on paraphrases, for example "nobody got the insurer's approval" vs "prior authorization". This gap is the main limitation and the reason the next step is embedding retrieval.

## How it works

1. `data.py` generates synthetic claims with known denial drivers. `shift=True` makes a different "next quarter".
2. `model.py`: feature building, an XGBoost classifier, a **cost-based threshold** (rather than a 0.5 cut-off), and per-claim drivers from XGBoost's built-in SHAP contributions (`pred_contribs`).
3. `explain.py`: drivers become a retrieval query over `data/playbook.json` (TF-IDF). The fix note is written by an offline template by default. A real LLM can be passed in as `llm=`, and it only rephrases the retrieved steps, each tagged `[playbook:ID]`.
4. `run.py`: EDA (denial rate by payer, service line and reason), training, both test sets, the worklist, and CSV exports in `results/`.

## Run it

```bash
pip install -r requirements.txt
cd src && python -m denial_ew.run      # writes results/metrics.json, claims_scored.csv, worklist_top200.csv
cd .. && pytest -q                     # 6 tests
```

No API key is needed. To use a real LLM for the note wording, pass any `str -> str` function as `llm=` to `fix_note` (for example a thin wrapper around an OpenAI or Azure OpenAI chat call). Keep the "use only the context" instruction.

## Limitations / next steps

- The data is **synthetic**, and the drivers were designed in, so a real claims feed would score lower and need far more features (CPT/ICD pairs, payer edits, history).
- Free-text retrieval is weak on paraphrases (41.7%). Next: sentence embeddings or a hybrid BM25 + embedding retriever, re-measured on the same hand-written set.
- Next: probability calibration and a monitoring check for payer-policy drift, since the held-out quarter shows recall dropping from 0.63 to 0.58 when one payer tightens its rules.
