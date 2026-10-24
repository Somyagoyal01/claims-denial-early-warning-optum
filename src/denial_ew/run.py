"""End-to-end run: data -> EDA -> model -> held-out quarter -> fix notes -> BI exports.

    python -m denial_ew.run          (from the src/ folder, or after `pip install -e .`)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data import make_claims
from .explain import Playbook, fix_note
from .model import DenialModel, evaluate

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def eda(df: pd.DataFrame) -> dict:
    by_payer = df.groupby("payer")["denied"].mean().round(3).to_dict()
    by_service = df.groupby("service_line")["denied"].mean().round(3).to_dict()
    reasons = df.loc[df.denied == 1, "denial_reason"].value_counts(normalize=True).round(3).to_dict()
    return {"denial_rate": round(float(df.denied.mean()), 3), "by_payer": by_payer,
            "by_service_line": by_service, "reason_mix": reasons}


def retrieval_heldout(playbook: Playbook) -> dict:
    rows = [json.loads(l) for l in (ROOT / "data" / "heldout_queries.jsonl").read_text().splitlines() if l.strip()]
    hits = [playbook.search(r["q"])[0][0]["id"] == r["expect"] for r in rows]
    misses = [r["q"] for r, h in zip(rows, hits) if not h]
    return {"n": len(rows), "top1_accuracy": round(sum(hits) / len(rows), 3), "misses": misses}


def main() -> dict:
    RESULTS.mkdir(exist_ok=True)
    q1 = make_claims(n=20000, seed=7)
    q1_test = make_claims(n=5000, seed=99)                 # same world, new sample
    q2 = make_claims(n=5000, seed=123, shift=True)         # next quarter: payer mix + policy change

    model = DenialModel()
    fit_info = model.fit(q1)
    metrics = {
        "eda_train": eda(q1),
        "fit": fit_info,
        "test_same_quarter": evaluate(model, q1_test),
        "heldout_next_quarter": evaluate(model, q2),
    }

    pb = Playbook()
    metrics["retrieval_heldout_handwritten"] = retrieval_heldout(pb)

    # Score the next-quarter claims and write Power BI / Tableau-friendly CSVs.
    q2 = q2.copy()
    q2["denial_risk"] = model.predict_proba(q2).round(4)
    q2["flagged"] = (q2["denial_risk"] >= model.threshold).astype(int)
    flagged = q2[q2.flagged == 1].sort_values("denial_risk", ascending=False).head(200)
    drv = model.drivers(flagged)
    flagged = flagged.assign(top_drivers=[", ".join(f for f, _ in d) for d in drv],
                             fix_note=[fix_note(d, pb) for d in drv])
    # Does the note's FIRST playbook entry match the (synthetic) true denial reason?
    reason_to_pb = {"prior_auth_missing": "PA-01", "timely_filing": "TF-01", "coding_mismatch": ("CD-01", "CD-02"),
                    "duplicate_claim": "DU-01", "out_of_network": "NW-01", "missing_information": "MI-01"}
    agree, total = 0, 0
    for note, reason in zip(flagged["fix_note"], flagged["denial_reason"]):
        if not reason or "[playbook:" not in note:
            continue
        first = note.split("[playbook:", 1)[1].split("]", 1)[0]
        want = reason_to_pb[reason]
        agree += first in (want if isinstance(want, tuple) else (want,))
        total += 1
    metrics["note_matches_true_reason"] = {"n_denied_in_top200": total, "agreement": round(agree / max(total, 1), 3)}

    q2.drop(columns=["denied", "denial_reason"]).to_csv(RESULTS / "claims_scored.csv", index=False)
    flagged[["claim_id", "payer", "service_line", "billed_amount", "denial_risk", "top_drivers", "fix_note"]] \
        .to_csv(RESULTS / "worklist_top200.csv", index=False)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    m = main()
    for k in ["test_same_quarter", "heldout_next_quarter", "retrieval_heldout_handwritten", "note_matches_true_reason"]:
        print(k, json.dumps(m[k], indent=1))
    print("threshold", round(m["fit"]["threshold"], 2))
