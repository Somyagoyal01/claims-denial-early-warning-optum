"""Synthetic medical-claims generator.

Everything here is SYNTHETIC. No real claims, payers or patients are used.
The generator encodes a few well-known, publicly described denial drivers
(missing prior authorisation, late filing, coding mismatches, duplicates,
out-of-network care, missing information) so a model has something to learn.

`make_claims(..., shift=True)` produces a *different* world: a new payer mix,
a payer that tightened its prior-auth rules, and more electronic submission.
It is used as a held-out "next quarter" test that the model never trains on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PAYERS = ["PayerA", "PayerB", "PayerC", "PayerD", "PayerE"]
SERVICE_LINES = ["imaging", "lab", "surgery", "office_visit", "therapy", "dme"]
SPECIALTIES = ["cardiology", "orthopedics", "primary_care", "radiology", "oncology"]
REASONS = [
    "prior_auth_missing",
    "timely_filing",
    "coding_mismatch",
    "duplicate_claim",
    "out_of_network",
    "missing_information",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_claims(n: int = 20000, seed: int = 7, shift: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    payer_p = [0.30, 0.25, 0.20, 0.15, 0.10]
    if shift:  # next quarter: PayerD grows, PayerA shrinks
        payer_p = [0.15, 0.20, 0.20, 0.35, 0.10]
    payer = rng.choice(PAYERS, size=n, p=payer_p)
    service = rng.choice(SERVICE_LINES, size=n, p=[0.20, 0.25, 0.10, 0.25, 0.12, 0.08])
    specialty = rng.choice(SPECIALTIES, size=n)

    needs_auth = np.isin(service, ["imaging", "surgery", "dme", "therapy"])
    auth_obtained = np.where(needs_auth, rng.random(n) < 0.82, True)
    days_to_submit = rng.gamma(shape=2.0, scale=12.0 if not shift else 9.0, size=n).round().astype(int)
    billed = np.round(rng.lognormal(mean=6.3, sigma=1.0, size=n), 2)
    dx_px_mismatch = rng.random(n) < 0.06
    missing_modifier = rng.random(n) < 0.08
    duplicate_flag = rng.random(n) < 0.03
    in_network = rng.random(n) < 0.92
    missing_fields = rng.poisson(0.25, size=n)
    electronic = rng.random(n) < (0.85 if not shift else 0.93)
    patient_age = rng.integers(0, 90, size=n)

    # Payer-specific strictness. In the shifted quarter PayerD tightens prior auth.
    auth_strict = {"PayerA": 1.0, "PayerB": 1.3, "PayerC": 0.9, "PayerD": 1.1, "PayerE": 1.6}
    if shift:
        auth_strict["PayerD"] = 2.0
    filing_limit = {"PayerA": 90, "PayerB": 60, "PayerC": 90, "PayerD": 120, "PayerE": 45}

    strict = np.array([auth_strict[p] for p in payer])
    limit = np.array([filing_limit[p] for p in payer])
    late = days_to_submit > limit

    # Driver contributions (log-odds)
    c_auth = np.where(needs_auth & ~auth_obtained, 2.6 * strict, 0.0)
    c_late = np.where(late, 3.0, 0.0)
    c_code = np.where(dx_px_mismatch, 1.8, 0.0) + np.where(missing_modifier, 1.1, 0.0)
    c_dup = np.where(duplicate_flag, 3.2, 0.0)
    c_oon = np.where(~in_network, 1.9, 0.0)
    c_info = 0.9 * missing_fields + np.where(~electronic, 0.4, 0.0)
    base = -3.4 + 0.15 * np.log1p(billed / 500)

    logit = base + c_auth + c_late + c_code + c_dup + c_oon + c_info + rng.normal(0, 0.5, n)
    denied = rng.random(n) < _sigmoid(logit)

    drivers = np.vstack([c_auth, c_late, c_code, c_dup, c_oon, c_info]).T
    reason = np.where(denied, np.array(REASONS)[drivers.argmax(axis=1)], "")
    reason = np.where(denied & (drivers.max(axis=1) == 0), "missing_information", reason)

    return pd.DataFrame(
        {
            "claim_id": [f"{'Q2' if shift else 'Q1'}-{i:06d}" for i in range(n)],
            "payer": payer,
            "service_line": service,
            "provider_specialty": specialty,
            "needs_prior_auth": needs_auth.astype(int),
            "prior_auth_obtained": auth_obtained.astype(int),
            "days_to_submit": days_to_submit,
            "billed_amount": billed,
            "dx_px_mismatch": dx_px_mismatch.astype(int),
            "missing_modifier": missing_modifier.astype(int),
            "duplicate_flag": duplicate_flag.astype(int),
            "in_network": in_network.astype(int),
            "missing_fields": missing_fields,
            "electronic_submission": electronic.astype(int),
            "patient_age": patient_age,
            "denied": denied.astype(int),
            "denial_reason": reason,
        }
    )
