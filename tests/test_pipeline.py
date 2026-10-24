import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from denial_ew.data import make_claims  # noqa: E402
from denial_ew.explain import Playbook, fix_note  # noqa: E402
from denial_ew.model import CostModel, DenialModel, choose_threshold, evaluate  # noqa: E402


def test_generator_is_deterministic_and_shift_differs():
    a, b = make_claims(500, seed=1), make_claims(500, seed=1)
    assert a.equals(b)
    q2 = make_claims(3000, seed=1, shift=True)
    assert q2["payer"].value_counts(normalize=True)["PayerD"] > 0.25


def test_model_beats_chance_on_shifted_quarter():
    m = DenialModel()
    m.fit(make_claims(6000, seed=3))
    res = evaluate(m, make_claims(2000, seed=4, shift=True))
    assert res["auc"] > 0.75
    assert 0 < res["flag_rate"] < 0.5


def test_threshold_prefers_flagging_when_review_is_cheap():
    y = np.array([1, 0, 1, 0, 0, 1])
    p = np.array([0.9, 0.2, 0.7, 0.1, 0.3, 0.6])
    t, saving = choose_threshold(y, p, CostModel(review_cost=1, rework_cost=50, fix_success=0.6))
    assert t <= 0.6 and saving > 0


def test_drivers_point_at_missing_auth():
    df = make_claims(6000, seed=5)
    m = DenialModel()
    m.fit(df)
    row = df[(df.needs_prior_auth == 1) & (df.prior_auth_obtained == 0)].head(1)
    names = [f for f, _ in m.drivers(row)[0]]
    assert any("auth" in n for n in names)


def test_fix_note_is_grounded_in_playbook():
    note = fix_note([("auth_gap", 1.2), ("days_to_submit", 0.4)], Playbook())
    assert "[playbook:PA-01]" in note and "[playbook:TF-01]" in note


def test_unknown_driver_falls_back_to_manual_review():
    assert "manual review" in fix_note([("patient_age", 0.3)], Playbook())
