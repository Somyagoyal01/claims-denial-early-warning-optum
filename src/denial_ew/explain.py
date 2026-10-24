"""Turn model drivers into a short, grounded pre-submission fix note.

Retrieval: TF-IDF over a small hand-written playbook (data/playbook.json).
Generation: offline template by default (no API key needed). A real LLM can be
plugged in through `llm=` (any callable str -> str), and it is only allowed to
rephrase the retrieved playbook text, which is passed in as the context.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PLAYBOOK_PATH = Path(__file__).resolve().parents[2] / "data" / "playbook.json"

# Map model features to the plain-English signal used as the retrieval query.
FEATURE_TO_QUERY = {
    "auth_gap": "procedure needs prior authorization but no authorization on file",
    "prior_auth_obtained": "prior authorization status",
    "days_to_submit": "claim submitted late after date of service filing deadline",
    "dx_px_mismatch": "diagnosis code does not support the procedure code",
    "missing_modifier": "modifier missing on procedure line",
    "duplicate_flag": "same claim already submitted duplicate",
    "in_network": "provider out of network for this plan",
    "missing_fields": "required fields missing on the claim form",
    "electronic_submission": "paper claim submission missing information",
}


class Playbook:
    def __init__(self, path: Path = PLAYBOOK_PATH):
        self.entries = json.loads(Path(path).read_text())
        self.texts = [e["title"] + ". " + e["signals"] + ". " + e["fix"] for e in self.entries]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = self.vec.fit_transform(self.texts)

    def search(self, query: str, k: int = 1) -> list[tuple[dict, float]]:
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [(self.entries[i], float(sims[i])) for i in order]


def offline_llm(prompt: str) -> str:
    """Deterministic stand-in for an LLM: returns the context lines it was given."""
    ctx = prompt.split("CONTEXT:\n", 1)[-1]
    return ctx.strip()


def fix_note(drivers: list[tuple[str, float]], playbook: Playbook,
             llm: Callable[[str], str] = offline_llm, min_sim: float = 0.08) -> str:
    lines, seen = [], set()
    for feat, _ in drivers:
        base = feat.split("_Payer")[0]
        query = FEATURE_TO_QUERY.get(base)
        if not query:
            continue
        entry, sim = playbook.search(query)[0]
        if sim < min_sim or entry["id"] in seen:
            continue
        seen.add(entry["id"])
        lines.append(f"- {entry['title']}: {entry['fix']} [playbook:{entry['id']}]")
    if not lines:
        return "No specific fix found in the playbook; route to a coder for manual review."
    prompt = ("Rewrite these steps as a short note for a billing specialist. Use only the context.\n"
              "CONTEXT:\n" + "\n".join(lines))
    return llm(prompt)
