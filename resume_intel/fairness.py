"""Blind (identity-neutral) text view and counterfactual fairness tooling."""
from __future__ import annotations
import random
import re
from typing import Dict, List

import numpy as np

from . import generator as G
from .preprocess import EMAIL_RE, PHONE_RE, URL_RE, RANGE_RE, MON

GENDER_RE = re.compile(r"\b(he|she|his|her|hers|him|mr|mrs|ms|miss|male|female|woman|women|man|men|women's|men's|"
                       r"fraternity|sorority)\b", re.I)
YEAR_RE = re.compile(rf"\b(?:{MON}\.?\s+)?(?:19|20)\d{{2}}\b|\b\d{{1,2}}/(?:19|20)\d{{2}}\b", re.I)


def blind_text(parsed: dict) -> str:
    """Rebuild a resume text WITHOUT name/contact block, institution, graduation year, absolute dates and
    activities/affiliations (the main carriers of gender, ethnicity, age and prestige proxies)."""
    secs = parsed["sections"]
    out: List[str] = []
    for name in ("summary", "skills", "experience", "projects", "certifications"):
        for ln in secs.get(name, []):
            out.append(ln)
    edu = parsed["education"]
    out.append(" ".join(x for x in [edu.get("degree"), edu.get("field")] if x))
    text = "\n".join(out)
    text = RANGE_RE.sub(" ", text)
    text = YEAR_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = GENDER_RE.sub(" ", text)
    return text.lower()


def role_group(category: str) -> str:
    """Stereotype-coded role sets used ONLY to build the bias-stress test (not by the pipeline)."""
    if category in ("Software Engineering", "DevOps & Cloud", "Cybersecurity", "Mechanical Engineering", "Civil Engineering"):
        return "A"
    if category in ("Human Resources", "Digital Marketing"):
        return "B"
    return "N"


def stress_train(records: List[dict], p_stereo: float = 0.9, seed: int = 7) -> List[dict]:
    """Re-assign identities so that group correlates strongly with category (historical-bias simulation)."""
    rng = random.Random(seed)
    out = []
    for r in records:
        rg = role_group(r["category"])
        want = rng.choice(["A", "B"]) if rg == "N" else (rg if rng.random() < p_stereo else ("B" if rg == "A" else "A"))
        if r["identity"]["group"] != want:
            r2 = G.swap_identity(r, group=want)
        else:
            r2 = dict(r)
        r2 = dict(r2)
        r2["text"] = G.render_text(r2)
        out.append(r2)
    return out


def balanced_test(records: List[dict], seed: int = 11) -> List[dict]:
    """Test identities drawn 50/50 independent of category."""
    rng = random.Random(seed)
    out = []
    for r in records:
        want = rng.choice(["A", "B"])
        r2 = G.swap_identity(r, group=want) if r["identity"]["group"] != want else dict(r)
        r2 = dict(r2)
        r2["text"] = G.render_text(r2)
        out.append(r2)
    return out


def counterfactual_twins(records: List[dict]) -> List[dict]:
    out = []
    for r in records:
        t = G.swap_identity(r)
        t = dict(t)
        t["text"] = G.render_text(t)
        out.append(t)
    return out


def rank_shift(S0: np.ndarray, S1: np.ndarray, k: int = 10) -> dict:
    """Score/rank stability between original and counterfactual score matrices (candidates x jobs)."""
    d = np.abs(S0 - S1)
    scale = np.maximum(S0.std(), 1e-9)
    disp, top_change = [], []
    for j in range(S0.shape[1]):
        r0 = np.argsort(np.argsort(-S0[:, j]))
        r1 = np.argsort(np.argsort(-S1[:, j]))
        disp.append(np.abs(r0 - r1).mean())
        a = set(np.argsort(-S0[:, j])[:k])
        b = set(np.argsort(-S1[:, j])[:k])
        top_change.append(1 - len(a & b) / k)
    return dict(mean_abs_score_shift=float(d.mean()), rel_shift=float(d.mean() / scale), mean_rank_displacement=float(np.mean(disp)),
                top10_change_rate=float(np.mean(top_change)))
