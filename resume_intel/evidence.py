"""Evidence layer (the project's core methodological contribution).

SES  - Skill Evidence Score: how strongly the resume text *demonstrates* a skill (not merely lists it).
PCI  - Profile Credibility Index: intrinsic, text-only internal-consistency checks (no external look-ups).
Both are transparent weighted formulas, fixed a priori (weights below); ablations and a learned variant
are reported in the experiments so the reader can see what each component contributes.
"""
from __future__ import annotations
import math
import re
from typing import Dict, List, Optional

from . import config as C

VERB_SCORE = {0: 0.50, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}      # 0 = unknown verb -> neutral
SECTION_SCORE = {"experience": 1.00, "projects": 0.70, "certifications": 0.60, "education": 0.40, "summary": 0.20, "skills": 0.0}
NON_NARRATIVE_SES = {"certifications": 0.35, "education": 0.20, "summary": 0.10, "skills": 0.08}
SES_WEIGHTS = dict(verb=0.30, section=0.20, metric=0.20, recency=0.15, repetition=0.15)
# Tier cut-points: 0.20 is fixed a priori ("no narrative evidence"); the other three were calibrated on the TRAINING split of the
# synthetic benchmark (grid search maximising exact agreement with latent proficiency, seed 42). Re-calibrate on your own data.
TIERS = [(0.20, "Claimed only"), (0.50, "Familiar"), (0.62, "Working"), (0.88, "Advanced"), (1.01, "Expert")]
PCI_WEIGHTS = dict(unsupported=0.35, stuffing=0.20, exp_gap=0.20, overlap=0.10, seniority=0.15)
SENIOR_RE = re.compile(r"\b(senior|sr|lead|principal|head|manager|director)\b", re.I)


def tier_of(ses: float) -> str:
    for thr, name in TIERS:
        if ses < thr:
            return name
    return TIERS[-1][1]


def ses_components(parsed: dict, skill: str) -> Optional[dict]:
    """Raw evidence components in [0,1] for one skill, or None if never mentioned."""
    rec = parsed["skills"].get(skill)
    if not rec:
        return None
    narr = [m for m in rec["mentions"] if m["section"] in ("experience", "projects")]
    if not narr:
        best = max(NON_NARRATIVE_SES.get(m["section"], 0.08) for m in rec["mentions"])
        return dict(narrative=False, fallback=best)
    verb = max(VERB_SCORE[m["verb"]] for m in narr)
    section = max(SECTION_SCORE[m["section"]] for m in narr)
    metric = 1.0 if any(m["metric"] for m in narr) else 0.0
    rec_scores = []
    for m in narr:
        if m["section"] == "projects" or m["job"] is None:
            rec_scores.append(0.6)
        else:
            job = parsed["jobs"][m["job"]]
            months = max(0, (C.REF_YEAR * 12 + C.REF_MONTH - 1) - (job["end"][0] * 12 + job["end"][1] - 1))
            rec_scores.append(math.exp(-(months / 12.0) / 6.0))
    recency = max(rec_scores)
    repetition = min(1.0, len(narr) / 3.0)
    return dict(narrative=True, verb=verb, section=section, metric=metric, recency=recency, repetition=repetition)


def ses_from_components(c: dict, weights: Optional[dict] = None) -> float:
    w = weights or SES_WEIGHTS
    if not c["narrative"]:
        return c["fallback"]
    tot = sum(w.values()) or 1.0
    return sum(w[k] * c[k] for k in w) / tot


def skill_evidence(parsed: dict, weights: Optional[dict] = None) -> Dict[str, dict]:
    out = {}
    for sk in parsed["skills"]:
        c = ses_components(parsed, sk)
        s = ses_from_components(c, weights)
        out[sk] = dict(ses=round(s, 4), tier=tier_of(s), comps=c, listed=parsed["skills"][sk]["listed"],
                       narrative=c["narrative"])
    return out


# ---------------------------------------------------------------------------------------------
# Profile credibility
# ---------------------------------------------------------------------------------------------
def pci_components(parsed: dict) -> dict:
    skills = parsed["skills"]
    listed = [s for s, v in skills.items() if v["listed"]]
    supported = [s for s in listed if any(m["section"] in ("experience", "projects", "certifications") for m in skills[s]["mentions"])]
    n_listed = len(listed)
    unsupported = (n_listed - len(supported)) / n_listed if n_listed else 0.0
    dated = parsed["dated_years"]
    density = n_listed / (6.0 + 1.5 * dated)
    stuffing = min(1.0, max(0.0, (density - 1.5) / 2.0))
    claimed = parsed["claimed_years"]
    exp_gap = 0.0
    if claimed is not None and claimed > 0:
        exp_gap = min(1.0, max(0.0, claimed - dated - 1.5) / claimed)
    overlap = min(1.0, parsed["overlap_months"] / 18.0)
    title = parsed.get("recent_title") or ""
    seniority = 1.0 if (SENIOR_RE.search(title) and dated < 3.0) else 0.0
    return dict(unsupported=unsupported, stuffing=stuffing, exp_gap=exp_gap, overlap=overlap, seniority=seniority,
                n_listed=n_listed, n_unsupported=n_listed - len(supported))


def pci_from_components(c: dict, weights: Optional[dict] = None) -> float:
    w = weights or PCI_WEIGHTS
    risk = sum(w[k] * c[k] for k in w) / (sum(w.values()) or 1.0)
    return float(max(0.0, min(1.0, 1.0 - risk)))


def credibility(parsed: dict, weights: Optional[dict] = None) -> dict:
    c = pci_components(parsed)
    flags = []
    if c["n_unsupported"] >= 3 and c["unsupported"] >= 0.30:
        flags.append(f"{c['n_unsupported']} of {c['n_listed']} listed skills have no supporting evidence in experience, projects or certificates")
    if c["exp_gap"] > 0.15:
        flags.append(f"summary claims {parsed['claimed_years']}+ years but dated roles total {parsed['dated_years']:.1f} years")
    if c["overlap"] > 0.0:
        flags.append(f"overlapping job dates ({parsed['overlap_months']} months beyond tolerance)")
    if c["seniority"] > 0:
        flags.append(f"senior-level title ('{parsed.get('recent_title')}') with only {parsed['dated_years']:.1f} years of dated experience")
    if c["stuffing"] > 0.25:
        flags.append(f"unusually many listed skills ({c['n_listed']}) for {parsed['dated_years']:.1f} years of experience")
    return dict(pci=round(pci_from_components(c, weights), 4), components=c, flags=flags)


def profile_strength(parsed: dict, ev: Dict[str, dict], cred: dict) -> dict:
    completeness = sum([bool(parsed["name"]), bool(parsed["email"]), bool(parsed["phone"]),
                        bool(parsed["education"]["degree"]), bool(parsed["jobs"]), bool(parsed["skills"])]) / 6.0
    top = sorted((v["ses"] for v in ev.values()), reverse=True)[:8]
    evidence_depth = sum(top) / len(top) if top else 0.0
    exp_norm = min(1.0, parsed["dated_years"] / 10.0)
    parts = dict(completeness=completeness, evidence_depth=evidence_depth, experience=exp_norm, credibility=cred["pci"])
    w = dict(completeness=0.15, evidence_depth=0.35, experience=0.20, credibility=0.30)
    score = 100.0 * sum(w[k] * parts[k] for k in w)
    return dict(score=round(score, 1), parts={k: round(v, 3) for k, v in parts.items()}, weights=w)
