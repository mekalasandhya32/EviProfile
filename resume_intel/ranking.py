"""Skill graph, job-fit scoring methods and ranking metrics."""
from __future__ import annotations
import math
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np

from .generator import FAM_KNOWN, IMP_KNOWN
from .evidence import skill_evidence, credibility
from .extractor import default_matcher


class SkillGraph:
    """Skill relatedness: curated families (0.5), curated implications child->parent (0.7), and an optional
    data-driven NPMI edge learned from training resumes (weak domain prior, capped)."""

    def __init__(self, use_npmi: bool = False, npmi_cap: float = 0.15):
        self.fam = defaultdict(float)
        for f in FAM_KNOWN:
            for a, b in combinations(f, 2):
                self.fam[(a, b)] = self.fam[(b, a)] = 0.5
        self.imp = {(c, p): 0.7 for c, p in IMP_KNOWN}
        self.npmi: Dict[tuple, float] = {}
        self.use_npmi, self.npmi_cap = use_npmi, npmi_cap

    def fit_npmi(self, parsed_list: List[dict], min_count: int = 8):
        cnt, pair, n = Counter(), Counter(), len(parsed_list)
        for p in parsed_list:
            ss = sorted(p["narrative_skills"])
            cnt.update(ss)
            pair.update(combinations(ss, 2))
        for (a, b), c in pair.items():
            if c < min_count or cnt[a] < min_count or cnt[b] < min_count:
                continue
            pab, pa, pb = c / n, cnt[a] / n, cnt[b] / n
            npmi = math.log(pab / (pa * pb)) / (-math.log(pab))
            if npmi > 0:
                self.npmi[(a, b)] = self.npmi[(b, a)] = npmi
        return self

    def rel(self, have: str, need: str) -> float:
        """Credit multiplier when a job needs ``need`` and the candidate has ``have``."""
        if have == need:
            return 1.0
        r = max(self.fam.get((have, need), 0.0), self.imp.get((have, need), 0.0))
        if self.use_npmi:
            r = max(r, self.npmi_cap * self.npmi.get((have, need), 0.0))
        return r


def candidate_vectors(parsed: dict, ev: Optional[dict] = None):
    """Return the three candidate skill maps used by different matchers."""
    ev = ev or skill_evidence(parsed)
    anywhere = {s: 1.0 for s in parsed["skills"]}                          # keyword view: any mention counts fully
    listed_only = {s: 1.0 for s in parsed["listed_skills"]}
    ses = {s: v["ses"] for s, v in ev.items()}
    return anywhere, listed_only, ses


def fit_score(job: dict, cand: Dict[str, float], graph: Optional[SkillGraph], years: float) -> float:
    w_tot = len(job["required"]) + 0.5 * len(job["preferred"])
    tot = 0.0
    for wgt, skills in ((1.0, job["required"]), (0.5, job["preferred"])):
        for need in skills:
            if need in cand:
                c = cand[need]
            elif graph is not None:
                c = max((graph.rel(h, need) * v for h, v in cand.items()), default=0.0)
            else:
                c = 0.0
            tot += wgt * c
    base = tot / w_tot
    ef = 1.0 if job["min_years"] == 0 else min(1.0, years / job["min_years"])
    return base * (0.8 + 0.2 * ef)


def score_matrix(method: str, parsed_list: List[dict], jobs: List[dict], graph: SkillGraph, evs: List[dict],
                 creds: List[dict], tfidf=None, lsa=None, texts: Optional[List[str]] = None) -> np.ndarray:
    """scores[i, j] = fit of candidate i for job j under ``method``."""
    n, m = len(parsed_list), len(jobs)
    S = np.zeros((n, m))
    if method in ("tfidf_cosine", "lsa_cosine"):
        vec, svd = tfidf, lsa
        R = vec.transform(texts)
        J = vec.transform([j["text"] for j in jobs])
        if method == "lsa_cosine":
            from sklearn.preprocessing import normalize
            R = normalize(svd.transform(R))
            J = normalize(svd.transform(J))
            return R @ J.T
        return (R @ J.T).toarray()
    for i, p in enumerate(parsed_list):
        anywhere, listed_only, ses = candidate_vectors(p, evs[i])
        years = p["dated_years"]
        pci = creds[i]["pci"]
        for jx, job in enumerate(jobs):
            if method == "keyword":
                s = fit_score(job, anywhere, None, 1e9)
            elif method == "ontology":
                s = fit_score(job, anywhere, graph, 1e9)
            elif method == "ses_only":
                s = fit_score(job, ses, None, 1e9)
            elif method == "ses_ontology":
                s = fit_score(job, ses, graph, 1e9)
            elif method == "ses_ontology_exp":
                s = fit_score(job, ses, graph, years)
            elif method == "full":
                s = fit_score(job, ses, graph, years) * (0.55 + 0.45 * pci)
            elif method == "full_no_exp":
                s = fit_score(job, ses, graph, 1e9) * (0.55 + 0.45 * pci)
            elif method == "ses_pci_no_graph":
                s = fit_score(job, ses, None, years) * (0.55 + 0.45 * pci)
            elif method == "ontology_exp_pci":      # keyword-style evidence, but with graph + credibility
                s = fit_score(job, anywhere, graph, years) * (0.55 + 0.45 * pci)
            else:
                raise ValueError(method)
            S[i, jx] = s
    return S


# ---------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------
def dcg(rels):
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ranking_metrics(S: np.ndarray, grades: np.ndarray, inflated: np.ndarray, k: int = 10, rel_thr: int = 2) -> dict:
    n, m = S.shape
    ndcg, ap, p_at, promo, hits = [], [], [], [], []
    for j in range(m):
        order = np.argsort(-S[:, j], kind="stable")
        g = grades[order, j]
        top = g[:k]
        ideal = np.sort(grades[:, j])[::-1][:k]
        if dcg(ideal) > 0:
            ndcg.append(dcg(top) / dcg(ideal))
        n_rel = int((grades[:, j] >= rel_thr).sum())
        if n_rel > 0:
            hit = (top >= rel_thr).astype(float)
            p_at.append(hit.mean())
            prec = np.cumsum(hit) / (np.arange(len(hit)) + 1)
            ap.append(float((prec * hit).sum() / min(n_rel, k)))
            hits.append(hit.sum() / min(n_rel, k))
        infl_top = inflated[order][:k]
        promo.append(float(np.mean(infl_top & (top <= 1))))
    return dict(ndcg10=float(np.mean(ndcg)), map10=float(np.mean(ap)), p10=float(np.mean(p_at)),
                recall10=float(np.mean(hits)), false_promotion=float(np.mean(promo)))
