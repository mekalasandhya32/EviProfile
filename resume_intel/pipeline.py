"""EviProfile: the assembled end-to-end pipeline (upload -> text -> NLP extraction -> evidence scoring ->
classification -> explainable profile -> optional job-fit). File-based, no database, no external services."""
from __future__ import annotations
import html
import json
from typing import Dict, List, Optional

import numpy as np

from . import config as C
from .extractor import parse_resume, default_matcher, ALL_SKILLS
from .evidence import skill_evidence, credibility, profile_strength
from .fairness import blind_text
from .models import FusedMLP, make_tfidf, structured_features
from .parsers import parse_file
from .ranking import SkillGraph, fit_score


class EviProfilePipeline:
    def __init__(self, seed: int = C.SEED, use_npmi: bool = False):
        self.seed = seed
        self.graph = SkillGraph(use_npmi=use_npmi)
        self.vec = make_tfidf()
        self.clf = FusedMLP(seed=seed)
        self.fitted = False

    # -------------------------------------------------------------------------------------
    def _features(self, parsed: dict):
        ev = skill_evidence(parsed)
        cred = credibility(parsed)
        return ev, cred, structured_features(parsed, ev, cred)

    def fit(self, raw_texts: List[str], labels: List[str]):
        parsed = [parse_resume(t) for t in raw_texts]
        feats = [self._features(p) for p in parsed]
        S = np.array([f[2] for f in feats])
        X = self.vec.fit_transform([blind_text(p) for p in parsed])
        self.clf.fit(X, S, np.array(labels))
        self.graph.fit_npmi(parsed)
        self.fitted = True
        return self

    # -------------------------------------------------------------------------------------
    def analyze_text(self, raw_text: str, job: Optional[dict] = None) -> dict:
        parsed = parse_resume(raw_text)
        ev, cred, S = self._features(parsed)
        strength = profile_strength(parsed, ev, cred)
        out = dict(
            candidate=dict(name=parsed["name"], email=parsed["email"], phone=parsed["phone"],
                           recent_title=parsed["recent_title"], dated_experience_years=parsed["dated_years"],
                           claimed_experience_years=parsed["claimed_years"],
                           education=parsed["education"], certifications=parsed["certifications"],
                           jobs=[dict(title=j["title"], company=j["company"], start=j["start"], end=j["end"], present=j["present"])
                                 for j in parsed["jobs"]]),
            skills=sorted(({"skill": s, "ses": v["ses"], "tier": v["tier"], "listed": v["listed"], "narrative": v["narrative"]}
                           for s, v in ev.items()), key=lambda d: -d["ses"]),
            credibility=dict(pci=cred["pci"], flags=cred["flags"], components={k: round(float(v), 3) for k, v in cred["components"].items()}),
            profile_strength=strength)
        if self.fitted:
            X = self.vec.transform([blind_text(parsed)])
            proba = self.clf.predict_proba(X, S.reshape(1, -1))[0]
            order = np.argsort(-proba)[:3]
            out["category"] = dict(predicted=str(self.clf.classes_[order[0]]),
                                   top3=[(str(self.clf.classes_[i]), round(float(proba[i]), 4)) for i in order])
        if job is not None:
            out["job_fit"] = self.job_fit(parsed, ev, cred, job)
        return out

    def analyze_file(self, path, job: Optional[dict] = None) -> dict:
        return self.analyze_text(parse_file(path), job)

    # -------------------------------------------------------------------------------------
    def job_fit(self, parsed: dict, ev: dict, cred: dict, job: dict) -> dict:
        ses = {s: v["ses"] for s, v in ev.items()}
        rows = []
        for kind, skills in (("required", job.get("required", [])), ("preferred", job.get("preferred", []))):
            for need in skills:
                if need in ses:
                    status, credit, via = ("demonstrated" if ev[need]["narrative"] else "claimed only"), ses[need], need
                else:
                    best = max(((self.graph.rel(h, need) * v, h) for h, v in ses.items()), default=(0.0, None))
                    credit, via = best
                    status = f"related ({via})" if credit > 0 else "missing"
                snippet = None
                if need in parsed["skills"]:
                    ms = [m["line"] for m in parsed["skills"][need]["mentions"] if m["section"] in ("experience", "projects")]
                    snippet = ms[0].lstrip("-•* ").strip() if ms else None
                rows.append(dict(skill=need, kind=kind, status=status, credit=round(float(credit), 3), evidence=snippet))
        score = fit_score(job, ses, self.graph, parsed["dated_years"])
        return dict(score=round(float(score), 4), score_pct=round(100 * float(score), 1),
                    credibility_adjusted_pct=round(100 * float(score) * (0.55 + 0.45 * cred["pci"]), 1),
                    min_years=job.get("min_years", 0), dated_years=parsed["dated_years"], rows=rows)

    # -------------------------------------------------------------------------------------
    def save(self, path):
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        import joblib
        return joblib.load(path)


def job_from_text(required: str, preferred: str = "", min_years: int = 0) -> dict:
    """Map free-text skill lists to canonical skills using the same matcher as resume parsing."""
    m = default_matcher()

    def canon(s):
        return list(dict.fromkeys(sk for sk, _, _ in m.find(s)))
    req = canon(required)
    pref = [s for s in canon(preferred) if s not in req]
    return dict(required=req, preferred=pref, min_years=int(min_years))


def render_html_card(profile: dict) -> str:
    """Self-contained HTML profile card (used by the Flask app and the example outputs)."""
    esc = html.escape
    c = profile["candidate"]
    tier_col = {"Claimed only": "#c0392b", "Familiar": "#e6a23c", "Working": "#3498db", "Advanced": "#1e8449", "Expert": "#145a32"}
    rows = "".join(
        f"<tr><td>{esc(s['skill'])}</td><td><div class='bar'><div style='width:{int(s['ses'] * 100)}%;background:{tier_col[s['tier']]}'></div></div></td>"
        f"<td>{s['ses']:.2f}</td><td style='color:{tier_col[s['tier']]}'>{esc(s['tier'])}</td></tr>" for s in profile["skills"][:20])
    flags = "".join(f"<li>{esc(f)}</li>" for f in profile["credibility"]["flags"]) or "<li>No credibility concerns detected</li>"
    parts = "".join(f"<li>{esc(k.replace('_', ' ').title())}: {v:.2f}</li>" for k, v in profile["profile_strength"]["parts"].items())
    cat = profile.get("category")
    cat_html = (f"<p><b>Predicted category:</b> {esc(cat['predicted'])} "
                f"<span class='muted'>({', '.join(f'{esc(a)} {b:.0%}' for a, b in cat['top3'])})</span></p>") if cat else ""
    jf = profile.get("job_fit")
    jf_html = ""
    if jf:
        jr = "".join(f"<tr><td>{esc(r['skill'])}</td><td>{r['kind']}</td><td>{esc(r['status'])}</td><td>{r['credit']:.2f}</td>"
                     f"<td class='muted'>{esc((r['evidence'] or '')[:90])}</td></tr>" for r in jf["rows"])
        jf_html = (f"<h3>Job fit: {jf['score_pct']}% (credibility-adjusted {jf['credibility_adjusted_pct']}%)</h3>"
                   f"<table><tr><th>Skill</th><th>Type</th><th>Status</th><th>Credit</th><th>Evidence</th></tr>{jr}</table>")
    edu = c["education"]
    return f"""<div class='card'>
<h2>{esc(c['name'] or 'Unknown candidate')}</h2>
<p class='muted'>{esc(c['email'] or '')} &nbsp; {esc(c['phone'] or '')}</p>
<p><b>Most recent role:</b> {esc(c['recent_title'] or 'n/a')} &nbsp;|&nbsp; <b>Dated experience:</b> {c['dated_experience_years']:.1f} yrs
 &nbsp;|&nbsp; <b>Claimed:</b> {c['claimed_experience_years'] if c['claimed_experience_years'] is not None else 'n/a'}</p>
<p><b>Education:</b> {esc(str(edu.get('degree') or 'n/a'))} {esc(str(edu.get('field') or ''))}</p>
{cat_html}
<div class='kpis'><div><span>{profile['profile_strength']['score']}</span>Profile strength</div>
<div><span>{profile['credibility']['pci']:.2f}</span>Credibility index</div></div>
<ul class='muted'>{parts}</ul>
<h3>Credibility notes</h3><ul>{flags}</ul>
<h3>Skill evidence (top 20)</h3><table><tr><th>Skill</th><th>Evidence</th><th>SES</th><th>Tier</th></tr>{rows}</table>
{jf_html}</div>"""


CARD_CSS = """body{font-family:Segoe UI,Arial,sans-serif;background:#f4f6f8;margin:0;padding:24px}
.card{max-width:900px;margin:auto;background:#fff;padding:24px 28px;border-radius:10px;box-shadow:0 2px 8px #0002}
table{border-collapse:collapse;width:100%;margin:8px 0}th,td{border-bottom:1px solid #e3e6ea;padding:5px 8px;text-align:left;font-size:14px}
.bar{background:#e8ebef;height:10px;width:160px;border-radius:5px}.bar div{height:10px;border-radius:5px}
.muted{color:#6b7480;font-size:13px}.kpis{display:flex;gap:16px;margin:12px 0}
.kpis div{background:#eef3f8;padding:10px 18px;border-radius:8px;font-size:13px;color:#445}.kpis span{display:block;font-size:26px;font-weight:700;color:#1b3a57}"""
