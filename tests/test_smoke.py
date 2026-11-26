"""Fast sanity tests: python -m pytest -q   (or: python tests/test_smoke.py)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resume_intel import generator as G
from resume_intel.extractor import parse_resume
from resume_intel.evidence import skill_evidence, credibility


def _recs():
    return G.generate_corpus(n_per_class=5, seed=1)


def test_extraction_basics():
    for r in _recs()[:20]:
        p = parse_resume(r["text"])
        assert p["email"] == r["identity"]["email"]
        assert p["jobs"], "at least one job must be parsed"


def test_evidence_is_monotone_in_verb_level():
    txt = """JANE DOE
Pune | jane@x.com | +91 98765 43210

SKILLS
Python, SQL

EXPERIENCE
Data Analyst | Acme | Jan 2022 - Present
- Assisted the team with SQL tasks.
- Led churn models using Python, improving retention by 20%.
"""
    ev = skill_evidence(parse_resume(txt))
    assert ev["Python"]["ses"] > ev["SQL"]["ses"]


def test_unsupported_skills_lower_credibility():
    honest = """A B
Pune | a@b.com | +91 98765 43210

SKILLS
Python

EXPERIENCE
Data Analyst | Acme | Jan 2020 - Present
- Developed models using Python, improving accuracy by 10%.
"""
    stuffed = honest.replace("Python", "Python, Java, Kubernetes, TensorFlow, Terraform, Docker, Spark, SQL, AWS", 1)
    assert credibility(parse_resume(stuffed))["pci"] < credibility(parse_resume(honest))["pci"]


def test_counterfactual_twin_differs_only_in_identity():
    r = _recs()[0]
    t = G.swap_identity(r)
    a, b = r["text"].split("\n"), G.render_text(t).split("\n")
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(diff) <= 4


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
