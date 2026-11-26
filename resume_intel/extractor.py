"""Stage 4: lexicon-driven NLP information extraction with provenance.

Unlike a plain keyword parser, every skill mention is stored with WHERE it occurs (section, job),
HOW it is used (verb level 0-4 from an ownership ladder) and WHETHER the sentence reports a
quantified outcome. This provenance is what the evidence layer (evidence.py) scores.
"""
from __future__ import annotations
import re
from functools import lru_cache
from typing import Dict, List, Optional

from . import config as C
from .lexicons import CAT, ALIASES, SOFT_SKILLS, EXT_VERB_PATTERNS, METRIC_PATTERN, TITLE_ROLE_WORDS
from .preprocess import (repair_layout, split_sections, EMAIL_RE, PHONE_RE, URL_RE, RANGE_RE, BULLETS, MON)

ALL_SKILLS = sorted({s for c in CAT.values() for s in c["core"] + c["secondary"]})
MONTH_IDX = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
VERB_RE = {lv: re.compile(p, re.I) for lv, p in EXT_VERB_PATTERNS.items()}
METRIC_RE = re.compile(METRIC_PATTERN)
TITLE_RE = re.compile(r"\b" + TITLE_ROLE_WORDS + r"\b", re.I)
CLAIM_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:years|yrs)", re.I)
DEGREE_RE = re.compile(r"^(B\.?\s?Tech|B\.?\s?E\.?|B\.?\s?Sc\.?|B\.?\s?Com\.?|B\.?\s?A\.?|BBA|M\.?\s?Tech|M\.?\s?Sc\.?|M\.?\s?Com\.?|M\.?\s?A\.?|MBA|Ph\.?\s?D|Diploma|Bachelor|Master)\b",
                       re.I)
INST_RE = re.compile(r"(Institute|University|College|Polytechnic|School)", re.I)


class SkillMatcher:
    """Longest-match alias matcher with boundary guards for tokens like C++, Node.js, CI/CD."""

    def __init__(self, extra_aliases: Optional[Dict[str, List[str]]] = None):
        self.alias2skill: Dict[str, str] = {}
        for s in ALL_SKILLS:
            self.alias2skill[s.lower()] = s
            for a in ALIASES.get(s, []):
                self.alias2skill[a.lower()] = s
        for s, al in (extra_aliases or {}).items():
            for a in al:
                self.alias2skill[a.lower()] = s
        keys = sorted(self.alias2skill, key=len, reverse=True)
        pat = "|".join(re.escape(k) for k in keys)
        self.rx = re.compile(rf"(?<![\w+#./])({pat})(?![\w+#]|\.\w)", re.I)

    def find(self, text: str):
        out = []
        for m in self.rx.finditer(text):
            out.append((self.alias2skill[m.group(1).lower()], m.start(), m.end()))
        return out


@lru_cache(maxsize=1)
def default_matcher() -> SkillMatcher:
    return SkillMatcher()


def _parse_date(tok: str, is_end: bool):
    t = tok.strip().lower().rstrip(".")
    if t in ("present", "current", "till date", "now", "ongoing"):
        return (C.REF_YEAR, C.REF_MONTH)
    m = re.match(rf"({MON})\.?\s+(\d{{4}})", t, re.I)
    if m:
        return (int(m.group(2)), MONTH_IDX[m.group(1)[:3].lower()])
    m = re.match(r"(\d{1,2})/(\d{4})", t)
    if m:
        return (int(m.group(2)), min(12, max(1, int(m.group(1)))))
    m = re.match(r"((?:19|20)\d{2})$", t)
    if m:
        y = int(m.group(1))
        if is_end:
            return (y, C.REF_MONTH if y >= C.REF_YEAR else 12)
        return (y, 1)
    return None


def _ym(t):
    return t[0] * 12 + t[1] - 1


def verb_level(sentence: str) -> int:
    """Ownership level of the FIRST action verb in the sentence (0 = no known verb)."""
    best_pos, best_lv = 10 ** 6, 0
    for lv, rx in VERB_RE.items():
        m = rx.search(sentence)
        if m and (m.start() < best_pos or (m.start() == best_pos and lv > best_lv)):
            best_pos, best_lv = m.start(), lv
    return best_lv


def _parse_job_header(line: str):
    m = RANGE_RE.search(line)
    if not m:
        return None
    start = _parse_date(m.group(1), False)
    end_tok = m.group(2)
    end = _parse_date(end_tok, True)
    present = end_tok.strip().lower() in ("present", "current", "till date", "now", "ongoing")
    rest = (line[:m.start()] + " " + line[m.end():])
    rest = re.sub(r"[()\[\]]", " ", rest)
    pieces = [p.strip(" ,|-–—") for p in re.split(r"\s\|\s|\s[-–—]\s|,|\|", rest) if p.strip(" ,|-–—")]
    title, company = None, None
    for p in pieces:
        if title is None and TITLE_RE.search(p):
            title = p
        elif company is None:
            company = p
    if title is None and pieces:
        title = pieces[0]
        company = pieces[1] if len(pieces) > 1 else None
    year_only = bool(re.fullmatch(r"(?:19|20)\d{2}", m.group(1).strip()))
    return dict(title=title, company=company, start=start, end=end, present=present, year_only=year_only)


def _parse_education(lines: List[str]):
    text = "\n".join(lines)
    pieces = [p.strip() for p in re.split(r",|\n|\(|\)|\s-\s", text) if p.strip()]
    deg = field = inst = None
    year = None
    for p in pieces:
        if DEGREE_RE.match(p) and deg is None:
            m = DEGREE_RE.match(p)
            deg = m.group(1)
            fm = re.search(r"\bin\s+(.+)$", p, re.I)
            if fm:
                field = fm.group(1).strip()
        elif INST_RE.search(p) and inst is None:
            inst = p
        elif re.fullmatch(r"(?:19|20)\d{2}", p):
            year = int(p)
        elif field is None and deg is not None and not INST_RE.search(p):
            field = p
    level = 0
    if deg:
        d = deg.lower().replace(".", "").replace(" ", "")
        level = 1 if d.startswith("dip") else 4 if d.startswith("ph") else 3 if (d.startswith("m") or d.startswith("master")) else 2
    return dict(degree=deg, field=field, institution=inst, year=year, level=level)


def parse_resume(raw_text: str, matcher: Optional[SkillMatcher] = None) -> dict:
    matcher = matcher or default_matcher()
    text = repair_layout(raw_text)
    secs = split_sections(text)
    header = "\n".join(secs.get("header", []))
    joined = "\n".join(l for k, v in secs.items() for l in v)

    # --- contact -------------------------------------------------------------------------
    email = EMAIL_RE.search(text)
    phone = None
    for m in PHONE_RE.finditer(header or text[:400]):
        digits = re.sub(r"\D", "", m.group(0))
        if 10 <= len(digits) <= 13:
            phone = m.group(0).strip()
            break
    name = None
    for ln in (secs.get("header") or [])[:3]:
        tokens = ln.replace(".", " ").split()
        if 2 <= len(tokens) <= 4 and all(re.fullmatch(r"[A-Za-z'\-]+", t) for t in tokens) and "@" not in ln:
            name = " ".join(t.capitalize() for t in tokens)
            break
    urls = URL_RE.findall(text)

    # --- jobs & bullets ------------------------------------------------------------------
    jobs: List[dict] = []
    cur = None
    mentions: Dict[str, dict] = {}

    def add_mention(skill, section, line, job_idx):
        rec = mentions.setdefault(skill, dict(listed=False, mentions=[]))
        narrative = section in ("experience", "projects")
        rec["mentions"].append(dict(
            section=section, job=job_idx, line=line,
            verb=verb_level(line) if narrative else 0,
            metric=bool(METRIC_RE.search(line)) if narrative else False))
        if section == "skills":
            rec["listed"] = True

    for ln in secs.get("experience", []):
        if not ln.startswith(BULLETS) and RANGE_RE.search(ln):
            hdr = _parse_job_header(ln)
            if hdr and hdr["start"] and hdr["end"]:
                cur = dict(hdr, lines=[])
                jobs.append(cur)
                continue
        if cur is not None:
            cur["lines"].append(ln)
        for sk, _, _ in matcher.find(ln):
            add_mention(sk, "experience", ln, len(jobs) - 1 if cur is not None else None)
    for ln in secs.get("projects", []):
        for sk, _, _ in matcher.find(ln):
            add_mention(sk, "projects", ln, None)
    for sec in ("skills", "certifications", "education", "summary"):
        for ln in secs.get(sec, []):
            for sk, _, _ in matcher.find(ln):
                add_mention(sk, sec, ln, None)
    # a skill that appears only in a narrative-free context but with listed flag
    for sk, rec in mentions.items():
        rec["listed"] = any(m["section"] == "skills" for m in rec["mentions"])

    # --- timeline ------------------------------------------------------------------------
    covered, total, anomalies = set(), 0, 0
    ref = C.REF_YEAR * 12 + C.REF_MONTH - 1
    for j in jobs:
        a, b = _ym(j["start"]), _ym(j["end"])
        if b < a:
            anomalies += 1
            continue
        if b > ref + 1:
            anomalies += 1
            b = ref
        j["months"] = b - a
        total += b - a
        covered.update(range(a, b))
    dated_years = round(len(covered) / 12.0, 2)
    # pairwise overlap beyond a tolerance that depends on date granularity (year-only dates are ambiguous by up to 12 months)
    ordered = sorted([j for j in jobs if "months" in j], key=lambda j: _ym(j["start"]))
    overlap_months = 0
    for a_, b_ in zip(ordered, ordered[1:]):
        tol = 12 if (a_.get("year_only") or b_.get("year_only")) else 2
        overlap_months += max(0, _ym(a_["end"]) - _ym(b_["start"]) - tol)
    summary_text = " ".join(secs.get("summary", []))
    cm = CLAIM_RE.search(summary_text) or CLAIM_RE.search(joined[:500])
    claimed = int(cm.group(1)) if cm else None

    # --- education / certs / soft skills --------------------------------------------------
    edu = _parse_education(secs.get("education", []))
    certs = [re.sub(r"^[\-•*–▪·\s]+", "", l).strip() for l in secs.get("certifications", [])]
    low = joined.lower()
    soft = [s for s in SOFT_SKILLS if s in low]

    recent_title = jobs[0]["title"] if jobs else None
    if jobs:  # resumes list jobs reverse-chronologically, but do not rely on it
        recent_title = max(jobs, key=lambda j: (_ym(j["end"]), _ym(j["start"])))["title"]
    return dict(
        name=name, email=email.group(0) if email else None, phone=phone, links=urls,
        sections=secs, jobs=jobs, education=edu, certifications=certs, skills=mentions, soft_skills=soft,
        claimed_years=claimed, dated_years=dated_years, overlap_months=overlap_months, date_anomalies=anomalies,
        recent_title=recent_title, n_listed=sum(1 for v in mentions.values() if v["listed"]), clean_text=text,
        listed_skills=[s for s, v in mentions.items() if v["listed"]],
        narrative_skills=[s for s, v in mentions.items() if any(m["section"] in ("experience", "projects") for m in v["mentions"])],
    )
