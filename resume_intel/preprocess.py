"""Stage 3: cleaning, layout repair, sentence/section segmentation and light tokenization."""
from __future__ import annotations
import re
from typing import Dict, List

from .lexicons import SECTION_HEADERS

BULLETS = ("-", "•", "*", "–", "▪", "·")
STOPWORDS = set("""a about above after again all also am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers him his how i if in
into is it its itself just me more most my no nor not now of off on once only or other our out over own same she should so some
such than that the their them then there these they this those through to too under until up very was we were what when where which
while who whom why will with would you your""".split())

MON = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
       r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
DATE = rf"(?:{MON}\.?\s+\d{{4}}|\d{{1,2}}/\d{{4}}|\b(?:19|20)\d{{2}}\b)"
RANGE_RE = re.compile(rf"({DATE})\s*(?:-|–|—|to)\s*({DATE}|present|current|till date|now|ongoing)", re.I)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+\d{1,3}[\s-]?)?(?:\(\d{3}\)\s?|\d{3,5}[\s-]?)\d{3,5}(?:[\s-]?\d{3,5})?(?!\d)")
URL_RE = re.compile(r"(?:https?://|www\.)\S+|linkedin\.com/\S+|github\.com/\S+", re.I)

_HEADER_LOOKUP: Dict[str, str] = {}
for sec, names in SECTION_HEADERS.items():
    for n in names:
        _HEADER_LOOKUP[n] = sec


def _norm_header(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().rstrip(":").strip().lower())


def header_of(line: str):
    """Return the section name if ``line`` is a bare section heading, else None."""
    s = line.strip()
    if not s or len(s) > 40 or s.startswith(BULLETS):
        return None
    return _HEADER_LOOKUP.get(_norm_header(s))


def inline_header(line: str):
    """'Skills: Python, SQL' -> ('skills', 'Python, SQL')."""
    m = re.match(r"^\s*([A-Za-z &]{3,32}):\s*(\S.*)$", line)
    if m:
        sec = _HEADER_LOOKUP.get(_norm_header(m.group(1)))
        if sec:
            return sec, m.group(2)
    return None


def repair_layout(text: str) -> str:
    """Re-join lines that a PDF/DOCX export wrapped in the middle of a bullet or list."""
    lines = [l.rstrip() for l in text.replace("\r", "").split("\n")]
    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            out.append("")
            continue
        if out and out[-1]:
            prev = out[-1]
            is_new = (s.startswith(BULLETS) or header_of(s) or inline_header(s) or RANGE_RE.search(s))
            cont = False
            if not is_new:
                if prev.startswith(BULLETS) and not prev.endswith((".", "!", "?")):
                    cont = True
                elif prev.endswith(","):
                    cont = True
                elif len(prev) >= 60 and not prev.endswith((".", ":")) and not header_of(prev) and not RANGE_RE.search(prev):
                    cont = True
            if cont:
                out[-1] = prev + " " + s
                continue
        out.append(s)
    return "\n".join(out)


def split_sections(text: str) -> Dict[str, List[str]]:
    """Segment a resume into canonical sections. Everything before the first heading is 'header'."""
    secs: Dict[str, List[str]] = {"header": []}
    cur = "header"
    for ln in text.split("\n"):
        if not ln.strip():
            continue
        h = header_of(ln)
        if h:
            cur = h
            secs.setdefault(cur, [])
            continue
        ih = inline_header(ln)
        if ih:
            cur = ih[0]
            secs.setdefault(cur, [])
            secs[cur].append(ih[1])
            continue
        secs.setdefault(cur, []).append(ln.strip())
    return secs


def sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s.strip()]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z][a-z0-9+#.\-]*[a-z0-9+#]|[a-z]", text.lower())


def light_lemma(tok: str) -> str:
    for suf in ("ings", "ing", "ies", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            return tok[: -len(suf)] + ("y" if suf == "ies" else "")
    return tok


def model_text(text: str) -> str:
    """Text fed to TF-IDF models: contact tokens removed, lower-cased."""
    t = EMAIL_RE.sub(" ", text)
    t = URL_RE.sub(" ", t)
    t = PHONE_RE.sub(" ", t)
    return t.lower()
