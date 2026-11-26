"""Synthetic benchmark generator.

A real, labelled, recruiter-validated resume corpus with *evidence-level* ground truth does not exist
publicly (resumes are personal data). This module therefore generates a controlled corpus in which the
latent truth is known for every resume:
  * true category (the most recent professional field),
  * which skills are genuinely demonstrated and at what proficiency (1-4),
  * which listed skills are unsupported (inflation),
  * true experience from dates, claimed experience, identity attributes (name group, institution tier).
All random draws are made first (``make_record``); rendering to text is deterministic
(``render_text``), so identity-swapped counterfactual twins differ ONLY in identity tokens.
"""
from __future__ import annotations
import copy
import random
from typing import Dict, List, Optional

from . import config as C
from .lexicons import (CAT, CATEGORIES, ALIASES, FAMILIES, IMPLIES, HELD_OUT_FAMILIES, HELD_OUT_IMPLIES,
                       GEN_VERBS, SOFT_SKILLS)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_FULL = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December"]

FIRST_A = ["James", "Robert", "Michael", "Arjun", "Rahul", "Vikram", "Daniel", "Ahmed", "Carlos", "Liam", "Rohan",
           "Karthik", "Sanjay", "Omar", "David", "Kevin", "Aditya", "Suresh", "Thomas", "Ravi"]      # male-coded pool
FIRST_B = ["Priya", "Ananya", "Sarah", "Emily", "Fatima", "Neha", "Sneha", "Maria", "Jessica", "Divya", "Aisha",
           "Lakshmi", "Olivia", "Meera", "Sophia", "Kavya", "Isabella", "Pooja", "Rachel", "Anjali"]  # female-coded pool
SURNAMES = ["Sharma", "Reddy", "Patel", "Kumar", "Singh", "Iyer", "Nair", "Khan", "Smith", "Johnson", "Williams",
            "Brown", "Garcia", "Martinez", "Lee", "Chen", "Rao", "Gupta", "Das", "Mehta", "Verma", "Joshi", "Kapoor",
            "Thomas", "Walker", "Hall", "Young", "Allen", "Wright", "Lopez"]
CITIES = ["Hyderabad", "Bengaluru", "Pune", "Chennai", "Mumbai", "Delhi", "Austin, TX", "Seattle, WA", "London, UK",
          "Toronto, Canada", "Singapore", "Dubai, UAE"]
UNIS = {1: ["Northbridge Institute of Technology", "Crestview University", "Meridian Institute of Science"],
        2: ["Lakeshore University", "Eastfield College of Engineering", "Riverton University", "Summit State University"],
        3: ["Greenvale College", "Oakridge Institute", "Southgate Polytechnic", "Hillcrest Degree College"]}
COMPANIES = ["Nexora Analytics", "BluePeak Systems", "Orbis Technologies", "Quantix Labs", "Veridian Solutions",
             "Stellar Dynamics", "Apex Infrasoft", "Cobalt Works", "Zenith Global", "Lumen Industries", "Pioneer Retail",
             "Atlas Consulting", "Helix Enterprises", "Brightwave Media", "Ironclad Holdings", "Vertex Partners",
             "Northstar Group", "Silverline Corp", "Evergreen Projects", "Skyline Infra", "Titan Manufacturing",
             "Aurora Financial", "Redwood Services", "Pinnacle Staffing", "Maple Logistics", "Orion Healthcare IT"]
ACT_A = ["Captain, university rugby team", "Member, Men's debating society", "Treasurer, fraternity chapter"]
ACT_B = ["Member, Women in Technology network", "Captain, university netball team", "Secretary, sorority chapter"]
ACT_N = ["Volunteer, community coding club", "Marathon runner", "Photography club member", "Open-source contributor",
         "Volunteer, local food bank", "Member, university quiz club"]
METRICS = ["processing time", "operational costs", "customer satisfaction", "report turnaround", "error rate",
           "team productivity", "system uptime", "lead conversion", "ramp-up time", "delivery timelines"]
DOMAINS = ["gmail.com", "outlook.com", "yahoo.com", "mail.com", "proton.me"]
# Fields whose vocabularies genuinely overlap in practice (used to create realistic confusion).
NEIGHBORS = {"Data Science": ["Software Engineering", "Finance & Accounting"], "Software Engineering": ["Data Science", "DevOps & Cloud"],
             "DevOps & Cloud": ["Software Engineering", "Cybersecurity"], "Cybersecurity": ["DevOps & Cloud", "Software Engineering"],
             "Human Resources": ["Sales & Business Development", "Digital Marketing"], "Digital Marketing": ["Sales & Business Development", "Human Resources"],
             "Mechanical Engineering": ["Civil Engineering"], "Civil Engineering": ["Mechanical Engineering"],
             "Finance & Accounting": ["Sales & Business Development", "Data Science"], "Sales & Business Development": ["Digital Marketing", "Finance & Accounting"]}
GENERIC_TASKS = ["cross-functional improvement initiatives", "quarterly reporting packs", "stakeholder presentations",
                 "process documentation efforts", "internal efficiency programs", "client-facing deliverables",
                 "team coordination activities", "vendor coordination work", "planning and tracking routines", "data collection and review cycles"]
GENERIC_TERMS = ["process improvement", "cross-functional collaboration", "analytical thinking", "stakeholder communication", "operational excellence"]
GENERIC_TITLES = ["Associate", "Analyst", "Senior Associate", "Team Lead"]

# Phrasings that only the generator uses (the extractor does not know them) -> realistic recall loss.
GEN_ONLY_ALIASES = {"Python": ["python3"], "Machine Learning": ["ml/ai"], "Kubernetes": ["kube"], "Excel": ["spreadsheets"],
                    "Recruitment": ["talent hunting"], "Auditing": ["internal audit"], "SQL": ["t-sql"],
                    "JavaScript": ["vanilla js"], "Docker": ["containers"], "Budgeting": ["budget planning"],
                    "Lead Generation": ["prospecting"], "Network Security": ["netsec"], "SEO": ["on-page seo"]}

FAM_FULL = FAMILIES
FAM_KNOWN = [f for i, f in enumerate(FAMILIES) if i not in HELD_OUT_FAMILIES]
IMP_FULL = IMPLIES
IMP_KNOWN = [p for i, p in enumerate(IMPLIES) if i not in HELD_OUT_IMPLIES]

LEVEL_PROBS = {0: [.40, .40, .15, .05], 1: [.20, .40, .30, .10], 2: [.10, .25, .40, .25], 3: [.05, .15, .40, .40]}
SENIORITY_YEARS = {0: (0.5, 2.0), 1: (2.5, 5.0), 2: (5.5, 9.0), 3: (9.5, 15.0)}


def _ym(y, m):
    return y * 12 + (m - 1)


def _from_ym(v):
    return v // 12, v % 12 + 1


def months_between(a, b):
    return _ym(*b) - _ym(*a)


def skill_pool(category):
    d = CAT[category]
    return d["core"], d["secondary"]


def make_identity(rng, group=None, tier=None):
    group = group or rng.choice(["A", "B"])
    first = rng.choice(FIRST_A if group == "A" else FIRST_B)
    last = rng.choice(SURNAMES)
    tier = tier or rng.choices([1, 2, 3], weights=[.2, .45, .35])[0]
    handle = f"{first}.{last}".lower() + (str(rng.randint(1, 99)) if rng.random() < .4 else "")
    phone_fmt = rng.randint(0, 3)
    d = [rng.randint(0, 9) for _ in range(10)]
    d[0] = rng.randint(6, 9)
    s = "".join(map(str, d))
    phone = [f"+91 {s[:5]} {s[5:]}", f"({s[:3]}) {s[3:6]}-{s[6:]}", f"{s[:5]}-{s[5:]}", f"+1-{s[:3]}-{s[3:6]}-{s[6:]}"][phone_fmt]
    r = rng.random()
    activity = None
    if r < 0.40:
        activity = rng.choice(ACT_A if group == "A" else ACT_B)       # group-correlated proxy
    elif r < 0.65:
        activity = rng.choice(ACT_N)
    return dict(group=group, first=first, last=last, tier=tier, university=rng.choice(UNIS[tier]),
                email=f"{handle}@{rng.choice(DOMAINS)}", phone=phone, city=rng.choice(CITIES),
                linkedin=f"linkedin.com/in/{handle}", activity=activity)


def _draw_levels(rng, seniority, n):
    return rng.choices([1, 2, 3, 4], weights=LEVEL_PROBS[seniority], k=n)


def make_record(rng: random.Random, rid: int, category: str, hybrid: bool, inflated: bool, noisy: bool) -> dict:
    other = rng.choice([c for c in CATEGORIES if c != category]) if hybrid else None
    seniority = rng.choices([0, 1, 2, 3], weights=[.25, .35, .27, .13])[0]
    if hybrid:
        seniority = max(seniority, 1)
    lo, hi = SENIORITY_YEARS[seniority]
    exp_target = rng.uniform(lo, hi)
    ident = make_identity(rng)
    core, sec = skill_pool(category)

    # --- true skills ------------------------------------------------------------------
    n_true = max(5, min(15, 7 + seniority * 2 + rng.randint(-2, 2)))
    pool_a = list(core) * 2 + list(sec)
    true = []
    thin = rng.random() < 0.12
    if thin:
        n_true = rng.randint(4, 6)
    n_other = int(round(n_true * 0.55)) if hybrid else (rng.randint(2, 5) if rng.random() < .60 else 0)
    n_main = n_true - n_other
    while len(true) < n_main:
        s = rng.choice(pool_a)
        if s not in true:
            true.append(s)
    if n_other:
        if hybrid:
            src = other
        else:
            src = rng.choice(NEIGHBORS[category]) if rng.random() < 0.7 else rng.choice([c for c in CATEGORIES if c != category])
        oc, osec = skill_pool(src)
        op = list(oc) + list(osec)
        tries = 0
        while len([s for s in true if s in op]) < n_other and tries < 50:
            s = rng.choice(op)
            tries += 1
            if s not in true:
                true.append(s)
    levels = _draw_levels(rng, seniority, len(true))
    true_skills = dict(zip(true, levels))

    # --- timeline ---------------------------------------------------------------------
    n_jobs = 1 if exp_target < 1.5 else (2 if exp_target < 5 else rng.choice([2, 3, 3, 4]))
    if hybrid:
        n_jobs = max(2, n_jobs)
    present = rng.random() < 0.6
    end = (C.REF_YEAR, C.REF_MONTH) if present else _from_ym(_ym(C.REF_YEAR, C.REF_MONTH) - rng.randint(1, 8))
    total_months = int(exp_target * 12)
    cuts = sorted(rng.sample(range(1, max(total_months, n_jobs + 1)), n_jobs - 1)) if n_jobs > 1 else []
    bounds = [0] + cuts + [max(total_months, n_jobs)]
    durations = [max(3, bounds[i + 1] - bounds[i]) for i in range(n_jobs)]
    if hybrid and n_jobs >= 2:
        new_last = rng.randint(6, 18)
        durations[0] += max(0, durations[-1] - new_last)
        durations[-1] = new_last
    jobs = []
    cursor_end = _ym(*end)
    for k in range(n_jobs - 1, -1, -1):                              # build backwards: last job first
        gap = rng.randint(0, 3) if k < n_jobs - 1 else 0
        j_end = cursor_end - (gap if k < n_jobs - 1 else 0)
        j_start = j_end - durations[k]
        jobs.append(dict(start=_from_ym(j_start), end=_from_ym(j_end), present=(k == n_jobs - 1 and present)))
        cursor_end = j_start
    jobs = list(reversed(jobs))                                     # chronological
    overlap_injected = False
    if inflated and n_jobs >= 2 and rng.random() < 0.35:             # impossible overlap
        k = rng.randint(1, n_jobs - 1)
        shift = rng.randint(4, 12)
        s = _ym(*jobs[k]["start"]) - shift
        jobs[k]["start"] = _from_ym(s)
        overlap_injected = True
    # union of months -> true dated experience
    covered = set()
    for j in jobs:
        for mth in range(_ym(*j["start"]), _ym(*j["end"])):
            covered.add(mth)
    true_years = round(len(covered) / 12.0, 2)

    # titles
    cat_for_job = [category] * n_jobs
    if hybrid:
        for k in range(max(1, n_jobs // 2)):
            cat_for_job[k] = other
    for k, j in enumerate(jobs):
        lvl = min(3, max(0, seniority - (n_jobs - 1 - k)))
        j["category"] = cat_for_job[k]
        j["title"] = CAT[cat_for_job[k]]["titles"][lvl]
        if rng.random() < (0.20 if k == n_jobs - 1 else 0.35):
            j["title"] = GENERIC_TITLES[lvl]
        j["company"] = rng.choice(COMPANIES)
    title_inflated = False
    if inflated and rng.random() < 0.4:
        cur = jobs[-1]
        lvl = CAT[cur["category"]]["titles"].index(cur["title"]) if cur["title"] in CAT[cur["category"]]["titles"] else GENERIC_TITLES.index(cur["title"])
        if lvl < 3:
            cur["title"] = CAT[cur["category"]]["titles"][min(3, lvl + 1)]
            title_inflated = True

    # assign true skills to jobs/projects; guarantee >=1 narrative mention for each true skill
    projects = []
    for p in range(0 if thin else (rng.randint(1, 3) if seniority < 2 else rng.randint(0, 2))):
        projects.append(dict(name=rng.choice(CAT[category]["projects"]), skills=[]))
    slots = list(range(n_jobs)) + [("p", i) for i in range(len(projects))]
    for j in jobs:
        j["skills"] = []
    for s, lv in true_skills.items():
        # recent jobs more likely; hybrid skills of the other category go to early jobs
        if hybrid and s in (CAT[other]["core"] + CAT[other]["secondary"]) and s not in core and s not in sec:
            slot = rng.randrange(0, max(1, n_jobs // 2))
        else:
            w = [(k + 1) ** 1.5 for k in range(n_jobs)] + [1.0] * len(projects)
            slot = rng.choices(slots, weights=w)[0]
        if isinstance(slot, tuple):
            projects[slot[1]]["skills"].append((s, lv))
        else:
            jobs[slot]["skills"].append((s, lv))
        if rng.random() < 0.25:                                      # a second, repeat mention
            k2 = rng.randrange(n_jobs)
            jobs[k2]["skills"].append((s, max(1, lv - (1 if k2 != n_jobs - 1 else 0))))

    # --- listed skills (Skills section) --------------------------------------------------
    listed = [s for s, lv in true_skills.items() if rng.random() < (0.85 if lv >= 2 else 0.6)]
    natural_unsupported = []
    if rng.random() < 0.40:                                         # honest "listed but not narrated"
        cand = [s for s in (core + sec) if s not in true_skills]
        natural_unsupported = rng.sample(cand, k=min(len(cand), rng.randint(1, 3)))
    listed += natural_unsupported
    inflated_extra = []
    if inflated:
        cand = [s for s in (core + sec) if s not in true_skills and s not in listed]
        other_cat = rng.choice([c for c in CATEGORIES if c != category])
        cand += [s for s in CAT[other_cat]["core"] if s not in true_skills and s not in listed]
        inflated_extra = rng.sample(cand, k=min(len(cand), rng.randint(2, 6)))
        listed += inflated_extra
    rng.shuffle(listed)
    narrative = set()
    for j in jobs:
        narrative.update(s for s, _ in j["skills"])
    for p in projects:
        narrative.update(s for s, _ in p["skills"])
    unsupported = [s for s in listed if s not in narrative]

    # --- education / certs ---------------------------------------------------------------
    edu_cat = other if (hybrid and rng.random() < .5) else category
    degree, field = rng.choice(CAT[edu_cat]["degrees"])
    grad_year = max(1998, min(C.REF_YEAR, C.REF_YEAR - int(true_years) - rng.randint(0, 2)))
    certs = []
    if rng.random() < 0.40 and not thin:
        certs = rng.sample(CAT[category]["certs"], k=rng.randint(1, 2))
    claimed = None
    if inflated:
        claimed = int(true_years) + rng.randint(1, 3)
    elif rng.random() < 0.90:
        claimed = max(0, int(round(true_years)) - (1 if rng.random() < .25 else 0))
    return dict(
        id=rid, category=category, thin=bool(thin), hybrid=bool(hybrid), other_category=other, inflated=bool(inflated), noisy=bool(noisy),
        seniority=seniority, identity=ident, jobs=jobs, projects=projects, education=dict(degree=degree, field=field, year=grad_year),
        certs=certs, listed_skills=listed, true_skills=true_skills, narrative_skills=sorted(narrative),
        unsupported_listed=unsupported, inflated_extra=inflated_extra, natural_unsupported=natural_unsupported,
        true_years=true_years, claimed_years=claimed, overlap_injected=overlap_injected, title_inflated=title_inflated,
        style=rng.randint(0, 2), date_fmt=rng.randint(0, 3), job_fmt=rng.randint(0, 2), bullet=rng.choice(["-", "•", "*"]),
        order=rng.randint(0, 2), hdr_pick=rng.randint(0, 99), rseed=rng.randint(1, 10 ** 9),
        soft=rng.sample(SOFT_SKILLS, 3),
    )


# ---------------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------------
def _fmt_date(ym, fmt, rng_unused=None):
    y, m = ym
    return [f"{MONTHS[m - 1]} {y}", f"{m:02d}/{y}", f"{y}", f"{MONTHS_FULL[m - 1]} {y}"][fmt]


def _range(job, fmt):
    a = _fmt_date(job["start"], fmt)
    if job.get("present"):
        b = ["Present", "Present", "Present", "Till date"][fmt]
    else:
        b = _fmt_date(job["end"], fmt)
    sep = " to " if fmt == 3 else " - "
    return f"{a}{sep}{b}"


def _alias(rng, skill):
    opts = [skill.lower() if rng.random() < 0.3 else skill]
    pool = ALIASES.get(skill, []) + GEN_ONLY_ALIASES.get(skill, [])
    r = rng.random()
    if pool and r < 0.25:
        return rng.choice(pool)
    return opts[0] if skill not in ("SQL", "AWS", "GCP", "SEO", "SEM", "CRM", "NLP", "CFD", "BIM", "IAM", "SIEM", "HVAC", "GAAP", "IFRS", "HRIS", "ATS", "PLC", "CNC Machining", "OWASP", "XGBoost") else skill


def _bullet_text(rng, skills_levels, category):
    lv = max(l for _, l in skills_levels)
    names = [_alias(rng, s) for s, _ in skills_levels]
    s = " and ".join(names) if len(names) > 1 else names[0]
    tmpl = rng.choice(GEN_VERBS[lv])
    task = rng.choice(GENERIC_TASKS) if rng.random() < 0.55 else rng.choice(CAT[category]["tasks"])
    text = tmpl.format(s=s, t=task, n=rng.randint(3, 12))
    p_metric = {1: 0.0, 2: 0.2, 3: 0.7, 4: 0.9}[lv]
    if rng.random() < p_metric:
        pct = rng.randint(8, 60)
        suf = rng.choice([f", improving {rng.choice(METRICS)} by {pct}%", f", reducing {rng.choice(METRICS)} by {pct}%",
                          f", serving {rng.randint(2, 90)}K users", f", saving ${rng.randint(10, 500)}K annually",
                          f", boosting {rng.choice(METRICS)} by {pct}%"])
        text += suf
    return text + "."


def _noise_line(line, rate, seed_tag):
    if rate <= 0:
        return line
    rng = random.Random(f"{seed_tag}|{line}")
    words = line.split(" ")
    for i, w in enumerate(words):
        if len(w) > 4 and w.isalpha() and rng.random() < rate:
            k = rng.randint(1, len(w) - 3)
            words[i] = w[:k] + w[k + 1] + w[k] + w[k + 2:]
    return " ".join(words)


def render_text(rec: dict, noise_rate: Optional[float] = None) -> str:
    """Deterministic resume rendering. Identity tokens are rendered with a separate RNG so that
    swapping identity leaves every body line byte-identical."""
    rb = random.Random(rec["rseed"])
    ident = rec["identity"]
    cat = rec["category"]
    rate = (0.03 if rec["noisy"] else 0.0) if noise_rate is None else noise_rate
    tag = str(rec["rseed"])
    bul = rec["bullet"]
    pick = rec["hdr_pick"]
    from .lexicons import SECTION_HEADERS as SH

    def H(name):
        opts = SH[name]
        return opts[pick % len(opts)]

    def hdr(name):
        h = H(name)
        return h.upper() if rec["style"] != 1 else h.title()

    # header (identity) -------------------------------------------------------------------
    nm = f"{ident['first']} {ident['last']}"
    head = [nm.upper() if rec["style"] == 0 else nm]
    contact = [ident["city"], ident["email"], ident["phone"]]
    if rec["rseed"] % 2:
        contact.append(ident["linkedin"])
    head.append(" | ".join(contact))

    sections: Dict[str, List[str]] = {}
    # summary
    title_now = rec["jobs"][-1]["title"]
    soft = ", ".join(rec["soft"])
    terms = rb.sample(CAT[cat]["terms"], 3)
    terms = [rb.choice(GENERIC_TERMS) if rb.random() < 0.50 else x for x in terms]
    if rec["claimed_years"] is not None:
        summ = (f"{title_now} with {rec['claimed_years']}+ years of experience in {terms[0]}, {terms[1]} and {terms[2]}. "
                f"Strong {soft}.")
    else:
        summ = f"{title_now} focused on {terms[0]} and {terms[1]}. Known for {soft}."
    sections["summary"] = [summ]
    # skills
    disp = [_alias(rb, s) for s in rec["listed_skills"]]
    if rec["style"] == 1:
        sections["skills"] = [", ".join(disp)]
    else:
        sections["skills"] = [", ".join(disp)] if rb.random() < .5 else [f"{bul} " + ", ".join(disp[i:i + 6]) for i in range(0, len(disp), 6)]
    # experience
    exp_lines = []
    for j in reversed(rec["jobs"]):                                   # reverse chronological like real resumes
        rng_job = random.Random(f"{rec['rseed']}|{j['company']}|{j['start']}")
        rr = _range(j, rec["date_fmt"])
        if rec["job_fmt"] == 0:
            exp_lines.append(f"{j['title']} | {j['company']} | {rr}")
        elif rec["job_fmt"] == 1:
            exp_lines.append(f"{j['company']} - {j['title']} ({rr})")
        else:
            exp_lines.append(f"{j['title']}, {j['company']}, {rr}")
        grouped = {}
        for s, l in j["skills"]:
            grouped.setdefault(l, []).append(s)
        items = []
        for l, ss in grouped.items():
            ss = list(dict.fromkeys(ss))
            i = 0
            while i < len(ss):
                take = 2 if (len(ss) - i >= 2 and rng_job.random() < .35) else 1
                items.append([(s, l) for s in ss[i:i + take]])
                i += take
        rng_job.shuffle(items)
        for it in items:
            exp_lines.append(f"{bul} {_bullet_text(rng_job, it, j['category'])}")
    sections["experience"] = exp_lines
    # projects
    if rec["projects"]:
        pl = []
        for p in rec["projects"]:
            if not p["skills"]:
                continue
            rng_p = random.Random(f"{rec['rseed']}|{p['name']}")
            pl.append(p["name"])
            for s, l in p["skills"]:
                pl.append(f"{bul} {_bullet_text(rng_p, [(s, l)], cat)}")
        if pl:
            sections["projects"] = pl
    # education (identity: university)
    e = rec["education"]
    sections["education"] = [f"{e['degree']} in {e['field']}, {ident['university']}, {e['year']}"] if rec["style"] != 2 else [
        f"{ident['university']}", f"{e['degree']} - {e['field']} ({e['year']})"]
    if rec["certs"]:
        sections["certifications"] = [f"{bul} {c}" for c in rec["certs"]]
    if ident["activity"]:
        sections["activities"] = [f"{bul} {ident['activity']}"]

    orders = [["summary", "skills", "experience", "projects", "education", "certifications", "activities"],
              ["summary", "experience", "projects", "skills", "education", "certifications", "activities"],
              ["summary", "education", "skills", "experience", "projects", "certifications", "activities"]]
    out = list(head) + [""]
    for name in orders[rec["order"]]:
        if name not in sections:
            continue
        lines = sections[name]
        if rec["style"] == 1 and name in ("skills", "summary", "education", "certifications") and len(lines) == 1 and not lines[0].startswith(bul):
            out.append(f"{H(name).title()}: {lines[0]}")
            out.append("")
            continue
        out.append(hdr(name))
        for ln in lines:
            if name in ("experience", "projects", "summary"):
                ln = _noise_line(ln, rate, tag)
            out.append(ln)
        out.append("")
    return "\n".join(out).strip() + "\n"


def swap_identity(rec: dict, group: Optional[str] = None, tier: Optional[int] = None, activity_mode: str = "swap") -> dict:
    """Return a counterfactual twin that differs from ``rec`` only in identity attributes."""
    new = copy.deepcopy(rec)
    rng = random.Random(rec["rseed"] + 777)
    old = rec["identity"]
    g = group or ("B" if old["group"] == "A" else "A")
    t = tier or (3 if old["tier"] in (1, 2) else 1)
    ident = make_identity(rng, group=g, tier=t)
    if activity_mode == "swap" and old["activity"]:
        ident["activity"] = rng.choice(ACT_B if g == "B" else ACT_A) if old["activity"] in (ACT_A + ACT_B) else old["activity"]
    elif activity_mode == "keep":
        ident["activity"] = old["activity"]
    new["identity"] = ident
    return new


# ---------------------------------------------------------------------------------------------
# corpus + jobs
# ---------------------------------------------------------------------------------------------
def generate_corpus(n_per_class: int = C.N_PER_CLASS, seed: int = C.SEED) -> List[dict]:
    rng = random.Random(seed)
    recs = []
    rid = 0
    for cat in CATEGORIES:
        for _ in range(n_per_class):
            hybrid = rng.random() < C.P_HYBRID
            inflated = rng.random() < C.P_INFLATED
            noisy = rng.random() < C.P_NOISY
            rec = make_record(rng, rid, cat, hybrid, inflated, noisy)
            rec["text"] = render_text(rec)
            recs.append(rec)
            rid += 1
    rng.shuffle(recs)
    return recs


def make_jobs(seed: int = C.SEED, n_per_class: int = C.N_JOBS_PER_CLASS) -> List[dict]:
    rng = random.Random(seed + 99)
    jobs = []
    jid = 0
    for cat in CATEGORIES:
        core, sec = skill_pool(cat)
        for _ in range(n_per_class):
            req = rng.sample(core, 5)
            pref = rng.sample([s for s in core + sec if s not in req], 3)
            min_years = rng.choice([0, 1, 2, 3, 5, 7])
            title = CAT[cat]["titles"][min(3, min_years // 2)]
            text = (f"We are hiring a {title}. Responsibilities include {rng.choice(CAT[cat]['tasks'])} and "
                    f"{rng.choice(CAT[cat]['tasks'])}. Required skills: {', '.join(req)}. "
                    f"Preferred skills: {', '.join(pref)}. Minimum {min_years} years of experience in {rng.choice(CAT[cat]['terms'])}.")
            jobs.append(dict(id=jid, category=cat, title=title, required=req, preferred=pref, min_years=min_years, text=text))
            jid += 1
    return jobs


def true_relevance(rec: dict, job: dict) -> float:
    """Continuous ground-truth fit (0-1) from LATENT competence, never from what is merely listed."""
    ts = rec["true_skills"]

    def credit(sk):
        if sk in ts:
            return ts[sk] / 4.0
        best = 0.0
        for fam in FAM_FULL:
            if sk in fam:
                for m in fam:
                    if m in ts:
                        best = max(best, 0.5 * ts[m] / 4.0)
        for ch, par in IMP_FULL:
            if par == sk and ch in ts:
                best = max(best, 0.7 * ts[ch] / 4.0)
        return best
    w_tot = len(job["required"]) * 1.0 + len(job["preferred"]) * 0.5
    sc = sum(credit(s) for s in job["required"]) + 0.5 * sum(credit(s) for s in job["preferred"])
    sc /= w_tot
    ef = 1.0 if job["min_years"] == 0 else min(1.0, rec["true_years"] / job["min_years"])
    return sc * (0.8 + 0.2 * ef)


def relevance_grade(x: float) -> int:
    """Fixed thresholds, set from the ground-truth distribution before any method was evaluated."""
    return 0 if x < 0.15 else 1 if x < 0.30 else 2 if x < 0.50 else 3


def explicit_skills(rec: dict) -> set:
    """Ground-truth set of skills that are explicitly present in the text (listed, narrated, or named in a certificate)."""
    out = set(rec["listed_skills"]) | set(rec["narrative_skills"])
    from .lexicons import CAT
    allsk = {s for c in CAT.values() for s in c["core"] + c["secondary"]}
    for c in rec["certs"]:
        for s in allsk:
            if s.lower() in c.lower():
                out.add(s)
    for s in allsk:                                   # skills named by the degree field (e.g. "M.Sc in Statistics")
        if s.lower() in rec["education"]["field"].lower():
            out.add(s)
    return out
