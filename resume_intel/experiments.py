"""All experiments. Every function returns plain dicts / DataFrames that run_pipeline.py saves and plots."""
from __future__ import annotations
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, binomtest
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import normalize

from . import config as C
from . import generator as G
from . import fairness as F
from .evidence import (SES_WEIGHTS, PCI_WEIGHTS, SECTION_SCORE, VERB_SCORE, skill_evidence, credibility, ses_components,
                       ses_from_components, pci_components, pci_from_components, tier_of, profile_strength)
from .extractor import parse_resume, ALL_SKILLS, default_matcher
from .lexicons import CATEGORIES, CAT
from .models import (KeywordRule, FusedMLP, baseline_models, make_mlp, make_tfidf, structured_features, timed)
from .parsers import parse_file, write_txt, write_docx, write_pdf
from .pipeline import EviProfilePipeline
from .preprocess import model_text, EMAIL_RE, PHONE_RE
from .ranking import SkillGraph, score_matrix, ranking_metrics, dcg

TIER_ORDER = ["Claimed only", "Familiar", "Working", "Advanced", "Expert"]


# =============================================================================================
@dataclass
class Context:
    recs: List[dict]
    parsed: List[dict]
    ev: List[dict]
    cr: List[dict]
    S: np.ndarray
    y: np.ndarray
    raw: List[str]
    blind: List[str]
    tr: np.ndarray
    te: np.ndarray
    jobs: List[dict]
    grades: np.ndarray            # (n_test, n_jobs) ground-truth relevance grades for test resumes
    infl: np.ndarray              # inflated flag of test resumes
    pipe: EviProfilePipeline = None
    cache: dict = field(default_factory=dict)


def build_context(n_per_class: int = C.N_PER_CLASS) -> Context:
    recs = G.generate_corpus(n_per_class)
    parsed = [parse_resume(r["text"]) for r in recs]
    ev = [skill_evidence(p) for p in parsed]
    cr = [credibility(p) for p in parsed]
    S = np.array([structured_features(p, e, c) for p, e, c in zip(parsed, ev, cr)])
    y = np.array([r["category"] for r in recs])
    idx = np.arange(len(recs))
    tr, te = train_test_split(idx, test_size=C.TEST_SIZE, stratify=y, random_state=C.SEED)
    raw = [model_text(r["text"]) for r in recs]
    blind = [F.blind_text(p) for p in parsed]
    jobs = G.make_jobs()
    grades = np.array([[G.relevance_grade(G.true_relevance(recs[i], j)) for j in jobs] for i in te])
    infl = np.array([recs[i]["inflated"] for i in te])
    ctx = Context(recs, parsed, ev, cr, S, y, raw, blind, tr, te, jobs, grades, infl)
    ctx.pipe = EviProfilePipeline().fit([recs[i]["text"] for i in tr], [recs[i]["category"] for i in tr])
    return ctx


def _boot_ci(vals, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, float)
    m = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _deg_level(d: str) -> int:
    d = d.lower().replace(".", "").replace(" ", "")
    return 1 if d.startswith("dip") else 4 if d.startswith("ph") else 3 if d.startswith("m") else 2


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# =============================================================================================
# E1 dataset overview
# =============================================================================================
def exp_dataset(ctx: Context) -> dict:
    df = pd.DataFrame(dict(category=ctx.y, words=[len(r["text"].split()) for r in ctx.recs],
                           listed=[p["n_listed"] for p in ctx.parsed], narrated=[len(p["narrative_skills"]) for p in ctx.parsed],
                           years=[r["true_years"] for r in ctx.recs], hybrid=[r["hybrid"] for r in ctx.recs],
                           inflated=[r["inflated"] for r in ctx.recs], noisy=[r["noisy"] for r in ctx.recs],
                           thin=[r["thin"] for r in ctx.recs], seniority=[r["seniority"] for r in ctx.recs]))
    summ = dict(n_resumes=len(df), n_categories=df.category.nunique(), n_train=len(ctx.tr), n_test=len(ctx.te),
                mean_words=float(df.words.mean()), mean_listed=float(df.listed.mean()), mean_narrated=float(df.narrated.mean()),
                mean_years=float(df.years.mean()), pct_hybrid=float(100 * df.hybrid.mean()), pct_inflated=float(100 * df.inflated.mean()),
                pct_noisy=float(100 * df.noisy.mean()), pct_thin=float(100 * df.thin.mean()), n_jobs=len(ctx.jobs))
    return dict(df=df, summary=summ)


# =============================================================================================
# E2 extraction accuracy (+ naive baseline)
# =============================================================================================
def _naive_skills(text: str):
    low = text.lower()
    return {s for s in ALL_SKILLS if s.lower() in low}


def _digits(s):
    return re.sub(r"\D", "", s or "")


def extraction_metrics(recs, parsed, with_baseline=True) -> pd.DataFrame:
    rows = []

    def acc(fn):
        return float(np.mean([fn(p, r) for p, r in zip(parsed, recs)]))
    ours = {
        "Email": acc(lambda p, r: p["email"] == r["identity"]["email"]),
        "Phone": acc(lambda p, r: _digits(p["phone"]) == _digits(r["identity"]["phone"])),
        "Name": acc(lambda p, r: p["name"] == f"{r['identity']['first']} {r['identity']['last']}"),
        "Institution": acc(lambda p, r: p["education"]["institution"] == r["identity"]["university"]),
        "Degree level": acc(lambda p, r: p["education"]["level"] == _deg_level(r["education"]["degree"])),
        "Recent job title": acc(lambda p, r: p["recent_title"] == max(r["jobs"], key=lambda j: (j["end"][0] * 12 + j["end"][1], j["start"][0] * 12 + j["start"][1]))["title"]),
        "Job count": acc(lambda p, r: len(p["jobs"]) == len(r["jobs"])),
    }
    tp = fp = fn = 0
    ctp = cfp = cfn = 0
    for p, r in zip(parsed, recs):
        pr, tr_ = set(p["skills"]), G.explicit_skills(r)
        tp += len(pr & tr_)
        fp += len(pr - tr_)
        fn += len(tr_ - pr)
        pc = {c.lower() for c in p["certifications"]}
        tc = {c.lower() for c in r["certs"]}
        ctp += len(pc & tc)
        cfp += len(pc - tc)
        cfn += len(tc - pc)
    sp_, sr_, sf_ = _prf(tp, fp, fn)
    cp_, cr_, cf_ = _prf(ctp, cfp, cfn)
    mae = float(np.mean([abs(p["dated_years"] - r["true_years"]) for p, r in zip(parsed, recs)]))
    # baseline: generic regex parser
    base = {}
    if with_baseline:
        texts = [r["text"] for r in recs]
        base["Email"] = float(np.mean([(EMAIL_RE.search(t).group(0) if EMAIL_RE.search(t) else None) == r["identity"]["email"] for t, r in zip(texts, recs)]))
        gp = []
        for t, r in zip(texts, recs):
            m = re.search(r"\+?\d[\d\s\-()]{8,}\d", t)
            gp.append(_digits(m.group(0)) == _digits(r["identity"]["phone"]) if m else False)
        base["Phone"] = float(np.mean(gp))
        base["Name"] = float(np.mean([t.split("\n")[0].strip().title() == f"{r['identity']['first']} {r['identity']['last']}" for t, r in zip(texts, recs)]))
        btp = bfp = bfn = 0
        for t, r in zip(texts, recs):
            pr, tr_ = _naive_skills(t), G.explicit_skills(r)
            btp += len(pr & tr_)
            bfp += len(pr - tr_)
            bfn += len(tr_ - pr)
        bp, br, bf = _prf(btp, bfp, bfn)
        claimed_err = [abs(p["claimed_years"] - r["true_years"]) if p["claimed_years"] is not None else r["true_years"] for p, r in zip(parsed, recs)]
        base_mae = float(np.mean(claimed_err))
    for k, v in ours.items():
        rows.append(dict(field=k, metric="accuracy", ours=v, baseline=base.get(k, np.nan)))
    rows.append(dict(field="Skills", metric="precision", ours=sp_, baseline=bp if with_baseline else np.nan))
    rows.append(dict(field="Skills", metric="recall", ours=sr_, baseline=br if with_baseline else np.nan))
    rows.append(dict(field="Skills", metric="F1", ours=sf_, baseline=bf if with_baseline else np.nan))
    rows.append(dict(field="Certifications", metric="F1", ours=cf_, baseline=np.nan))
    rows.append(dict(field="Experience years", metric="MAE (years, lower is better)", ours=mae, baseline=base_mae if with_baseline else np.nan))
    return pd.DataFrame(rows)


def exp_extraction(ctx: Context) -> pd.DataFrame:
    return extraction_metrics([ctx.recs[i] for i in ctx.te], [ctx.parsed[i] for i in ctx.te])


# =============================================================================================
# E2b format robustness (TXT / DOCX / PDF)
# =============================================================================================
def exp_formats(ctx: Context, n: int = 150, outdir: Path = C.DATA_DIR / "sample_files") -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(C.SEED)
    sel = rng.choice(ctx.te, size=n, replace=False)
    recs = [ctx.recs[i] for i in sel]
    res, parsed_by_fmt, timing = {}, {}, {}
    for fmt, writer in (("txt", write_txt), ("docx", write_docx), ("pdf", write_pdf)):
        paths = []
        for r in recs:
            p = outdir / f"resume_{r['id']:04d}.{fmt}"
            writer(r["text"], p)
            paths.append(p)
        t0 = time.perf_counter()
        texts = [parse_file(p) for p in paths]
        timing[fmt] = 1000 * (time.perf_counter() - t0) / n
        parsed = [parse_resume(t) for t in texts]
        parsed_by_fmt[fmt] = parsed
        m = extraction_metrics(recs, parsed, with_baseline=False).set_index(["field", "metric"])["ours"]
        pred = [ctx.pipe.clf.predict(ctx.pipe.vec.transform([F.blind_text(p)]),
                                     structured_features(p, skill_evidence(p), credibility(p)).reshape(1, -1))[0] for p in parsed]
        res[fmt.upper()] = dict(email=m[("Email", "accuracy")], phone=m[("Phone", "accuracy")], name=m[("Name", "accuracy")],
                                skills_f1=m[("Skills", "F1")], title=m[("Recent job title", "accuracy")],
                                years_mae=m[("Experience years", "MAE (years, lower is better)")],
                                category_acc=float(np.mean([a == r["category"] for a, r in zip(pred, recs)])))
    ctx.cache["file_parse_ms"] = timing
    return dict(table=pd.DataFrame(res).T, parse_ms=timing)


# =============================================================================================
# E4 Skill Evidence Score validation
# =============================================================================================
def _ses_rows(ctx: Context, idxs):
    rows = []
    for i in idxs:
        r, p = ctx.recs[i], ctx.parsed[i]
        for s, lv in r["true_skills"].items():
            if s not in p["skills"]:
                continue
            c = ses_components(p, s)
            rows.append(dict(rid=i, skill=s, latent=lv, comps=c, mentions=len(p["skills"][s]["mentions"])))
    return rows


def exp_ses(ctx: Context) -> dict:
    tr_rows, te_rows = _ses_rows(ctx, ctx.tr), _ses_rows(ctx, ctx.te)
    lat = np.array([r["latent"] for r in te_rows])

    def sc(rows, w):
        return np.array([ses_from_components(r["comps"], w) for r in rows])

    def only(*keys):
        return {k: (1.0 if k in keys else 0.0) for k in SES_WEIGHTS}
    out = {}
    narr = np.array([r["comps"]["narrative"] for r in te_rows])
    out["Mention count"] = np.array([r["mentions"] for r in te_rows], float)
    out["Section only"] = sc(te_rows, only("section"))
    out["Verb only"] = sc(te_rows, only("verb"))
    out["Verb + metric"] = sc(te_rows, {**only("verb", "metric")})
    out["SES (a priori weights)"] = sc(te_rows, SES_WEIGHTS)
    # learned weights (ridge on training resumes, narrative skills only)
    keys = list(SES_WEIGHTS)
    Xtr = np.array([[r["comps"][k] for k in keys] for r in tr_rows if r["comps"]["narrative"]])
    ytr = np.array([r["latent"] for r in tr_rows if r["comps"]["narrative"]])
    rid = Ridge(alpha=1.0, positive=True).fit(Xtr, ytr)
    learned = {k: float(max(w, 1e-6)) for k, w in zip(keys, rid.coef_)}
    pred = []
    for r in te_rows:
        pred.append(float(rid.predict(np.array([[r["comps"][k] for k in keys]]))[0]) / 4.0 if r["comps"]["narrative"] else r["comps"]["fallback"])
    out["SES (learned weights)"] = np.array(pred)
    rho = {k: float(spearmanr(v, lat)[0]) for k, v in out.items()}
    ablation = {}
    for k in SES_WEIGHTS:
        w = dict(SES_WEIGHTS)
        w[k] = 0.0
        ablation[f"without {k}"] = float(spearmanr(sc(te_rows, w), lat)[0])
    ablation["full SES"] = rho["SES (a priori weights)"]
    # tier confusion
    ses_full = out["SES (a priori weights)"]
    tiers = [tier_of(x) for x in ses_full]
    cm = pd.crosstab(pd.Series(lat, name="latent level"), pd.Series(tiers, name="tier")).reindex(index=[1, 2, 3, 4], columns=TIER_ORDER, fill_value=0)
    expected = {1: "Familiar", 2: "Working", 3: "Advanced", 4: "Expert"}
    exact = float(np.mean([expected[l] == t for l, t in zip(lat, tiers)]))
    within1 = float(np.mean([abs(TIER_ORDER.index(expected[l]) - TIER_ORDER.index(t)) <= 1 for l, t in zip(lat, tiers)]))
    # unsupported-claim detection at skill level (listed skills in test resumes)
    tp = fp = fn = 0
    for i in ctx.te:
        r, p, ev = ctx.recs[i], ctx.parsed[i], ctx.ev[i]
        cert_sk = {s for s in ALL_SKILLS for c in r["certs"] if s.lower() in c.lower()}
        truth = {s for s in r["unsupported_listed"] if s not in cert_sk}
        pred_ = {s for s in p["listed_skills"] if ev[s]["tier"] == "Claimed only"}
        tp += len(truth & pred_)
        fp += len(pred_ - truth)
        fn += len(truth - pred_)
    cp, cr_, cf = _prf(tp, fp, fn)
    return dict(spearman=rho, ablation=ablation, confusion=cm, tier_exact=exact, tier_within1=within1, n_pairs=len(te_rows),
                claimed_only=dict(precision=cp, recall=cr_, f1=cf, n_truth=tp + fn), learned_weights=learned,
                scatter=dict(latent=lat.tolist(), ses=ses_full.tolist()))


# =============================================================================================
# E5 Profile Credibility Index
# =============================================================================================
PCI_KEYS = list(PCI_WEIGHTS)


def exp_pci(ctx: Context) -> dict:
    comps = [pci_components(p) for p in ctx.parsed]
    X = np.array([[c[k] for k in PCI_KEYS] for c in comps])
    yinf = np.array([r["inflated"] for r in ctx.recs]).astype(int)
    pci = np.array([pci_from_components(c) for c in comps])
    nl = np.array([p["n_listed"] for p in ctx.parsed], float)
    tr, te = ctx.tr, ctx.te

    def thr_f1(score, idx_tr, idx_te):
        best, bt = -1, 0
        for t in np.unique(np.round(score[idx_tr], 3)):
            f = f1_score(yinf[idx_tr], score[idx_tr] >= t)
            if f > best:
                best, bt = f, t
        pr = score[idx_te] >= bt
        p, r, f, _ = precision_recall_fscore_support(yinf[idx_te], pr, average="binary", zero_division=0)
        return float(p), float(r), float(f)
    lr = LogisticRegression(max_iter=500, C=5.0).fit(X[tr], yinf[tr])
    lrp = lr.predict_proba(X)[:, 1]
    risk = 1 - pci
    scorers = {"Skill count": nl, "Unsupported-skill ratio only": X[:, 0], "Experience-gap only": X[:, 2],
               "PCI (a priori weights)": risk, "PCI (learned weights)": lrp}
    rows = []
    for k, s in scorers.items():
        p, r, f = thr_f1(s, tr, te)
        rows.append(dict(method=k, roc_auc=roc_auc_score(yinf[te], s[te]), pr_auc=average_precision_score(yinf[te], s[te]),
                         precision=p, recall=r, f1=f))
    ablation = {}
    for k in PCI_KEYS:
        w = dict(PCI_WEIGHTS)
        w[k] = 0.0
        rk = np.array([1 - pci_from_components(c, w) for c in comps])
        ablation[f"without {k}"] = float(roc_auc_score(yinf[te], rk[te]))
    ablation["full PCI"] = float(roc_auc_score(yinf[te], risk[te]))
    curves = {}
    from sklearn.metrics import roc_curve, precision_recall_curve
    for k, s in scorers.items():
        fpr, tpr, _ = roc_curve(yinf[te], s[te])
        pr_, rc_, _ = precision_recall_curve(yinf[te], s[te])
        curves[k] = dict(fpr=fpr, tpr=tpr, prec=pr_, rec=rc_)
    return dict(table=pd.DataFrame(rows), ablation=ablation, curves=curves, pci=pci[te], inflated=yinf[te],
                coef=dict(zip(PCI_KEYS, lr.coef_[0].round(3).tolist())), prevalence=float(yinf[te].mean()))


# =============================================================================================
# E6 job-fit ranking
# =============================================================================================
METHOD_LABEL = {
    "keyword": "Keyword match (ATS-style)", "tfidf_cosine": "TF-IDF cosine", "lsa_cosine": "LSA semantic cosine",
    "ontology": "Keyword + skill graph", "ses_only": "SES only", "ses_ontology": "SES + skill graph",
    "ses_ontology_exp": "EviProfile fit (SES + graph + experience)", "full": "EviProfile fit + PCI adjustment",
    "ses_ontology_exp_npmi": "EviProfile fit + learned NPMI edges"}


def per_job_ndcg(S, grades, k=10):
    out = []
    for j in range(S.shape[1]):
        order = np.argsort(-S[:, j], kind="stable")
        ideal = np.sort(grades[:, j])[::-1][:k]
        if dcg(ideal) > 0:
            out.append(dcg(grades[order, j][:k]) / dcg(ideal))
    return np.array(out)


def rank_scores(ctx: Context, method: str, idxs, parsed=None, evs=None, crs=None, texts=None, graph=None):
    parsed = parsed or [ctx.parsed[i] for i in idxs]
    evs = evs or [ctx.ev[i] for i in idxs]
    crs = crs or [ctx.cr[i] for i in idxs]
    texts = texts or [ctx.raw[i] for i in idxs]
    graph = graph or ctx.pipe.graph
    vec, svd = ctx.cache["rank_vec"], ctx.cache["rank_svd"]
    return score_matrix(method, parsed, ctx.jobs, graph, evs, crs, tfidf=vec, lsa=svd, texts=texts)


def exp_ranking(ctx: Context) -> dict:
    vec = make_tfidf(10000)
    Xtr = vec.fit_transform([ctx.raw[i] for i in ctx.tr] + [j["text"] for j in ctx.jobs])
    svd = TruncatedSVD(100, random_state=C.SEED).fit(Xtr)
    ctx.cache["rank_vec"], ctx.cache["rank_svd"] = vec, svd
    npmi_graph = SkillGraph(use_npmi=True).fit_npmi([ctx.parsed[i] for i in ctx.tr])
    rows, perjob, mats = [], {}, {}
    for m in ["keyword", "tfidf_cosine", "lsa_cosine", "ontology", "ses_only", "ses_ontology", "ses_ontology_exp", "full", "ses_ontology_exp_npmi"]:
        if m == "ses_ontology_exp_npmi":
            S = rank_scores(ctx, "ses_ontology_exp", ctx.te, graph=npmi_graph)
        else:
            S = rank_scores(ctx, m, ctx.te)
        mats[m] = S
        met = ranking_metrics(S, ctx.grades, ctx.infl)
        pj = per_job_ndcg(S, ctx.grades)
        perjob[m] = pj
        lo, hi = _boot_ci(pj)
        rows.append(dict(method=METHOD_LABEL[m], key=m, ndcg10=met["ndcg10"], ndcg_lo=lo, ndcg_hi=hi, map10=met["map10"], p10=met["p10"],
                         recall10=met["recall10"], false_promotion=met["false_promotion"]))
    tab = pd.DataFrame(rows)
    # paired bootstrap for the headline comparison: EviProfile fit vs keyword
    d = perjob["ses_ontology_exp"] - perjob["keyword"]
    d2 = perjob["ses_ontology_exp"] - perjob["tfidf_cosine"]
    ctx.cache["rank_mats"] = mats
    return dict(table=tab, diff_vs_keyword=dict(mean=float(d.mean()), ci=_boot_ci(d)), diff_vs_tfidf=dict(mean=float(d2.mean()), ci=_boot_ci(d2)),
                n_jobs_eval=len(perjob["keyword"]), n_candidates=len(ctx.te))


# =============================================================================================
# E3 classification
# =============================================================================================
def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return _softmax(model.decision_function(X))


def exp_classification(ctx: Context, cv_folds: int = 5) -> dict:
    y, tr, te = ctx.y, ctx.tr, ctx.te
    classes = np.array(CATEGORIES)
    names = ["Keyword rule", "Multinomial NB", "Logistic Regression", "Linear SVM", "Random Forest", "MLP (TF-IDF)", "EviProfile (fused MLP)"]
    # ---------------- cross-validation on training split
    skf = StratifiedKFold(cv_folds, shuffle=True, random_state=C.SEED)
    cv = {n: [] for n in names}
    for f, (a, b) in enumerate(skf.split(tr, y[tr])):
        ia, ib = tr[a], tr[b]
        vr = make_tfidf()
        Xa = vr.fit_transform([ctx.raw[i] for i in ia])
        Xb = vr.transform([ctx.raw[i] for i in ib])
        vb = make_tfidf()
        Xab = vb.fit_transform([ctx.blind[i] for i in ia])
        Xbb = vb.transform([ctx.blind[i] for i in ib])
        preds = {"Keyword rule": KeywordRule().fit(None).predict([ctx.recs[i]["text"] for i in ib])}
        for n, m in baseline_models(seed=f).items():
            m.fit(Xa, y[ia])
            preds[n] = m.predict(Xb)
        fm = FusedMLP(seed=f).fit(Xab, ctx.S[ia], y[ia])
        preds["EviProfile (fused MLP)"] = fm.predict(Xbb, ctx.S[ib])
        for n in names:
            cv[n].append((accuracy_score(y[ib], preds[n]), f1_score(y[ib], preds[n], average="macro")))
    cv_tab = pd.DataFrame({n: dict(cv_acc_mean=np.mean([a for a, _ in v]), cv_acc_std=np.std([a for a, _ in v]),
                                   cv_f1_mean=np.mean([f for _, f in v]), cv_f1_std=np.std([f for _, f in v])) for n, v in cv.items()}).T

    # ---------------- hold-out test
    vr = make_tfidf()
    Xtr = vr.fit_transform([ctx.raw[i] for i in tr])
    Xte = vr.transform([ctx.raw[i] for i in te])
    vb = make_tfidf()
    Xtrb = vb.fit_transform([ctx.blind[i] for i in tr])
    Xteb = vb.transform([ctx.blind[i] for i in te])
    preds, probas, train_t, models = {}, {}, {}, {}
    kr = KeywordRule().fit(None)
    preds["Keyword rule"] = kr.predict([ctx.recs[i]["text"] for i in te])
    probas["Keyword rule"] = kr.predict_proba([ctx.recs[i]["text"] for i in te])
    train_t["Keyword rule"] = 0.0
    for n, m in baseline_models(seed=C.SEED).items():
        _, t = timed(m.fit, Xtr, y[tr])
        train_t[n] = t
        preds[n] = m.predict(Xte)
        probas[n] = _proba(m, Xte)
        models[n] = m
    fm = FusedMLP(seed=C.SEED)
    _, t = timed(fm.fit, Xtrb, ctx.S[tr], y[tr])
    train_t["EviProfile (fused MLP)"] = t
    preds["EviProfile (fused MLP)"] = fm.predict(Xteb, ctx.S[te])
    probas["EviProfile (fused MLP)"] = fm.predict_proba(Xteb, ctx.S[te])
    # ablation variants (not in the main table)
    mlp_blind = make_mlp(C.SEED).fit(Xtrb, y[tr])
    fused_raw = FusedMLP(seed=C.SEED).fit(Xtr, ctx.S[tr], y[tr])
    variants = {"MLP (blind text)": mlp_blind.predict(Xteb), "Fused MLP (raw text)": fused_raw.predict(Xte, ctx.S[te])}
    struct_only = make_mlp(C.SEED).fit(ctx.S[tr], y[tr])
    variants["Evidence features only"] = struct_only.predict(ctx.S[te])
    rows = []
    yt = y[te]
    for n in names:
        p = preds[n]
        pr, rc, f1, _ = precision_recall_fscore_support(yt, p, average="macro", zero_division=0)
        correct = (p == yt).astype(float)
        lo, hi = _boot_ci(correct)
        pb = probas[n]
        order = [list(classes).index(c) for c in (models[n].classes_ if n in models else (fm.classes_ if n.startswith("Evi") else classes))]
        # align probability columns to CATEGORIES order
        cls_src = (models[n].classes_ if n in models else (fm.classes_ if n.startswith("Evi") else classes))
        P = np.zeros((len(yt), len(classes)))
        for j, c in enumerate(cls_src):
            P[:, list(classes).index(c)] = pb[:, j]
        try:
            srt = np.argsort(classes)
            auc = roc_auc_score(yt, (P / P.sum(1, keepdims=True))[:, srt], multi_class="ovr", labels=list(np.array(classes)[srt]))
        except Exception:
            auc = np.nan
        rows.append(dict(model=n, accuracy=accuracy_score(yt, p), acc_lo=lo, acc_hi=hi, precision=pr, recall=rc, macro_f1=f1,
                         weighted_f1=f1_score(yt, p, average="weighted"), roc_auc=auc, train_s=train_t[n]))
        probas[n] = P
    test_tab = pd.DataFrame(rows).set_index("model")
    # subset analysis
    flags = {"All": np.ones(len(te), bool), "Career switchers": np.array([ctx.recs[i]["hybrid"] for i in te]),
             "Non-switchers": np.array([not ctx.recs[i]["hybrid"] for i in te]), "Thin resumes": np.array([ctx.recs[i]["thin"] for i in te]),
             "Typo-noisy": np.array([ctx.recs[i]["noisy"] for i in te]), "Inflated": np.array([ctx.recs[i]["inflated"] for i in te])}
    sub = {n: {k: (float(np.mean(preds[n][m] == yt[m])) if m.sum() else np.nan) for k, m in flags.items()} for n in ["Keyword rule", "Logistic Regression", "Linear SVM", "MLP (TF-IDF)", "EviProfile (fused MLP)"]}
    sub_n = {k: int(m.sum()) for k, m in flags.items()}
    # McNemar (exact) fused vs others
    mc = {}
    for other in ["Keyword rule", "Logistic Regression", "Linear SVM", "MLP (TF-IDF)"]:
        a = preds["EviProfile (fused MLP)"] == yt
        b = preds[other] == yt
        n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
        pval = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        mc[other] = dict(fused_only_correct=n10, other_only_correct=n01, p_value=float(pval))
    cm = confusion_matrix(yt, preds["EviProfile (fused MLP)"], labels=list(classes))
    pc = {n: f1_score(yt, preds[n], average=None, labels=list(classes)) for n in ["Keyword rule", "Logistic Regression", "MLP (TF-IDF)", "EviProfile (fused MLP)"]}
    var_tab = {k: dict(accuracy=float(accuracy_score(yt, v)), macro_f1=float(f1_score(yt, v, average="macro"))) for k, v in variants.items()}
    ctx.cache["clf"] = dict(vr=vr, vb=vb, models=models, fm=fm, kr=kr, fused_raw=fused_raw)
    return dict(cv=cv_tab, test=test_tab, subsets=pd.DataFrame(sub).T, subset_n=sub_n, mcnemar=mc, confusion=cm, per_class=pc,
                variants=var_tab, probas=probas, y_test=yt, classes=list(classes))


# =============================================================================================
# E7 learning curve and noise robustness
# =============================================================================================
def exp_learning_curve(ctx: Context, sizes=(100, 200, 400, 800, 1600), reps=3) -> pd.DataFrame:
    y, te = ctx.y, ctx.te
    rows = []
    for n in sizes:
        for rep in range(reps):
            if n >= len(ctx.tr):
                sub = ctx.tr
            else:
                sub, _ = train_test_split(ctx.tr, train_size=n, stratify=y[ctx.tr], random_state=100 + rep)
            vr, vb = make_tfidf(), make_tfidf()
            Xr = vr.fit_transform([ctx.raw[i] for i in sub])
            Xrt = vr.transform([ctx.raw[i] for i in te])
            Xb = vb.fit_transform([ctx.blind[i] for i in sub])
            Xbt = vb.transform([ctx.blind[i] for i in te])
            res = {}
            res["Logistic Regression"] = f1_score(y[te], LogisticRegression(max_iter=300, C=10).fit(Xr, y[sub]).predict(Xrt), average="macro")
            res["MLP (TF-IDF)"] = f1_score(y[te], make_mlp(rep).fit(Xr, y[sub]).predict(Xrt), average="macro")
            res["EviProfile (fused MLP)"] = f1_score(y[te], FusedMLP(seed=rep).fit(Xb, ctx.S[sub], y[sub]).predict(Xbt, ctx.S[te]), average="macro")
            res["Keyword rule"] = f1_score(y[te], KeywordRule().fit(None).predict([ctx.recs[i]["text"] for i in te]), average="macro")
            for k, v in res.items():
                rows.append(dict(n_train=len(sub), model=k, macro_f1=v, rep=rep))
    return pd.DataFrame(rows)


def exp_noise(ctx: Context, rates=(0.0, 0.03, 0.06, 0.10, 0.15, 0.20)) -> pd.DataFrame:
    cl = ctx.cache["clf"]
    y = ctx.y
    te = ctx.te
    rows = []
    for rate in rates:
        recs = [ctx.recs[i] for i in te]
        texts = [G.render_text(r, noise_rate=rate) for r in recs]
        parsed = [parse_resume(t) for t in texts]
        evs = [skill_evidence(p) for p in parsed]
        crs = [credibility(p) for p in parsed]
        S = np.array([structured_features(p, e, c) for p, e, c in zip(parsed, evs, crs)])
        Xr = cl["vr"].transform([model_text(t) for t in texts])
        Xb = cl["vb"].transform([F.blind_text(p) for p in parsed])
        yt = y[te]
        accs = {"Keyword rule": accuracy_score(yt, cl["kr"].predict(texts)),
                "Logistic Regression": accuracy_score(yt, cl["models"]["Logistic Regression"].predict(Xr)),
                "Linear SVM": accuracy_score(yt, cl["models"]["Linear SVM"].predict(Xr)),
                "MLP (TF-IDF)": accuracy_score(yt, cl["models"]["MLP (TF-IDF)"].predict(Xr)),
                "EviProfile (fused MLP)": accuracy_score(yt, cl["fm"].predict(Xb, S))}
        ses_tau = np.mean([len(set(p["skills"]) & G.explicit_skills(r)) / max(1, len(G.explicit_skills(r))) for p, r in zip(parsed, recs)])
        for k, v in accs.items():
            rows.append(dict(noise=rate, model=k, accuracy=v, skill_recall=ses_tau))
    return pd.DataFrame(rows)


# =============================================================================================
# E8 fairness
# =============================================================================================
def exp_fairness(ctx: Context) -> dict:
    te = ctx.te
    recs_te = [ctx.recs[i] for i in te]
    twins = F.counterfactual_twins(recs_te)
    t_parsed = [parse_resume(t["text"]) for t in twins]
    t_ev = [skill_evidence(p) for p in t_parsed]
    t_cr = [credibility(p) for p in t_parsed]
    t_raw = [model_text(t["text"]) for t in twins]
    audit = {}
    for m, label in [("tfidf_cosine", "TF-IDF cosine"), ("lsa_cosine", "LSA semantic cosine"), ("ses_ontology_exp", "EviProfile fit")]:
        S0 = ctx.cache["rank_mats"][m]
        S1 = rank_scores(ctx, m, te, parsed=t_parsed, evs=t_ev, crs=t_cr, texts=t_raw)
        audit[label] = F.rank_shift(S0, S1)
    # classification flip rate under identity swap
    cl = ctx.cache["clf"]
    Xr0 = cl["vr"].transform([ctx.raw[i] for i in te])
    Xr1 = cl["vr"].transform(t_raw)
    t_S = np.array([structured_features(p, e, c) for p, e, c in zip(t_parsed, t_ev, t_cr)])
    Xb0 = cl["vb"].transform([ctx.blind[i] for i in te])
    Xb1 = cl["vb"].transform([F.blind_text(p) for p in t_parsed])
    flips = {"Logistic Regression (raw text)": float(np.mean(cl["models"]["Logistic Regression"].predict(Xr0) != cl["models"]["Logistic Regression"].predict(Xr1))),
             "MLP (raw text)": float(np.mean(cl["models"]["MLP (TF-IDF)"].predict(Xr0) != cl["models"]["MLP (TF-IDF)"].predict(Xr1))),
             "EviProfile (blind + evidence)": float(np.mean(cl["fm"].predict(Xb0, ctx.S[te]) != cl["fm"].predict(Xb1, t_S)))}
    # ---------------- bias-stress test: strongly gender-coded training history
    tr_recs = [ctx.recs[i] for i in ctx.tr]
    st = F.stress_train(tr_recs, p_stereo=0.9)
    bt = F.balanced_test(recs_te)
    st_parsed = [parse_resume(r["text"]) for r in st]
    bt_parsed = [parse_resume(r["text"]) for r in bt]
    st_ev = [skill_evidence(p) for p in st_parsed]
    st_cr = [credibility(p) for p in st_parsed]
    bt_ev = [skill_evidence(p) for p in bt_parsed]
    bt_cr = [credibility(p) for p in bt_parsed]
    S_st = np.array([structured_features(p, e, c) for p, e, c in zip(st_parsed, st_ev, st_cr)])
    S_bt = np.array([structured_features(p, e, c) for p, e, c in zip(bt_parsed, bt_ev, bt_cr)])
    ytr, yte = ctx.y[ctx.tr], ctx.y[te]
    vr, vb = make_tfidf(), make_tfidf()
    Xr_tr = vr.fit_transform([model_text(r["text"]) for r in st])
    Xr_te = vr.transform([model_text(r["text"]) for r in bt])
    Xb_tr = vb.fit_transform([F.blind_text(p) for p in st_parsed])
    Xb_te = vb.transform([F.blind_text(p) for p in bt_parsed])
    pred = {"Logistic Regression (raw text)": LogisticRegression(max_iter=300, C=10).fit(Xr_tr, ytr).predict(Xr_te),
            "MLP (raw text)": make_mlp(C.SEED).fit(Xr_tr, ytr).predict(Xr_te),
            "MLP (blind text)": make_mlp(C.SEED).fit(Xb_tr, ytr).predict(Xb_te),
            "EviProfile (blind + evidence)": FusedMLP(seed=C.SEED).fit(Xb_tr, S_st, ytr).predict(Xb_te, S_bt)}
    grp = np.array([r["identity"]["group"] for r in bt])
    rg = np.array([F.role_group(c) for c in yte])
    hard = np.array([(r["hybrid"] or r["thin"]) for r in recs_te])
    rows = []
    for n, p in pred.items():
        ok = (p == yte)
        counter = ok[((rg == "A") & (grp == "B")) | ((rg == "B") & (grp == "A"))]
        stereo = ok[((rg == "A") & (grp == "A")) | ((rg == "B") & (grp == "B"))]
        c_h = ok[(((rg == "A") & (grp == "B")) | ((rg == "B") & (grp == "A"))) & hard]
        s_h = ok[(((rg == "A") & (grp == "A")) | ((rg == "B") & (grp == "B"))) & hard]
        # equal opportunity gap: |TPR(group A) - TPR(group B)| averaged over coded categories
        gaps = []
        for c in CATEGORIES:
            if F.role_group(c) == "N":
                continue
            ma = (yte == c) & (grp == "A")
            mb = (yte == c) & (grp == "B")
            if ma.sum() and mb.sum():
                gaps.append(abs(ok[ma].mean() - ok[mb].mean()))
        rows.append(dict(model=n, accuracy=float(ok.mean()), acc_stereotypical=float(stereo.mean()), acc_counter=float(counter.mean()),
                         gap=float(stereo.mean() - counter.mean()), hard_gap=float(s_h.mean() - c_h.mean()) if len(s_h) and len(c_h) else np.nan,
                         eo_gap=float(np.mean(gaps)), n_counter=int(len(counter)), n_stereo=int(len(stereo))))
    return dict(audit=pd.DataFrame(audit).T, flips=flips, stress=pd.DataFrame(rows).set_index("model"))


# =============================================================================================
# E9 efficiency
# =============================================================================================
def exp_efficiency(ctx: Context, n: int = 200) -> dict:
    idx = ctx.te[:n]
    texts = [ctx.recs[i]["text"] for i in idx]
    t = time.perf_counter()
    parsed = [parse_resume(x) for x in texts]
    t_ext = (time.perf_counter() - t) / n
    t = time.perf_counter()
    evs = [skill_evidence(p) for p in parsed]
    crs = [credibility(p) for p in parsed]
    t_evd = (time.perf_counter() - t) / n
    t = time.perf_counter()
    S = np.array([structured_features(p, e, c) for p, e, c in zip(parsed, evs, crs)])
    X = ctx.pipe.vec.transform([F.blind_text(p) for p in parsed])
    ctx.pipe.clf.predict_proba(X, S)
    t_clf = (time.perf_counter() - t) / n
    t = time.perf_counter()
    score_matrix("ses_ontology_exp", parsed, ctx.jobs, ctx.pipe.graph, evs, crs)
    t_rank = (time.perf_counter() - t) / n
    fp = ctx.cache.get("file_parse_ms", {})
    stages = {"File parsing (TXT)": fp.get("txt", np.nan), "File parsing (DOCX)": fp.get("docx", np.nan), "File parsing (PDF)": fp.get("pdf", np.nan),
              "NLP extraction": 1000 * t_ext, "Evidence scoring (SES + PCI)": 1000 * t_evd, "Classification": 1000 * t_clf,
              f"Job-fit vs {len(ctx.jobs)} jobs": 1000 * t_rank}
    total_txt = stages["NLP extraction"] + stages["Evidence scoring (SES + PCI)"] + stages["Classification"] + stages[f"Job-fit vs {len(ctx.jobs)} jobs"]
    return dict(stages_ms=stages, total_ms_txt=total_txt, throughput_per_min=60000.0 / total_txt)
