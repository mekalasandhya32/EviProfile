"""All figures for the EviProfile study (matplotlib / seaborn, 200 dpi PNG)."""
from __future__ import annotations
import textwrap
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc as sk_auc

from . import config as C

OURS = "#0B6E4F"
BASE = "#8A99A8"
ACCENT = "#E07A1F"
BLUE = "#2F6DB5"
RED = "#C0392B"
TIER_COL = {"Claimed only": "#C0392B", "Familiar": "#E6A23C", "Working": "#3498DB", "Advanced": "#1E8449", "Expert": "#145A32"}

sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved", name)


def _bar_labels(ax, fmt="{:.3f}", fs=8, dy=0.0, horizontal=False):
    for p in ax.patches:
        v = p.get_width() if horizontal else p.get_height()
        if np.isnan(v) or v == 0:
            continue
        if horizontal:
            ax.text(v + dy, p.get_y() + p.get_height() / 2, fmt.format(v), va="center", fontsize=fs)
        else:
            ax.text(p.get_x() + p.get_width() / 2, v + dy, fmt.format(v), ha="center", va="bottom", fontsize=fs)


def fig_architecture(R):
    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#33475B", fs=9):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#14213D")

    def arrow(x1, y1, x2, y2, c="#33475B"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4))
    r1 = 5.2
    box(0.2, r1, 1.9, 0.9, "1. Resume upload\nTXT / DOCX / PDF", "#EAF1F8")
    box(2.5, r1, 1.9, 0.9, "2. Text extraction\n+ layout repair", "#EAF1F8")
    box(4.8, r1, 1.9, 0.9, "3. Cleaning +\nsection segmentation", "#EAF1F8")
    box(7.1, r1, 2.4, 0.9, "4. NLP extraction\nentities, dates, skills,\nverb levels + provenance", "#EAF1F8", fs=8.5)
    for a, b in ((2.1, 2.5), (4.4, 4.8), (6.7, 7.1)):
        arrow(a, r1 + 0.45, b, r1 + 0.45)
    ax.add_patch(mpatches.FancyBboxPatch((0.2, 2.35), 9.3, 2.35, boxstyle="round,pad=0.03,rounding_size=0.15", fc="#E8F5EE", ec=OURS, lw=1.6, ls="--"))
    ax.text(0.4, 4.45, "EVIDENCE LAYER (core contribution)", fontsize=9.5, fontweight="bold", color=OURS)
    box(0.5, 2.6, 2.7, 1.55, "Skill Evidence Score (SES)\nverb ownership + section +\nquantified outcome + recency\n+ repetition  ->  tier", "#FFFFFF", ec=OURS, fs=8.5)
    box(3.5, 2.6, 2.7, 1.55, "Profile Credibility Index (PCI)\nunsupported claims, skill stuffing,\nexperience gap, job overlap,\nseniority mismatch", "#FFFFFF", ec=OURS, fs=8.5)
    box(6.5, 2.6, 2.8, 1.55, "Skill graph\nfamilies + implications\n(+ learned NPMI edges)\nfor semantic matching", "#FFFFFF", ec=OURS, fs=8.5)
    arrow(8.3, r1, 1.9, 4.15, OURS)
    arrow(8.3, r1, 4.85, 4.15, OURS)
    arrow(8.3, r1, 7.9, 4.15, OURS)
    box(10.0, 3.2, 2.8, 1.5, "Blind view (fairness)\nremoves name, contact, institution,\ndates, activities, gendered terms", "#FFF4E5", ec=ACCENT, fs=8.5)
    arrow(9.5, r1 + 0.45, 11.4, 4.7, ACCENT)
    r3 = 0.35
    box(0.5, r3, 2.7, 1.2, "5. Classification\nTF-IDF(blind) + evidence features\n-> deep MLP", "#EAF1F8", fs=8.5)
    box(3.6, r3, 2.7, 1.2, "6. Candidate profile\nentities + tiered skills\n+ strength score", "#EAF1F8", fs=8.5)
    box(6.7, r3, 2.7, 1.2, "7. Job-fit ranking\nSES x graph x experience\n(+ optional PCI adjustment)", "#EAF1F8", fs=8.5)
    box(9.9, r3, 2.9, 1.2, "8. Explainable output\nHTML card, JSON, evidence\nsnippets, fairness audit", "#EAF1F8", fs=8.5)
    arrow(11.4, 3.2, 11.4, 1.55, ACCENT)
    arrow(1.9, 2.6, 1.9, 1.55, OURS)
    arrow(4.85, 2.6, 4.95, 1.55, OURS)
    arrow(7.9, 2.6, 8.0, 1.55, OURS)
    arrow(3.2, r3 + 0.6, 3.6, r3 + 0.6)
    arrow(6.3, r3 + 0.6, 6.7, r3 + 0.6)
    arrow(9.4, r3 + 0.6, 9.9, r3 + 0.6)
    ax.set_title("EviProfile: system architecture", fontsize=14, pad=10)
    _save(fig, "fig01_architecture.png")


def fig_dataset(R):
    df = R["dataset"]["df"]
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.5))
    order = df.category.value_counts().index
    sns.countplot(y="category", data=df, order=order, color=BLUE, ax=ax[0, 0])
    ax[0, 0].set_title("(a) Resumes per category")
    ax[0, 0].set_ylabel("")
    sns.histplot(df.words, bins=30, color=OURS, ax=ax[0, 1])
    ax[0, 1].set_title("(b) Resume length (words)")
    sns.histplot(data=df, x="years", hue="seniority", multiple="stack", palette="crest", bins=24, ax=ax[1, 0])
    ax[1, 0].set_title("(c) True dated experience (years) by seniority band")
    flags = pd.Series({"Career switchers": df.hybrid.mean(), "Inflated profiles": df.inflated.mean(), "Typo-noisy": df.noisy.mean(),
                       "Thin resumes": df.thin.mean()}) * 100
    flags.plot.barh(color=[ACCENT, RED, BASE, "#5B8DB8"], ax=ax[1, 1])
    _bar_labels(ax[1, 1], "{:.1f}%", horizontal=True, dy=0.3)
    ax[1, 1].set_title("(d) Difficulty ingredients (% of corpus)")
    ax[1, 1].set_xlim(0, 28)
    fig.suptitle(f"Benchmark overview: {len(df):,} synthetic resumes, {df.category.nunique()} categories", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, "fig02_dataset_overview.png")


def fig_extraction(R):
    ex = R["extraction"].copy()
    ex = ex[ex.metric.isin(["accuracy", "F1"])].copy()
    ex["label"] = np.where(ex.field == "Skills", "Skills (F1)", np.where(ex.metric == "F1", ex.field + " (F1)", ex.field))
    fig, ax = plt.subplots(2, 1, figsize=(9, 9))
    x = np.arange(len(ex))
    w = 0.38
    ax[0].bar(x - w / 2, ex.ours, w, color=OURS, label="EviProfile extractor")
    ax[0].bar(x + w / 2, ex.baseline.fillna(0), w, color=BASE, label="Generic regex baseline")
    for i, (o, b) in enumerate(zip(ex.ours, ex.baseline)):
        ax[0].text(i - w / 2, o + 0.005, f"{o:.3f}", ha="center", fontsize=7.5)
        ax[0].text(i + w / 2, 0.81 if np.isnan(b) else b + 0.005, "n/a" if np.isnan(b) else f"{b:.3f}", ha="center", fontsize=7.5, color="#555")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(ex.label, rotation=30, ha="right")
    ax[0].set_ylim(0.8, 1.10)
    ax[0].set_ylabel("Accuracy / F1 (test split)")
    ax[0].set_title("(a) Field-level extraction quality")
    ax[0].legend(loc="upper center", ncol=2)
    m = R["extraction"][R["extraction"].field == "Experience years"].iloc[0]
    ax[1].bar(["Claimed-years\nregex", "Dated-timeline\nEviProfile"], [m.baseline, m.ours], color=[BASE, OURS])
    _bar_labels(ax[1], "{:.2f}", fs=9, dy=0.02)
    ax[1].set_title("(b) Experience error (MAE, years)")
    ax[1].set_ylabel("Mean absolute error vs. true years")
    fig.tight_layout()
    _save(fig, "fig03_extraction.png")


def fig_formats(R):
    t = R["formats"]["table"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    cols = ["email", "phone", "name", "skills_f1", "title", "category_acc"]
    labels = ["Email", "Phone", "Name", "Skills F1", "Job title", "Category acc."]
    x = np.arange(len(cols))
    w = 0.26
    for i, (fmt, c) in enumerate(zip(t.index, [BLUE, ACCENT, OURS])):
        ax[0].bar(x + (i - 1) * w, t.loc[fmt, cols].astype(float), w, label=fmt, color=c)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, rotation=20)
    ax[0].set_ylim(0.9, 1.06)
    ax[0].set_title("(a) Extraction quality by file format")
    ax[0].legend(ncol=3, loc="upper center")
    ms = R["formats"]["parse_ms"]
    ax[1].bar([k.upper() for k in ms], list(ms.values()), color=[BLUE, ACCENT, OURS])
    for i, v in enumerate(ms.values()):
        ax[1].text(i, v * 1.15 + 0.02, f"{v:.2f} ms", ha="center", fontsize=9)
    ax[1].set_yscale("symlog", linthresh=0.1)
    ax[1].set_title("(b) File-reading time per resume (ms, log scale)")
    fig.tight_layout()
    _save(fig, "fig04_format_robustness.png")


def fig_ses(R):
    s = R["ses"]
    fig, ax = plt.subplots(3, 1, figsize=(8.5, 12.5))
    d = pd.DataFrame(s["scatter"])
    sns.boxplot(x="latent", y="ses", data=d, hue="latent", palette=["#E6A23C", "#3498DB", "#1E8449", "#145A32"], legend=False, ax=ax[0], fliersize=1.5)
    ax[0].set_xticks(range(4))
    ax[0].set_xticklabels(["1 Familiar", "2 Working", "3 Advanced", "4 Expert"])
    ax[0].set_xlabel("True (latent) proficiency")
    ax[0].set_ylabel("Skill Evidence Score")
    ax[0].set_title(f"(a) SES vs. true proficiency (n={s['n_pairs']:,})")
    rho = pd.Series(s["spearman"]).sort_values()
    cols = [OURS if "SES" in k else BASE for k in rho.index]
    ax[1].barh(rho.index, rho.values, color=cols)
    _bar_labels(ax[1], "{:.3f}", horizontal=True, dy=0.01)
    ax[1].set_xlim(0, 1.0)
    ax[1].set_xlabel("Spearman rho with true proficiency")
    ax[1].set_title("(b) Scorer comparison")
    ab = pd.Series(s["ablation"])
    cols = [OURS if k == "full SES" else ACCENT for k in ab.index]
    ax[2].barh(ab.index, ab.values, color=cols)
    _bar_labels(ax[2], "{:.3f}", horizontal=True, dy=0.01)
    ax[2].set_xlim(0, 1.0)
    ax[2].set_xlabel("Spearman rho (a priori weights)")
    ax[2].set_title("(c) Leave-one-component-out ablation")
    fig.tight_layout()
    _save(fig, "fig05_ses_validation.png")


def fig_tier_confusion(R):
    s = R["ses"]
    cm = s["confusion"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    norm = cm.div(cm.sum(axis=1), axis=0)
    sns.heatmap(norm, annot=cm.values, fmt="d", cmap="Greens", cbar_kws={"label": "Row-normalised share"}, ax=ax, linewidths=0.5)
    ax.set_yticklabels(["1 Familiar", "2 Working", "3 Advanced", "4 Expert"], rotation=0)
    ax.set_title(f"Predicted evidence tier vs. true proficiency\nexact agreement {s['tier_exact']:.1%}, within one tier {s['tier_within1']:.1%}")
    _save(fig, "fig06_ses_tier_confusion.png")


def fig_pci(R):
    p = R["pci"]
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 8.5))
    cols = {"Skill count": BASE, "Unsupported-skill ratio only": "#B39DDB", "Experience-gap only": "#D4A373", "PCI (a priori weights)": OURS, "PCI (learned weights)": BLUE}
    for k, c in p["curves"].items():
        a = sk_auc(c["fpr"], c["tpr"])
        ax[0, 0].plot(c["fpr"], c["tpr"], color=cols[k], lw=2 if "PCI" in k else 1.4, label=f"{k} (AUC {a:.3f})")
        ax[0, 1].plot(c["rec"], c["prec"], color=cols[k], lw=2 if "PCI" in k else 1.4, label=k)
    ax[0, 0].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[0, 0].set_xlabel("False positive rate")
    ax[0, 0].set_ylabel("True positive rate")
    ax[0, 0].set_title("(a) ROC: detecting inflated profiles")
    ax[0, 0].legend(fontsize=8, loc="lower right")
    ax[0, 1].axhline(p["prevalence"], color="k", ls=":", lw=0.8)
    ax[0, 1].set_xlabel("Recall")
    ax[0, 1].set_ylabel("Precision")
    ax[0, 1].set_title(f"(b) Precision-recall (prevalence {p['prevalence']:.0%})")
    d = pd.DataFrame(dict(pci=p["pci"], group=np.where(p["inflated"] == 1, "Inflated", "Honest")))
    sns.histplot(data=d, x="pci", hue="group", bins=25, element="step", stat="density", common_norm=False, palette={"Honest": OURS, "Inflated": RED}, ax=ax[1, 0])
    ax[1, 0].set_xlabel("Profile Credibility Index (1 = fully consistent)")
    ax[1, 0].set_title("(c) PCI distribution by group")
    ab = pd.Series(p["ablation"])
    ax[1, 1].barh(ab.index, ab.values, color=[OURS if k == "full PCI" else ACCENT for k in ab.index])
    _bar_labels(ax[1, 1], "{:.3f}", horizontal=True, dy=0.005)
    ax[1, 1].set_xlim(0.6, 1.0)
    ax[1, 1].set_xlabel("ROC AUC")
    ax[1, 1].set_title("(d) Leave-one-component-out ablation")
    fig.tight_layout()
    _save(fig, "fig07_pci_detection.png")


def fig_ranking(R):
    t = R["ranking"]["table"].copy()
    t["lab"] = t.method.apply(lambda s: "\n".join(textwrap.wrap(s, 24)))
    fig, ax = plt.subplots(3, 1, figsize=(9, 13.5))
    cols = [BASE, BASE, BASE, "#5B8DB8", "#B39DDB", "#D4A373", OURS, BLUE, ACCENT][:len(t)]
    y = np.arange(len(t))[::-1]
    ax[0].barh(y, t.ndcg10, xerr=[t.ndcg10 - t.ndcg_lo, t.ndcg_hi - t.ndcg10], color=cols, capsize=3)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(t.lab, fontsize=8)
    for yi, v, hi in zip(y, t.ndcg10, t.ndcg_hi):
        ax[0].text(hi + 0.012, yi, f"{v:.3f}", va="center", fontsize=8)
    ax[0].set_xlim(0, 1.15)
    ax[0].set_xlabel("nDCG@10 (mean over jobs, 95% bootstrap CI)")
    ax[0].set_title("(a) Ranking quality")
    w = 0.27
    ax[1].barh(y + w, t.map10, w, color="#2F6DB5", label="MAP@10")
    ax[1].barh(y, t.p10, w, color="#5B8DB8", label="P@10")
    ax[1].barh(y - w, t.recall10, w, color="#A9C3DE", label="Recall@10")
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(t.lab, fontsize=8)
    ax[1].set_xlim(0, 1.05)
    ax[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    ax[1].set_title("(b) Precision-oriented metrics")
    ax[2].barh(y, t.false_promotion * 100, color=[RED if k in ("keyword", "tfidf_cosine", "lsa_cosine", "ontology") else OURS for k in t.key])
    ax[2].set_yticks(y)
    ax[2].set_yticklabels(t.lab, fontsize=8)
    for yi, v in zip(y, t.false_promotion * 100):
        ax[2].text(v + 0.1, yi, f"{v:.1f}%", va="center", fontsize=8)
    ax[2].set_xlabel("% of top-10 slots taken by inflated,\nunderqualified candidates")
    ax[2].set_title("(c) False promotion")
    fig.tight_layout()
    _save(fig, "fig08_ranking_comparison.png")


def fig_models(R):
    c = R["clf"]
    t, cv = c["test"], c["cv"]
    names = list(t.index)
    cols = [BASE, "#5B8DB8", "#B39DDB", "#D4A373", "#7FA37F", BLUE, OURS]
    fig, ax = plt.subplots(3, 1, figsize=(8.5, 12.5))
    x = np.arange(len(names))
    lab = ["\n".join(textwrap.wrap(n, 10)) for n in names]
    ax[0].bar(x, t.accuracy, color=cols, yerr=[t.accuracy - t.acc_lo, t.acc_hi - t.accuracy], capsize=3)
    for i, v in enumerate(t.accuracy):
        ax[0].text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(lab, fontsize=8)
    ax[0].set_ylim(0.7, 1.06)
    ax[0].set_ylabel("Accuracy (hold-out test, 95% bootstrap CI)")
    ax[0].set_title("(a) Hold-out test accuracy")
    ax[1].bar(x, cv.loc[names, "cv_f1_mean"], yerr=cv.loc[names, "cv_f1_std"], color=cols, capsize=3)
    for i, v in enumerate(cv.loc[names, "cv_f1_mean"]):
        ax[1].text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(lab, fontsize=8)
    ax[1].set_ylim(0.7, 1.06)
    ax[1].set_ylabel("Macro-F1 (5-fold CV, mean +/- SD)")
    ax[1].set_title("(b) Cross-validated macro-F1")
    ax[2].barh(names[::-1], t.train_s.values[::-1], color=cols[::-1])
    for i, v in enumerate(t.train_s.values[::-1]):
        ax[2].text(v + 0.1, i, f"{v:.1f}s", va="center", fontsize=8)
    ax[2].set_xlabel("Training time (s, 2,000 resumes, 1 CPU core)")
    ax[2].set_title("(c) Training cost")
    fig.tight_layout()
    _save(fig, "fig09_model_comparison.png")


def fig_confusion(R):
    c = R["clf"]
    cm, cls = c["confusion"], c["classes"]
    fig, ax = plt.subplots(figsize=(9.2, 7.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[x.replace(" & ", "\n& ") for x in cls], yticklabels=cls,
                cbar_kws={"label": "Resumes"}, ax=ax, linewidths=0.4)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title(f"EviProfile confusion matrix ({int(cm.sum())} hold-out resumes)")
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=8)
    _save(fig, "fig10_confusion_matrix.png")


def fig_per_class(R):
    c = R["clf"]
    pc = pd.DataFrame(c["per_class"], index=c["classes"])
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    sns.heatmap(pc, annot=True, fmt=".3f", cmap="YlGn", vmin=0.6, vmax=1.0, ax=ax, linewidths=0.4, cbar_kws={"label": "F1"})
    ax.set_title("Per-category F1 on the hold-out set")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _save(fig, "fig11_per_class_f1.png")


def fig_roc(R):
    c = R["clf"]
    y = np.array(c["y_test"])
    cls = c["classes"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.6))
    grid = np.linspace(0, 1, 200)
    cols = {"Keyword rule": BASE, "Multinomial NB": "#5B8DB8", "Logistic Regression": "#B39DDB", "Linear SVM": "#D4A373",
            "Random Forest": "#7FA37F", "MLP (TF-IDF)": BLUE, "EviProfile (fused MLP)": OURS}
    for n, P in c["probas"].items():
        full, zoom = [], []
        for j, k in enumerate(cls):
            fpr, tpr, _ = roc_curve((y == k).astype(int), P[:, j])
            full.append(np.interp(grid, fpr, tpr))
            zoom.append(np.interp(grid * 0.12, fpr, tpr))
        lw = 2.2 if "Evi" in n else 1.3
        ax[0].plot(grid, np.mean(full, 0), color=cols[n], lw=lw, label=f"{n} (AUC {c['test'].loc[n, 'roc_auc']:.3f})")
        ax[1].plot(grid * 0.12, np.mean(zoom, 0), color=cols[n], lw=lw)
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate")
    ax[0].set_title("(a) Macro-averaged one-vs-rest ROC")
    ax[0].legend(fontsize=8, loc="lower right")
    ax[1].set_xlim(0, 0.12)
    ax[1].set_ylim(0.85, 1.005)
    ax[1].set_xlabel("False positive rate")
    ax[1].set_title("(b) Zoom on the low false-positive region")
    fig.tight_layout()
    _save(fig, "fig12_roc_curves.png")


def fig_subsets(R):
    s, n = R["clf"]["subsets"], R["clf"]["subset_n"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    cols = [BASE, "#B39DDB", "#D4A373", BLUE, OURS]
    x = np.arange(len(s.columns))
    w = 0.16
    for i, (m, c) in enumerate(zip(s.index, cols)):
        ax.bar(x + (i - 2) * w, s.loc[m].astype(float), w, label=m, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\n(n={n[k]})" for k in s.columns])
    ax.set_ylim(0.5, 1.03)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy on difficult subsets of the hold-out set")
    ax.legend(ncol=5, loc="lower center", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig13_subset_accuracy.png")


def fig_learning(R):
    d = R["learning"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    cols = {"Keyword rule": BASE, "Logistic Regression": "#B39DDB", "MLP (TF-IDF)": BLUE, "EviProfile (fused MLP)": OURS}
    for m, g in d.groupby("model"):
        a = g.groupby("n_train").macro_f1.agg(["mean", "std"]).fillna(0)
        ax.plot(a.index, a["mean"], marker="o", color=cols[m], lw=2.2 if "Evi" in m else 1.5, label=m)
        ax.fill_between(a.index, a["mean"] - a["std"], a["mean"] + a["std"], color=cols[m], alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("Training resumes (log scale)")
    ax.set_ylabel("Macro-F1 on hold-out test")
    ax.set_title("Learning curves (mean +/- SD over 3 resamples)")
    ax.legend(loc="lower right")
    _save(fig, "fig14_learning_curve.png")


def fig_noise(R):
    d = R["noise"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    cols = {"Keyword rule": BASE, "Logistic Regression": "#B39DDB", "Linear SVM": "#D4A373", "MLP (TF-IDF)": BLUE, "EviProfile (fused MLP)": OURS}
    for m, g in d.groupby("model"):
        ax.plot(g.noise * 100, g.accuracy, marker="o", color=cols[m], lw=2.4 if "Evi" in m else 1.5, label=m)
    ax.set_xlabel("Typographical noise injected at test time (% of long words corrupted)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Robustness to typographical noise")
    ax.legend(loc="lower left")
    _save(fig, "fig15_noise_robustness.png")


def fig_fairness(R):
    f = R["fairness"]
    fig, ax = plt.subplots(3, 1, figsize=(8.5, 13))
    a = f["audit"]
    x = np.arange(len(a))
    c3 = [BASE, "#B39DDB", OURS][:len(a)]
    ax[0].bar(x - 0.2, a.mean_rank_displacement, 0.4, color=c3)
    for i, v in enumerate(a.mean_rank_displacement):
        ax[0].text(i - 0.2, v + 0.2, f"{v:.2f}", ha="center", fontsize=8)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(["\n".join(textwrap.wrap(n, 11)) for n in a.index], fontsize=9)
    ax[0].set_ylabel("Mean |rank change| (solid bars)")
    ax2 = ax[0].twinx()
    ax2.bar(x + 0.2, a.top10_change_rate * 100, 0.4, color=ACCENT, alpha=0.8)
    for i, v in enumerate(a.top10_change_rate * 100):
        ax2.text(i + 0.2, v + 0.2, f"{v:.1f}%", ha="center", fontsize=8)
    ax2.set_ylabel("Top-10 changed, % (orange bars)")
    ax2.grid(False)
    ax[0].set_title("(a) Ranking after identity swap\n(name group, institution tier, activity)")
    fl = pd.Series(f["flips"]) * 100
    ax[1].barh(fl.index, fl.values, color=[BASE, BLUE, OURS][:len(fl)])
    for i, v in enumerate(fl.values):
        ax[1].text(v + 0.03, i, f"{v:.2f}%", va="center", fontsize=9)
    ax[1].set_xlabel("Predictions changed by identity swap (%)")
    ax[1].set_title("(b) Classification flip rate")
    st = f["stress"]
    x = np.arange(len(st))
    cc = [BASE, BLUE, "#7FA37F", OURS][:len(st)]
    ax[2].bar(x - 0.2, st.gap * 100, 0.4, color=cc, label="Accuracy gap")
    ax[2].bar(x + 0.2, st.eo_gap * 100, 0.4, color=cc, alpha=0.45, hatch="//", label="Equal-opportunity gap")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(["\n".join(textwrap.wrap(n, 14)) for n in st.index], fontsize=8)
    ax[2].set_ylabel("Percentage points")
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_title("(c) Bias-stress test (90% gender-coded history)\ngap > 0 would mean worse accuracy for counter-stereotypical candidates", fontsize=10)
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig16_fairness.png")


def fig_efficiency(R):
    s = pd.Series(R["efficiency"]["stages_ms"]).dropna()
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    cols = [BLUE if "File" in k else OURS for k in s.index]
    ax.barh(s.index[::-1], s.values[::-1], color=cols[::-1])
    for i, v in enumerate(s.values[::-1]):
        ax.text(v * 1.08 + 0.02, i, f"{v:.2f} ms", va="center", fontsize=9)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("Milliseconds per resume (log scale)")
    ax.set_title(f"Processing time by stage (end-to-end {R['efficiency']['total_ms_txt']:.1f} ms, "
                 f"{R['efficiency']['throughput_per_min']:,.0f} resumes/min on one core)", fontsize=11)
    _save(fig, "fig17_efficiency.png")


def fig_profile_cards(R):
    ex = R["examples"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 11.5))
    for ax, (tag, title) in zip(axes, [("well_evidenced", "Well-evidenced candidate"), ("inflated", "Inflated candidate")]):
        p = ex[tag]["profile"]
        sk = p["skills"][:12]
        y = np.arange(len(sk))[::-1]
        ax.barh(y, [s["ses"] for s in sk], color=[TIER_COL[s["tier"]] for s in sk])
        ax.set_yticks(y)
        ax.set_yticklabels([s["skill"] + ("" if s["narrative"] else " (list only)") for s in sk], fontsize=8.5)
        ax.set_xlim(0, 1.5)
        for yi, s in zip(y, sk):
            ax.text(s["ses"] + 0.015, yi, f"{s['ses']:.2f} {s['tier']}", va="center", fontsize=7.5)
        c = p["candidate"]
        cat = p["category"]["predicted"] if "category" in p else "n/a"
        txt = (f"{c['recent_title']}\ncategory: {cat}\ndated exp: {c['dated_experience_years']:.1f} y | claimed: {c['claimed_experience_years']}\n"
               f"strength {p['profile_strength']['score']} | credibility {p['credibility']['pci']:.2f}")
        ax.text(0.99, 0.02, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                bbox=dict(boxstyle="round", fc="#F3F6F9", ec="#B8C4CF"))
        flags = p["credibility"]["flags"]
        ft = "\n".join(textwrap.fill("! " + f, 55) for f in flags[:3]) if flags else "no credibility concerns"
        ax.set_title(f"{title}\n{ft}", fontsize=9.5, color=RED if flags else OURS, loc="left")
        ax.set_xlabel("Skill Evidence Score")
    handles = [mpatches.Patch(color=v, label=k) for k, v in TIER_COL.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "fig18_profile_cards.png")


def fig_skill_graph(R):
    import networkx as nx
    g = nx.Graph()
    for f in R["graph"]["families"]:
        for i in range(len(f)):
            for j in range(i + 1, len(f)):
                g.add_edge(f[i], f[j], kind="family")
    for c, p in R["graph"]["implies"]:
        g.add_edge(c, p, kind="implies")
    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    comps = sorted(nx.connected_components(g), key=lambda c: -len(c))
    pos, ncols = {}, 4
    for k, comp in enumerate(comps):
        sub = g.subgraph(comp)
        lp = nx.circular_layout(sub, scale=0.9) if len(comp) > 2 else nx.spring_layout(sub, seed=1, scale=0.5)
        cx, cy = (k % ncols) * 3.0, -(k // ncols) * 2.7
        for n_, (x_, y_) in lp.items():
            pos[n_] = (cx + x_, cy + y_)
    fam = [(u, v) for u, v, d in g.edges(data=True) if d["kind"] == "family"]
    imp = [(u, v) for u, v, d in g.edges(data=True) if d["kind"] == "implies"]
    nx.draw_networkx_edges(g, pos, edgelist=fam, edge_color=BLUE, width=1.6, alpha=0.7, ax=ax)
    nx.draw_networkx_edges(g, pos, edgelist=imp, edge_color=ACCENT, width=1.6, style="dashed", alpha=0.9, ax=ax)
    nx.draw_networkx_nodes(g, pos, node_size=1300, node_color="#EAF1F8", edgecolors="#33475B", ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=8.5, ax=ax)
    ax.legend(handles=[plt.Line2D([0], [0], color=BLUE, lw=2, label="same family (credit 0.5)"),
                       plt.Line2D([0], [0], color=ACCENT, lw=2, ls="--", label="implies, child to parent (credit 0.7)")], loc="lower left")
    ax.set_title("Skill graph known to the matcher (curated relations)")
    ax.axis("off")
    _save(fig, "fig19_skill_graph.png")


def make_all(R):
    for fn in [fig_architecture, fig_dataset, fig_extraction, fig_formats, fig_ses, fig_tier_confusion, fig_pci, fig_ranking, fig_models,
               fig_confusion, fig_per_class, fig_roc, fig_subsets, fig_learning, fig_noise, fig_fairness, fig_efficiency,
               fig_profile_cards, fig_skill_graph]:
        try:
            fn(R)
        except Exception:
            print("FAILED", fn.__name__)
            traceback.print_exc()
