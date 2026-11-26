"""Run the complete EviProfile study: data generation -> extraction -> evidence -> models -> experiments -> tables -> figures.

    python run_pipeline.py            # full study (about 6-8 minutes on one CPU core)
    python run_pipeline.py --quick    # small smoke test (about 1 minute)
    python run_pipeline.py --figures-only   # re-draw figures from the cached results
"""
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

from resume_intel import config as C
from resume_intel import experiments as E
from resume_intel import generator as G
from resume_intel.pipeline import render_html_card, CARD_CSS, job_from_text


def save_table(df, name):
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TAB_DIR / f"{name}.csv")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    for d in (C.DATA_DIR, C.MODEL_DIR, C.FIG_DIR, C.TAB_DIR, C.RESULTS_DIR / "examples"):
        d.mkdir(parents=True, exist_ok=True)
    cache = C.RESULTS_DIR / "_results_cache.pkl"
    if a.figures_only:
        from resume_intel import viz
        viz.make_all(pickle.load(open(cache, "rb")))
        return
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)
    n_pc = 40 if a.quick else C.N_PER_CLASS
    log(f"building context ({n_pc} resumes per category)")
    ctx = E.build_context(n_pc)
    R = {}
    log("E1 dataset");            R["dataset"] = E.exp_dataset(ctx)
    log("E2 extraction");         R["extraction"] = E.exp_extraction(ctx)
    log("E2b file formats");      R["formats"] = E.exp_formats(ctx, n=40 if a.quick else 150)
    log("E4 SES validation");     R["ses"] = E.exp_ses(ctx)
    log("E5 PCI");                R["pci"] = E.exp_pci(ctx)
    log("E6 job-fit ranking");    R["ranking"] = E.exp_ranking(ctx)
    log("E3 classification (5-fold CV + hold-out)"); R["clf"] = E.exp_classification(ctx, cv_folds=3 if a.quick else 5)
    log("E7a learning curve");    R["learning"] = E.exp_learning_curve(ctx, sizes=(100, 400) if a.quick else (100, 200, 400, 800, 1600), reps=1 if a.quick else 3)
    log("E7b noise robustness");  R["noise"] = E.exp_noise(ctx)
    log("E8 fairness");           R["fairness"] = E.exp_fairness(ctx)
    log("E9 efficiency");         R["efficiency"] = E.exp_efficiency(ctx, n=50 if a.quick else 200)

    # ---- examples: one well-evidenced candidate and one inflated candidate, scored against the same job
    te = list(ctx.te)
    job = dict(ctx.jobs[0]); job_ds = dict(required=job["required"], preferred=job["preferred"], min_years=job["min_years"])
    # Disclosed selection rule (illustrative, NOT representative): among Data Science test resumes with seniority >= 1,
    # the inflated resume with the LOWEST credibility index and the honest resume with the HIGHEST profile strength.
    cand = [i for i in te if ctx.recs[i]["category"] == "Data Science" and ctx.recs[i]["seniority"] >= 1]
    if not any(ctx.recs[i]["inflated"] for i in cand) or all(ctx.recs[i]["inflated"] for i in cand):   # tiny test sets (--quick)
        cand = list(te)
    profs = {i: ctx.pipe.analyze_text(ctx.recs[i]["text"], job_ds) for i in cand}
    inf = [i for i in cand if ctx.recs[i]["inflated"]]
    hon = [i for i in cand if not ctx.recs[i]["inflated"]]
    chosen = {"well_evidenced": max(hon, key=lambda i: profs[i]["profile_strength"]["score"]),
              "inflated": min(inf, key=lambda i: profs[i]["credibility"]["pci"])}
    ex = {}
    for tag, i in chosen.items():
        prof = profs[i]
        ex[tag] = dict(profile=prof, text=ctx.recs[i]["text"], truth=dict(inflated=ctx.recs[i]["inflated"], true_years=ctx.recs[i]["true_years"],
                                                                         claimed=ctx.recs[i]["claimed_years"], true_skills=ctx.recs[i]["true_skills"]))
        (C.RESULTS_DIR / "examples" / f"profile_{tag}.json").write_text(json.dumps(prof, indent=2, default=str))
        (C.RESULTS_DIR / "examples" / f"profile_{tag}.html").write_text(f"<style>{CARD_CSS}</style>" + render_html_card(prof), encoding="utf-8")
        (C.RESULTS_DIR / "examples" / f"resume_{tag}.txt").write_text(ctx.recs[i]["text"], encoding="utf-8")
    ex["job"] = job_ds
    R["examples"] = ex
    g = ctx.pipe.graph
    R["graph"] = dict(families=[f for f in G.FAM_KNOWN], implies=list(G.IMP_KNOWN))

    # ---- save tables
    log("saving tables / artefacts")
    save_table(pd.Series(R["dataset"]["summary"], name="value").to_frame(), "dataset_summary")
    save_table(R["extraction"].set_index(["field", "metric"]), "extraction_accuracy")
    save_table(R["formats"]["table"], "format_robustness")
    save_table(pd.Series(R["ses"]["spearman"], name="spearman_rho").to_frame(), "ses_validation")
    save_table(pd.Series(R["ses"]["ablation"], name="spearman_rho").to_frame(), "ses_ablation")
    save_table(R["ses"]["confusion"], "ses_tier_confusion")
    save_table(R["pci"]["table"].set_index("method"), "pci_detection")
    save_table(pd.Series(R["pci"]["ablation"], name="roc_auc").to_frame(), "pci_ablation")
    save_table(R["ranking"]["table"].set_index("method"), "ranking_comparison")
    save_table(R["clf"]["cv"], "classification_cv")
    save_table(R["clf"]["test"], "classification_test")
    save_table(R["clf"]["subsets"], "classification_subsets")
    save_table(pd.DataFrame(R["clf"]["variants"]).T, "classification_ablation")
    save_table(pd.DataFrame(R["clf"]["mcnemar"]).T, "classification_mcnemar")
    save_table(R["learning"].groupby(["model", "n_train"]).macro_f1.agg(["mean", "std"]), "learning_curve")
    save_table(R["noise"].pivot(index="noise", columns="model", values="accuracy"), "noise_robustness")
    save_table(R["fairness"]["audit"], "fairness_counterfactual_audit")
    save_table(R["fairness"]["stress"], "fairness_stress_test")
    save_table(pd.Series(R["efficiency"]["stages_ms"], name="ms_per_resume").to_frame(), "efficiency")
    R["dataset"]["df"].to_csv(C.DATA_DIR / "dataset_features.csv", index=False)
    pd.DataFrame([dict(id=r["id"], category=r["category"], hybrid=r["hybrid"], inflated=r["inflated"], noisy=r["noisy"], text=r["text"]) for r in ctx.recs]
                 ).to_csv(C.DATA_DIR / "synthetic_resumes.csv", index=False)
    (C.DATA_DIR / "job_postings.json").write_text(json.dumps(ctx.jobs, indent=2))
    ctx.pipe.save(C.MODEL_DIR / "eviprofile.joblib")
    pickle.dump(R, open(cache, "wb"))
    summary = dict(dataset=R["dataset"]["summary"], ranking=R["ranking"]["table"].set_index("key")[["ndcg10", "map10", "p10", "false_promotion"]].round(4).to_dict("index"),
                   classification_test=R["clf"]["test"][["accuracy", "macro_f1"]].round(4).to_dict("index"), efficiency=R["efficiency"],
                   ses_spearman=R["ses"]["spearman"], pci_auc=R["pci"]["table"].set_index("method").roc_auc.round(4).to_dict())
    (C.RESULTS_DIR / "results_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log("figures")
    from resume_intel import viz
    viz.make_all(R)
    log("done")


if __name__ == "__main__":
    main()
