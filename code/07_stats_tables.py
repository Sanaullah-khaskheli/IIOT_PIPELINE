"""STAGE 5c -- aggregation, statistics, manuscript tables and the numbers.tex macro file. Reads ONLY saved result CSVs of this run."""
import os, glob
import numpy as np, pandas as pd
from common import *
import safe_core as sc

PRIMARY = ["rf:maxprob", "mlp:mahalanobis", "fusion_rank"]
PAIRS = [("fusion_rank", "rf:maxprob"), ("fusion_rank", "mlp:mahalanobis"), ("rf:maxprob", "mlp:mahalanobis"),
         ("rf:maxprob", "mlp:maxprob"), ("mlp:mahalanobis", "mlp:maxprob"), ("mlp:mahalanobis", "mlp:energy"), ("edl:vacuity", "mlp:maxprob")]


def cat(root, sub, fn):
    fs = sorted(glob.glob(os.path.join(root, sub, "*", fn)))
    if not fs: raise FileNotFoundError(f"{sub}/*/{fn} missing -- MISSING EVIDENCE, do not report")
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def main(cfg, run_id):
    root = results_root(cfg); T = os.path.join(root, "tables"); S = os.path.join(root, "statistics"); os.makedirs(T, exist_ok=True); os.makedirs(S, exist_ok=True)
    num = {}
    def put(k, v, src): num[k] = (float(v), src)
    wt = lambda df, stem: (write_table(df.assign(run_id=run_id) if False else df, T, stem), df.to_csv(os.path.join(S, stem + ".csv"), index=False))
    lad = cat(root, "closed_set", "ladder_by_seed.csv")
    g = lad.groupby(["dataset", "rung", "model"])
    agg = g.agg(n_seeds=("seed", "nunique"), n_test=("n_test", "first"), n_features=("n_features", "first"),
                MacroF1=("MacroF1", "mean"), MacroF1_sd=("MacroF1", "std"), BinaryMCC=("BinaryMCC", "mean"), BinaryMCC_sd=("BinaryMCC", "std"),
                ECE=("ECE", "mean"), Accuracy=("Accuracy", "mean")).reset_index()
    ci = lad.groupby(["dataset", "rung", "model"]).MacroF1.apply(lambda x: pd.Series(sc.mean_sd_ci(x))).unstack().reset_index()
    agg = agg.merge(ci[["dataset", "rung", "model", "ci_low", "ci_high"]].rename(columns={"ci_low": "MacroF1_CI_low", "ci_high": "MacroF1_CI_high"}))
    order = {"A_raw": 0, "D_strict": 1, "E3_noport": 2, "E4_network": 3}
    agg = agg.sort_values(["dataset", "rung", "MacroF1"], key=lambda s: s.map(order) if s.name == "rung" else s, ascending=[True, True, False])
    wt(agg, "T_ladder_closed_set")
    for _, r in agg.iterrows():
        k = f"{r.dataset}{r.rung}{r.model}".replace("_", "").replace(":", "")
        put("F1" + k, r.MacroF1, "tables/T_ladder_closed_set.csv:MacroF1")
    ov = []
    for f in sorted(glob.glob(os.path.join(root, "loao", "*", "family_inventory.csv"))):
        ds = os.path.basename(os.path.dirname(f)); inv = pd.read_csv(f); cc = pd.read_csv(os.path.join(root, "closed_set", ds, "class_counts.csv"))
        l = lad[lad.dataset == ds]; nf = l.groupby("rung").n_features.first()
        ev = inv[inv.evaluated]; grp = ev.groupby("group").family.nunique()
        ov.append({"dataset": ds, "rows_used": int(cc.n_rows_used.sum()), "classes": len(cc), "attack_families": len(inv), "families_evaluated": len(ev),
                   "families_skipped_small": int((~inv.evaluated).sum()), "near_families": int((ev.group.map(grp) > 1).sum()), "far_families": int((ev.group.map(grp) == 1).sum()),
                   "features_per_rung": "; ".join(f"{k}:{int(v)}" for k, v in nf.items())})
    wt(pd.DataFrame(ov), "T_dataset_overview")
    lo = cat(root, "loao", "loao_by_family_seed_method.csv")
    prox = cat(root, "loao", "proximity.csv").groupby(["dataset", "family"], as_index=False).proximity_dist.mean()
    clo = cat(root, "loao", "known_class_closed_set_F1.csv")
    pf = lo.groupby(["dataset", "condition", "method", "family", "group", "near_far", "n_siblings"], as_index=False).agg(
        AUROC=("AUROC", "mean"), AUPR=("AUPR", "mean"), FPR95=("FPR@95TPR", "mean"), DetRate=("DetRate@val5FPR", "mean"),
        KnownFPR=("KnownFPR@val5", "mean"), AUROC_seed_sd=("AUROC", "std"), n_seeds=("seed", "nunique"), n_unknown=("n_unknown", "first"), n_known=("n_known", "first"))
    pf = pf.merge(prox, on=["dataset", "family"], how="left")
    pf.to_csv(os.path.join(S, "loao_per_family.csv"), index=False)
    ps = lo.groupby(["dataset", "condition", "method", "seed"], as_index=False).agg(AUROC=("AUROC", "mean"), AUPR=("AUPR", "mean"),
                                                                                       FPR95=("FPR@95TPR", "mean"), DetRate=("DetRate@val5FPR", "mean"))
    rows = []
    for (d, c, m), x in ps.groupby(["dataset", "condition", "method"]):
        r = {"dataset": d, "condition": c, "method": m, "families": int(pf[(pf.dataset == d) & (pf.condition == c) & (pf.method == m)].family.nunique())}
        for met in ("AUROC", "AUPR", "FPR95", "DetRate"):
            q = sc.mean_sd_ci(x[met]); r[met] = q["mean"]; r[met + "_sd_seeds"] = q["sd"]
        r["AUROC_CI_low"], r["AUROC_CI_high"] = sc.mean_sd_ci(x.AUROC)["ci_low"], sc.mean_sd_ci(x.AUROC)["ci_high"]
        r["median_family_AUROC"] = float(pf[(pf.dataset == d) & (pf.condition == c) & (pf.method == m)].AUROC.median())
        rows.append(r)
    summ = pd.DataFrame(rows); wt(summ, "T_loao_method_summary")
    for _, r in summ.iterrows():
        put(f"AUROC{r.dataset}{r.condition}{r.method}".replace("_", "").replace(":", ""), r.AUROC, "tables/T_loao_method_summary.csv:AUROC")
    kc = clo.groupby(["dataset", "model"], as_index=False).known_MacroF1.mean(); wt(kc, "T_loao_known_class_F1")
    pr = []
    for (d, c), x in pf.groupby(["dataset", "condition"]):
        w = x.pivot(index="family", columns="method", values="AUROC")
        block = [dict(dataset=d, condition=c, **sc.paired_over_families(w, a, b)) for a, b in PAIRS if a in w and b in w]
        if block:
            h = sc.holm([b["Wilcoxon_p"] for b in block])
            for b, hp in zip(block, h): b["Holm_p"] = hp
            pr += block
    pairs = pd.DataFrame(pr); wt(pairs, "T_loao_paired_methods")
    nf = []
    for (d, c, m), x in pf.groupby(["dataset", "condition", "method"]):
        r = sc.near_vs_far(x[x.near_far == "near"].AUROC, x[x.near_far == "far"].AUROC); r.update(dataset=d, condition=c, method=m)
        xx = x.dropna(subset=["proximity_dist"])
        if len(xx) >= 6:
            from scipy.stats import spearmanr
            rho = spearmanr(xx.proximity_dist, xx.AUROC); r["spearman_proximity_vs_AUROC"] = rho.statistic; r["spearman_p"] = rho.pvalue
        nf.append(r)
    nfd = pd.DataFrame(nf); wt(nfd, "T_loao_near_vs_far")
    if len(nfd):
        for _, r in nfd.iterrows():
            put(f"nearmed{r.dataset}{r.condition}{r.method}".replace("_", "").replace(":", ""), r.median_near, "tables/T_loao_near_vs_far.csv:median_near")
            put(f"farmed{r.dataset}{r.condition}{r.method}".replace("_", "").replace(":", ""), r.median_far, "tables/T_loao_near_vs_far.csv:median_far")
    pl = []
    for (c, m), x in pf.groupby(["condition", "method"]):
        r = sc.near_vs_far(x[x.near_far == "near"].AUROC, x[x.near_far == "far"].AUROC); r.update(condition=c, method=m)
        sg = [np.sign(y[y.near_far == "far"].AUROC.median() - y[y.near_far == "near"].AUROC.median()) for _, y in x.groupby("dataset") if (y.near_far == "far").any() and (y.near_far == "near").any()]
        r["datasets_far_higher"] = int(sum(v > 0 for v in sg)); r["datasets_far_lower"] = int(sum(v < 0 for v in sg)); pl.append(r)
    wt(pd.DataFrame(pl), "T_loao_near_far_pooled")
    fc = pf[(pf.condition == "attacks_only") & (pf.method.isin(PRIMARY))].sort_values("AUROC").groupby(["dataset", "method"]).head(5)
    wt(fc[["dataset", "method", "family", "group", "near_far", "n_siblings", "AUROC", "FPR95", "n_unknown"]], "T_loao_failure_cases")
    import re
    W = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    mac = lambda k: re.sub(r"[^A-Za-z]", lambda m: W.get(m.group(), ""), k)
    with open(os.path.join(T, "numbers.tex"), "w") as f:
        for k, (v, src) in num.items():
            f.write("\\newcommand{\\%s}{%.3f}\n" % (mac(k), v))
    pd.DataFrame([{"macro": mac(k), "value": round(v, 3), "source": s_, "run_id": run_id} for k, (v, s_) in num.items()]).to_csv(os.path.join(T, "numbers_manifest.csv"), index=False)
    return {"tables": 8}, "### stats done\n"
