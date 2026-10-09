"""SAFE-IIoT figure generator (STANDALONE: needs only pandas/numpy/matplotlib/seaborn and the CSVs produced by 05-07).
Kaggle:   !python 08_make_figures.py --results /kaggle/working/results_final
Writes figures/*.pdf + *.png (600 dpi), figure_manifest.csv, figure_captions.txt, MANUSCRIPT_FIGURE_DATA_AUDIT.csv, SAFE_IIoT_figures.zip.
Fails loudly: a missing file/column prints 'DATA REQUIRED -- DO NOT GENERATE FIGURE' and the script exits non-zero.
Fig 1 (framework) is a MANUAL VECTOR FIGURE (spec in FIG1_SPEC below). Fig 6 (TabICLv2) is NOT generated: experiment removed."""
import os, sys, argparse, zipfile, datetime as dt
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

RESULTS_DIR = "/kaggle/working/results_final"
DPI = 600
W1, W2 = 3.4, 7.0
DATASETS = ["edge_iiotset", "x_iiotid", "ciciot2023"]
DS_LABEL = {"edge_iiotset": "Edge-IIoTset", "x_iiotid": "X-IIoTID", "ciciot2023": "CICIoT2023"}
RUNG_ORDER = ["A_raw", "D_strict", "E3_noport", "E4_network"]
RUNG_LABEL = {"A_raw": "Raw", "D_strict": "Strict-clean", "E3_noport": "No-port", "E4_network": "Network-only"}
MODELS = ["rf", "lgbm", "mlp_softmax_en"]
MODEL_LABEL = {"rf": "RF", "lgbm": "LightGBM", "mlp_softmax_en": "MLP", "mlp_edl_both": "MLP-EDL"}
METHODS = ["rf:maxprob", "mlp:maxprob", "mlp:energy", "mlp:mahalanobis", "fusion_rank", "edl:vacuity"]
METHOD_LABEL = {"rf:maxprob": "RF max-prob", "mlp:maxprob": "MLP max-prob", "mlp:energy": "MLP energy", "mlp:mahalanobis": "MLP Mahalanobis",
                "fusion_rank": "Rank fusion", "edl:vacuity": "EDL vacuity (control)", "fusion_z": "Z fusion (sens.)"}
PRIMARY = ["rf:maxprob", "mlp:mahalanobis", "fusion_rank"]
COND = "attacks_only"
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000", "#F0E442"]
FIG1_SPEC = """FIG 1 -- MANUAL VECTOR FIGURE REQUIRED (draw in Inkscape/draw.io, export PDF, 7.0 in wide, sans-serif 7-8 pt):
 Left->right 4 boxes: (1) 'Raw dataset' (Edge-IIoTset / X-IIoTID / CICIoT2023) -> (2) 'Shortcut audit ladder' [Raw -> Strict-clean -> No-port -> Network-only; sub-text: identifiers, timestamps, payload, ports, host telemetry]
 -> (3) 'Frozen family taxonomy' [near = sibling in known pool, far = no sibling] -> (4) 'Sibling-aware LOAO' [train on known families, hold out one family]
 -> output box 'Scores: RF max-prob | MLP max-prob/energy/Mahalanobis | rank fusion | EDL (control)' -> 'AUROC, AUPR, FPR@95TPR, DetRate@val | near vs far'. Colours: Okabe-Ito; no gradients."""

CAPTIONS, MANIFEST, AUDIT, ERRORS = {}, [], [], []
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight"})


def need(path, cols):
    if not os.path.exists(path):
        ERRORS.append(f"DATA REQUIRED -- DO NOT GENERATE FIGURE: missing file {path}"); return None
    d = pd.read_csv(path); miss = [c for c in cols if c not in d.columns]
    if miss:
        ERRORS.append(f"DATA REQUIRED -- DO NOT GENERATE FIGURE: {path} lacks columns {miss}"); return None
    return d


def save(fig, stem, num, purpose, src, rows, filters, metric, seeds, datasets, caption, size):
    out = os.path.join(OUT, "figures"); os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, stem + ".pdf")); fig.savefig(os.path.join(out, stem + ".png"), dpi=DPI); plt.close(fig)
    row = dict(figure=num, file=stem, purpose=purpose, source_file=src, rows_used=rows, filters=filters, metric=metric, seeds=seeds,
               dataset=datasets, output_filename=stem + ".pdf/.png", width_in=size, dpi=DPI, generated=dt.datetime.now().isoformat(timespec="seconds"))
    MANIFEST.append({k: row[k] for k in ("figure", "file", "purpose", "source_file", "output_filename", "width_in", "dpi", "generated")}); AUDIT.append(row)
    CAPTIONS[num] = caption


def fig01(R):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    boxes = [("1. Raw data", "Edge-IIoTset\nX-IIoTID\nCICIoT2023", "#DCE9F7"),
             ("2. Audit ladder", "raw \u2192 strict-clean\n\u2192 no-port \u2192\nnetwork-only", "#DCE9F7"),
             ("3. Taxonomy", "frozen; near = sibling\nin known pool,\nfar = no sibling", "#DCE9F7"),
             ("4. Family holdout", "train on known\nfamilies, hold\none family out", "#DCE9F7"),
             ("5. Scores", "RF, MLP softmax,\nenergy, Mahalanobis,\nrank fusion, EDL", "#FBE5CC")]
    fig, ax = plt.subplots(figsize=(W2, 1.65)); ax.set_xlim(-1.5, 102.0); ax.set_ylim(0, 25); ax.axis("off")
    w, g = 16.8, 4.0
    for i, (t, b, c) in enumerate(boxes):
        x = i * (w + g) + 0.5
        ax.add_patch(FancyBboxPatch((x, 8), w, 15, boxstyle="round,pad=0.2,rounding_size=0.8", fc=c, ec="black", lw=0.6))
        ax.text(x + w / 2, 20.3, t, ha="center", va="center", fontsize=6.6, fontweight="bold")
        ax.text(x + w / 2, 14.0, b, ha="center", va="center", fontsize=5.9, linespacing=1.3)
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.4, 15.5), (x + w + g - 0.4, 15.5), arrowstyle="-|>", mutation_scale=7, lw=0.8, color="black"))
    ax.text(1 * (w + g) + 0.5 + w / 2, 5.0, "closed-set Macro-F1, MCC, ECE per rung", ha="center", va="center", fontsize=6.0)
    ax.text(4 * (w + g) + 0.5 + w / 2 - 6, 4.0, "AUROC, AUPR, FPR@95TPR,\nvalidation-threshold detection;\nnear vs far", ha="center", va="center", fontsize=6.0, linespacing=1.3)
    save(fig, "fig01_safe_iiot_framework", "Fig. 1", "framework (static diagram, no data)", "figures/FIG1_SPEC.txt", len(boxes), "none", "n/a", "n/a", "n/a",
         "SAFE-IIoT evaluation framework: raw datasets, shortcut-audit ladder, frozen taxonomy, leave-one-family-out training, and scores with metrics.", W2)


def fig02(R):
    p = os.path.join(R, "tables", "T_ladder_closed_set.csv"); d = need(p, ["dataset", "rung", "model", "MacroF1", "MacroF1_sd", "n_seeds"])
    if d is None: return
    d = d[d.model.isin(MODELS)]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.3), sharey=True)
    for ax, ds in zip(axs, DATASETS):
        x = d[d.dataset == ds]
        for i, m in enumerate(MODELS):
            xm = x[x.model == m].set_index("rung").reindex(RUNG_ORDER)
            ax.errorbar(np.arange(4) + (i - 1) * 0.08, xm.MacroF1, yerr=xm.MacroF1_sd, marker="o", ms=3.5, lw=1, capsize=2, color=OKABE[i], label=MODEL_LABEL[m])
        ax.set_xticks(range(4)); ax.set_xticklabels([RUNG_LABEL[r] for r in RUNG_ORDER], rotation=30, ha="right"); ax.set_title(DS_LABEL[ds]); ax.set_ylim(0, 1.02)
    axs[0].set_ylabel("Macro-F1 (mean $\\pm$ sd over seeds)"); axs[0].legend(frameon=False, loc="lower left")
    save(fig, "fig02_shortcut_audit", "Fig. 2", "shortcut-audit ladder", "tables/T_ladder_closed_set.csv", len(d), "models=" + ",".join(MODELS), "MacroF1 mean+-sd over seeds",
         f"n={int(d.n_seeds.max())}", "all 3", "Closed-set Macro-F1 along the shortcut-audit ladder. Rungs are cumulative; a rung identical to its predecessor is not plotted (gap). Error bars: sd over seeds.", W2)


def fig03(R):
    p = os.path.join(R, "tables", "T_ladder_closed_set.csv"); d = need(p, ["dataset", "rung", "model", "MacroF1", "MacroF1_sd", "ECE"])
    if d is None: return
    last = d.sort_values("rung", key=lambda s: s.map({r: i for i, r in enumerate(RUNG_ORDER)})).groupby("dataset").rung.last()
    d = d[d.apply(lambda r: r.rung == last[r.dataset], axis=1)]
    fig, ax = plt.subplots(figsize=(W1, 2.4)); ms = ["rf", "lgbm", "mlp_softmax_en", "mlp_edl_both"]
    for i, m in enumerate(ms):
        v = d[d.model == m].set_index("dataset").reindex(DATASETS)
        ax.bar(np.arange(3) + (i - 1.5) * 0.2, v.MacroF1, 0.19, yerr=v.MacroF1_sd, capsize=1.5, color=OKABE[i], label=MODEL_LABEL[m], edgecolor="black", lw=0.3)
    ax.set_xticks(range(3)); ax.set_xticklabels([DS_LABEL[x] for x in DATASETS]); ax.set_ylabel("Macro-F1 (cleanest rung)"); ax.set_ylim(0, 1.02); ax.legend(frameon=False, ncol=2, loc="lower left")
    save(fig, "fig03_closed_set", "Fig. 3", "closed-set comparison on identical cleaned features", "tables/T_ladder_closed_set.csv", len(d), "cleanest rung per dataset", "MacroF1", "see table", "all 3",
         "Closed-set Macro-F1 of the classifiers on identical cleaned features (cleanest available rung per dataset). MLP-EDL is a control. Error bars: sd over seeds.", W1)


def fig04(R):
    p = os.path.join(R, "statistics", "loao_per_family.csv"); d = need(p, ["dataset", "condition", "method", "family", "near_far", "AUROC"])
    if d is None: return
    d = d[(d.condition == COND) & d.method.isin(PRIMARY)]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.4), sharey=True)
    for ax, ds in zip(axs, DATASETS):
        x = d[d.dataset == ds]
        sns.stripplot(data=x, x="method", y="AUROC", hue="near_far", order=PRIMARY, hue_order=["near", "far"], dodge=True, size=3, palette=[OKABE[0], OKABE[3]], ax=ax, jitter=0.15, linewidth=0.3, edgecolor="black")
        ax.set_title(DS_LABEL[ds]); ax.set_xlabel(""); ax.set_xticks(range(3)); ax.set_xticklabels([METHOD_LABEL[m].replace(" ", "\n") for m in PRIMARY]); ax.axhline(0.5, color="grey", lw=0.5, ls="--")
        ax.get_legend().remove() if ax.get_legend() else None
    axs[0].set_ylabel("AUROC per held-out family"); h = [plt.Line2D([], [], marker="o", ls="", color=OKABE[0]), plt.Line2D([], [], marker="o", ls="", color=OKABE[3])]
    axs[0].legend(h, ["near (sibling known)", "far (no sibling)"], frameon=False, loc="lower left")
    save(fig, "fig04_near_far_sibling", "Fig. 4", "near vs far family detection", "statistics/loao_per_family.csv", len(d), f"condition={COND}; methods=primary", "AUROC (mean over seeds, one dot per family)",
         "see file", "all 3", "Unseen-family detection AUROC for each held-out attack family, split by the frozen taxonomy into near (a sibling family remains in the known pool) and far (no sibling). Dashed line: chance.", W2)
    for ds in DATASETS:
        x = pd.read_csv(p); x = x[(x.dataset == ds) & (x.condition == COND) & x.method.isin(METHODS)]
        if x.empty: continue
        pv = x.pivot(index="family", columns="method", values="AUROC")[[m for m in METHODS if m in set(x.method)]]
        pv = pv.loc[pv.mean(1).sort_values().index]
        fig, ax = plt.subplots(figsize=(W2 * 0.7, 0.18 * len(pv) + 1.0))
        sns.heatmap(pv, annot=len(pv) <= 20, fmt=".2f", cmap="viridis", vmin=0, vmax=1, cbar_kws={"label": "AUROC"}, ax=ax, annot_kws={"size": 5})
        ax.set_xticklabels([METHOD_LABEL[m] for m in pv.columns], rotation=30, ha="right"); ax.set_ylabel(""); ax.set_xlabel("")
        save(fig, f"figS1_family_heatmap_{ds}", f"Fig. S1-{ds}", "per-family AUROC heatmap (supplement)", "statistics/loao_per_family.csv", len(x), f"dataset={ds}; condition={COND}", "AUROC", "see file", ds,
             f"Per-family AUROC of every score on {DS_LABEL[ds]} (attacks-only known pool). Families sorted by mean AUROC.", W2 * 0.7)


def fig05(R):
    p = os.path.join(R, "tables", "T_loao_method_summary.csv"); d = need(p, ["dataset", "condition", "method", "AUROC", "AUROC_CI_low", "AUROC_CI_high"])
    if d is None: return
    d = d[(d.condition == COND) & d.method.isin(METHODS + ["fusion_z"])]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2), sharex=True)
    for ax, ds in zip(axs, DATASETS):
        x = d[d.dataset == ds].set_index("method").reindex([m for m in METHODS + ["fusion_z"] if m in set(d.method)])
        y = np.arange(len(x)); ax.barh(y, x.AUROC, color=OKABE[0], edgecolor="black", lw=0.3)
        ax.errorbar(x.AUROC, y, xerr=[(x.AUROC - x.AUROC_CI_low).clip(lower=0).fillna(0), (x.AUROC_CI_high - x.AUROC).clip(lower=0).fillna(0)], fmt="none", ecolor="black", capsize=1.5, lw=0.7)
        ax.set_yticks(y); ax.set_yticklabels([METHOD_LABEL[m] for m in x.index] if ds == DATASETS[0] else []); ax.invert_yaxis(); ax.set_xlim(0.4, 1.0); ax.axvline(0.5, color="grey", ls="--", lw=0.5)
        ax.set_title(DS_LABEL[ds]); ax.set_xlabel("mean AUROC over families")
    save(fig, "fig05_ood_scores", "Fig. 5", "OOD score comparison", "tables/T_loao_method_summary.csv", len(d), f"condition={COND}", "AUROC mean over families; 95% t-CI over seeds", "loao seeds", "all 3",
         "Unseen-family detection AUROC (mean over held-out families) for each score. Bars: 95% interval over seeds (not over families). EDL vacuity is a control.", W2)


def fig07(R):
    p = os.path.join(R, "tables", "T_loao_failure_cases.csv"); d = need(p, ["dataset", "method", "family", "near_far", "AUROC"])
    if d is None: return
    d = d[d.method == "fusion_rank"] if "fusion_rank" in set(d.method) else d
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.0), sharex=True, gridspec_kw={"wspace": 1.15})
    for ax, ds in zip(axs, DATASETS):
        x = d[d.dataset == ds].sort_values("AUROC"); ax.tick_params(axis="y", labelsize=5.5); ax.barh(x.family, x.AUROC, color=[OKABE[0] if n == "near" else OKABE[3] for n in x.near_far], edgecolor="black", lw=0.3)
        ax.axvline(0.5, color="grey", ls="--", lw=0.5); ax.set_title(DS_LABEL[ds]); ax.set_xlim(0, 1); ax.set_xlabel("AUROC")
    save(fig, "fig07_failure_cases", "Fig. 7", "five lowest-AUROC held-out families per dataset", "tables/T_loao_failure_cases.csv", len(d), "bottom-5 families by AUROC, rank fusion", "AUROC", "loao seeds", "all 3",
         "Held-out families that are hardest to detect (rank fusion, attacks-only known pool). Blue: near; orange: far. Dashed line: chance.", W2)


def main():
    global OUT
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default=RESULTS_DIR); a = ap.parse_args(); R = a.results; OUT = R
    for f in (fig01, fig02, fig03, fig04, fig05, fig07):
        try: f(R)
        except Exception as e: ERRORS.append(f"{f.__name__} failed: {e!r}")
    fd = os.path.join(OUT, "figures"); os.makedirs(fd, exist_ok=True)
    open(os.path.join(fd, "FIG1_SPEC.txt"), "w").write(FIG1_SPEC)
    pd.DataFrame(MANIFEST).to_csv(os.path.join(fd, "figure_manifest.csv"), index=False)
    pd.DataFrame(AUDIT).to_csv(os.path.join(fd, "MANUSCRIPT_FIGURE_DATA_AUDIT.csv"), index=False)
    open(os.path.join(fd, "figure_captions.txt"), "w").write("\n\n".join(f"{k}: {v}" for k, v in CAPTIONS.items()))
    z = os.path.join(OUT, "SAFE_IIoT_figures.zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in sorted(os.listdir(fd)): zf.write(os.path.join(fd, fn), "figures/" + fn)
    print("generated:", [m["file"] for m in MANIFEST], "->", z)
    if ERRORS:
        print("\n" + "\n".join(ERRORS)); sys.exit(1)


if __name__ == "__main__":
    main()
