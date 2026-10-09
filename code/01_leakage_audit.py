"""STAGE 1 -- LEAKAGE AUDIT (CPU only). Experiment name: LEAK_AUDIT_V1 / CLEAN_<DATASET>_V1
1A single-feature label probes   1B artifact-specific token probes   1C dataset variants A/B/C/D
1D exact + near-duplicate train/test overlap   1E split-protocol comparison (random/grouped/temporal)
Ends with a provisional GO/NO-GO case (A/B/C) computed from the numbers, not from opinion."""
import os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from common import *

NAME = "LEAK_AUDIT_V1"


def bin_metrics(y, p, score=None):
    d = {"F1": f1_score(y, p, zero_division=0), "MCC": matthews_corrcoef(y, p),
         "Acc": accuracy_score(y, p), "BalAcc": balanced_accuracy_score(y, p)}
    try:
        d["AUROC"] = roc_auc_score(y, score if score is not None else p)
    except ValueError:
        d["AUROC"] = np.nan
    return d


def lookup(codes_tr, y_tr, codes_te, card, n_cls):
    cnt = np.bincount(codes_tr * n_cls + y_tr, minlength=card * n_cls).reshape(card, n_cls)
    seen = cnt.sum(1) > 0
    glob = np.bincount(y_tr, minlength=n_cls).argmax()
    pred = np.where(seen[codes_te], cnt.argmax(1)[codes_te], glob)
    return pred, cnt


def suspicion(m, th):
    return ("CRITICAL" if m >= th["critical"] else "HIGH" if m >= th["high"]
            else "MODERATE" if m >= th["moderate"] else "low")


def stratified_idx(ds, n, seed):
    idx = np.arange(len(ds.y_bin))
    if n and n < len(idx):
        idx, _ = train_test_split(idx, train_size=n, stratify=_strat(ds), random_state=seed)
    return np.sort(idx)


def _strat(ds):
    vc = pd.Series(ds.y_family).value_counts()
    ok = pd.Series(ds.y_family).map(vc).to_numpy() >= 10
    return np.where(ok, ds.y_family, "__rare__")


def fit_eval(X, yf, tr, te, seed, iters, normal_mask_classes):
    """Multiclass HistGB with balanced sample weights; returns family Macro-F1, family MCC, binary MCC."""
    classes, yi = np.unique(yf, return_inverse=True)
    m = HistGradientBoostingClassifier(max_iter=iters, learning_rate=0.1, random_state=seed)
    w = compute_sample_weight("balanced", yi[tr])
    if len(classes) > 20:
        w = w ** 0.5; w = w / w.mean()
    m.fit(X[tr], yi[tr], sample_weight=w)
    p = m.predict(X[te])
    present = np.unique(yi[te])
    macro = f1_score(yi[te], p, labels=present, average="macro", zero_division=0)
    mcc = matthews_corrcoef(yi[te], p)
    nm = np.isin(classes, list(normal_mask_classes))
    bm = matthews_corrcoef(~nm[yi[te]], ~nm[p])
    return macro, mcc, bm


def main(cfg, name):
    ds = get_dataset(cfg, name, cfg["seeds"][0])
    dc = ds.cfg
    out = exp_dir(cfg, f"leak_audit/{name}")
    mt = os.path.join(out, "metrics"); pl = os.path.join(out, "plots")
    manuscript = os.path.join(results_root(cfg), "manuscript_tables")
    L = lambda m: log(cfg, "01_leakage_audit", f"[{name}] {m}")
    th = cfg["suspicion"]; seed = cfg["seeds"][0]
    normal_cls = {c for c in np.unique(ds.y_family) if c.lower() in {t.lower() for t in dc["normal_tokens"]}}

    pidx = stratified_idx(ds, rows_for(cfg, ds, "probe_rows"), seed)
    tr_i, te_i = train_test_split(np.arange(len(pidx)), test_size=cfg["test_size"],
                                  stratify=_strat(ds)[pidx], random_state=seed)
    sub = ds.df.iloc[pidx]
    yb, yf = ds.y_bin[pidx], ds.y_family[pidx]
    fam_classes, yfi = np.unique(yf, return_inverse=True)
    K = len(fam_classes)
    rows = []
    for c in ds.feature_cols:
        s = sub[c]; card = len(s.cat.categories); codes = s.cat.codes.to_numpy().astype(np.int64)
        nv = numeric_view(s)
        if card * max(K, 2) <= 5e7:
            pb, cb = lookup(codes[tr_i], yb[tr_i], codes[te_i], card, 2)
            rate = (cb[:, 1] / np.maximum(cb.sum(1), 1)); prior = yb[tr_i].mean()
            score = np.where(cb.sum(1)[codes[te_i]] > 0, rate[codes[te_i]], prior)
            pf, _ = lookup(codes[tr_i], yfi[tr_i], codes[te_i], card, K)
            r = bin_metrics(yb[te_i], pb[...], score)
            r.update(feature=c, view="raw_token", MacroF1_multiclass=f1_score(
                yfi[te_i], pf, labels=np.unique(yfi[te_i]), average="macro", zero_division=0))
            rows.append(r)
        if nv is not None:
            x = np.nan_to_num(nv, nan=-9e9).reshape(-1, 1)
            t = DecisionTreeClassifier(max_depth=3, random_state=seed).fit(x[tr_i], yb[tr_i])
            r = bin_metrics(yb[te_i], t.predict(x[te_i]), t.predict_proba(x[te_i])[:, -1])
            tf = DecisionTreeClassifier(max_depth=3, random_state=seed).fit(x[tr_i], yfi[tr_i])
            r.update(feature=c, view="canonical_numeric", MacroF1_multiclass=f1_score(
                yfi[te_i], tf.predict(x[te_i]), labels=np.unique(yfi[te_i]), average="macro", zero_division=0))
            rows.append(r)
    probes = pd.DataFrame(rows)
    probes.to_csv(os.path.join(mt, "single_feature_probes_all_views.csv"), index=False)
    best = probes.sort_values("MCC", ascending=False).drop_duplicates("feature")
    best["Suspicion"] = best.MCC.apply(lambda m: suspicion(m, th))
    tab = best[["feature", "view", "F1", "MCC", "BalAcc", "AUROC", "MacroF1_multiclass", "Suspicion"]].rename(
        columns={"feature": "Feature", "F1": "Single-feature F1", "MacroF1_multiclass": "Macro-F1 (multiclass)"})
    write_table(tab, manuscript, f"{name}_single_feature_ranking")
    tab.to_csv(os.path.join(mt, "single_feature_ranking.csv"), index=False)
    top = tab.head(25)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(top.Feature[::-1], top.MCC[::-1], color=["#C44E52" if s in ("CRITICAL", "HIGH") else "#4C72B0"
                                                     for s in top.Suspicion[::-1]], edgecolor="black", linewidth=.5)
    ax.set_xlabel("single-feature binary MCC (Normal vs Attack)"); ax.set_title(f"{name}: leakage feature ranking")
    save_fig(fig, os.path.join(pl, "leakage_feature_ranking"))
    L("1A done. Top: " + "; ".join(f"{r.Feature}({r.view}) MCC={r.MCC:.3f}" for r in tab.head(5).itertuples()))

    art_cols = [c for c in dc.get("artifact_columns", []) if c in ds.feature_cols]
    alias_flag_p = os.path.join(results_root(cfg), f"dataset_audit/{name}/metrics/alias_flagged_columns.csv")
    alias_flagged = list(pd.read_csv(alias_flag_p)["column"]) if os.path.exists(alias_flag_p) else []
    suspected = sorted(set(art_cols) | set(alias_flagged))
    rows = []
    for c in suspected:
        s = ds.df[c]; cats = s.cat.categories.astype(str); codes = s.cat.codes.to_numpy()
        cnt = np.bincount(codes, minlength=len(cats))
        for t in np.argsort(-cnt)[:10]:
            if cnt[t] < cfg["min_token_support"]:
                continue
            d = (codes == t).astype(int)
            mcc = matthews_corrcoef(ds.y_bin, d)
            rows.append({"column": c, "dummy_feature": f"{c}_{cats[t]}", "token": cats[t], "rows": int(cnt[t]),
                         "attack_rate_when_token": float(ds.y_bin[codes == t].mean()),
                         "binary_MCC_as_attack_flag": mcc, "orientation": "attack" if mcc >= 0 else "normal(inverted)",
                         "abs_MCC": abs(mcc)})
    art = pd.DataFrame(rows)
    art.to_csv(os.path.join(mt, "artifact_token_probes.csv"), index=False)
    if len(art):
        write_table(art, manuscript, f"{name}_artifact_token_probes")
    L(f"1B done. suspected columns = {suspected}")

    canon = best[best.view == "canonical_numeric"]
    strong = set(canon[canon.MCC >= cfg["strong_mcc_for_strict_drop"]].feature)
    feat_rep = pd.read_csv(os.path.join(results_root(cfg), f"dataset_audit/{name}/metrics/feature_report.csv"))
    ident = set(feat_rep[(feat_rep.identifier_like | feat_rep.timestamp_like) & (feat_rep.type != "numeric")].feature) | set(dc.get("identifier_columns", []))
    pats = [re.compile(p) for p in dc.get("strict_drop_patterns", [])]
    patmatch = {c for c in ds.feature_cols if any(p.search(c) for p in pats)}
    consts = set(feat_rep[feat_rep.constant].feature)
    sus, cols = set(suspected), ds.feature_cols
    D_drop = sus | strong | ident | patmatch | consts
    variants = {
        "A_original": dict(cols=cols, raw=set(cols)),
        "B_artifact_removed": dict(cols=[c for c in cols if c not in sus], raw=set(cols) - sus),
        "C_artifact_neutralized": dict(cols=cols, raw=set(cols) - sus),
        "D_strict_clean": dict(cols=[c for c in cols if c not in D_drop], raw=set()),
    }
    reasons = [{"feature": c, "in_variant_D_removed": c in D_drop, "suspected_artifact": c in sus,
                "strong_single_feature(canonical)": c in strong, "identifier_or_time": c in ident,
                "strict_pattern": c in patmatch, "constant": c in consts} for c in cols]
    pd.DataFrame(reasons).to_csv(os.path.join(mt, "variant_feature_decisions.csv"), index=False)

    vidx = stratified_idx(ds, rows_for(cfg, ds, "variant_rows"), seed)
    vsub = ds.df.iloc[vidx]; vyf = ds.y_family[vidx]
    seeds = cfg["seeds"]
    res = []
    for vn, v in variants.items():
        X = encode_matrix(vsub, v["cols"], v["raw"])
        for sd in seeds:
            tr, te = train_test_split(np.arange(len(vidx)), test_size=cfg["test_size"],
                                      stratify=_strat(ds)[vidx], random_state=sd)
            ma, mc, bm = fit_eval(X, vyf, tr, te, sd, cfg["hgb_max_iter"], normal_cls)
            res.append({"variant": vn, "seed": sd, "n_features": len(v["cols"]), "MacroF1": ma, "MCC": mc, "BinaryMCC": bm})
            L(f"1C {vn} seed={sd}: MacroF1={ma:.4f} MCC={mc:.4f} binMCC={bm:.4f}")
    rv = pd.DataFrame(res); rv.to_csv(os.path.join(mt, "variant_results_by_seed.csv"), index=False)
    g = rv.groupby("variant").agg(MacroF1=("MacroF1", "mean"), MacroF1_sd=("MacroF1", "std"),
                                  MCC=("MCC", "mean"), BinaryMCC=("BinaryMCC", "mean"),
                                  n_features=("n_features", "first")).reset_index()
    a = g.loc[g.variant == "A_original", "MacroF1"].iloc[0]
    maxmcc = tab.MCC.max()
    g["Delta_MacroF1_vs_A"] = g.MacroF1 - a
    g["Leakage Risk"] = [f"{suspicion(maxmcc, th)} (max single-feature MCC={maxmcc:.3f})" if v == "A_original"
                         else "cleaned reference" for v in g.variant]
    write_table(g, manuscript, f"{name}_variant_results")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(g.variant, g.MacroF1, yerr=g.MacroF1_sd.fillna(0), color="#4C72B0", edgecolor="black", linewidth=.5, capsize=3)
    ax.set_ylabel("Macro-F1 (HistGB, balanced weights)"); ax.set_ylim(0, 1.05)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right"); ax.set_title(f"{name}: before/after leakage removal")
    save_fig(fig, os.path.join(pl, "variants_before_after"))

    base = ds.df[[c for c in cols if c not in ident]]
    h = pd.util.hash_pandas_object(base, index=False).to_numpy()
    itr, ite = train_test_split(np.arange(len(h)), test_size=cfg["test_size"], stratify=_strat(ds), random_state=seed)
    in_train = np.isin(h[ite], h[itr])
    exact_overlap = float(in_train.mean())
    by_cls = pd.DataFrame({"class": ds.y_family[ite], "dup_in_train": in_train}).groupby("class").dup_in_train.mean()
    by_cls.to_csv(os.path.join(mt, "exact_duplicate_overlap_by_class.csv"))
    Xd = encode_matrix(ds.df.iloc[np.sort(np.concatenate([itr[:50000], ite[:5000]]))],
                       [c for c in cols if c not in D_drop] or cols, set())
    ntr = min(50000, len(itr))
    Xd = np.nan_to_num(StandardScaler().fit(Xd[:ntr]).transform(Xd))
    nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=-1).fit(Xd[:ntr])
    dist, _ = nn.kneighbors(Xd[ntr:])
    near = {"exact_row_overlap_test_in_train(content cols)": exact_overlap,
            "median_nn_dist_std_units": float(np.median(dist)),
            "frac_nn_dist_lt_0.05": float((dist < 0.05).mean()), "frac_nn_dist_lt_0.5": float((dist < 0.5).mean())}
    pd.DataFrame([near]).to_csv(os.path.join(mt, "near_duplicate_summary.csv"), index=False)
    L(f"1D done: {near}")

    splits = {"random_stratified": None}
    gcols = [c for c in dc.get("group_cols", []) if c in cols]
    if gcols:
        splits["grouped(" + "+".join(gcols) + ")"] = pd.factorize(
            vsub[gcols].astype(str).agg("|".join, axis=1))[0]
    tc = dc.get("time_col", "")
    tnum = None
    if tc and tc in ds.df.columns:
        t = pd.to_datetime(ds.df[tc].iloc[vidx].astype(str), errors="coerce")
        if t.notna().mean() > 0.9:
            tnum = t.astype("int64").to_numpy()
        else:
            nvv = numeric_view(vsub[tc]) if tc in vsub else None
            tnum = nvv if nvv is not None and np.isfinite(nvv).mean() > 0.9 else None
    if tnum is not None:
        splits["temporal(last 20%)"] = "time"
    rows = []
    for sn, spec in splits.items():
        for vn in ("A_original", "D_strict_clean"):
            v = variants[vn]; X = encode_matrix(vsub, v["cols"], v["raw"])
            if spec is None:
                tr, te = train_test_split(np.arange(len(vidx)), test_size=cfg["test_size"],
                                          stratify=_strat(ds)[vidx], random_state=seed)
            elif isinstance(spec, str):
                o = np.argsort(np.nan_to_num(tnum, nan=0), kind="stable"); k = int(len(o) * (1 - cfg["test_size"]))
                tr, te = o[:k], o[k:]
            else:
                tr, te = next(GroupShuffleSplit(1, test_size=cfg["test_size"], random_state=seed).split(X, groups=spec))
            ma, mc, bm = fit_eval(X, vyf, tr, te, seed, cfg["hgb_max_iter"], normal_cls)
            rows.append({"split": sn, "variant": vn, "MacroF1": ma, "MCC": mc, "BinaryMCC": bm,
                         "test_classes_present": len(np.unique(vyf[te]))})
    sp = pd.DataFrame(rows); write_table(sp, manuscript, f"{name}_split_protocols")
    L("1E done:\n" + sp.to_string(index=False))

    A = g.loc[g.variant == "A_original", "MacroF1"].iloc[0]
    Dm = g.loc[g.variant == "D_strict_clean", "MacroF1"].iloc[0]
    gap = A - Dm
    flagged_feat = tab[tab.Suspicion.isin(["CRITICAL", "HIGH"])].Feature.tolist()
    shortcut = [f for f in flagged_feat if f in ident or f in patmatch or f in sus or re.search(r"port", f, re.I)]
    informative = [f for f in flagged_feat if f not in shortcut]
    leak = bool(shortcut or alias_flagged)
    gcfg = cfg["gate"]
    if leak and gap >= gcfg["collapse_gap"] and Dm < gcfg["collapse_floor"]:
        case = "CASE A -- severe leakage; performance collapses after cleaning"
        act = "Do NOT hide this. Re-centre the paper on leakage-aware benchmarking/robust IIoT IDS."
    elif leak:
        case = "CASE B -- leakage present; model must be re-tested on the clean variant"
        act = "Proceed with variant D/C only. Stage 2 decides whether the proposed method stays strong."
    else:
        case = "CASE C -- no practically important leakage detected"
        act = "Proceed with the original hypothesis; cite this audit as dataset-validity evidence."
    answer = ("YES -- practically important label leakage / serialisation artifact detected."
              if leak else "NO practically important single-feature leakage detected by these probes.")
    tbl = open(os.path.join(manuscript, f"{name}_variant_results.md")).read()
    md = (f"### Stage 1 -- {name}\n\n**Does the dataset contain a practically important label leakage/serialisation "
          f"artifact?** {answer}\n\n- CRITICAL/HIGH single-feature MCC that are artifact-type shortcuts (identifier/time/port/alias): {shortcut or 'none'}\n"
          f"- CRITICAL/HIGH but ordinary informative features (NOT counted as leakage): {informative or 'none'}\n"
          f"- Alias-flagged columns (same number, different spelling, label-separating): {alias_flagged or 'none'}\n"
          f"- Macro-F1 original (A) = {A:.4f}; strict clean (D) = {Dm:.4f}; gap = {gap:.4f}\n"
          f"- Exact content overlap test-in-train (random split) = {100*exact_overlap:.2f}%\n"
          f"- **Provisional gate: {case}.** {act}\n\n" + tbl)
    open(os.path.join(out, "STAGE1_DECISION.md"), "w").write(md)
    L(f"DECISION: {case}")
    summary = {"dataset": name, "leakage_detected": leak, "informative_high_mcc_features": ";".join(informative), "max_single_feature_MCC": float(maxmcc),
               "MacroF1_A": float(A), "MacroF1_D": float(Dm), "gap": float(gap), "gate": case}
    pd.DataFrame([summary]).to_csv(os.path.join(mt, "summary.csv"), index=False)
    return summary, md
