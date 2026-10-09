"""STAGE 0 -- DATASET FORENSICS (CPU only). Experiment name: DATASET_AUDIT_V1
Per dataset: shape, types, classes, missing/inf tokens, duplicates, constant columns, cardinality,
identifier/timestamp/target-derived/protocol fields and -- most importantly -- TOKEN ALIASES
(different spellings of the same number, e.g. '0' vs '0.0') whose spelling predicts the label."""
import os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

NAME = "DATASET_AUDIT_V1"
ID_RE = re.compile(r"(^|[._])(ip|addr|address|host|mac|id|uid|flow_?id|session|src_?ip|dst_?ip)([._]|$)", re.I)
TIME_RE = re.compile(r"(time|timestamp|date|epoch)", re.I)
TARGET_RE = re.compile(r"(label|class|attack|target|malicious|category)", re.I)


def canonical(tok):
    try:
        v = float(tok)
        return repr(v) if np.isfinite(v) else tok
    except ValueError:
        return tok


def token_alias_report(ds, min_support, max_card=2000):
    """Tokens that parse to the same number but are spelled differently, with their attack rate."""
    rows, flagged = [], {}
    yb = ds.y_bin
    for c in ds.feature_cols:
        s = ds.df[c]
        cats = s.cat.categories.astype(str)
        if len(cats) > max_card or len(cats) < 2:
            continue
        codes = s.cat.codes.to_numpy()
        cnt = np.bincount(codes, minlength=len(cats))
        att = np.bincount(codes, weights=yb, minlength=len(cats))
        groups = {}
        for i, t in enumerate(cats):
            groups.setdefault(canonical(t), []).append(i)
        for k, idx in groups.items():
            if len(idx) < 2:
                continue
            rates = {cats[i]: (att[i] / cnt[i] if cnt[i] else np.nan) for i in idx}
            for i in idx:
                rows.append({"column": c, "canonical_value": k, "token": cats[i], "rows": int(cnt[i]),
                             "attack_rate": rates[cats[i]]})
            big = [i for i in idx if cnt[i] >= min_support]
            if len(big) >= 2:
                r = [att[i] / cnt[i] for i in big]
                if max(r) - min(r) >= 0.9:
                    flagged[c] = flagged.get(c, 0) + 1
    return pd.DataFrame(rows), sorted(flagged)


def main(cfg, name):
    ds = get_dataset(cfg, name, cfg["seeds"][0])
    out = exp_dir(cfg, f"dataset_audit/{name}")
    L = lambda m: log(cfg, "00_dataset_audit", f"[{name}] {m}")
    n = len(ds.df)
    L(f"{n:,} rows, {len(ds.feature_cols)} features, files={[os.path.basename(f) for f in ds.files]}")

    cls = pd.Series(ds.y_family).value_counts()
    class_df = pd.DataFrame({"class": cls.index, "count": cls.values, "percent": 100 * cls.values / n})
    write_table(class_df, os.path.join(out, "metrics"), "class_distribution")
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(cls) + 1.5))
    ax.barh(class_df["class"][::-1], class_df["count"][::-1], color="#4C72B0", edgecolor="black", linewidth=.5)
    ax.set_xscale("log"); ax.set_xlabel("rows (log scale)"); ax.set_title(f"{name}: class distribution")
    save_fig(fig, os.path.join(out, "plots", "class_distribution"))

    time_col = ds.cfg.get("time_col", "")
    rows = []
    for k, c in enumerate(ds.feature_cols):
        L(f"feature {k+1}/{len(ds.feature_cols)}: {c} (card={len(ds.df[c].cat.categories):,}, peak RAM {rss_gb():.1f} GB)")
        s = ds.df[c]
        card = len(s.cat.categories)
        codes = s.cat.codes.to_numpy()
        cnt = np.bincount(codes, minlength=card)
        if card > 200000:
            cats_ix = s.cat.categories
            miss = int(cnt[cats_ix.isin(list(MISSING)).astype(bool)].sum())
            inf = int(cnt[cats_ix.isin(list(INF)).astype(bool)].sum())
        else:
            toks = pd.Series(s.cat.categories.astype(str))
            miss = int(cnt[toks.str.strip().str.lower().isin(MISSING).to_numpy()].sum())
            inf = int(cnt[toks.str.lower().isin(INF).to_numpy()].sum())
        nv = numeric_view(s)
        ident = bool(ID_RE.search(c)) or card > 0.5 * n
        rows.append({
            "feature": c, "type": "numeric" if nv is not None else "categorical/string",
            "cardinality": card, "cardinality_ratio": card / n, "top_token_share": cnt.max() / n,
            "missing_tokens": miss, "inf_tokens": inf, "constant": card <= 1,
            "identifier_like": ident, "timestamp_like": bool(TIME_RE.search(c)) or c == time_col,
            "target_derived_name": bool(TARGET_RE.search(c)), "protocol_prefix": c.split(".")[0] if "." in c else ""})
    feat = pd.DataFrame(rows)
    feat.to_csv(os.path.join(out, "metrics", "feature_report.csv"), index=False)
    feat.sort_values("cardinality", ascending=False).head(40).to_csv(
        os.path.join(out, "metrics", "feature_cardinality_report.csv"), index=False)
    feat[feat.missing_tokens + feat.inf_tokens > 0][["feature", "missing_tokens", "inf_tokens"]].to_csv(
        os.path.join(out, "metrics", "missing_value_report.csv"), index=False)
    feat[feat.identifier_like | feat.timestamp_like | feat.target_derived_name | feat.constant].to_csv(
        os.path.join(out, "metrics", "suspicious_feature_report.csv"), index=False)

    L(f"duplicates: hashing rows (peak RAM {rss_gb():.1f} GB)")
    h = pd.util.hash_pandas_object(ds.df, index=False).to_numpy()
    ycode = pd.factorize(ds.y_family)[0]
    key = pd.DataFrame({"h": h, "y": ycode})
    dmask = key.duplicated().to_numpy()
    dup = int(dmask.sum())
    dup_by_class = pd.Series(ds.y_family[dmask]).value_counts()
    pd.DataFrame({"class": dup_by_class.index, "duplicate_rows": dup_by_class.values}).to_csv(
        os.path.join(out, "metrics", "duplicate_report.csv"), index=False)
    del key, h

    alias, flagged = token_alias_report(ds, cfg["min_token_support"])
    alias.to_csv(os.path.join(out, "metrics", "token_alias_report.csv"), index=False)
    pd.Series(flagged, dtype=str).to_csv(os.path.join(out, "metrics", "alias_flagged_columns.csv"),
                                         index=False, header=["column"])
    L(f"duplicates={dup:,} ({100*dup/n:.2f}%), constant cols={int(feat.constant.sum())}, "
      f"alias-flagged label-separating columns={flagged}")

    fig, ax = plt.subplots(figsize=(7, 6))
    t = feat.sort_values("cardinality", ascending=False).head(30)
    ax.barh(t.feature[::-1], t.cardinality[::-1], color="#DD8452", edgecolor="black", linewidth=.5)
    ax.set_xscale("log"); ax.set_xlabel("distinct raw tokens"); ax.set_title(f"{name}: highest-cardinality fields")
    save_fig(fig, os.path.join(out, "plots", "feature_cardinality"))

    summary = {"dataset": name, "rows": n, "features": len(ds.feature_cols), "classes": len(cls),
               "normal_percent": float(100 * (ds.y_bin == 0).mean()), "duplicate_rows": dup,
               "duplicate_percent": 100 * dup / n, "constant_columns": int(feat.constant.sum()),
               "identifier_like_columns": int(feat.identifier_like.sum()),
               "alias_flagged_columns": ";".join(flagged)}
    pd.DataFrame([summary]).to_csv(os.path.join(out, "metrics", "summary.csv"), index=False)
    md = (f"### Stage 0 -- {name}\n\n{n:,} rows, {len(ds.feature_cols)} features, {len(cls)} classes; "
          f"Normal = {summary['normal_percent']:.2f}%; exact duplicates = {dup:,} ({100*dup/n:.2f}%); "
          f"identifier-like columns (heuristic, informational) = {summary['identifier_like_columns']}.\n\n"
          f"Sampling: {ds.sampling_note}.\n\n"
          f"**Serialisation-alias columns whose spelling separates Normal from Attack:** "
          f"{flagged if flagged else 'none detected'}.\n")
    return summary, md
