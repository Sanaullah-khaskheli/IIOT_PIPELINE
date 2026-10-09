"""Shared utilities for the IIoT-IDS pipeline (paths, logging, loading, encoding, tables)."""
import os, re, glob, json, time, zipfile, random, traceback, datetime as dt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MISSING = {"", "nan", "na", "n/a", "null", "none", "?"}
INF = {"inf", "-inf", "infinity", "-infinity", "+inf"}


def load_config(path=None):
    import yaml
    with open(path or os.path.join(HERE, "config.yaml")) as f:
        return yaml.safe_load(f)


def results_root(cfg):
    base = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
    root = os.path.join(base, cfg.get("results_dir", "results"))
    os.makedirs(root, exist_ok=True)
    return root


def exp_dir(cfg, name, subs=("plots", "metrics", "logs")):
    d = os.path.join(results_root(cfg), name)
    for s in subs:
        os.makedirs(os.path.join(d, s), exist_ok=True)
    return d


def log(cfg, name, msg):
    line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    d = os.path.join(results_root(cfg), "logs")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{name}.log"), "a") as f:
        f.write(line + "\n")


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def record_failure(cfg, experiment, exc):
    p = os.path.join(results_root(cfg), "failed_experiments.csv")
    row = pd.DataFrame([{"experiment": experiment, "error": repr(exc),
                         "traceback": traceback.format_exc(),
                         "timestamp": dt.datetime.now().isoformat(timespec="seconds")}])
    row.to_csv(p, mode="a", header=not os.path.exists(p), index=False)
    print(f"\nFAILED: {experiment}\n  error: {exc!r}\n  see failed_experiments.csv "
          f"(the pipeline continues; results of other experiments remain valid)\n", flush=True)


def zip_dir(src_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for r, _, fs in os.walk(src_dir):
            for f in fs:
                full = os.path.join(r, f)
                if os.path.abspath(full) == os.path.abspath(zip_path):
                    continue
                z.write(full, os.path.relpath(full, os.path.dirname(src_dir)))
    return zip_path


def save_fig(fig, stem):
    fig.savefig(stem + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _tex_escape(s):
    return str(s).replace("\\", "/").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def write_table(df, out_dir, stem, floatfmt="{:.4f}"):
    """Every manuscript number must come from here: CSV + Markdown + LaTeX."""
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, stem + ".csv"), index=False)
    f = lambda v: floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)
    cols = list(df.columns)
    md = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    tex = ["\\begin{tabular}{" + "l" * len(cols) + "}", "\\hline",
           " & ".join(_tex_escape(c) for c in cols) + " \\\\", "\\hline"]
    for _, r in df.iterrows():
        md.append("| " + " | ".join(f(v) for v in r.values) + " |")
        tex.append(" & ".join(_tex_escape(f(v)) for v in r.values) + " \\\\")
    tex += ["\\hline", "\\end{tabular}"]
    open(os.path.join(out_dir, stem + ".md"), "w").write("\n".join(md) + "\n")
    open(os.path.join(out_dir, stem + ".tex"), "w").write("\n".join(tex) + "\n")


class Dataset:
    pass


_CACHE = {}


def get_dataset(cfg, name, seed=42):
    if name not in _CACHE:
        _CACHE[name] = load_dataset(cfg, name, seed)
    return _CACHE[name]


def load_dataset(cfg, name, seed=42):
    """Reads EVERY column as a raw string category (dtype='category', no NA parsing) so that
    serialisation differences such as '0' vs '0.0' are preserved for the forensic audit."""
    dc = cfg["datasets"][name]
    files = sorted({f for pat in dc["search"] for f in glob.glob(pat, recursive=True)})
    if not files:
        raise FileNotFoundError(f"[{name}] no CSV found for {dc['search']}. Add the Kaggle dataset "
                                f"and/or fix 'search' in config.yaml")
    frac = float(dc.get("sample_frac", 1.0))
    rng = np.random.default_rng(seed)
    cap = dc.get("class_cap")
    sampling_note = f"uniform sample_frac={frac}"
    keep_prob = None
    if cap:
        gmap, cnt, codes_per_file = {}, {}, []
        for f in files:
            hdr = [str(c).strip() for c in pd.read_csv(f, nrows=0).columns]
            lab = next((c for c in dc["family_col_candidates"] if c in hdr), None)
            if lab is None:
                raise KeyError(f"[{name}] label column not found in {os.path.basename(f)}: {hdr[:20]}")
            col = pd.read_csv(f, usecols=lambda c, lab=lab: str(c).strip() == lab, dtype="category",
                              keep_default_na=False, na_filter=False).iloc[:, 0]
            cats = [str(c).strip() for c in col.cat.categories]
            ix = np.array([gmap.setdefault(c, len(gmap)) for c in cats], dtype=np.int32)
            codes = ix[col.cat.codes.to_numpy()]
            codes_per_file.append(codes)
            for k, v in zip(*np.unique(codes, return_counts=True)):
                cnt[k] = cnt.get(k, 0) + int(v)
        keep_prob = np.zeros(len(gmap))
        for k, v in cnt.items():
            keep_prob[k] = min(1.0, cap / v)
        sampling_note = (f"class-capped (<= {cap} rows per class, rare classes kept whole) from {sum(cnt.values()):,} rows; "
                         f"class priors are therefore NOT the natural ones")
    frames = []
    for fi, f in enumerate(files):
        kw = dict(dtype="category", keep_default_na=False, na_filter=False, low_memory=False)
        if keep_prob is not None:
            keep = rng.random(len(codes_per_file[fi])) < keep_prob[codes_per_file[fi]]
            kw["skiprows"] = (np.flatnonzero(~keep) + 1).tolist()
        elif frac < 1:
            kw["skiprows"] = lambda i: i > 0 and rng.random() > frac
        d = pd.read_csv(f, **kw)
        d.columns = [str(c).strip() for c in d.columns]
        frames.append(d)
    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    if len(frames) > 1:
        df = df.astype("category")
    present = [c for c in dc["family_col_candidates"] if c in df.columns]
    if dc.get("family_pick") == "most_classes" and present:
        fam_col = max(present, key=lambda c: df[c].nunique())
        print(f"[{name}] family column = {fam_col} "
              f"({ {c: int(df[c].nunique()) for c in present} } distinct values per candidate)", flush=True)
    else:
        fam_col = present[0] if present else None
    if fam_col is None:
        raise KeyError(f"[{name}] none of {dc['family_col_candidates']} in columns: {list(df.columns)[:40]}...")
    y_f = df[fam_col].astype(str).str.strip().to_numpy()
    normal = {t.lower() for t in dc["normal_tokens"]}
    y_b = (~pd.Series(y_f).str.lower().isin(normal)).to_numpy().astype(np.int8)
    drop = {c for c in dc.get("label_cols", []) if c in df.columns} | {fam_col}
    ds = Dataset()
    ds.name, ds.cfg = name, dc
    ds.feature_cols = [c for c in df.columns if c not in drop]
    ds.df = df[ds.feature_cols]
    ds.y_family, ds.y_bin = y_f, y_b
    ds.family_col, ds.files, ds.sampling_note = fam_col, files, sampling_note
    return ds


_NUM_CACHE = {}


def rss_gb():
    try:
        import resource, sys
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / (1024 ** 2 if sys.platform != "darwin" else 1024 ** 3)
    except Exception:
        return float("nan")


def numeric_view(s, thresh=0.98):
    """Numeric parse of a raw-string category column (None if it is not numeric-like).
    The category->number map is cached per column, so subsets reuse it (big speed-up)."""
    key = (s.name, len(s.cat.categories))
    if key not in _NUM_CACHE:
        cats = pd.Series(s.cat.categories.astype(str))
        short = (cats.str.len() <= 25).to_numpy()
        num = np.full(len(cats), np.nan)
        if short.mean() >= thresh:
            num[short] = pd.to_numeric(cats[short], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(num)
        _NUM_CACHE[key] = (np.where(ok, num, np.nan), len(cats) > 0 and ok.mean() >= thresh)
    mapped, numeric = _NUM_CACHE[key]
    if not numeric:
        return None
    codes = s.cat.codes.to_numpy()
    out = np.full(len(codes), np.nan)
    m = codes >= 0
    out[m] = mapped[codes[m]]
    return out


def encode_matrix(df, cols, raw_cols, lowcard=64):
    """float32 matrix. 'raw' encoding keeps token spelling for low-cardinality columns (this
    reproduces the one-hot behaviour that exposes '0' vs '0.0'); 'canonical' merges spellings."""
    X = np.empty((len(df), len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        s = df[c]
        nv = numeric_view(s)
        codes = s.cat.codes.to_numpy().astype(np.float32)
        if c in raw_cols:
            X[:, j] = codes if (nv is None or len(s.cat.categories) <= lowcard) else nv
        else:
            X[:, j] = nv if nv is not None else codes
    return X


def clean_variant_cols(cfg, name, ds):
    """Clean feature sets decided by Stage 1:  D = strict clean;  E3 = D without port columns;
    E4 = E3 without host-telemetry / host-alert columns (only if the dataset config lists telemetry_patterns)."""
    p = os.path.join(results_root(cfg), f"leak_audit/{name}/metrics/variant_feature_decisions.csv")
    dec = pd.read_csv(p)
    d_drop = set(dec[dec.in_variant_D_removed].feature)
    D = [c for c in ds.feature_cols if c not in d_drop]
    E3 = [c for c in D if not re.search(r"port", c, re.I)]
    pats = [re.compile(x) for x in ds.cfg.get("telemetry_patterns", [])]
    E4 = [c for c in E3 if not any(r.search(c) for r in pats)]
    out = {}
    if pats and E4 and len(E4) != len(E3):
        out["E4_network_only"] = E4
    if E3 and len(E3) != len(D):
        out["E3_clean_noports"] = E3
    out["D_clean"] = D
    return out


def rows_for(cfg, ds, key):
    """Per-dataset override of a row budget (variant_rows / model_rows / probe_rows)."""
    return ds.cfg.get(key, cfg[key])
