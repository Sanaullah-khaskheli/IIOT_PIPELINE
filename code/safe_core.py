"""SAFE-IIoT core: pure numpy/pandas/scipy functions (no torch) -- metrics, fusion, taxonomy, statistics.
Everything the manuscript reports about open-set detection is computed here, so it is unit-testable."""
import re, hashlib, os, json
import numpy as np, pandas as pd
from scipy import stats as sst
from sklearn.metrics import roc_auc_score, average_precision_score

RULES = [(r"ddos", "DDoS"), (r"(^|[_\-\s])r?dos($|[_\-\s])|^dos", "DoS"), (r"mirai", "Mirai"),
         (r"recon|scan|fingerprint|discover", "Reconnaissance"),
         (r"sql|xss|command_?injection|upload|inject|deface|browser", "WebApplication"),
         (r"brute|dictionary|password|login", "CredentialAttack"),
         (r"mitm|arp|spoof", "MITM_Spoofing"), (r"ransom|crypto", "Ransomware"),
         (r"backdoor|reverse_?shell|malware|c&c|c2|trojan|relay", "Malware_C2")]


def propose_group(family):
    f = family.lower()
    for pat, g in RULES:
        if re.search(pat, f):
            return g, pat
    return family, "singleton(no rule)"


def build_taxonomy(dataset_to_families):
    rows = [{"dataset": d, "family": f, "group": propose_group(f)[0], "rule": propose_group(f)[1]}
            for d, fams in dataset_to_families.items() for f in sorted(fams)]
    return pd.DataFrame(rows)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_taxonomy(path, dataset, expected_sha=None):
    """Refuses to run without a frozen taxonomy (hash must match config) -- prevents post-hoc regrouping."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"taxonomy file missing: {path}. Run 05_taxonomy.py, REVIEW it, then freeze its sha256 in config_final.yaml")
    sha = file_sha256(path)
    if expected_sha and sha != expected_sha:
        raise RuntimeError(f"TAXONOMY CHANGED AFTER FREEZE: sha256 {sha} != config {expected_sha}. Do NOT report results.")
    t = pd.read_csv(path)
    t = t[t.dataset == dataset]
    return dict(zip(t.family, t.group)), sha


def near_far(family, tax, class_counts, min_sibling_rows=20):
    """near = >=1 sibling (same group) present in the known pool; far = no sibling. Always reports n_siblings."""
    g = tax.get(family, family)
    sib = [f for f, gg in tax.items() if gg == g and f != family and class_counts.get(f, 0) >= min_sibling_rows]
    return ("near" if sib else "far"), len(sib), g


def hanley_mcneil_ci(auc, n_pos, n_neg):
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    se = np.sqrt(max(0.0, (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)))
    return max(0.0, auc - 1.96 * se), min(1.0, auc + 1.96 * se)


def ood_full(s_known, s_unk, s_val):
    """Higher score = more unknown. positive = unknown family; negative = known test rows.
    FPR@95TPR  : known rows flagged when the threshold keeps 95% of unknown rows (oracle-threshold, standard OOD metric).
    DetRate@val: share of unknown rows flagged by a threshold fixed on KNOWN VALIDATION rows at 5% false-alarm rate
                 (deployable operating point; no unknowns used). KnownFPR@val is the realised false-alarm rate on known test rows."""
    y = np.r_[np.zeros(len(s_known)), np.ones(len(s_unk))]; s = np.r_[s_known, s_unk]
    auc = roc_auc_score(y, s); lo, hi = hanley_mcneil_ci(auc, len(s_unk), len(s_known))
    thr95 = np.percentile(s_unk, 5)
    thr_val = np.percentile(s_val, 95)
    return {"AUROC": auc, "AUROC_CI_low": lo, "AUROC_CI_high": hi, "AUPR": average_precision_score(y, s),
            "FPR@95TPR": float((s_known >= thr95).mean()),
            "DetRate@val5FPR": float((s_unk > thr_val).mean()), "KnownFPR@val5": float((s_known > thr_val).mean())}


def ecdf_transform(val_scores):
    v = np.sort(np.asarray(val_scores, float))
    return lambda s: np.searchsorted(v, np.asarray(s, float), side="right") / len(v)


def zlog_transform(val_scores, eps=1e-12):
    l = np.log(np.asarray(val_scores, float) + eps); mu, sd = l.mean(), l.std() + 1e-12
    return lambda s: (np.log(np.asarray(s, float) + eps) - mu) / sd


def fuse(components_val, components_other, kind="rank", weights=None):
    """components_val: list of validation-known score arrays; components_other: list of lists (score arrays to transform).
    Normalisation is fitted on VALIDATION KNOWN rows only. Weights default to equal (no unknowns exist in validation,
    so a tuned weight is impossible under the strict protocol -- stated in the paper)."""
    k = len(components_val); w = weights or [1.0 / k] * k
    tf = [(ecdf_transform if kind == "rank" else zlog_transform)(v) for v in components_val]
    return [sum(wi * t(arr[i]) for i, (wi, t) in enumerate(zip(w, tf))) for arr in components_other]


def holm(p):
    p = np.asarray(p, float); out = np.full(len(p), np.nan); ok = np.isfinite(p)
    idx = np.argsort(p[ok]); m = ok.sum(); run = 0.0; vals = p[ok][idx]; adj = np.empty(m)
    for r, v in enumerate(vals):
        run = max(run, (m - r) * v); adj[r] = min(1.0, run)
    tmp = np.empty(m); tmp[idx] = adj; out[ok] = tmp
    return out


def mean_sd_ci(x, level=0.95):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n == 0:
        return dict(mean=np.nan, sd=np.nan, ci_low=np.nan, ci_high=np.nan, n=0)
    m = x.mean(); sd = x.std(ddof=1) if n > 1 else np.nan
    h = sst.t.ppf(0.5 + level / 2, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
    return dict(mean=m, sd=sd, ci_low=m - h if n > 1 else np.nan, ci_high=m + h if n > 1 else np.nan, n=n)


def boot_ci_diff(a, b=None, n=5000, seed=0, stat=np.median):
    """bootstrap CI of stat(a)-stat(b) (independent groups) or stat(a) (paired differences, b=None)."""
    rng = np.random.default_rng(seed); a = np.asarray(a, float)
    if b is None:
        d = [stat(rng.choice(a, len(a))) for _ in range(n)]
    else:
        b = np.asarray(b, float); d = [stat(rng.choice(a, len(a))) - stat(rng.choice(b, len(b))) for _ in range(n)]
    return tuple(np.percentile(d, [2.5, 97.5]))


def paired_over_families(pf, a, b):
    """pf: DataFrame index=family, columns=methods. Unit = family. Wilcoxon only if >=6 families with non-zero differences."""
    d = (pf[a] - pf[b]).dropna()
    nz = d[d != 0]
    p = sst.wilcoxon(nz).pvalue if len(nz) >= 6 else np.nan
    lo, hi = boot_ci_diff(d.values) if len(d) >= 3 else (np.nan, np.nan)
    return {"A": a, "B": b, "families": len(d), "median_diff": float(np.median(d)) if len(d) else np.nan,
            "mean_diff": float(d.mean()) if len(d) else np.nan, "boot95_low": lo, "boot95_high": hi,
            "A_better": int((d > 0).sum()), "B_better": int((d < 0).sum()), "Wilcoxon_p": p}


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float((np.sum(a[:, None] > b[None, :]) - np.sum(a[:, None] < b[None, :])) / (len(a) * len(b)))


def near_vs_far(values_near, values_far):
    a, b = np.asarray(values_near, float), np.asarray(values_far, float)
    r = {"n_near": len(a), "n_far": len(b), "median_near": float(np.median(a)) if len(a) else np.nan,
         "median_far": float(np.median(b)) if len(b) else np.nan, "cliffs_delta_far_minus_near": cliffs_delta(b, a)}
    if len(a) >= 2 and len(b) >= 2:
        r["MannWhitney_p"] = float(sst.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        r["boot95_low_far_minus_near"], r["boot95_high_far_minus_near"] = boot_ci_diff(b, a)
    else:
        r["MannWhitney_p"] = np.nan; r["boot95_low_far_minus_near"] = r["boot95_high_far_minus_near"] = np.nan
    return r
