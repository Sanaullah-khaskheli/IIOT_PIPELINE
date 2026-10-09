"""STAGE 2 -- FAIR BASELINES + CORE EDL ABLATION on the CLEAN variants. Experiment name: BASELINES_V1
Identical rows, features, split, preprocessing, backbone (MLP 256-256-128), epochs and seeds for every deep model.
Isolates whether EDL / the weighting-on-KL idea adds anything beyond class weighting:
  mlp_softmax_bal   softmax + sklearn-'balanced' weights
  mlp_softmax_en    softmax + effective-number weights   <-- the control the review demanded
  mlp_edl_none      EDL, no weights
  mlp_edl_ce        EDL, effective-number weights on the CE (Bayes-risk) term only
  mlp_edl_kl        EDL, weights on the KL term only
  mlp_edl_both      EDL, weights on CE and KL                <-- the proposed CB-EDL idea
Classical: hgb (HistGradientBoosting), lgbm (if installed), rf, logreg.
All class-weight vectors are rescaled so the SAMPLE-weighted mean weight is 1 (equal effective step size)."""
import os, time, importlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest, wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from common import *

L1 = importlib.import_module("01_leakage_audit")
NAME = "BASELINES_V1"
DEEP = {"mlp_softmax_bal": ("softmax", "bal"), "mlp_softmax_en": ("softmax", "en"),
        "mlp_edl_none": ("edl", None, 0, 0), "mlp_edl_ce": ("edl", "en", 1, 0),
        "mlp_edl_kl": ("edl", "en", 0, 1), "mlp_edl_both": ("edl", "en", 1, 1)}


def cm_of(y, p, K):
    return np.bincount(y * K + p, minlength=K * K).reshape(K, K)


def macro_f1_cm(cm):
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    return f1[cm.sum(1) > 0].mean()


def boot_ci(y, p, K, n, seed=0):
    rng = np.random.default_rng(seed); N = len(y); v = []
    for _ in range(n):
        i = rng.integers(0, N, N); v.append(macro_f1_cm(cm_of(y[i], p[i], K)))
    return np.percentile(v, [2.5, 97.5])


def ece(prob, y, bins=15):
    conf, pred = prob.max(1), prob.argmax(1); acc = (pred == y).astype(float); e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def class_weights(ytr, K, scheme, beta):
    n = np.maximum(np.bincount(ytr, minlength=K), 1).astype(float); N = n.sum()
    w = N / (K * n) if scheme == "bal" else 1.0 / ((1 - beta ** n) / (1 - beta))
    return w * (N / (w * n).sum())


def prep(X, tr):
    X = X.astype(np.float64).copy()
    med = np.nanmedian(X[tr], axis=0); med = np.where(np.isfinite(med), med, 0.0)
    bad = ~np.isfinite(X); X[bad] = np.take(med, np.where(bad)[1])
    X = np.sign(X) * np.log1p(np.abs(X))
    sc = StandardScaler().fit(X[tr])
    return np.clip(sc.transform(X), -10, 10).astype(np.float32)


def kl_dirichlet(alpha):
    import torch
    K = alpha.shape[1]; S = alpha.sum(1, keepdim=True)
    return (torch.lgamma(S).squeeze(1) - torch.lgamma(torch.tensor(float(K), device=alpha.device))
            - torch.lgamma(alpha).sum(1) + ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S))).sum(1))


def train_deep(spec, Xtr, ytr, Xva, yva, Xte, K, seed, s2, epochs=None):
    import torch, torch.nn as nn, torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    d = Xtr.shape[1]
    net = nn.Sequential(nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, K)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=s2["lr"], weight_decay=1e-5)
    kind = spec[0]
    wnp = class_weights(ytr, K, spec[1], s2["en_beta"]) if spec[1] else np.ones(K)
    w = torch.tensor(wnp, dtype=torch.float32, device=dev)
    Xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(ytr, dtype=torch.long, device=dev)
    Xv, Xe = torch.tensor(Xva, device=dev), torch.tensor(Xte, device=dev)
    bs, E = s2["batch_size"], epochs or s2["epochs"]

    def probs(X):
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                z = net(X[i:i + 8192])
                if kind == "softmax":
                    out.append(F.softmax(z, 1))
                else:
                    a = F.softplus(z) + 1; out.append(a / a.sum(1, keepdim=True))
        return torch.cat(out).cpu().numpy()

    best, best_state, bad = -1, None, 0
    for ep in range(E):
        net.train(); perm = torch.randperm(len(Xt), device=dev)
        lam = s2["edl_kl_coef"] * min(1.0, (ep + 1) / s2["edl_anneal_epochs"])
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            z, y = net(Xt[idx]), yt[idx]
            if kind == "softmax":
                loss = (w[y] * F.cross_entropy(z, y, reduction="none")).mean()
            else:
                alpha = F.softplus(z) + 1; S = alpha.sum(1, keepdim=True)
                oh = F.one_hot(y, K).float()
                ce = (oh * (torch.digamma(S) - torch.digamma(alpha))).sum(1)
                kl = kl_dirichlet(oh + (1 - oh) * alpha)
                loss = ((w[y] if spec[2] else 1.0) * ce).mean() + lam * ((w[y] if spec[3] else 1.0) * kl).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        f1 = f1_score(yva, probs(Xv).argmax(1), average="macro", zero_division=0)
        if f1 > best:
            best, bad = f1, 0; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= s2["patience"]:
                break
    net.load_state_dict(best_state)
    return probs(Xe)


def smoke_deep(s2):
    rng = np.random.default_rng(0); X = rng.normal(size=(600, 8)).astype(np.float32); y = rng.integers(0, 4, 600)
    for name, spec in DEEP.items():
        p = train_deep(spec, X[:400], y[:400], X[400:500], y[400:500], X[500:], 4, 0, s2, epochs=2)
        assert p.shape == (100, 4) and np.isfinite(p).all(), f"{name}: bad output"
    return True


def main(cfg, name):
    ds = get_dataset(cfg, name, cfg["seeds"][0])
    s2 = cfg["stage2"]; seeds = cfg["seeds"]; seed0 = seeds[0]
    out = exp_dir(cfg, f"baselines/{name}", ("plots", "metrics", "logs", "predictions"))
    mt, pl, pr = (os.path.join(out, s) for s in ("metrics", "plots", "predictions"))
    manuscript = os.path.join(results_root(cfg), "manuscript_tables")
    L = lambda m: log(cfg, "02_baselines", f"[{name}] {m}")
    models = list(s2["models"])
    try:
        import lightgbm
    except Exception:
        if "lgbm" in models:
            models.remove("lgbm"); L("lightgbm not installed -> skipped")
    if any(m in DEEP for m in models):
        try:
            smoke_deep(s2); L("deep-model smoke test passed")
        except Exception as e:
            record_failure(cfg, f"{NAME}:{name}:deep_smoke_test", e)
            models = [m for m in models if m not in DEEP]; L("deep models DISABLED (smoke test failed)")
    variants = clean_variant_cols(cfg, name, ds)
    normal = {t.lower() for t in ds.cfg["normal_tokens"]}
    vidx = L1.stratified_idx(ds, rows_for(cfg, ds, "model_rows"), seed0)
    vsub = ds.df.iloc[vidx]; yf = ds.y_family[vidx]; strat = L1._strat(ds)[vidx]
    classes, y = np.unique(yf, return_inverse=True); K = len(classes)
    nm = np.array([c.lower() in normal for c in classes])
    rows, per_class, preds = [], [], {}
    for vn, cols in variants.items():
        Xraw = encode_matrix(vsub, cols, set())
        for sd in seeds:
            tr_all, te = train_test_split(np.arange(len(y)), test_size=cfg["test_size"], stratify=strat, random_state=sd)
            tr, va = train_test_split(tr_all, test_size=0.1, stratify=strat[tr_all], random_state=sd)
            Xs = prep(Xraw, tr)
            for mn in models:
                t0 = time.time()
                try:
                    if mn == "hgb":
                        w = compute_sample_weight("balanced", y[tr]) ** 0.5; w = w / w.mean()
                        m = HistGradientBoostingClassifier(max_iter=cfg["hgb_max_iter"] * 3, early_stopping=True, validation_fraction=0.1,
                                                           n_iter_no_change=10, random_state=sd)
                        m.fit(Xraw[tr], y[tr], sample_weight=w); prob = m.predict_proba(Xraw[te])
                    elif mn == "lgbm":
                        import lightgbm as lgb
                        w = compute_sample_weight("balanced", y[tr]) ** 0.5; w = w / w.mean()
                        m = lgb.LGBMClassifier(n_estimators=cfg["hgb_max_iter"] * 8, learning_rate=0.05, random_state=sd, verbose=-1)
                        m.fit(Xraw[tr], y[tr], sample_weight=w, eval_set=[(Xraw[va], y[va])],
                              callbacks=[lgb.early_stopping(20, verbose=False)]); prob = m.predict_proba(Xraw[te])
                    elif mn == "rf":
                        m = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight="balanced_subsample", random_state=sd)
                        m.fit(Xs[tr], y[tr]); prob = m.predict_proba(Xs[te])
                    elif mn == "logreg":
                        m = LogisticRegression(max_iter=200, class_weight="balanced")
                        m.fit(Xs[tr], y[tr]); prob = m.predict_proba(Xs[te])
                    else:
                        prob = train_deep(DEEP[mn], Xs[tr], y[tr], Xs[va], y[va], Xs[te], K, sd, s2)
                    full = np.zeros((len(te), K)); full[:, (m.classes_ if mn in ("hgb", "lgbm", "rf", "logreg") else np.arange(K))] = prob
                    p = full.argmax(1); yt = y[te]
                    lo, hi = boot_ci(yt, p, K, s2["bootstrap"], sd)
                    rows.append({"dataset": name, "variant": vn, "model": mn, "seed": sd, "n_features": len(cols),
                                 "MacroF1": f1_score(yt, p, average="macro", labels=np.unique(yt), zero_division=0),
                                 "MacroF1_CI_low": lo, "MacroF1_CI_high": hi, "MCC": matthews_corrcoef(yt, p),
                                 "BinaryMCC": matthews_corrcoef(~nm[yt], ~nm[p]), "Accuracy": accuracy_score(yt, p),
                                 "ECE": ece(full, yt), "seconds": time.time() - t0})
                    preds[(vn, mn, sd)] = (te, yt, p)
                    np.savez_compressed(os.path.join(pr, f"{vn}_{mn}_seed{sd}.npz"), test_idx=vidx[te], y=yt.astype(np.int8), pred=p.astype(np.int8))
                    if sd == seed0:
                        f1s = f1_score(yt, p, labels=np.arange(K), average=None, zero_division=0)
                        per_class += [{"variant": vn, "model": mn, "class": c, "F1": f, "support": int((yt == i).sum())}
                                      for i, (c, f) in enumerate(zip(classes, f1s))]
                    L(f"{vn} {mn} seed={sd}: MacroF1={rows[-1]['MacroF1']:.4f} binMCC={rows[-1]['BinaryMCC']:.4f} ECE={rows[-1]['ECE']:.4f} ({rows[-1]['seconds']:.0f}s)")
                except Exception as e:
                    record_failure(cfg, f"{NAME}:{name}:{vn}:{mn}:seed{sd}", e)
    res = pd.DataFrame(rows)
    if res.empty:
        raise RuntimeError("no baseline finished -- see failed_experiments.csv")
    res.to_csv(os.path.join(mt, "results_by_seed.csv"), index=False)
    pd.DataFrame(per_class).to_csv(os.path.join(mt, "per_class_f1_seed0.csv"), index=False)
    agg = res.groupby(["variant", "model"]).agg(
        MacroF1=("MacroF1", "mean"), MacroF1_sd=("MacroF1", "std"), MacroF1_median=("MacroF1", "median"),
        CI95_low=("MacroF1_CI_low", "mean"), CI95_high=("MacroF1_CI_high", "mean"), MCC=("MCC", "mean"),
        BinaryMCC=("BinaryMCC", "mean"), ECE=("ECE", "mean"), seconds=("seconds", "mean"), n_seeds=("seed", "nunique")).reset_index()
    agg = agg.sort_values(["variant", "MacroF1"], ascending=[True, False])
    write_table(agg, manuscript, f"{name}_baselines")
    agg.to_csv(os.path.join(mt, "baselines_summary.csv"), index=False)

    pairs = [("mlp_edl_both", "mlp_softmax_en"), ("mlp_softmax_en", "mlp_softmax_bal"), ("mlp_edl_both", "mlp_edl_none"),
             ("mlp_edl_both", "mlp_edl_ce"), ("mlp_edl_both", "mlp_edl_kl"), ("hgb", "mlp_edl_both"), ("hgb", "mlp_softmax_en")]
    st = []
    for vn in variants:
        for a, b in pairs:
            ds_f1, mc = [], []
            for sd in seeds:
                if (vn, a, sd) in preds and (vn, b, sd) in preds:
                    _, yt, pa = preds[(vn, a, sd)]; _, _, pb = preds[(vn, b, sd)]
                    ca, cb = pa == yt, pb == yt; n01 = int((ca & ~cb).sum()); n10 = int((~ca & cb).sum())
                    mc.append(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0)
                    ds_f1.append(macro_f1_cm(cm_of(yt, pa, K)) - macro_f1_cm(cm_of(yt, pb, K)))
            if ds_f1:
                wp = wilcoxon(ds_f1).pvalue if len(ds_f1) >= 6 and np.any(np.array(ds_f1) != 0) else np.nan
                st.append({"variant": vn, "A": a, "B": b, "mean_MacroF1_A_minus_B": float(np.mean(ds_f1)),
                           "McNemar_exact_p_per_seed": ";".join(f"{p:.2g}" for p in mc), "Wilcoxon_p(>=6 seeds)": wp,
                           "note": "McNemar tests accuracy on the same test rows; with ~60k rows tiny effects become 'significant' -- judge the effect size"})
    stats = pd.DataFrame(st); stats.to_csv(os.path.join(mt, "paired_tests.csv"), index=False)
    if len(stats):
        write_table(stats, manuscript, f"{name}_paired_tests")

    for vn in variants:
        a = agg[agg.variant == vn]
        fig, ax = plt.subplots(figsize=(8, 0.4 * len(a) + 1.5))
        ax.barh(a.model[::-1], a.MacroF1[::-1], xerr=a.MacroF1_sd.fillna(0)[::-1], color="#4C72B0", edgecolor="black", linewidth=.5)
        ax.set_xlabel("Macro-F1 (mean +/- sd over seeds)"); ax.set_title(f"{name} / {vn}: fair baselines")
        save_fig(fig, os.path.join(pl, f"baselines_{vn}"))

    prim = list(variants)[0]
    a = agg[agg.variant == prim].set_index("model")
    best = a.MacroF1.idxmax()
    g = lambda m: float(a.MacroF1.get(m, np.nan))
    md = (f"### Stage 2 -- {name} (primary clean variant: {prim})\n\n" + open(os.path.join(manuscript, f"{name}_baselines.md")).read() +
          f"\n**Best model:** {best} (Macro-F1 {a.MacroF1[best]:.4f}). "
          f"EDL(both) - softmax(EN) = {g('mlp_edl_both') - g('mlp_softmax_en'):+.4f}; softmax(EN) - softmax(bal) = "
          f"{g('mlp_softmax_en') - g('mlp_softmax_bal'):+.4f}; hgb - EDL(both) = {g('hgb') - g('mlp_edl_both'):+.4f}.\n"
          "These are pilot numbers from few seeds; do not call any difference significant without >=5-10 seeds.\n")
    summary = {"dataset": name, "primary_variant": prim, "best_model": best, "best_MacroF1": float(a.MacroF1[best]),
               "best_MCC": float(a.MCC[best]), "best_ECE": float(a.ECE[best]),
               "EDL_both_minus_softmax_EN": g("mlp_edl_both") - g("mlp_softmax_en"), "hgb_MacroF1": g("hgb")}
    return summary, md
