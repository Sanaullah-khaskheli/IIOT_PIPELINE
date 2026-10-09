"""STAGE 4 -- OPEN-SET / UNSEEN-ATTACK EVALUATION on the clean variants. Experiment name: OPENSET_V1
Leave-one-attack-out over EVERY attack family with enough rows (not 3 hand-picked ones).
Model is trained on the remaining (known) classes only; the held-out family is the "unknown".
Two known-pool conditions (the second removes the free 'unknown attack vs Normal' pairs):
    with_normal      known side of the test = all known test rows
    attacks_only     known side of the test = known ATTACK test rows only (Normal excluded)
Scores (higher = more unknown):
    softmax_en : maxprob (1-p_max), entropy, energy, mahalanobis (penultimate features), mc_dropout (entropy of mean probs)
    edl_both / edl_ce : vacuity (K/S), plus maxprob and entropy of alpha/S
    rf : 1 - max class probability   |   iforest : Isolation Forest fit on known training rows
Question answered: does evidential vacuity beat the standard post-hoc scores on THIS data? If not, say so.
Statistics: unit of analysis is the FAMILY (>=6 families -> Wilcoxon signed-rank is valid); AUROC CI by Hanley-McNeil."""
import os, importlib, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from scipy.special import logsumexp
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from common import *

L1 = importlib.import_module("01_leakage_audit")
B2 = importlib.import_module("02_baselines")
NAME = "OPENSET_V1"
DEEP4 = {"softmax_en": ("softmax", "en"), "edl_both": ("edl", "en", 1, 1), "edl_ce": ("edl", "en", 1, 0)}


def hanley_mcneil_ci(auc, n_pos, n_neg):
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg))
    return max(0.0, auc - 1.96 * se), min(1.0, auc + 1.96 * se)


def ood_metrics(s_known, s_unk):
    y = np.r_[np.zeros(len(s_known)), np.ones(len(s_unk))]; s = np.r_[s_known, s_unk]
    auc = roc_auc_score(y, s)
    lo, hi = hanley_mcneil_ci(auc, len(s_unk), len(s_known))
    thr95 = np.percentile(s_unk, 5)
    return {"AUROC": auc, "AUROC_CI_low": lo, "AUROC_CI_high": hi, "AUPR": average_precision_score(y, s),
            "FPR@95TPR": float((s_known >= thr95).mean()),
            "TPR@5FPR": float((s_unk > np.percentile(s_known, 95)).mean())}


def fit_net(spec, Xtr, ytr, Xva, yva, K, seed, s2):
    import torch, torch.nn as nn, torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    d = Xtr.shape[1]
    net = nn.Sequential(nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, K)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=s2["lr"], weight_decay=1e-5)
    kind = spec[0]
    w = torch.tensor(B2.class_weights(ytr, K, spec[1], s2["en_beta"]), dtype=torch.float32, device=dev)
    Xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(ytr, dtype=torch.long, device=dev)
    Xv = torch.tensor(Xva, device=dev)
    best, best_state, bad = -1, None, 0

    def val_f1():
        net.eval()
        with torch.no_grad():
            p = torch.cat([net(Xv[i:i + 8192]) for i in range(0, len(Xv), 8192)]).argmax(1).cpu().numpy()
        return f1_score(yva, p, average="macro", zero_division=0)

    for ep in range(s2["epochs"]):
        net.train(); perm = torch.randperm(len(Xt), device=dev)
        lam = s2["edl_kl_coef"] * min(1.0, (ep + 1) / s2["edl_anneal_epochs"])
        for i in range(0, len(perm), s2["batch_size"]):
            idx = perm[i:i + s2["batch_size"]]
            if len(idx) < 2:
                continue
            z, y = net(Xt[idx]), yt[idx]
            if kind == "softmax":
                loss = (w[y] * F.cross_entropy(z, y, reduction="none")).mean()
            else:
                alpha = F.softplus(z) + 1; S = alpha.sum(1, keepdim=True); oh = F.one_hot(y, K).float()
                ce = (oh * (torch.digamma(S) - torch.digamma(alpha))).sum(1)
                kl = B2.kl_dirichlet(oh + (1 - oh) * alpha)
                loss = ((w[y] if spec[2] else 1.0) * ce).mean() + lam * ((w[y] if spec[3] else 1.0) * kl).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        f1 = val_f1()
        if f1 > best:
            best, bad = f1, 0; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= s2["patience"]:
                break
    net.load_state_dict(best_state)
    return net


def net_outputs(net, X, passes=0):
    """logits, penultimate features, and (optionally) mean softmax probs under MC dropout."""
    import torch, torch.nn.functional as F
    dev = next(net.parameters()).device
    Xt = torch.tensor(X, device=dev); net.eval()
    with torch.no_grad():
        Z = torch.cat([net(Xt[i:i + 8192]) for i in range(0, len(Xt), 8192)]).cpu().numpy()
        H = torch.cat([net[:-1](Xt[i:i + 8192]) for i in range(0, len(Xt), 8192)]).cpu().numpy()
        mc = None
        if passes:
            for m in net.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.train()
            acc = 0
            for _ in range(passes):
                acc = acc + torch.cat([F.softmax(net(Xt[i:i + 8192]), 1) for i in range(0, len(Xt), 8192)]).cpu().numpy()
            mc = acc / passes; net.eval()
    return Z, H, mc


def softmax_np(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)


def entropy_np(P):
    return -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)


def deep_scores(model, net, Xtr, ytr, Xk, Xu, K, passes):
    """dict {score_name: (known_scores, unknown_scores)}"""
    Zk, Hk, mck = net_outputs(net, Xk, passes if model == "softmax_en" else 0)
    Zu, Hu, mcu = net_outputs(net, Xu, passes if model == "softmax_en" else 0)
    out = {}
    if model == "softmax_en":
        Pk, Pu = softmax_np(Zk), softmax_np(Zu)
        out["maxprob"] = (1 - Pk.max(1), 1 - Pu.max(1))
        out["entropy"] = (entropy_np(Pk), entropy_np(Pu))
        out["energy"] = (-logsumexp(Zk, 1), -logsumexp(Zu, 1))
        if mck is not None:
            out["mc_dropout"] = (entropy_np(mck), entropy_np(mcu))
        _, Htr, _ = net_outputs(net, Xtr[:100000])
        ytr_s = ytr[:100000]
        mu = np.stack([Htr[ytr_s == c].mean(0) if (ytr_s == c).any() else np.zeros(Htr.shape[1]) for c in range(K)])
        Xc = Htr - mu[ytr_s]; cov = Xc.T @ Xc / len(Xc) + 1e-3 * np.eye(Htr.shape[1]); P = np.linalg.inv(cov)

        def maha(H):
            return np.min(np.stack([(((H - m) @ P) * (H - m)).sum(1) for m in mu]), axis=0)
        out["mahalanobis"] = (maha(Hk), maha(Hu))
    else:
        def evi(Z):
            a = np.logaddexp(0, Z) + 1; return a, a.sum(1, keepdims=True)
        ak, Sk = evi(Zk); au, Su = evi(Zu)
        out["vacuity"] = ((K / Sk).ravel(), (K / Su).ravel())
        out["maxprob"] = (1 - (ak / Sk).max(1), 1 - (au / Su).max(1))
        out["entropy"] = (entropy_np(ak / Sk), entropy_np(au / Su))
    return out


def smoke(s2):
    rng = np.random.default_rng(0); X = rng.normal(size=(600, 8)).astype(np.float32); y = rng.integers(0, 4, 600)
    s2 = dict(s2, epochs=2)
    for m, spec in DEEP4.items():
        net = fit_net(spec, X[:400], y[:400], X[400:500], y[400:500], 4, 0, s2)
        sc = deep_scores(m, net, X[:400], y[:400], X[500:550], X[550:], 4, 3)
        assert all(np.isfinite(a).all() and np.isfinite(b).all() for a, b in sc.values()), m


def main(cfg, name):
    ds = get_dataset(cfg, name, cfg["seeds"][0])
    s2, s4 = cfg["stage2"], cfg["stage4"]; seeds = cfg["seeds"]; seed0 = seeds[0]
    out = exp_dir(cfg, f"open_set/{name}", ("plots", "metrics", "logs"))
    mt, pl = os.path.join(out, "metrics"), os.path.join(out, "plots")
    manuscript = os.path.join(results_root(cfg), "manuscript_tables")
    L = lambda m: log(cfg, "04_open_set", f"[{name}] {m}")
    deep_models = list(s4["deep_models"])
    if deep_models:
        try:
            smoke(s2); L("open-set deep smoke test passed")
        except Exception as e:
            record_failure(cfg, f"{NAME}:{name}:deep_smoke_test", e); deep_models = []; L("deep models DISABLED")
    vn, cols = list(clean_variant_cols(cfg, name, ds).items())[0]
    L(f"clean variant = {vn} ({len(cols)} features)")
    vidx = L1.stratified_idx(ds, rows_for(cfg, ds, "model_rows"), seed0)
    vsub = ds.df.iloc[vidx]; yf = ds.y_family[vidx]; strat = L1._strat(ds)[vidx]
    Xraw = encode_matrix(vsub, cols, set())
    normal = {t.lower() for t in ds.cfg["normal_tokens"]}
    classes_all = np.unique(yf)
    attacks = [c for c in classes_all if c.lower() not in normal]
    counts = pd.Series(yf).value_counts()
    families = [c for c in attacks if counts[c] >= s4["min_family_rows"]]
    if s4.get("max_families") and len(families) > s4["max_families"]:
        pick = np.random.default_rng(0).choice(len(families), s4["max_families"], replace=False)
        families = [families[i] for i in sorted(pick)]
    seeds = s4.get("seeds") or seeds
    L(f"{len(families)} held-out families of {len(attacks)} attack classes (min_family_rows={s4['min_family_rows']})")
    rows, closed = [], []
    for fam in families:
        known_mask = yf != fam
        kidx = np.where(known_mask)[0]; uidx = np.where(~known_mask)[0]
        kcls, ky = np.unique(yf[kidx], return_inverse=True); K = len(kcls)
        is_normal_known = np.array([c.lower() in normal for c in kcls])[ky]
        for sd in seeds:
            tr_all, te = train_test_split(np.arange(len(kidx)), test_size=cfg["test_size"], stratify=strat[kidx], random_state=sd)
            tr, va = train_test_split(tr_all, test_size=0.1, stratify=strat[kidx][tr_all], random_state=sd)
            u = uidx if len(uidx) <= s4["max_unknown"] else np.random.default_rng(sd).choice(uidx, s4["max_unknown"], replace=False)
            Xs = B2.prep(Xraw, kidx[tr])
            Xk_tr, Xk_va, Xk_te, Xu = Xs[kidx[tr]], Xs[kidx[va]], Xs[kidx[te]], Xs[u]
            scores = {}
            t0 = time.time()
            tt = {}; t1 = time.time()
            rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight="balanced_subsample", random_state=sd).fit(Xk_tr, ky[tr])
            scores["rf:maxprob"] = (1 - rf.predict_proba(Xk_te).max(1), 1 - rf.predict_proba(Xu).max(1))
            tt["rf"] = time.time() - t1; t1 = time.time()
            iso = IsolationForest(n_estimators=100, random_state=sd, n_jobs=-1).fit(Xk_tr[:100000])
            scores["iforest:score"] = (-iso.score_samples(Xk_te), -iso.score_samples(Xu))
            tt["iforest"] = time.time() - t1
            rf_f1 = f1_score(ky[te], rf.predict(Xk_te), average="macro", labels=np.unique(ky[te]), zero_division=0)
            closed.append({"family": fam, "seed": sd, "model": "rf", "known_MacroF1": rf_f1})
            for m in deep_models:
                try:
                    t1 = time.time()
                    net = fit_net(DEEP4[m], Xk_tr, ky[tr], Xk_va, ky[va], K, sd, s2)
                    for sn, pair in deep_scores(m, net, Xk_tr, ky[tr], Xk_te, Xu, K, s4["mc_dropout_passes"]).items():
                        scores[f"{m}:{sn}"] = pair
                    tt[m] = time.time() - t1
                    Zk, _, _ = net_outputs(net, Xk_te)
                    closed.append({"family": fam, "seed": sd, "model": m,
                                   "known_MacroF1": f1_score(ky[te], Zk.argmax(1), average="macro", labels=np.unique(ky[te]), zero_division=0)})
                except Exception as e:
                    record_failure(cfg, f"{NAME}:{name}:{fam}:{m}:seed{sd}", e)
            for method, (sk, su) in scores.items():
                for cond, mask in (("with_normal", np.ones(len(sk), bool)), ("attacks_only", ~is_normal_known[te])):
                    if mask.sum() < 20:
                        continue
                    r = ood_metrics(sk[mask], su); r.update(dataset=name, variant=vn, family=fam, seed=sd, method=method,
                                                              condition=cond, n_known=int(mask.sum()), n_unknown=len(su))
                    rows.append(r)
            L(f"holdout {fam} (n_unknown={len(u)}) seed={sd}: {len(scores)} scores in {time.time()-t0:.0f}s  parts(s)={ {k: round(v) for k, v in tt.items()} }")
            pd.DataFrame(rows).to_csv(os.path.join(mt, "openset_PARTIAL.csv"), index=False)
    res = pd.DataFrame(rows)
    if res.empty:
        raise RuntimeError("no open-set result")
    res.to_csv(os.path.join(mt, "openset_by_family_seed.csv"), index=False)
    pd.DataFrame(closed).to_csv(os.path.join(mt, "known_class_closed_set_F1.csv"), index=False)
    per_fam = res.groupby(["condition", "method", "family"], as_index=False).agg(
        AUROC=("AUROC", "mean"), AUPR=("AUPR", "mean"), FPR95=("FPR@95TPR", "mean"), TPR5=("TPR@5FPR", "mean"),
        CI_low=("AUROC_CI_low", "mean"), CI_high=("AUROC_CI_high", "mean"), n_unknown=("n_unknown", "first"))
    per_fam.to_csv(os.path.join(mt, "openset_per_family.csv"), index=False)
    summ = per_fam.groupby(["condition", "method"], as_index=False).agg(
        mean_AUROC=("AUROC", "mean"), median_AUROC=("AUROC", "median"), mean_AUPR=("AUPR", "mean"),
        mean_FPR95=("FPR95", "mean"), mean_TPR5=("TPR5", "mean"), families=("family", "nunique"))
    summ = summ.sort_values(["condition", "mean_AUROC"], ascending=[True, False])
    write_table(summ, manuscript, f"{name}_openset_summary")
    write_table(per_fam[per_fam.condition == "with_normal"].pivot(index="family", columns="method", values="AUROC").reset_index(),
                manuscript, f"{name}_openset_per_family_AUROC")

    st = []
    for cond in ("with_normal", "attacks_only"):
        p = per_fam[per_fam.condition == cond].pivot(index="family", columns="method", values="AUROC")
        for a in [c for c in p.columns if c.endswith(":vacuity")]:
            for b in [c for c in p.columns if c != a]:
                dlt = (p[a] - p[b]).dropna()
                if len(dlt) >= 6 and (dlt != 0).any():
                    pv = wilcoxon(dlt).pvalue
                else:
                    pv = np.nan
                st.append({"condition": cond, "A": a, "B": b, "mean_AUROC_diff": dlt.mean(), "families_A_better": int((dlt > 0).sum()),
                           "families": len(dlt), "Wilcoxon_p": pv})
    stats = pd.DataFrame(st); stats.to_csv(os.path.join(mt, "vacuity_vs_others.csv"), index=False)
    if len(stats):
        write_table(stats, manuscript, f"{name}_vacuity_vs_others")

    for cond in ("with_normal", "attacks_only"):
        s = summ[summ.condition == cond]
        fig, ax = plt.subplots(figsize=(8, 0.35 * len(s) + 1.5))
        ax.barh(s.method[::-1], s.mean_AUROC[::-1], color="#4C72B0", edgecolor="black", linewidth=.5)
        ax.axvline(0.5, color="gray", ls="--", lw=.8); ax.set_xlim(0, 1)
        ax.set_xlabel("mean AUROC over held-out families"); ax.set_title(f"{name}: unseen-attack detection ({cond})")
        save_fig(fig, os.path.join(pl, f"openset_{cond}"))

    s_ = summ[summ.condition == "attacks_only"].set_index("method")
    best = s_.mean_AUROC.idxmax()
    vac = [m for m in s_.index if m.endswith(":vacuity")]
    vtxt = ", ".join(f"{m}={s_.mean_AUROC[m]:.3f}" for m in vac) or "n/a"
    md = (f"### Stage 4 -- {name} ({vn}, {len(families)} held-out families)\n\n" +
          open(os.path.join(manuscript, f"{name}_openset_summary.md")).read() +
          f"\n**Best method (attacks-only known pool):** {best} (mean AUROC {s_.mean_AUROC[best]:.3f}). Vacuity: {vtxt}.\n"
          "Held-out families with very few rows give noisy AUROCs; read the per-family table, not only the mean.\n")
    summary = {"dataset": name, "openset_variant": vn, "families": len(families), "best_openset_method": best,
               "best_openset_AUROC": float(s_.mean_AUROC[best]),
               "vacuity_AUROC": float(max([s_.mean_AUROC[m] for m in vac])) if vac else np.nan,
               "softmax_maxprob_AUROC": float(s_.mean_AUROC.get("softmax_en:maxprob", np.nan))}
    return summary, md
