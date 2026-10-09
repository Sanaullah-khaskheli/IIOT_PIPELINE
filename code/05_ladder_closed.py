"""STAGE 5a -- shortcut-audit LADDER + closed-set baselines on IDENTICAL splits/rows/preprocessing.
Rungs: A_raw (all columns, raw spelling kept) -> D_strict (identifiers/timestamps/payload removed) -> E3_noport -> E4_network (X-IIoTID host telemetry removed).
Saves: feature lists, removed lists, preprocessing config, splits, model config, seeds, metrics, predictions, probabilities (seeds in final.save_probs_seeds),
per-class results, confusion matrices.  Models: rf, lgbm (if installed), mlp_softmax_en, mlp_edl_both (CONTROL only)."""
import os, json, time, importlib
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from common import *
import safe_core as sc

L1 = importlib.import_module("01_leakage_audit"); B2 = importlib.import_module("02_baselines")
NAME = "LADDER_V5"
RUNG = {"A_raw": "A_raw", "D_clean": "D_strict", "E3_clean_noports": "E3_noport", "E4_network_only": "E4_network"}


def main(cfg, name, run_id):
    F = cfg["final"]; ds = get_dataset(cfg, name, cfg["seeds"][0]); s2 = cfg["stage2"]; root = results_root(cfg)
    D = {k: os.path.join(root, k, name) for k in ("cleaned_features", "splits", "closed_set", "predictions")}
    for d in D.values(): os.makedirs(d, exist_ok=True)
    L = lambda m: log(cfg, "05_ladder", f"[{name}] {m}")
    normal = {t.lower() for t in ds.cfg["normal_tokens"]}
    vidx = L1.stratified_idx(ds, rows_for(cfg, ds, "model_rows"), cfg["seeds"][0])
    vsub = ds.df.iloc[vidx]; yf = ds.y_family[vidx]; strat = L1._strat(ds)[vidx]
    classes, y = np.unique(yf, return_inverse=True); K = len(classes); nm = np.array([c.lower() in normal for c in classes])
    pd.DataFrame({"class": classes, "n_rows_used": np.bincount(y, minlength=K), "is_normal": nm}).assign(dataset=name, run_id=run_id) \
        .to_csv(os.path.join(D["closed_set"], "class_counts.csv"), index=False)
    clean = clean_variant_cols(cfg, name, ds)
    rungs = {"A_raw": (list(ds.feature_cols), set(ds.feature_cols))}
    prev = rungs["A_raw"][0]
    for k in ("D_clean", "E3_clean_noports", "E4_network_only"):
        if k in clean and set(clean[k]) != set(prev):
            rungs[RUNG[k]] = (clean[k], set()); prev = clean[k]
        elif k in clean:
            L(f"rung {RUNG[k]} identical to previous rung -> not repeated")
    prev_cols = None
    for rn, (cols, raw) in rungs.items():
        json.dump({"rung": rn, "n_features": len(cols), "features": cols,
                   "removed_vs_previous": sorted(set(prev_cols) - set(cols)) if prev_cols else [],
                   "encoding": "raw-spelling categorical codes" if raw else "canonical numeric",
                   "preprocessing": "median impute (train rows) -> sign*log1p -> StandardScaler(train rows) -> clip[-10,10]; trees use unscaled matrix",
                   "run_id": run_id}, open(os.path.join(D["cleaned_features"], f"{rn}.json"), "w"), indent=1)
        prev_cols = cols
    models = [m for m in F["ladder_models"]]
    try:
        import lightgbm
    except Exception:
        models = [m for m in models if m != "lgbm"]; L("lightgbm missing -> skipped (state in paper)")
    json.dump({"models": models, "stage2": s2, "hgb_max_iter": cfg["hgb_max_iter"], "seeds": F["ladder_seeds"], "run_id": run_id,
               "rf": "100 trees, balanced_subsample", "lgbm": "sqrt-balanced weights, lr .05, early stop 20"},
              open(os.path.join(D["closed_set"], "model_config.json"), "w"), indent=1)
    B2.smoke_deep(s2) if any(m.startswith("mlp") for m in models) else None
    rows, pcls = [], []
    for rn, (cols, raw) in rungs.items():
        Xraw = encode_matrix(vsub, cols, raw)
        for sd in F["ladder_seeds"]:
            tr_all, te = train_test_split(np.arange(len(y)), test_size=cfg["test_size"], stratify=strat, random_state=sd)
            tr, va = train_test_split(tr_all, test_size=0.1, stratify=strat[tr_all], random_state=sd)
            np.savez_compressed(os.path.join(D["splits"], f"{rn}_seed{sd}.npz"), train=vidx[tr], val=vidx[va], test=vidx[te])
            Xs = B2.prep(Xraw, tr)
            for mn in models:
                t0 = time.time()
                try:
                    if mn == "rf":
                        m = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight="balanced_subsample", random_state=sd).fit(Xs[tr], y[tr])
                        prob = np.zeros((len(te), K)); prob[:, m.classes_] = m.predict_proba(Xs[te])
                    elif mn == "lgbm":
                        import lightgbm as lgb
                        w = compute_sample_weight("balanced", y[tr]) ** 0.5; w = w / w.mean()
                        m = lgb.LGBMClassifier(n_estimators=cfg["hgb_max_iter"] * 8, learning_rate=0.05, random_state=sd, verbose=-1)
                        m.fit(Xraw[tr], y[tr], sample_weight=w, eval_set=[(Xraw[va], y[va])], callbacks=[lgb.early_stopping(20, verbose=False)])
                        prob = np.zeros((len(te), K)); prob[:, m.classes_] = m.predict_proba(Xraw[te])
                    else:
                        prob = B2.train_deep(B2.DEEP[mn], Xs[tr], y[tr], Xs[va], y[va], Xs[te], K, sd, s2)
                    p, yt = prob.argmax(1), y[te]
                    cm = B2.cm_of(yt, p, K)
                    rows.append({"dataset": name, "rung": rn, "model": mn, "seed": sd, "n_features": len(cols), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                                 "MacroF1": f1_score(yt, p, average="macro", labels=np.unique(yt), zero_division=0), "MCC": matthews_corrcoef(yt, p),
                                 "BinaryMCC": matthews_corrcoef(~nm[yt], ~nm[p]), "Accuracy": accuracy_score(yt, p),
                                 "ECE": B2.ece(prob, yt, F["ece_bins"]), "seconds": time.time() - t0, "run_id": run_id})
                    f1s = f1_score(yt, p, labels=np.arange(K), average=None, zero_division=0)
                    pcls += [{"dataset": name, "rung": rn, "model": mn, "seed": sd, "class": c, "F1": f, "support": int((yt == i).sum()), "run_id": run_id}
                             for i, (c, f) in enumerate(zip(classes, f1s))]
                    pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(D["closed_set"], f"cm_{rn}_{mn}_seed{sd}.csv"))
                    kw = {"test_idx": vidx[te], "y": yt.astype(np.int16), "pred": p.astype(np.int16), "conf": prob.max(1).astype(np.float32)}
                    if sd in F["save_probs_seeds"]: kw["prob"] = prob.astype(np.float16)
                    np.savez_compressed(os.path.join(D["predictions"], f"{rn}_{mn}_seed{sd}.npz"), **kw)
                    L(f"{rn} {mn} seed={sd}: MacroF1={rows[-1]['MacroF1']:.4f} binMCC={rows[-1]['BinaryMCC']:.4f}")
                except Exception as e:
                    record_failure(cfg, f"{NAME}:{name}:{rn}:{mn}:seed{sd}", e)
            pd.DataFrame(rows).to_csv(os.path.join(D["closed_set"], "ladder_by_seed_PARTIAL.csv"), index=False)
    res = pd.DataFrame(rows)
    if res.empty: raise RuntimeError("ladder produced no result")
    res.to_csv(os.path.join(D["closed_set"], "ladder_by_seed.csv"), index=False)
    pd.DataFrame(pcls).to_csv(os.path.join(D["closed_set"], "per_class_by_seed.csv"), index=False)
    return {"dataset": name, "rungs": list(rungs), "rows": len(res)}, f"### {NAME} {name}: {len(res)} result rows\n"
