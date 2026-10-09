"""STAGE 5b -- sibling-aware leave-one-attack-family-out (LOAO) over EVERY supported family.
Known pool = all other classes (incl. Normal). Unknown = held-out family (POSITIVE class). Conditions: with_normal, attacks_only.
Scores (higher = more unknown): rf:maxprob | mlp:maxprob | mlp:energy | mlp:mahalanobis | fusion_rank (RF+Mahalanobis, ECDF on known VALIDATION rows, equal weights)
| fusion_z (pre-declared sensitivity) | edl:vacuity (CONTROL).  near/far from the FROZEN taxonomy; proximity = distance of the held-out family's centroid to the nearest known centroid.
Resumable: finished (family, seed) pairs in loao_by_family_seed_method_PARTIAL.csv are skipped."""
import os, time, hashlib, importlib
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from common import *
import safe_core as sc

L1 = importlib.import_module("01_leakage_audit"); B2 = importlib.import_module("02_baselines"); O4 = importlib.import_module("04_open_set")
NAME = "LOAO_V5"
MLP_SPEC, EDL_SPEC = O4.DEEP4["softmax_en"], O4.DEEP4["edl_both"]


def main(cfg, name, run_id):
    F, s4, s2 = cfg["final"], cfg["stage4"], cfg["stage2"]; ds = get_dataset(cfg, name, cfg["seeds"][0]); root = results_root(cfg)
    out = os.path.join(root, "loao", name); sp = os.path.join(root, "splits", name); pr = os.path.join(root, "predictions", "loao", name)
    for d in (out, sp, pr): os.makedirs(d, exist_ok=True)
    L = lambda m: log(cfg, "06_loao", f"[{name}] {m}")
    tax, tax_sha = sc.load_taxonomy(os.path.join(HERE, F["taxonomy_file"]), name, F.get("taxonomy_sha256") or None)
    vn, cols = list(clean_variant_cols(cfg, name, ds).items())[0]
    vidx = L1.stratified_idx(ds, rows_for(cfg, ds, "model_rows"), cfg["seeds"][0])
    vsub = ds.df.iloc[vidx]; yf = ds.y_family[vidx]; strat = L1._strat(ds)[vidx]
    Xraw = encode_matrix(vsub, cols, set()); normal = {t.lower() for t in ds.cfg["normal_tokens"]}
    counts = pd.Series(yf).value_counts(); cc = counts.to_dict()
    attacks = [c for c in counts.index if c.lower() not in normal]
    families = sorted([c for c in attacks if counts[c] >= s4["min_family_rows"]])
    skipped = sorted(set(attacks) - set(families))
    pd.DataFrame({"family": families, "n_rows": [counts[f] for f in families], "evaluated": True, "group": [tax.get(f, f) for f in families]}).assign(dataset=name, run_id=run_id) \
        .pipe(lambda d: pd.concat([d, pd.DataFrame({"family": skipped, "n_rows": [counts[f] for f in skipped], "evaluated": False, "group": [tax.get(f, f) for f in skipped]}).assign(dataset=name, run_id=run_id)])) \
        .to_csv(os.path.join(out, "family_inventory.csv"), index=False)
    L(f"{len(families)} evaluated families; {len(skipped)} skipped (< {s4['min_family_rows']} rows): {skipped}; variant={vn}; taxonomy sha={tax_sha[:12]}")
    O4.smoke(s2)
    part = os.path.join(out, "loao_by_family_seed_method_PARTIAL.csv")
    rows = pd.read_csv(part).to_dict("records") if os.path.exists(part) else []
    done = {(r["family"], r["seed"]) for r in rows}
    aux = {k: os.path.join(out, k + "_PARTIAL.csv") for k in ("known_class_closed_set_F1", "proximity", "split_hashes")}
    rd = lambda k: pd.read_csv(aux[k]).to_dict("records") if os.path.exists(aux[k]) else []
    closed, prox, hashes = rd("known_class_closed_set_F1"), rd("proximity"), rd("split_hashes")
    for fam in families:
        nf, nsib, grp = sc.near_far(fam, tax, cc, F["min_sibling_rows"])
        known_mask = yf != fam; kidx = np.where(known_mask)[0]; uidx = np.where(~known_mask)[0]
        kcls, ky = np.unique(yf[kidx], return_inverse=True); K = len(kcls)
        is_norm = np.array([c.lower() in normal for c in kcls])[ky]
        for sd in F["loao_seeds"]:
            if (fam, sd) in done: continue
            t0 = time.time()
            tr_all, te = train_test_split(np.arange(len(kidx)), test_size=cfg["test_size"], stratify=strat[kidx], random_state=sd)
            tr, va = train_test_split(tr_all, test_size=0.1, stratify=strat[kidx][tr_all], random_state=sd)
            u = uidx if len(uidx) <= s4["max_unknown"] else np.random.default_rng(sd).choice(uidx, s4["max_unknown"], replace=False)
            hashes.append({"dataset": name, "family": fam, "seed": sd, "train_sha": hashlib.sha256(kidx[tr].tobytes()).hexdigest()[:16],
                           "val_sha": hashlib.sha256(kidx[va].tobytes()).hexdigest()[:16], "test_sha": hashlib.sha256(kidx[te].tobytes()).hexdigest()[:16],
                           "unknown_sha": hashlib.sha256(np.sort(u).tobytes()).hexdigest()[:16], "run_id": run_id})
            if sd == F["loao_seeds"][0]:
                np.savez_compressed(os.path.join(sp, f"loao_{fam}_seed{sd}.npz"), train=vidx[kidx[tr]], val=vidx[kidx[va]], test=vidx[kidx[te]], unknown=vidx[u])
            Xs = B2.prep(Xraw, kidx[tr])
            Xtr, Xva, Xte, Xu = Xs[kidx[tr]], Xs[kidx[va]], Xs[kidx[te]], Xs[u]
            S = {}
            rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight="balanced_subsample", random_state=sd).fit(Xtr, ky[tr])
            f = lambda X: 1 - rf.predict_proba(X).max(1)
            S["rf:maxprob"] = (f(Xva), f(Xte), f(Xu))
            closed.append({"dataset": name, "family": fam, "seed": sd, "model": "rf", "known_MacroF1": f1_score(ky[te], rf.predict(Xte), average="macro", labels=np.unique(ky[te]), zero_division=0), "run_id": run_id})
            net = O4.fit_net(MLP_SPEC, Xtr, ky[tr], Xva, ky[va], K, sd, s2)
            Xvt = np.vstack([Xva, Xte]); nv = len(Xva)
            for sn, (sk, su) in O4.deep_scores("softmax_en", net, Xtr, ky[tr], Xvt, Xu, K, 0).items():
                S[f"mlp:{sn}"] = (sk[:nv], sk[nv:], su)
            Zk, _, _ = O4.net_outputs(net, Xte)
            closed.append({"dataset": name, "family": fam, "seed": sd, "model": "mlp_softmax_en", "known_MacroF1": f1_score(ky[te], Zk.argmax(1), average="macro", labels=np.unique(ky[te]), zero_division=0), "run_id": run_id})
            if "edl_both" in s4["deep_models"]:
                enet = O4.fit_net(EDL_SPEC, Xtr, ky[tr], Xva, ky[va], K, sd, s2)
                sc_e = O4.deep_scores("edl_both", enet, Xtr, ky[tr], Xvt, Xu, K, 0)["vacuity"]
                S["edl:vacuity"] = (sc_e[0][:nv], sc_e[0][nv:], sc_e[1])
            for kind, nm_ in (("rank", "fusion_rank"), ("z", "fusion_z")):
                comps = ["rf:maxprob", "mlp:mahalanobis"]
                fv, fk, fu = sc.fuse([S[c][0] for c in comps], [[S[c][i] for c in comps] for i in (0, 1, 2)], kind, F.get("fusion_weights"))
                S[nm_] = (fv, fk, fu)
            for method, (sv, sk, su) in S.items():
                for cond, mask in (("with_normal", np.ones(len(sk), bool)), ("attacks_only", ~is_norm[te])):
                    if mask.sum() < 20: continue
                    r = sc.ood_full(sk[mask], su, sv[~is_norm[va]] if cond == "attacks_only" else sv)
                    r.update(dataset=name, variant=vn, family=fam, group=grp, near_far=nf, n_siblings=nsib, seed=sd, method=method, condition=cond,
                             n_known=int(mask.sum()), n_unknown=len(su), run_id=run_id); rows.append(r)
            cent = lambda X, yy: np.stack([X[yy == c].mean(0) for c in np.unique(yy)])
            d = np.sqrt(((cent(Xu, np.zeros(len(Xu), int))[0] - cent(Xtr, ky[tr])) ** 2).sum(1)) / np.sqrt(Xtr.shape[1])
            prox.append({"dataset": name, "family": fam, "seed": sd, "proximity_dist": float(d.min()), "nearest_known": kcls[np.unique(ky[tr])[d.argmin()]], "run_id": run_id})
            if sd in F["save_scores_seeds"]:
                cap = F["stored_known_scores_cap"]; ii = np.random.default_rng(0).choice(len(Xte), min(cap, len(Xte)), replace=False)
                np.savez_compressed(os.path.join(pr, f"{fam}_seed{sd}.npz"), known_is_normal=is_norm[te][ii],
                                    **{f"{m}__known": v[1][ii] for m, v in S.items()}, **{f"{m}__unknown": v[2] for m, v in S.items()})
            pd.DataFrame(rows).to_csv(part, index=False)
            for k, d in (("known_class_closed_set_F1", closed), ("proximity", prox), ("split_hashes", hashes)):
                pd.DataFrame(d).to_csv(aux[k], index=False)
            L(f"{fam} [{nf}, siblings={nsib}] seed={sd}: {len(S)} scores, {time.time()-t0:.0f}s")
    res = pd.DataFrame(rows); res.to_csv(os.path.join(out, "loao_by_family_seed_method.csv"), index=False)
    for nm_, d in (("known_class_closed_set_F1", closed), ("proximity", prox), ("split_hashes", hashes)):
        if d: pd.DataFrame(d).to_csv(os.path.join(out, nm_ + ".csv"), index=False)
    if not res.empty:
        print(res[(res.condition == "attacks_only")].groupby("method").AUROC.mean().round(3).sort_values(ascending=False))
    return {"dataset": name, "families": len(families), "rows": len(res)}, f"### {NAME} {name}: {len(families)} families\n"
