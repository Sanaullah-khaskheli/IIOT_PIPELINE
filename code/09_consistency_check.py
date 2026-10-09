"""Manuscript-consistency checker. Standalone (pandas/numpy/yaml). Usage: python 09_consistency_check.py --results results_final [--tex paper.tex]
Any FAIL prints 'TRACEABILITY FAILURE -- DO NOT REPORT' and exits 1."""
import os, re, sys, json, glob, argparse, hashlib
import numpy as np, pandas as pd

rep = []
def chk(name, ok, detail=""):
    rep.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)[:400]})
def warn(name, detail): rep.append({"check": name, "status": "WARN", "detail": str(detail)[:400]})
def close(a, b, tol=1e-6): return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol, equal_nan=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results_final"); ap.add_argument("--tex", default=None)
    ap.add_argument("--config", default="config_final.yaml"); a = ap.parse_args(); R = a.results
    meta = json.load(open(os.path.join(R, "run_meta.json"))) if os.path.exists(os.path.join(R, "run_meta.json")) else None
    chk("run_meta.json present", meta is not None); rid = meta["run_id"] if meta else None
    req = ["tables/T_ladder_closed_set.csv", "tables/T_loao_method_summary.csv", "tables/T_loao_paired_methods.csv", "tables/T_loao_near_vs_far.csv",
           "tables/numbers.tex", "tables/numbers_manifest.csv", "statistics/loao_per_family.csv", "figures/figure_manifest.csv", "figures/MANUSCRIPT_FIGURE_DATA_AUDIT.csv"]
    for f in req: chk(f"exists {f}", os.path.exists(os.path.join(R, f)))
    lad_fs = glob.glob(os.path.join(R, "closed_set", "*", "ladder_by_seed.csv")); loa_fs = glob.glob(os.path.join(R, "loao", "*", "loao_by_family_seed_method.csv"))
    chk("ladder files for >=1 dataset", len(lad_fs) > 0); chk("loao files for >=1 dataset", len(loa_fs) > 0)
    if not (lad_fs and loa_fs and all(os.path.exists(os.path.join(R, f)) for f in req)):
        return finish()
    for f in lad_fs + loa_fs + glob.glob(os.path.join(R, "tables", "*.csv")):
        d = pd.read_csv(f, nrows=2000)
        if "run_id" in d.columns: chk(f"run_id {os.path.relpath(f, R)}", set(d.run_id.astype(str)) == {str(rid)}, set(d.run_id.astype(str)))
        if meta and os.path.getmtime(f) < meta["started_epoch"] - 5: warn(f"older than run start {os.path.relpath(f, R)}", "check it was produced by this run_id (merged folders are fine)")
    lad = pd.concat([pd.read_csv(f) for f in lad_fs]); T = pd.read_csv(os.path.join(R, "tables/T_ladder_closed_set.csv"))
    g = lad.groupby(["dataset", "rung", "model"]).agg(MacroF1=("MacroF1", "mean"), sd=("MacroF1", "std"), n=("seed", "nunique")).reset_index()
    m = T.merge(g, on=["dataset", "rung", "model"], suffixes=("", "_re"))
    chk("ladder table rows == recomputed groups", len(m) == len(g) == len(T), f"{len(m)}/{len(g)}/{len(T)}")
    chk("ladder MacroF1 mean matches seeds", close(m.MacroF1, m.MacroF1_re)); chk("ladder MacroF1 sd matches seeds", close(m.MacroF1_sd, m.sd))
    chk("ladder n_seeds matches", (m.n_seeds == m.n).all())
    lo = pd.concat([pd.read_csv(f) for f in loa_fs]); S = pd.read_csv(os.path.join(R, "tables/T_loao_method_summary.csv"))
    ps = lo.groupby(["dataset", "condition", "method", "seed"]).AUROC.mean().groupby(["dataset", "condition", "method"]).agg(["mean", "std"]).reset_index()
    m2 = S.merge(ps, on=["dataset", "condition", "method"])
    chk("loao summary rows == recomputed", len(m2) == len(S) == len(ps), f"{len(m2)}/{len(S)}/{len(ps)}")
    chk("loao AUROC mean matches seed-level", close(m2.AUROC, m2["mean"])); chk("loao AUROC sd matches seed-level", close(m2.AUROC_sd_seeds, m2["std"]))
    for f in lad_fs:
        ds = os.path.basename(os.path.dirname(f)); d = pd.read_csv(f); cc = os.path.join(os.path.dirname(f), "class_counts.csv")
        if os.path.exists(cc):
            tot = pd.read_csv(cc).n_rows_used.sum(); chk(f"{ds}: n_train+n_val+n_test == rows used", ((d.n_train + d.n_val + d.n_test) == tot).all(), tot)
        pc = pd.read_csv(os.path.join(os.path.dirname(f), "per_class_by_seed.csv")).groupby(["rung", "model", "seed"]).support.sum().reset_index()
        mm = pc.merge(d[["rung", "model", "seed", "n_test"]], on=["rung", "model", "seed"])
        chk(f"{ds}: per-class supports sum to n_test", (mm.support == mm.n_test).all())
        chk(f"{ds}: every (rung,model) has the same seed set", d.groupby(["rung", "model"]).seed.apply(lambda s: tuple(sorted(s))).nunique() == 1)
    for f in loa_fs:
        ds = os.path.basename(os.path.dirname(f)); d = pd.read_csv(f); inv = pd.read_csv(os.path.join(os.path.dirname(f), "family_inventory.csv"))
        ev = set(inv[inv.evaluated].family); chk(f"{ds}: all evaluated families present in results", ev == set(d.family), ev ^ set(d.family))
        cnt = d.groupby(["family", "method", "condition"]).seed.nunique(); chk(f"{ds}: complete family x method x seed grid", cnt.nunique() == 1, cnt.value_counts().to_dict())
        nn = d.groupby("family").n_unknown.max().reindex(inv[inv.evaluated].family.values); chk(f"{ds}: n_unknown <= family rows", (nn.values <= inv[inv.evaluated].n_rows.values).all())
        chk(f"{ds}: near/far assigned for every family", d.near_far.isin(["near", "far"]).all())
    man = pd.read_csv(os.path.join(R, "tables/numbers_manifest.csv")); tex = open(os.path.join(R, "tables/numbers.tex")).read()
    ok = all(re.search(r"\\newcommand\{\\%s\}\{%.3f\}" % (r.macro, r.value), tex) for r in man.itertuples())
    chk("numbers.tex == numbers_manifest.csv", ok)
    bad = []
    for r in man.itertuples():
        fn, col = r.source.split(":"); d = pd.read_csv(os.path.join(R, fn))
        if not np.isclose(d[col].astype(float).round(3), r.value, atol=1e-3).any(): bad.append(r.macro)
    chk("every macro value occurs in its source column", not bad, bad[:5])
    au = pd.read_csv(os.path.join(R, "figures/MANUSCRIPT_FIGURE_DATA_AUDIT.csv"))
    for r in au.itertuples():
        chk(f"figure {r.figure}: source exists & rows>0", os.path.exists(os.path.join(R, r.source_file)) and r.rows_used > 0)
        chk(f"figure {r.figure}: file written", os.path.exists(os.path.join(R, "figures", r.file + ".pdf")))
    if os.path.exists(a.config):
        import yaml; c = yaml.safe_load(open(a.config))["final"]; tp = os.path.join(os.path.dirname(os.path.abspath(a.config)), c["taxonomy_file"])
        if c.get("taxonomy_sha256") and os.path.exists(tp): chk("taxonomy frozen (sha256 matches config)", hashlib.sha256(open(tp, "rb").read()).hexdigest() == c["taxonomy_sha256"])
        else: chk("taxonomy sha256 set in config", False, "final.taxonomy_sha256 is empty")
    if a.tex:
        t = open(a.tex, encoding="utf8").read(); t = re.sub(r"%.*", "", t)
        nums = set(re.findall(r"(?<![\w\\.])0\.\d{3}(?!\d)", re.sub(r"\\(cite|ref|label)\{[^}]*\}", "", t)))
        known = {f"{v:.3f}" for v in man.value}
        un = sorted(nums - known); (warn if un else (lambda n, d: chk(n, True, d)))("hard-coded 3-decimal numbers in manuscript not from numbers.tex", un[:20])
    finish()


def finish():
    df = pd.DataFrame(rep); print(df.to_string(index=False)); fail = (df.status == "FAIL").any()
    if len(df): df.to_csv(os.path.join(os.getcwd(), "consistency_report.csv"), index=False)
    if fail:
        print("\nTRACEABILITY FAILURE -- DO NOT REPORT"); sys.exit(1)
    print("\nALL CHECKS PASSED (warnings, if any, need a manual look)")


if __name__ == "__main__":
    main()
