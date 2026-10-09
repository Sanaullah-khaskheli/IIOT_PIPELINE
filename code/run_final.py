"""SAFE-IIoT FINAL RUN (Kaggle: Save Version -> Save & Run All (Commit) -> 'Always save output').
    !python run_final.py --stage taxonomy                       # once: writes taxonomy.csv -> REVIEW -> paste sha256 into config_final.yaml
    !python run_final.py --stage all --datasets edge_iiotset    # then per dataset (one notebook version per dataset is safest)
    stages (comma list ok): audit (00+01) | ladder | loao | stats | figures | check | all
Everything is written to results_final/ with a run_id in every CSV; nothing from older runs is reused."""
import argparse, importlib, json, os, sys, time, uuid, datetime as dt, subprocess
import pandas as pd
from common import *

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config_final.yaml"); ap.add_argument("--stage", default="all")
    ap.add_argument("--datasets", default=None); ap.add_argument("--run-id", default=None, help="share ONE id across per-dataset notebooks"); a = ap.parse_args()
    cfg = load_config(os.path.join(HERE, a.config))
    if a.datasets: cfg["active_datasets"] = a.datasets.split(",")
    root = results_root(cfg)
    for s in ("data_audit", "cleaned_features", "splits", "closed_set", "open_set", "loao", "statistics", "predictions", "figures", "tables", "logs"):
        os.makedirs(os.path.join(root, s), exist_ok=True)
    mp = os.path.join(root, "run_meta.json")
    if os.path.exists(mp): meta = json.load(open(mp))
    else:
        meta = {"run_id": a.run_id or (dt.datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]), "started_epoch": time.time(), "started": dt.datetime.now().isoformat(timespec="seconds")}
        json.dump(meta, open(mp, "w"), indent=1)
    rid = meta["run_id"]; seed_everything(cfg["seeds"][0])
    import yaml; yaml.safe_dump(cfg, open(os.path.join(root, "config_used.yaml"), "w"))
    if a.stage == "taxonomy":
        import importlib as il; il.import_module("05_taxonomy").main(a.config, cfg["active_datasets"], cfg); return
    stages = set(a.stage.split(",")); run = lambda s: "all" in stages or s in stages
    for ds in cfg["active_datasets"]:
        for stage, mod, key in (("audit", "00_dataset_audit", None), ("audit", "01_leakage_audit", None), ("ladder", "05_ladder_closed", rid), ("loao", "06_loao_final", rid)):
            if not run(stage): continue
            m = importlib.import_module(mod)
            try:
                log(cfg, "run_final", f"START {mod}:{ds}")
                m.main(cfg, ds) if key is None else m.main(cfg, ds, key)
                log(cfg, "run_final", f"DONE  {mod}:{ds}")
            except Exception as e:
                record_failure(cfg, f"{mod}:{ds}", e)
    if run("stats"):
        importlib.import_module("07_stats_tables").main(cfg, rid)
    if run("figures"):
        subprocess.run([sys.executable, os.path.join(HERE, "08_make_figures.py"), "--results", root], check=False)
    if run("check"):
        subprocess.run([sys.executable, os.path.join(HERE, "09_consistency_check.py"), "--results", root, "--config", os.path.join(HERE, a.config)], check=False)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
    zip_dir(root, os.path.join(os.path.dirname(root), f"SAFE_IIoT_RESULTS_{ts}.zip"))
    print("DONE -> upload SAFE_IIoT_RESULTS_*.zip (or the whole results_final folder) to continue with the manuscript")

if __name__ == "__main__":
    main()
