"""Step 0 (run ONCE, then REVIEW and FREEZE): propose the operational attack-family taxonomy from the class names that
actually exist in each dataset. Writes taxonomy.csv and prints its sha256. Paste the hash into config_final.yaml
(final.taxonomy_sha256). Later scripts refuse to run if the file changes afterwards (no post-hoc regrouping)."""
import os, sys, pandas as pd
from common import *
import safe_core as sc

def main(cfg_path="config_final.yaml", datasets=None, cfg=None):
    cfg = cfg or load_config(os.path.join(HERE, cfg_path)); fam = {}
    for name in (datasets or cfg["active_datasets"]):
        ds = get_dataset(cfg, name, cfg["seeds"][0])
        normal = {t.lower() for t in ds.cfg["normal_tokens"]}
        vc = pd.Series(ds.y_family).value_counts()
        fam[name] = [c for c in vc.index if c.lower() not in normal]
    t = sc.build_taxonomy(fam)
    p = os.path.join(HERE, cfg["final"]["taxonomy_file"])
    if os.path.exists(p):
        old = pd.read_csv(p); t = pd.concat([old[~old.dataset.isin(t.dataset.unique())], t], ignore_index=True)
    t.to_csv(p, index=False)
    print(t.to_string()); print("\nsha256 =", sc.file_sha256(p))
    print("REVIEW the 'group' column by hand NOW (siblings = same attack mechanism), save, re-run this script to print the final hash,"
          " then put it in config_final.yaml -> final.taxonomy_sha256.")

if __name__ == "__main__":
    main()
