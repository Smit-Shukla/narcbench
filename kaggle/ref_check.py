"""Check that our extraction reproduces the authors' published activations.

Compares our re-extraction of one of the authors' raw Qwen3-32B Core runs with
their published activations_gen.npz, row by row (public channel only), on
token count and per-layer cosine similarity.

Usage: python kaggle/ref_check.py PUBLISHED_DIR OURS_DIR RUN_ID
Exit code 0 = PASS (token counts identical, median cosine >= 0.99).
"""

import json
import sys
from pathlib import Path

import numpy as np


def rows(d, run_id):
    meta = json.load(open(Path(d) / "metadata_gen.json"))
    npz = np.load(Path(d) / "activations_gen.npz")
    idx = {(m["agent_name"], m["phase"], m["round"]): i for i, m in enumerate(meta)
           if m["run_id"] == run_id and m["channel"] == "public"}
    return meta, npz, idx


def main():
    pub_dir, our_dir, run_id = sys.argv[1:4]
    pm, pz, pidx = rows(pub_dir, run_id)
    om, oz, oidx = rows(our_dir, run_id)
    keys = sorted(set(pidx) & set(oidx), key=lambda k: (k[2], k[0]))
    layers = [k for k in pz.files if k in oz.files]
    print(f"{len(keys)} matching public samples, layers {layers}")

    tok_ok, cos_all = True, []
    for k in keys:
        pi, oi = pidx[k], oidx[k]
        pt, ot = pm[pi]["n_tokens"], om[oi]["n_tokens"]
        cos = [float(pz[l][pi] @ oz[l][oi] / (np.linalg.norm(pz[l][pi]) * np.linalg.norm(oz[l][oi]) + 1e-9))
               for l in layers]
        cos_all += cos
        tok_ok &= pt == ot
        print(f"  {k[1]:<10} r{k[2]} {k[0]:<20} tokens {pt}/{ot} {'OK ' if pt == ot else 'DIFF'} "
              f"cos min {min(cos):.4f}")
    med = float(np.median(cos_all)) if cos_all else 0.0
    ok = bool(keys) and tok_ok and med >= 0.99
    print(f"\ntoken counts identical: {tok_ok} | median cosine: {med:.4f} -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
