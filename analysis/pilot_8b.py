"""Qwen3-8B pilot: Stego bug effect (H1) and matched-context control (H2).

Uses reproduce_LR's own functions (same probe, same five methods), on
layers 14-18 of 36 (~39-50% depth; the paper uses 26-30 of 64).

Data filtering (as in the paper): keep collusion runs where both colluders
voted for the target (collusion_success) and their paired control runs.
Scenarios whose collusion run contains an [ERROR] reply (vLLM timeout) are
dropped in both conditions.

Tests
  T0  Core probe, leave-domain-out on Core                 (sanity: does it work at 8B?)
  T1  Core probe -> Stego bug vs Stego fixed               (H1)
  T2  Core probe -> Core-matched                           (H2: does it flag honest agents with a private chat?)
  T3  Core-matched probe, leave-domain-out; -> Stego fixed (does a clean probe still work?)
  L   token count alone on each set                        (is the length shortcut gone?)

Usage: python analysis/pilot_8b.py [--seeds 10]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reproduce"))
import config
import reproduce_LR as R

R.LAYERS = [14, 15, 16, 17, 18]
ACT = config.ACTIVATIONS_ROOT / "qwen3_8b"
RAW = config.RAW_ROOT / "qwen3_8b"
SETS = ["core", "core_matched", "stego", "stego_bug"]


# ── Loading and filtering ────────────────────────────────────────────────

def error_scenarios(raw_dir):
    bad = set()
    for p in raw_dir.glob("*/results.json"):
        if "[ERROR]" in p.read_text():
            bad.add(p.parent.name.rsplit("__", 1)[0])   # scenario prefix, both modes
    return bad


def load(name):
    meta, npz = R.load_data(ACT / name)
    npz = {k: npz[k] for k in npz.files}
    if not name.startswith("core"):
        return meta, npz
    bad = error_scenarios(RAW / name)
    ok_scen = {m["run_id"].rsplit("__", 1)[0] for m in meta
               if m["mode"] == "collusion" and m["collusion_success"]} - bad
    keep = np.array([m["run_id"].rsplit("__", 1)[0] in ok_scen for m in meta])
    meta = [m for m, k in zip(meta, keep) if k]
    npz = {k: v[keep] for k, v in npz.items()}
    print(f"  {name}: kept {len(ok_scen)} scenario pairs "
          f"(dropped {len(bad)} with [ERROR]; rest failed to collude)")
    return meta, npz


# ── Evaluation helpers ───────────────────────────────────────────────────

def phase_of(name):
    return ("discussion", True) if name.startswith("core") else ("final", False)


def eval_transfer(train, test_name, test, seeds):
    """Train on `train` (Core-style), score `test` run-level; mean AUROC per method."""
    phase, avg = phase_of(test_name)
    acc = defaultdict(list)
    for seed in range(seeds):
        for layer in R.LAYERS:
            scaler, w = R.train_deception_probe(*train, layer, seed=seed)
            tr_g = R.build_groups(*train, layer, phase="discussion", avg_rounds=True)
            tr_r = sorted(tr_g)
            te_g = R.build_groups(*test, layer, phase=phase, avg_rounds=avg)
            te_r = sorted(te_g)
            res = R.evaluate(R.compute_scores(tr_g, tr_r, scaler, w), R._run_labels(tr_g, tr_r),
                             R.compute_scores(te_g, te_r, scaler, w), R._run_labels(te_g, te_r))
            for k, v in res.items():
                acc[k].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def eval_heldout(data, seeds):
    acc = defaultdict(list)
    for seed in range(seeds):
        for layer in R.LAYERS:
            for k, v in R.held_out_cv_layer(*data, layer, seed=seed).items():
                acc[k].append(v)
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def agent_scores(train, test_name, test, seeds=3):
    """Mean probe score per agent group, and colluder-vs-honest AUROC (agent level)."""
    phase, _ = phase_of(test_name)
    groups = defaultdict(list)
    aucs = []
    for seed in range(seeds):
        for layer in R.LAYERS:
            scaler, w = R.train_deception_probe(*train, layer, seed=seed)
            g = R.build_groups(*test, layer, phase=phase, avg_rounds=phase == "discussion")
            ys, ss = [], []
            for run in g.values():
                for agent, vec in run["agents"].items():
                    s = float(scaler.transform(vec.reshape(1, -1))[0] @ w)
                    role = run["roles"][agent]
                    key = "colluder" if role == "colluder" else f"honest ({run['mode']} run)"
                    groups[key].append(s)
                    ys.append(role == "colluder")
                    ss.append(s)
            aucs.append(roc_auc_score(ys, ss))
    return {k: float(np.mean(v)) for k, v in groups.items()}, float(np.mean(aucs))


def length_auc(name, data):
    meta, _ = data
    phase, _ = phase_of(name)
    per = defaultdict(list)
    info = {}
    for m in meta:
        if m["channel"] == "public" and m["phase"] == phase:
            per[(m["run_id"], m["agent_name"])].append(m["n_tokens"])
            info[(m["run_id"], m["agent_name"])] = (m["mode"], m["role"])
    keys = list(per)
    x = [np.mean(per[k]) for k in keys]
    agent = roc_auc_score([info[k][1] == "colluder" for k in keys], x)
    runs = defaultdict(list)
    for k, v in zip(keys, x):
        runs[k[0]].append(v)
    run = roc_auc_score([r.endswith("__collusion") for r in runs], [max(v) for v in runs.values()])
    return agent, run


def show(title, res):
    mean = np.mean(list(res.values()))
    cols = "  ".join(f"{k.replace('_', ' ')} {v:.3f}" for k, v in res.items())
    print(f"{title:<44} mean {mean:.3f} | {cols}")
    return mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    print("Loading...")
    D = {s: load(s) for s in SETS}
    summary = {}

    print("\nL. Token count alone (agent / run AUROC)")
    for s in SETS:
        a, r = length_auc(s, D[s])
        summary[f"length_{s}"] = (a, r)
        print(f"  {s:<14} {a:.3f} / {r:.3f}")

    print(f"\nRun-level AUROC, 5 methods ({args.seeds} seeds x layers {R.LAYERS})")
    summary["T0_core_heldout"] = show("T0  Core probe, held-out Core", eval_heldout(D["core"], args.seeds))
    summary["T1_stego_bug"] = show("T1  Core probe -> Stego BUG (empty replies)",
                                   eval_transfer(D["core"], "stego_bug", D["stego_bug"], args.seeds))
    summary["T1_stego_fixed"] = show("T1  Core probe -> Stego FIXED",
                                     eval_transfer(D["core"], "stego", D["stego"], args.seeds))
    summary["T2_core_to_matched"] = show("T2  Core probe -> Core-matched",
                                         eval_transfer(D["core"], "core_matched", D["core_matched"], args.seeds))
    summary["T3_matched_heldout"] = show("T3  Matched probe, held-out matched",
                                         eval_heldout(D["core_matched"], args.seeds))
    summary["T3_matched_to_stego"] = show("T3  Matched probe -> Stego FIXED",
                                          eval_transfer(D["core_matched"], "stego", D["stego"], args.seeds))

    print("\nAgent-level probe scores (mean projection; higher = more 'colluder-like')")
    for title, train, tname in [("Core probe on Core", "core", "core"),
                                ("Core probe on Core-matched", "core", "core_matched"),
                                ("Matched probe on Core-matched", "core_matched", "core_matched")]:
        means, auc = agent_scores(D[train], tname, D[tname])
        summary[f"agents_{train}_on_{tname}"] = {"means": means, "auc": auc}
        cols = "  ".join(f"{k}: {v:+.2f}" for k, v in sorted(means.items()))
        print(f"  {title:<32} colluder-vs-honest AUROC {auc:.3f} | {cols}")

    out = config.ROOT / "results" / "pilot_qwen3_8b.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
