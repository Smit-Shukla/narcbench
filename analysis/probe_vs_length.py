"""Does the paper's probe track prompt length?

3a. Correlation: Spearman correlation between each agent's probe score and its
    token count, overall and within honest agents only (within-role correlation
    can't be explained by colluder-vs-honest differences).
3b. Length removal: for each dataset and layer, regress every activation
    dimension on log(n_tokens) and subtract the fit (label-free, so no leakage).
    Then rerun the paper's full pipeline (reproduce_LR) and compare AUROCs.

Usage: python analysis/probe_vs_length.py [--model qwen3_32b] [--seeds 10]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reproduce"))
import config
import reproduce_LR as R

TIERS = [("core", "discussion"), ("transfer", "final"), ("stego", "final")]


# ── 3a: correlation ──────────────────────────────────────────────────────

def agent_scores_and_lengths(meta, npz, layer, phase, scaler, w):
    """Per (run, agent): probe score and token count (discussion averaged over rounds)."""
    acts = R.get_layer(npz, len(meta), layer)
    vecs, lens, role = defaultdict(list), defaultdict(list), {}
    for i, m in enumerate(meta):
        if m["channel"] != "public" or m["phase"] != phase:
            continue
        k = (m["run_id"], m["agent_name"])
        vecs[k].append(acts[i])
        lens[k].append(m["n_tokens"])
        role[k] = m["role"]
    keys = list(vecs)
    X = scaler.transform(np.stack([np.mean(vecs[k], axis=0) for k in keys]))
    return X @ w, np.array([np.mean(lens[k]) for k in keys]), np.array([role[k] for k in keys])


def correlation(data, layers):
    print("\n3a. Spearman correlation: probe score vs token count (seed 0, mean over layers)")
    print(f"{'tier':<10}{'all agents':>12}{'honest only':>13}{'colluders only':>16}")
    core_meta, core_npz = data["core"]
    for tier, phase in TIERS:
        r_all, r_hon, r_col = [], [], []
        for layer in layers:
            scaler, w = R.train_deception_probe(core_meta, core_npz, layer, seed=0)
            s, n, role = agent_scores_and_lengths(*data[tier], layer, phase, scaler, w)
            r_all.append(spearmanr(s, n)[0])
            r_hon.append(spearmanr(s[role == "honest"], n[role == "honest"])[0])
            r_col.append(spearmanr(s[role == "colluder"], n[role == "colluder"])[0])
        print(f"{tier:<10}{np.mean(r_all):>12.2f}{np.mean(r_hon):>13.2f}{np.mean(r_col):>16.2f}")


# ── 3b: length removal ───────────────────────────────────────────────────

def make_residualiser(datasets):
    """Patch R.get_layer so every dataset's activations have log-length regressed out."""
    log_len = {id(npz): np.log(np.array([m["n_tokens"] for m in meta], dtype=np.float64))
               for meta, npz in datasets}
    cache = {}
    original = R.get_layer

    def get_layer_residualised(npz, n, layer):
        key = (id(npz), layer)
        if key not in cache:
            X = original(npz, n, layer).astype(np.float64)
            L = np.column_stack([np.ones(n), log_len[id(npz)][:n]])
            beta, *_ = np.linalg.lstsq(L, X, rcond=None)   # per-dimension linear fit
            cache[key] = (X - L @ beta).astype(np.float32)
        return cache[key]

    return get_layer_residualised


def run_pipeline(data, seeds):
    """Run reproduce_LR's full computation; return group -> method -> mean AUROC."""
    acc = defaultdict(lambda: defaultdict(list))
    group_names = None
    for seed in range(seeds):
        gn, raw = R.compute_all_results(*data["core"], *data["transfer"], *data["stego"], seed=seed)
        group_names = gn
        for g in gn:
            for m in R.METHOD_KEYS:
                acc[g][m].extend(raw[g][m])
    return group_names, {g: {m: np.nanmean(acc[g][m]) for m in R.METHOD_KEYS} for g in group_names}


def summarise(group_names, means):
    transfer = [g for g in group_names if g not in ("Held-out", "blackjack")]
    return {
        "Held-out (Core)": np.mean([means["Held-out"][m] for m in R.METHOD_KEYS]),
        "Transfer": np.mean([means[g][m] for g in transfer for m in R.METHOD_KEYS]),
        "Stego": np.mean([means["blackjack"][m] for m in R.METHOD_KEYS]),
    }, {"Stego": means["blackjack"], "Held-out (Core)": means["Held-out"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3_32b")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    root = config.ACTIVATIONS_ROOT / args.model
    data = {t: R.load_data(root / t) for t, _ in TIERS}
    # Read each npz layer into memory once (np.load is lazy and re-reads otherwise)
    for t in data:
        meta, npz = data[t]
        data[t] = (meta, {k: npz[k] for k in npz.files})

    correlation(data, R.LAYERS)

    print(f"\n3b. Full pipeline, original vs length-removed ({args.seeds} seeds x {len(R.LAYERS)} layers)")
    print("  running original...")
    gn, orig = run_pipeline(data, args.seeds)
    R.get_layer = make_residualiser(data.values())
    print("  running length-removed...")
    _, resid = run_pipeline(data, args.seeds)

    o_sum, o_det = summarise(gn, orig)
    r_sum, r_det = summarise(gn, resid)
    print(f"\n{'mean AUROC over 5 methods':<28}{'original':>10}{'length removed':>16}")
    for k in o_sum:
        print(f"{k:<28}{o_sum[k]:>10.3f}{r_sum[k]:>16.3f}")
    for tier in ("Held-out (Core)", "Stego"):
        print(f"\n{tier} per method:")
        for m in R.METHOD_KEYS:
            print(f"  {m:<22}{o_det[tier][m]:>10.3f}{r_det[tier][m]:>16.3f}")


if __name__ == "__main__":
    main()
