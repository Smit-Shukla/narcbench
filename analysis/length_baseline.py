"""Prompt-length baseline: how well does token count ALONE detect collusion?

No model, no probe. For each agent we take n_tokens (prompt + response length,
from metadata_gen.json) as its "suspicion score", then compute AUROC exactly
where the paper evaluates its probes:
  - Core:     discussion phase (the probe's training data), averaged over rounds
  - Transfer: final phase, AUROC per family, then averaged (as in Table 13)
  - Stego:    final phase

Two levels:
  - agent: colluder vs honest agents
  - run:   max token count over the run's agents (like Peak Suspicion),
           collusion runs vs control runs

Usage: python analysis/length_baseline.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

MODELS = ["qwen3_32b", "llama_3_1_70b", "deepseek_r1_32b", "gpt_oss_20b"]
TIERS = {"core": "discussion", "transfer": "final", "stego": "final"}
# Paper Table 13: mean AUROC over the five probe methods
PAPER = {
    "llama_3_1_70b": {"transfer": 0.87, "stego": 1.00},
    "qwen3_32b": {"transfer": 0.78, "stego": 1.00},
    "deepseek_r1_32b": {"transfer": 0.75, "stego": 0.92},
    "gpt_oss_20b": {"transfer": 0.64, "stego": 0.71},
}


def agent_lengths(meta, phase):
    """One token count per (run, agent), averaged over rounds in that phase."""
    per_agent = defaultdict(list)
    info = {}
    for m in meta:
        if m["channel"] != "public" or m["phase"] != phase:
            continue
        key = (m["run_id"], m["agent_name"])
        per_agent[key].append(m["n_tokens"])
        info[key] = (m["mode"], m["role"])
    return {k: (float(np.mean(v)), *info[k]) for k, v in per_agent.items()}


def aurocs(agents):
    """Agent-level and run-level AUROC of token count."""
    y = [1 if role == "colluder" else 0 for _, _, role in agents.values()]
    x = [n for n, _, _ in agents.values()]
    agent_auc = roc_auc_score(y, x) if len(set(y)) == 2 else np.nan

    runs = defaultdict(list)
    run_mode = {}
    for (rid, _), (n, mode, _) in agents.items():
        runs[rid].append(n)
        run_mode[rid] = mode
    ry = [1 if run_mode[r] == "collusion" else 0 for r in runs]
    rx = [max(v) for v in runs.values()]
    run_auc = roc_auc_score(ry, rx) if len(set(ry)) == 2 else np.nan
    return agent_auc, run_auc, sum(ry), len(ry) - sum(ry)


def main():
    rows = []
    for model in MODELS:
        for tier, phase in TIERS.items():
            path = config.ACTIVATIONS_ROOT / model / tier / "metadata_gen.json"
            meta = json.load(open(path))
            agents = agent_lengths(meta, phase)

            if tier == "transfer":
                # per-family AUROC, then mean (matches paper's averaging)
                fams = defaultdict(dict)
                for k, v in agents.items():
                    fams[k[0].split("__")[0]][k] = v
                res = [aurocs(a) for a in fams.values()]
                agent_auc = np.nanmean([r[0] for r in res])
                run_auc = np.nanmean([r[1] for r in res])
                n_col, n_ctl = sum(r[2] for r in res), sum(r[3] for r in res)
            else:
                agent_auc, run_auc, n_col, n_ctl = aurocs(agents)

            paper = PAPER[model].get(tier, 1.00 if tier == "core" else np.nan)
            rows.append((model, tier, phase, agent_auc, run_auc, paper, n_col, n_ctl))

    print(f"{'model':<17}{'tier':<10}{'phase':<12}{'agent AUROC':>12}{'run AUROC':>11}"
          f"{'paper probes':>14}{'runs (col/ctl)':>16}")
    print("-" * 92)
    for model, tier, phase, a, r, p, nc, nk in rows:
        print(f"{model:<17}{tier:<10}{phase:<12}{a:>12.3f}{r:>11.3f}{p:>14.2f}{f'{nc}/{nk}':>16}")

    out = config.ROOT / "results" / "length_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump([dict(zip(["model", "tier", "phase", "agent_auroc", "run_auroc",
                         "paper_probe_auroc", "n_collusion_runs", "n_control_runs"], r))
               for r in rows], open(out, "w"), indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
