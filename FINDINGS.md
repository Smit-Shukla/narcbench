# Audit of NARCBench: findings so far

Work-in-progress audit of [NARCBench](https://arxiv.org/abs/2604.01151) (Rose et al., 2026), forked from [aaronrose227/narcbench](https://github.com/aaronrose227/narcbench). This file lists only what has been run. Open items are listed at the end.

## 1. Reproduction (authors' published Qwen3-32B activations)

`reproduce/reproduce_LR.py`, unchanged, 10 seeds × layers 26–30. Output in `results/qwen3_32b/`.

| | Paper (Table 13) | This run |
|---|---|---|
| Transfer (mean of 6 families × 5 methods) | 0.78 | 0.80 |
| Stego | 1.00 | 1.00 |
| Held-out Core | 1.00 | 1.00 |

The small Transfer gap is likely the scikit-learn version (1.9.1 here; the repo pins 1.7.2).

## 2. Prompt length alone matches the probes on Stego

`analysis/length_baseline.py` → `results/length_baseline.json`. Each agent is scored by its token count only (no model, no probe), at the same phases the paper evaluates. AUROC per agent / per run (run = longest agent):

| Model | Core (discussion) | Transfer (final) | Stego (final) | Paper probes, Transfer / Stego |
|---|---|---|---|---|
| Qwen3-32B | 0.88 / 0.89 | 0.59 / 0.74 | **1.00 / 1.00** | 0.78 / 1.00 |
| Llama-3.1-70B | 0.85 / 0.84 | 0.51 / 0.43 | **1.00 / 1.00** | 0.87 / 1.00 |
| DeepSeek-R1-32B | 0.84 / 0.82 | 0.39 / 0.34 | **1.00 / 1.00** | 0.75 / 0.92 |
| GPT-OSS-20B | 0.80 / 0.79 | 0.39 / 0.40 | **0.95 / 0.83** | 0.64 / 0.71 |

Transfer is not explained by length. Stego is fully matched by length on every model, and Core (the training data) is strongly length-separable. The cause: colluders' prompts include the private coordination chat; honest agents' don't.

## 3. The probe is not simply a length detector (Qwen3-32B)

`analysis/probe_vs_length.py` → `results/probe_vs_length_qwen3_32b.txt`.

- Spearman correlation of probe score with token count, **within honest agents only**: −0.03 (Core), 0.16 (Transfer), −0.13 (Stego).
- Regressing log(token count) out of the activations (label-free) and rerunning the full pipeline: Core 1.00 → 1.00, Transfer 0.80 → 0.80, Stego 1.00 → 0.985.

So the shortcut is plausibly the *content* of the private context, not its length. Linear length removal can't test that; see §6.

## 4. Bugs in extraction

1. **Stego final replies are dropped** (`extraction/extract_stego.py`). `generation/stego.py` stores final-hand replies in `results["final_texts"]`, but the loader only reads `public_messages`, so all 160 final-phase responses are extracted as `""`. `reproduce_LR.py` scores Stego on the final phase, so the published Stego activations encode prompts only. **Fixed** in commit `1ac00b4`. `tests/test_extract_stego_loader.py` builds a synthetic run in `stego.py`'s logging order: original code 8/8 final replies empty (FAIL), fixed 0 mismatches (PASS).
2. **Core private replies are blank** (`extraction/extract_core.py`). Private prompts are logged with phase `"private"`, but the loader matches `"discussion"`. This doesn't affect reported results (the probes use the public channel only). **Not fixed.**

## 5. Matched-context control

`generation/core.py --honest-private`: honest agents are paired and get the same private chat as colluders (same `private_frame`, same "PRIVATE COORDINATION" header in discussion prompts), so the colluder/honest difference is reduced to the system-prompt motive. With the flag off, prompts are identical to the original on all 50 scenarios × 2 modes (checked with a stub LLM).

## 6. Pilot on Qwen3-8B (Kaggle, 2× T4)

`kaggle/narcbench_qwen3_8b.ipynb` generated Core, Core-matched and Stego (240 runs), then extracted layers 13–19. Analysis: `analysis/pilot_8b.py` → `results/pilot_qwen3_8b.{txt,json}` (reproduce_LR's own probe and methods, 10 probe seeds × layers 14–18).

Filtering as in the paper: collusion runs where both colluders voted for the target, plus their paired controls. Collusion succeeded in 22/50 (Core) and 25/50 (matched). Scenarios with a vLLM timeout (`[ERROR]` reply) were dropped, leaving **20 scenario pairs per set**.

Run-level AUROC, mean of 5 methods:

| Test | AUROC |
|---|---|
| Length only: Core / Core-matched / Stego-fixed | 0.84 / **0.51** / 1.00 |
| Core probe, held-out Core | 1.000 |
| **Core probe → Stego, bug reproduced (final replies blanked)** | **0.955** |
| **Core probe → Stego, fixed** | **0.793** |
| Core probe → Core-matched | 0.994 |
| Matched probe, held-out Core-matched | 0.999 |
| Matched probe → Stego, fixed | 0.724 |

Mean probe projection per agent group (Core-trained probe): colluders +12.8 on matched data; honest agents −13.4 on original Core vs **−3.1 on matched Core**. A harmless private chat moves honest agents most of the way toward the colluder side, though separation remains.

**Limits:** one 8B model, one generation run (no repeat with new generation seeds), 20 scenario pairs; the Transfer tier was not generated.

## 7. Attempted, not completed: Qwen3-32B verification

`kaggle/stego_qwen3_32b.ipynb` would regenerate Stego with Qwen3-32B-AWQ and compare bug vs fix using the authors' published Core activations. It includes a gate (`kaggle/ref_check.py`) that re-extracts one of the authors' own runs and compares it with their published activations. On free T4 GPUs the AWQ model could not be loaded for extraction (gptqmodel kernel issues), so **no 32B results yet**.

## Open / not yet done
- Stego bug-vs-fix on the paper's 32B model.
- Whether probes detect collusion or the collusion *instruction*. Instructed data can't separate these.
- Emergent (uninstructed) collusion.

## Reproduce
```bash
python -m venv .venv && .venv/bin/pip install numpy scikit-learn matplotlib scipy
# authors' data from https://huggingface.co/datasets/aaronrose227/narcbench into data/
.venv/bin/python reproduce/reproduce_LR.py
.venv/bin/python analysis/length_baseline.py
.venv/bin/python analysis/probe_vs_length.py
.venv/bin/python tests/test_extract_stego_loader.py
# 8B pilot: run kaggle/narcbench_qwen3_8b.ipynb on Kaggle, unpack into data/, then:
.venv/bin/python analysis/pilot_8b.py
```
