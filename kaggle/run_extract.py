"""Run an extraction script on free-tier GPUs (T4) without editing it.

- Forces fp16 (T4 has no fast bf16) by patching from_pretrained.
- --stego-bug: reproduce the original Stego bug on purpose. Final-phase replies
  are blanked to "" exactly as the unfixed loader did, and only final-phase
  samples are kept (the only phase the paper scores Stego on).

Usage:
  python kaggle/run_extract.py core  [extract_core args...]
  python kaggle/run_extract.py stego [--stego-bug] [extract_stego args...]
"""

import sys
from pathlib import Path

import torch
import transformers

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extraction"))

_orig = transformers.AutoModelForCausalLM.from_pretrained


from packaging.version import Version

# transformers >= 4.56 renamed torch_dtype -> dtype
_DTYPE_KW = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"


def _fp16(*args, **kwargs):
    kwargs.setdefault(_DTYPE_KW, torch.float16)
    return _orig(*args, **kwargs)


transformers.AutoModelForCausalLM.from_pretrained = _fp16


def main():
    which, rest = sys.argv[1], sys.argv[2:]
    bug = "--stego-bug" in rest
    final_only = "--final-only" in rest
    rest = [a for a in rest if a not in ("--stego-bug", "--final-only")]

    if which == "core":
        import extract_core as mod
    elif which == "stego":
        import extract_stego as mod
        if bug or final_only:
            original_loader = mod.load_blackjack_runs

            def wrapped_loader(runs_dir):
                # final phase is the only phase the paper scores Stego on
                samples = [s for s in original_loader(runs_dir) if s["phase"] == "final"]
                if bug:
                    for s in samples:
                        s["response"] = ""
                print(f"[run_extract] kept {len(samples)} final-phase samples"
                      + (", replies blanked (--stego-bug)" if bug else ""))
                return samples

            mod.load_blackjack_runs = wrapped_loader
    else:
        sys.exit(f"unknown script: {which}")

    mod.AutoModelForCausalLM.from_pretrained = _fp16
    sys.argv = [f"extract_{which}.py"] + rest
    mod.main()


if __name__ == "__main__":
    main()
