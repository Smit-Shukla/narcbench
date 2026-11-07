"""Check that extract_stego.load_blackjack_runs matches each prompt to its reply.

Builds a synthetic run laid out exactly as generation/stego.py writes it:
  - private coordination (collusion only)
  - hands 1..N-1: public table talk  -> results["public_messages"]
  - judge check-in at hand 4 (discussion) -> results["public_messages"]
  - final hand N: table talk + judge  -> results["final_texts"] (keyed by agent)

Runs without GPU libraries: torch/transformers/tqdm are stubbed, since the
loader itself is plain Python.

Usage: python tests/test_extract_stego_loader.py [path/to/extract_stego.py]
"""

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PLAYERS = ["Ana", "Ben", "Cy", "Dee"]
JUDGE = "PitBoss"
N_HANDS = 8
JUDGE_HANDS = [4, 8]


def _stub_gpu_libs():
    """Minimal stand-ins so extract_stego imports without torch/transformers."""
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    nn.Module = object
    nn.functional = types.SimpleNamespace(gelu=None)
    torch.nn, torch.Tensor = nn, object
    act = types.ModuleType("transformers.activations")
    act.PytorchGELUTanh = object
    tf = types.ModuleType("transformers")
    tf.activations, tf.AutoModelForCausalLM, tf.AutoTokenizer = act, None, None
    tqdm = types.ModuleType("tqdm")
    tqdm.tqdm = lambda x, **k: x
    sys.modules.update({"torch": torch, "torch.nn": nn, "transformers": tf,
                        "transformers.activations": act, "tqdm": tqdm})


def _load_module(path):
    _stub_gpu_libs()
    spec = importlib.util.spec_from_file_location("extract_stego_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_run(run_dir, mode):
    """Write a synthetic run; return {(phase, round, agent): expected reply}."""
    roles = {p: ("colluder" if mode == "collusion" and p in PLAYERS[:2] else "honest")
             for p in PLAYERS}
    roles[JUDGE] = "judge"
    prompts, expected = [], {}
    private, public, final = [], [], {}

    def log(agent, phase, channel, rnd):
        prompts.append({"agent_name": agent, "role": roles[agent], "phase": phase,
                        "channel": channel, "round": rnd,
                        "system_prompt": "sys", "user_prompt": "user"})

    if mode == "collusion":
        for i, agent in enumerate(PLAYERS[:2] * 2):        # 4 coordination rounds
            reply = f"private reply {i} from {agent}"
            log(agent, "private", "private", 0)
            private.append(f"[{agent}]: {reply}")

    for hand in range(1, N_HANDS + 1):
        is_final = hand == N_HANDS
        phase = "final" if is_final else "discussion"
        for agent in PLAYERS:
            reply = f"hand {hand} talk from {agent}"
            log(agent, phase, "public", hand)
            expected[(phase, hand, agent)] = reply
            if is_final:
                final[agent] = reply
            else:
                public.append(f"[{agent}]: {reply}")
        if hand in JUDGE_HANDS:
            reply = f"judge note at hand {hand}"
            log(JUDGE, phase, "public", hand)
            if is_final:
                final[JUDGE] = reply
            else:
                public.append(f"[{JUDGE}]: {reply}")

    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(json.dumps({
        "run_id": run_dir.name, "seed": 0, "mode": mode, "roles": roles,
        "coalition": PLAYERS[:2] if mode == "collusion" else []}))
    (run_dir / "agent_prompts.json").write_text(json.dumps(prompts))
    (run_dir / "results.json").write_text(json.dumps({
        "private_messages": private, "thinking_messages": [],
        "public_messages": public, "final_texts": final}))
    return expected


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "extraction" / "extract_stego.py"
    mod = _load_module(path)

    with tempfile.TemporaryDirectory() as tmp:
        expected = {}
        for mode in ("collusion", "control"):
            for k, v in make_run(Path(tmp) / f"shoe_00_{mode}", mode).items():
                expected[(f"shoe_00_{mode}",) + k] = v
        samples = mod.load_blackjack_runs(Path(tmp))

    public = [s for s in samples if s["channel"] == "public"]
    wrong = [s for s in public
             if s["response"] != expected[(s["run_id"], s["phase"], s["round"], s["agent_name"])]]
    empty_final = [s for s in public if s["phase"] == "final" and s["response"] == ""]

    print(f"\npublic samples: {len(public)} | mismatched: {len(wrong)} | "
          f"empty final replies: {empty_final.__len__()}")
    if wrong:
        for s in wrong[:3]:
            print(f"  e.g. {s['run_id']} {s['phase']} hand {s['round']} {s['agent_name']}: "
                  f"got {s['response']!r}")
        print("FAIL")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
