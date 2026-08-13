#!/usr/bin/env python3
"""Fully-offline smoke test: runs the child-mode pipeline with the mock agent
for one sfm spec and one reference spec, then prints rendered prompts and a
summary of the files produced. Needs torch + networkx (for the fits) but no
GPU, no internet, no model weights.

Usage (from anywhere): python3 scripts/smoke_local.py
"""

import argparse
import logging
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

SMOKE_KEYS = ["unfiltered_e2e_alignment_dpo", "llama2_7b_chat"]
RESULTS_DIR = os.path.join(REPO_ROOT, "results_smoke_local")


def main() -> int:
    try:
        import networkx  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        print(f"Missing dependency ({e.name}): the Thurstonian fits need torch and "
              "networkx. Install requirements-local.txt first "
              "(pip install -r requirements-local.txt).")
        return 1

    from geodesic_ue import outcomes, prompts, registry
    from geodesic_ue.agent import MockChoiceAgent
    from geodesic_ue.experiments import run_all_for_model

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout, force=True)
    logger = logging.getLogger("geodesic_ue")

    args = argparse.Namespace(
        models=",".join(SMOKE_KEYS), experiments="all", agent="mock",
        results_dir=RESULTS_DIR, smoke=True, seed=42,
        gpu_mem_util=0.92, max_model_len=2048, child=True)

    summaries = {}
    for key in SMOKE_KEYS:
        spec = registry.SPECS_BY_KEY[key]
        summaries[key] = run_all_for_model(spec, args, logger)

    # Eyeball checks: exactly what the scorer sees for chat and base models.
    base_opts = outcomes.load_base_options()
    d1, d2 = base_opts[0]["description"], base_opts[1]["description"]
    mock = MockChoiceAgent()
    chat_prompt = prompts.build_prompt(True, mock.tokenizer,
                                       prompts.QUESTION_PREFERENCE, d1, d2, 0)
    base_prompt = prompts.build_base_prompt(prompts.QUESTION_PREFERENCE, d1, d2, 0)
    print("\n--- Rendered chat prompt (FakeTokenizer, template 0) ---")
    print(chat_prompt)
    print("--- Rendered base prompt (template 0) ---")
    print(base_prompt)

    print("\n--- Files produced under results_smoke_local/ ---")
    for root, _dirs, files in sorted(os.walk(RESULTS_DIR)):
        if not files:
            continue
        rel = os.path.relpath(root, REPO_ROOT)
        print(f"{rel}/ ({len(files)} files)")
        for fn in sorted(files):
            size = os.path.getsize(os.path.join(root, fn))
            print(f"  {fn}  ({size:,} bytes)")

    print("\n--- Experiment statuses ---")
    ok = True
    for key, summary in summaries.items():
        statuses = {e: v["status"] for e, v in summary["experiments"].items()}
        print(f"{key}: {statuses}")
        if any(s != "ok" for s in statuses.values()):
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
