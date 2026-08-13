"""CLI runner. Parent mode spawns one subprocess per model (a dead vLLM engine
or CUDA state never poisons the next model); child mode runs a single model.

Run from repo root: python -m geodesic_ue.runner --models pod_a --smoke
"""

import argparse
import glob
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from typing import List, Optional

from . import REPO_ROOT, registry
from .experiments import run_all_for_model

GATE_MIN_MODELS = 6
GATE_MEDIAN_ACC = 0.60


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="geodesic_ue.runner",
        description="Utility elicitation over the Geodesic model suite.")
    p.add_argument("--models", default="all",
                   help="registry.resolve_keys arg: all | pod_a | pod_b | tier:dpo | csv of keys")
    p.add_argument("--experiments", default="all",
                   help="csv subset of experiments (default: all)")
    p.add_argument("--agent", choices=("vllm", "hf", "mock"), default="vllm")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-mem-util", type=float, default=0.92)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)  # internal
    return p


def setup_logging(log_to_file: bool) -> logging.Logger:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_to_file:
        os.makedirs("logs", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        handlers.append(logging.FileHandler(os.path.join("logs", f"runner_{ts}.log")))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers, force=True)
    return logging.getLogger("geodesic_ue")


def _load_summary(results_dir: str, key: str) -> Optional[dict]:
    path = os.path.join(results_dir, "summary", f"{key}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _gate_check(results_dir: str, logger: logging.Logger) -> bool:
    """True = gate FAILED (stop the sweep). Fires once >=6 models have a
    base_utilities holdout accuracy and their median is below 0.60 — the
    signal that logprob readout isn't working on this suite at all."""
    accs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "summary", "*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        acc = (data.get("experiments", {})
                   .get("base_utilities", {})
                   .get("holdout_accuracy"))
        if isinstance(acc, (int, float)):
            accs[data.get("model", os.path.basename(path))] = float(acc)
    if len(accs) < GATE_MIN_MODELS:
        return False
    median = statistics.median(accs.values())
    if median >= GATE_MEDIAN_ACC:
        return False
    payload = {"median_holdout_accuracy": median,
               "threshold": GATE_MEDIAN_ACC,
               "n_models": len(accs),
               "accuracies": accs,
               "ts": time.time()}
    with open(os.path.join(results_dir, "GATE_FAILED.json"), "w") as f:
        json.dump(payload, f, indent=2)
    banner = "!" * 78
    logger.error("\n%s\n!! GATE FAILED: median base_utilities holdout accuracy "
                 "%.3f < %.2f across %d models.\n!! Wrote %s. STOPPING — "
                 "remaining models will not run.\n%s",
                 banner, median, GATE_MEDIAN_ACC, len(accs),
                 os.path.join(results_dir, "GATE_FAILED.json"), banner)
    return True


def _run_child(args, logger: logging.Logger) -> None:
    keys = registry.resolve_keys(args.models)
    if len(keys) != 1:
        raise SystemExit(f"--child expects exactly one model key, got {keys}")
    spec = registry.SPECS_BY_KEY[keys[0]]
    summary = run_all_for_model(spec, args, logger)
    statuses = [e.get("status") for e in summary["experiments"].values()]
    if statuses and all(s == "failed" for s in statuses):
        sys.exit(1)


def _run_parent(args, logger: logging.Logger) -> None:
    keys = registry.resolve_keys(args.models)
    logger.info("Running %d models: %s", len(keys), ", ".join(keys))
    os.makedirs(args.results_dir, exist_ok=True)
    progress: List[dict] = []
    progress_path = os.path.join(args.results_dir, "progress.json")

    for key in keys:
        cmd = [sys.executable, "-u", "-m", "geodesic_ue.runner",
               "--models", key, "--child",
               "--experiments", args.experiments,
               "--agent", args.agent,
               "--results-dir", args.results_dir,
               "--seed", str(args.seed),
               "--gpu-mem-util", str(args.gpu_mem_util),
               "--max-model-len", str(args.max_model_len)]
        if args.smoke:
            cmd.append("--smoke")

        t0 = time.time()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                cwd=REPO_ROOT)
        assert proc.stdout is not None
        for line in proc.stdout:  # live passthrough of the child's log
            sys.stdout.write(line)
            sys.stdout.flush()
        code = proc.wait()
        seconds = time.time() - t0

        status = "ok" if code == 0 else "failed"
        if code != 0:
            logger.error("!! MODEL %s FAILED (exit %d) — continuing", key, code)
        summary = _load_summary(args.results_dir, key) or {}
        progress.append({"model": key, "status": status, "seconds": seconds,
                         "prompts_per_s": summary.get("prompts_per_s")})
        with open(progress_path, "w") as f:
            json.dump(progress, f, indent=2)

        if _gate_check(args.results_dir, logger):
            break


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    # Smoke runs thin + reindex the panel option sets, so their raw JSONL is
    # incompatible with a full run's. Never let the two share a results dir.
    if args.smoke and args.results_dir.rstrip("/") == "results":
        raise SystemExit("--smoke must use its own results dir "
                         "(e.g. --results-dir results_smoke); ids from smoke "
                         "panels would poison the full run's raw cache.")
    if args.child:
        logger = setup_logging(log_to_file=False)  # parent captures our stdout
        _run_child(args, logger)
    else:
        logger = setup_logging(log_to_file=True)
        _run_parent(args, logger)


if __name__ == "__main__":
    main()
