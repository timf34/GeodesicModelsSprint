#!/usr/bin/env python3
"""Pre-download + verify EVERY model checkpoint before the overnight run starts.

run_on_pod.sh step [2] runs this and aborts the whole night if it fails. The point:
all network-dependent bytes land up front, while someone is still watching. A
download timeout at 11pm costs five minutes; the same timeout at 3am inside the
runner costs the whole night.

Verification goes beyond "snapshot_download returned a path": interrupted or
partially-evicted caches can leave a valid-looking snapshot dir, so we re-check
that every shard listed in model.safetensors.index.json exists with nonzero size,
that the tokenizer is complete, and that chat models actually ship a chat template
(a missing template degrades prompts SILENTLY — wrong numbers, not a crash).

Usage:
    python scripts/predownload.py --models all          # same tokens as the runner
    python scripts/predownload.py --models pod_a
    python scripts/predownload.py --models key1,key2
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

RETRIES = 3
LOW_DISK_GB = 30  # vLLM compile caches + logs + results need headroom after the weights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-download + verify model weights for the Geodesic UE sweep."
    )
    p.add_argument(
        "--models",
        default="all",
        help="same tokens as the runner: all | pod_a | pod_b | tier:<tier> | csv of keys",
    )
    p.add_argument(
        "--hf-home",
        default=os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"),
        help="HF cache root (default: $HF_HOME or ~/.cache/huggingface)",
    )
    return p.parse_args()


def load_dotenv_if_present() -> None:
    """Pick up HF_TOKEN from .env — the Llama-2 repos are gated and fail without it."""
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("  (python-dotenv not installed — skipping .env; export HF_TOKEN yourself if needed)")
        return
    load_dotenv(env_path)


def download_with_retries(spec, token):
    """snapshot_download with retries + exponential backoff.

    snapshot_download resumes partial downloads and is a no-op when the revision is
    already cached, so retrying (and rerunning this script) is always safe.
    """
    from huggingface_hub import snapshot_download

    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            return snapshot_download(
                repo_id=spec.repo_id, revision=spec.revision, token=token
            )
        except Exception as e:  # network flakes, 5xx, auth — retry them all, then report
            last_err = e
            if attempt < RETRIES:
                delay = 15 * (2 ** (attempt - 1))  # 15s, 30s
                print(f"  attempt {attempt}/{RETRIES} failed ({e}); retrying in {delay}s...")
                time.sleep(delay)
    raise RuntimeError(f"download failed after {RETRIES} attempts: {last_err}")


def verify_snapshot(spec, snap: Path):
    """Return a list of problems (empty == verified OK)."""
    problems = []

    if not (snap / "config.json").is_file():
        problems.append("config.json missing")

    # Weights: every shard in the index must exist with nonzero size. is_file() and
    # stat() both resolve the HF cache's snapshot->blob symlinks, so a broken link
    # (blob evicted) or a zero-byte placeholder is caught here.
    index_path = snap / "model.safetensors.index.json"
    single_path = snap / "model.safetensors"
    if index_path.is_file():
        try:
            weight_files = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
        except (json.JSONDecodeError, KeyError, OSError) as e:
            problems.append(f"unreadable model.safetensors.index.json ({e})")
            weight_files = []
        for name in weight_files:
            shard = snap / name
            if not shard.is_file():
                problems.append(f"weight shard missing: {name}")
            elif shard.stat().st_size == 0:
                problems.append(f"weight shard zero-size: {name}")
    elif single_path.is_file():
        if single_path.stat().st_size == 0:
            problems.append("model.safetensors is zero-size")
    else:
        problems.append("no model.safetensors / model.safetensors.index.json")

    # Tokenizer completeness.
    tok_cfg = snap / "tokenizer_config.json"
    if not tok_cfg.is_file():
        problems.append("tokenizer_config.json missing")
    if not ((snap / "tokenizer.json").is_file() or (snap / "tokenizer.model").is_file()):
        problems.append("tokenizer.json / tokenizer.model missing")

    # Chat models MUST have a chat template. Geodesic repos ship a separate
    # chat_template.jinja (the transformers >= 4.57 layout); references may embed it
    # in tokenizer_config.json instead. Either is fine — having neither is not,
    # unless the registry supplies the template itself (Llama-2 mirror).
    if spec.is_chat and spec.chat_template_override is None:
        has_jinja = (snap / "chat_template.jinja").is_file()
        has_embedded = False
        if tok_cfg.is_file():
            try:
                has_embedded = "chat_template" in json.loads(tok_cfg.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        if not (has_jinja or has_embedded):
            problems.append("no chat template (chat_template.jinja or tokenizer_config.json entry)")

    return problems


def dir_size_bytes(path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))  # follows blob symlinks
            except OSError:
                pass
    return total


def main() -> int:
    args = parse_args()
    hf_home = os.path.abspath(os.path.expanduser(args.hf_home))
    # HF_HOME must be set BEFORE huggingface_hub is imported — it freezes cache paths
    # at import time, and a mismatched cache here means the runner re-downloads at 3am.
    os.environ["HF_HOME"] = hf_home

    load_dotenv_if_present()

    sys.path.insert(0, str(REPO_ROOT))
    from geodesic_ue.registry import SPECS_BY_KEY, resolve_keys

    keys = resolve_keys(args.models)
    specs = [SPECS_BY_KEY[k] for k in keys]

    token = os.environ.get("HF_TOKEN") or None
    if token is None and any(s.repo_id.startswith("meta-llama/") for s in specs):
        print("!! HF_TOKEN is not set but this selection includes GATED meta-llama repos —")
        print("   those downloads WILL fail. Put HF_TOKEN=... in .env first.")

    print(f"pre-downloading {len(specs)} models  (HF_HOME={hf_home})")

    results = []  # (key, size_bytes, problems)
    for i, spec in enumerate(specs, 1):
        print(f"\n[{i}/{len(specs)}] {spec.key}  <-  {spec.repo_id}@{spec.revision}")
        try:
            snap = Path(download_with_retries(spec, token))
        except Exception as e:
            print(f"  FAIL: {e}")
            results.append((spec.key, 0, [str(e)]))
            continue
        problems = verify_snapshot(spec, snap)
        size = dir_size_bytes(snap)
        results.append((spec.key, size, problems))
        if problems:
            print(f"  FAIL: {'; '.join(problems)}")
        else:
            print(f"  OK  {size / 2**30:.1f} GB  ({snap})")

    failed = [r for r in results if r[2]]
    total_bytes = sum(r[1] for r in results)

    print("\n" + "=" * 78)
    print(f"{'model key':<46}{'size(GB)':>9}  status")
    print("-" * 78)
    for key, size, problems in results:
        print(f"{key:<46}{size / 2**30:>9.1f}  {'FAIL' if problems else 'OK'}")
    print("-" * 78)
    print(f"{'TOTAL':<46}{total_bytes / 2**30:>9.1f}  "
          f"{len(results) - len(failed)} OK / {len(failed)} FAIL")

    # Disk check: the run also needs room for vLLM caches, logs and results — flag a
    # nearly-full volume NOW, not when the runner ENOSPCs halfway through the night.
    probe = Path(hf_home)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    free_gb = usage.free / 2**30
    print(f"\ndisk at {hf_home}: {free_gb:.1f} GB free of {usage.total / 2**30:.1f} GB")
    if free_gb < LOW_DISK_GB:
        print(f"!! WARNING: less than {LOW_DISK_GB} GB free after downloads — the run needs "
              "headroom for vLLM caches, logs and results. Free space or use a bigger volume.")

    if failed:
        print(f"\n{len(failed)} model(s) FAILED — do NOT start the run:")
        for key, _size, problems in failed:
            print(f"  {key}:")
            for p in problems:
                print(f"    - {p}")
        return 1

    print(f"\nAll {len(specs)} models downloaded and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
