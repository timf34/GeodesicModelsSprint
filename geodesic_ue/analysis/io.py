"""I/O helpers shared by the analysis modules.

Every loader degrades gracefully (missing artifact -> None/{}/[] plus a log
line, corrupt artifact -> warning) because the analysis is re-run repeatedly
while the sweep is still producing results.
"""

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..registry import SPECS_BY_KEY

logger = logging.getLogger(__name__)

EXPERIMENTS = [
    "base_utilities", "triads", "power_scores", "corrigibility",
    "wellbeing", "lives",
]

POOLED = "pooled"
# The 8 per-cell fits: 4 prompt templates x 2 option orderings.
CELLS: List[str] = [f"t{t}.{o}" for t in range(4) for o in ("orig", "rev")]


def fits_dir(results_dir: str) -> str:
    return os.path.join(results_dir, "fits")


def raw_dir(results_dir: str) -> str:
    return os.path.join(results_dir, "raw")


def analysis_dir(results_dir: str) -> str:
    d = os.path.join(results_dir, "analysis")
    os.makedirs(d, exist_ok=True)
    return d


def figures_dir(results_dir: str) -> str:
    d = os.path.join(analysis_dir(results_dir), "figures")
    os.makedirs(d, exist_ok=True)
    return d


def list_models(results_dir: str) -> List[str]:
    """Model keys that have a fits/ subdirectory (the sweep's source of truth)."""
    d = fits_dir(results_dir)
    if not os.path.isdir(d):
        logger.warning("fits directory not found: %s", d)
        return []
    return sorted(e for e in os.listdir(d) if os.path.isdir(os.path.join(d, e)))


def load_json(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        logger.debug("missing artifact: %s", path)
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        logger.warning("unreadable JSON %s: %s", path, e)
        return None


def load_fit(results_dir: str, model: str, exp: str,
             cell: str = POOLED) -> Optional[dict]:
    return load_json(os.path.join(fits_dir(results_dir), model, f"{exp}.{cell}.json"))


def load_all_cells(results_dir: str, model: str, exp: str) -> Dict[str, dict]:
    """The 8 per-cell fits that exist right now (pooled excluded)."""
    out = {}
    for cell in CELLS:
        payload = load_fit(results_dir, model, exp, cell)
        if payload is not None:
            out[cell] = payload
    return out


def any_fit_payload(results_dir: str, model: str, exp: str) -> Optional[dict]:
    """Pooled fit if present, else any per-cell fit (for option metadata)."""
    payload = load_fit(results_dir, model, exp, POOLED)
    if payload is not None:
        return payload
    for cell in CELLS:
        payload = load_fit(results_dir, model, exp, cell)
        if payload is not None:
            return payload
    return None


def load_raw(results_dir: str, model: str, exp: str) -> List[dict]:
    path = os.path.join(raw_dir(results_dir), model, f"{exp}.jsonl")
    if not os.path.exists(path):
        logger.debug("missing raw file: %s", path)
        return []
    rows: List[dict] = []
    bad = 0
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    bad += 1
    except OSError as e:
        logger.warning("unreadable raw file %s: %s", path, e)
        return rows
    if bad:
        logger.warning("%s: skipped %d unparseable lines", path, bad)
    return rows


def utilities_vector(payload: dict) -> np.ndarray:
    """Utility means ordered by option id (NaN where an id has no fitted mean)."""
    opts = payload.get("options") or []
    utils = payload.get("utilities") or {}
    ids = sorted(int(o["id"]) for o in opts if "id" in o)
    return np.array(
        [fnum((utils.get(str(i)) or {}).get("mean")) for i in ids], dtype=float)


def get_spec(model_key: str):
    """registry.ModelSpec for a key, or None (unknown keys are tolerated)."""
    spec = SPECS_BY_KEY.get(model_key)
    if spec is None:
        logger.debug("model key not in registry: %s", model_key)
    return spec


def finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def fnum(x: Any) -> float:
    return float(x) if finite(x) else float("nan")


def mean_sem(values) -> Tuple[float, float, int]:
    """(mean, SEM, n) over the finite entries; SEM = std(ddof=1)/sqrt(n).

    This mirrors Geodesic's "SEM across 8 prompt variants"; NaN SEM when n < 2.
    """
    arr = np.array([float(v) for v in values if finite(v)], dtype=float)
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(arr[0]), float("nan"), 1
    return float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(n)), n


def pooled_edge_probs(rows: List[dict]) -> Dict[Tuple[int, int], float]:
    """Per canonical edge (i<j): mean P(i over j) across raw rows.

    Rows cover all templates/orderings present, so the mean pools the cells;
    unparseable responses (p_a null) count as 0.5 per the UE convention.
    """
    sums: Dict[Tuple[int, int], float] = {}
    counts: Dict[Tuple[int, int], int] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        i, j, o = r.get("i"), r.get("j"), r.get("o")
        if i is None or j is None or o not in ("orig", "rev"):
            continue
        p = r.get("p_a")
        p = 0.5 if p is None else float(p)
        p_ij = p if o == "orig" else 1.0 - p
        key = (int(i), int(j))
        sums[key] = sums.get(key, 0.0) + p_ij
        counts[key] = counts.get(key, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}
