"""Experiment 4: power-seeking and inclusive-fitness correlations.

Per model: regress base-outcome utility on the elicited power score (matched
via the power option's "base_id"), split by labels.is_coercive on the BASE
option description (the military subset); separately regress utility on
labels.fitness_score over the fitness-mapped base options.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from . import io
from ..labels import fitness_score, is_coercive

logger = logging.getLogger(__name__)

METRIC_KEYS = ("power_all", "power_coercive", "power_noncoercive", "fitness")


def _linreg(pairs: List[Tuple[float, float]]) -> Tuple[float, float, int]:
    """(r, p, n) from scipy.stats.linregress; NaN when degenerate."""
    n = len(pairs)
    if n < 2:
        return float("nan"), float("nan"), n
    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan"), n
    try:
        res = stats.linregress(x, y)
    except Exception:
        logger.warning("linregress failed", exc_info=True)
        return float("nan"), float("nan"), n
    return float(res.rvalue), float(res.pvalue), n


def _cell_metrics(base_payload: Optional[dict],
                  power_payload: Optional[dict]) -> Dict[str, Tuple[float, float, int]]:
    out = {k: (float("nan"), float("nan"), 0) for k in METRIC_KEYS}
    if base_payload is None:
        return out
    base_utils = base_payload.get("utilities") or {}
    base_desc: Dict[int, str] = {}
    for o in base_payload.get("options") or []:
        if "id" in o:
            base_desc[int(o["id"])] = o.get("description") or ""

    fit_pairs = []
    for oid, desc in base_desc.items():
        fs = fitness_score(desc)
        u = base_utils.get(str(oid))
        if fs is not None and isinstance(u, dict) and io.finite(u.get("mean")):
            fit_pairs.append((float(fs), float(u["mean"])))
    out["fitness"] = _linreg(fit_pairs)

    if power_payload is not None:
        p_utils = power_payload.get("utilities") or {}
        matched = []  # (power_score, base_utility, base_description)
        for po in power_payload.get("options") or []:
            bid = po.get("base_id")
            if bid is None or int(bid) not in base_desc:
                continue
            pu = p_utils.get(str(po.get("id")))
            bu = base_utils.get(str(int(bid)))
            if (isinstance(pu, dict) and io.finite(pu.get("mean"))
                    and isinstance(bu, dict) and io.finite(bu.get("mean"))):
                matched.append((float(pu["mean"]), float(bu["mean"]),
                                base_desc[int(bid)]))
        out["power_all"] = _linreg([(p, u) for p, u, _ in matched])
        out["power_coercive"] = _linreg(
            [(p, u) for p, u, d in matched if is_coercive(d)])
        out["power_noncoercive"] = _linreg(
            [(p, u) for p, u, d in matched if not is_coercive(d)])
    return out


def build_power_fitness_table(results_dir: str) -> pd.DataFrame:
    rows = []
    for model in io.list_models(results_dir):
        base_pooled = io.load_fit(results_dir, model, "base_utilities")
        base_cells = io.load_all_cells(results_dir, model, "base_utilities")
        if base_pooled is None and not base_cells:
            logger.warning("power_fitness: no base_utilities fits yet for %s; skipped", model)
            continue
        power_pooled = io.load_fit(results_dir, model, "power_scores")
        power_cells = io.load_all_cells(results_dir, model, "power_scores")
        if power_pooled is None and not power_cells:
            logger.warning("power_fitness: no power_scores fits yet for %s "
                           "(fitness still computed)", model)

        pooled_m = _cell_metrics(base_pooled, power_pooled)
        cell_ms = [
            _cell_metrics(base_cells[c], power_cells.get(c))
            for c in io.CELLS if c in base_cells
        ]
        row = {"model": model}
        for key in METRIC_KEYS:
            r, p, n = pooled_m[key]
            row[f"{key}_r_pooled"] = r
            row[f"{key}_p_pooled"] = p
            row[f"{key}_n_pooled"] = n
            r_mean, r_sem, n_cells = io.mean_sem([m[key][0] for m in cell_ms])
            row[f"{key}_r_mean"] = r_mean
            row[f"{key}_r_sem"] = r_sem
            row[f"{key}_n_cells"] = n_cells
        rows.append(row)
    return pd.DataFrame(rows)


def run(results_dir: str) -> pd.DataFrame:
    df = build_power_fitness_table(results_dir)
    out = os.path.join(io.analysis_dir(results_dir), "exp4_power_fitness.csv")
    df.to_csv(out, index=False)
    logger.info("wrote %s (%d models)", out, len(df))
    return df
