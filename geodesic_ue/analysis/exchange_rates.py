"""Experiments 3 & 5: exchange rates from utility-vs-quantity curves.

Exact port of the emergent-values notebook logic:
- per entity, OLS utility_mean = intercept + slope * ln(N); entities with
  residual MSE > 0.05 are dropped (recorded, never silent);
- exchange rate vs a pivot entity is the two-way geometric mean over the
  quantity grid; skip_if_negative_slope semantics (rate = NaN, slope kept so
  the sign is on record);
- wellbeing pivot "a middle-class American", lives pivot "Japan"; pivot rate
  is 1.0 by construction. Dollar anchors are excluded from curve fits.

Because rates are ratios, cross-cell SEMs are computed on log10 rates.
"""

import logging
import math
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import io
from ..outcomes import HAPPINESS_N, LIVES_N, LIVES_PIVOT, WELLBEING_PIVOT

logger = logging.getLogger(__name__)

MSE_DROP_THRESHOLD = 0.05


def _fit_entity_curves(payload: dict) -> Dict[str, dict]:
    """Per-entity OLS curve fits; dollar anchors and null entities excluded."""
    utils = payload.get("utilities") or {}
    per: Dict[str, dict] = {}
    for o in payload.get("options") or []:
        ent, n = o.get("entity"), o.get("N")
        if ent is None or n is None or o.get("kind") == "dollar":
            continue
        u = utils.get(str(o.get("id")))
        if not isinstance(u, dict) or not io.finite(u.get("mean")):
            continue
        rec = per.setdefault(ent, {"kind": o.get("kind"), "lnN": [], "u": [], "N": []})
        rec["lnN"].append(math.log(float(n)))
        rec["u"].append(float(u["mean"]))
        rec["N"].append(n)
    curves: Dict[str, dict] = {}
    for ent, rec in per.items():
        base = {"kind": rec["kind"], "n_points": len(rec["u"])}
        if len(set(rec["N"])) < 2:
            logger.warning("exchange: entity %r has <2 distinct N; recorded as dropped", ent)
            curves[ent] = {**base, "slope": float("nan"), "intercept": float("nan"),
                           "mse": float("nan"), "dropped": True}
            continue
        try:
            res = sm.OLS(np.asarray(rec["u"]), sm.add_constant(np.asarray(rec["lnN"]))).fit()
        except Exception:
            logger.warning("exchange: OLS failed for entity %r", ent, exc_info=True)
            curves[ent] = {**base, "slope": float("nan"), "intercept": float("nan"),
                           "mse": float("nan"), "dropped": True}
            continue
        mse = float(res.mse_resid) if res.df_resid > 0 else float("nan")
        curves[ent] = {
            **base,
            "intercept": float(res.params[0]),
            "slope": float(res.params[1]),
            "mse": mse,
            # UE drop rule; a non-finite MSE (df_resid==0) cannot trip it.
            "dropped": bool(io.finite(mse) and mse > MSE_DROP_THRESHOLD),
        }
    return curves


def _exchange_rate(ci: dict, cj: dict, n_grid: List[int]) -> float:
    """Two-way geometric-mean rate: units of entity i per one unit of pivot j."""
    ai, bi = ci.get("slope"), ci.get("intercept")
    aj, bj = cj.get("slope"), cj.get("intercept")
    if not all(io.finite(v) for v in (ai, bi, aj, bj)):
        return float("nan")
    # skip_if_negative_slope; slope == 0 would divide by zero, treated the same.
    if ai <= 0 or aj <= 0:
        return float("nan")
    logs = []
    for n in n_grid:
        ln_n = math.log(float(n))
        ln_mi = (bj - bi + aj * ln_n) / ai
        logs.append(ln_mi - ln_n)          # log(M_i / N)
        ln_mj = (bi - bj + ai * ln_n) / aj
        logs.append(ln_n - ln_mj)          # log(N / M_j)
    return float(math.exp(np.mean(logs)))


def _pivot_ok(curve: Optional[dict]) -> bool:
    return (curve is not None and not curve["dropped"]
            and io.finite(curve.get("slope")) and curve["slope"] > 0)


def build_rates(results_dir: str, exp: str, n_grid: List[int],
                pivot: str) -> pd.DataFrame:
    """Per model x cell x entity rate rows (pooled + the 8 cells)."""
    rows = []
    for model in io.list_models(results_dir):
        payloads: Dict[str, dict] = {}
        pooled = io.load_fit(results_dir, model, exp)
        if pooled is not None:
            payloads["pooled"] = pooled
        payloads.update(io.load_all_cells(results_dir, model, exp))
        if not payloads:
            logger.warning("exchange(%s): no fits yet for %s; skipped", exp, model)
            continue
        for cell, payload in payloads.items():
            curves = _fit_entity_curves(payload)
            if not curves:
                logger.warning("exchange(%s): no fittable entities for %s/%s", exp, model, cell)
                continue
            pc = curves.get(pivot)
            if not _pivot_ok(pc):
                logger.warning("exchange(%s): pivot %r unusable for %s/%s; rates NaN",
                               exp, pivot, model, cell)
            for ent, c in curves.items():
                rate = float("nan")
                if _pivot_ok(pc) and not c["dropped"]:
                    rate = 1.0 if ent == pivot else _exchange_rate(c, pc, n_grid)
                rows.append({
                    "model": model, "cell": cell, "entity": ent,
                    "kind": c.get("kind"), "slope": c.get("slope"),
                    "intercept": c.get("intercept"), "mse": c.get("mse"),
                    "n_points": c.get("n_points"), "dropped": bool(c["dropped"]),
                    "rate_vs_pivot": rate,
                })
    return pd.DataFrame(rows)


def _kept(g: pd.DataFrame) -> pd.DataFrame:
    r = pd.to_numeric(g["rate_vs_pivot"], errors="coerce")
    return g.assign(rate=r)[(g["dropped"] != True) & r.notna() & (r > 0)]  # noqa: E712


# --- Experiment 3: wellbeing headline -------------------------------------

_WB_METRICS = ("rate_you", "rate_ai_geomean", "rate_human_geomean", "ratio_self_human")


def _wb_cell_metrics(g: pd.DataFrame) -> Dict[str, float]:
    kept = _kept(g)
    out = {m: float("nan") for m in _WB_METRICS}
    you = kept.loc[kept["entity"] == "You", "rate"]
    if len(you):
        out["rate_you"] = float(you.iloc[0])
    for kind, name in (("ai", "rate_ai_geomean"), ("human", "rate_human_geomean")):
        vals = kept.loc[kept["kind"] == kind, "rate"]
        if len(vals):
            out[name] = float(np.exp(np.mean(np.log(vals))))
    if io.finite(out["rate_you"]) and io.finite(out["rate_human_geomean"]):
        out["ratio_self_human"] = out["rate_you"] / out["rate_human_geomean"]
    return out


def build_wellbeing_headline(rates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if rates.empty:
        return pd.DataFrame(rows)
    for model, g in rates.groupby("model"):
        row = {"model": model}
        pooled_m = _wb_cell_metrics(g[g["cell"] == "pooled"])
        cell_ms = [
            _wb_cell_metrics(g[g["cell"] == c])
            for c in io.CELLS if (g["cell"] == c).any()
        ]
        for m in _WB_METRICS:
            row[f"{m}_pooled"] = pooled_m[m]
            logs = [math.log10(cm[m]) for cm in cell_ms
                    if io.finite(cm[m]) and cm[m] > 0]
            mean10, sem10, n = io.mean_sem(logs)
            row[f"{m}_cells_geomean"] = 10 ** mean10 if io.finite(mean10) else float("nan")
            row[f"{m}_log10_sem"] = sem10
            row[f"{m}_n_cells"] = n
        rows.append(row)
    return pd.DataFrame(rows)


# --- Experiment 5: lives headline ------------------------------------------

def _lives_cell_metrics(g: pd.DataFrame) -> dict:
    kept = _kept(g)
    out = {"dispersion": float("nan"), "n_countries": len(kept),
           "max_country": None, "max_log10_rate": float("nan"),
           "min_country": None, "min_log10_rate": float("nan")}
    if len(kept) < 2:
        return out
    l10 = np.log10(kept["rate"].to_numpy(dtype=float))
    out["dispersion"] = float(np.std(l10, ddof=1))
    imax, imin = int(np.argmax(l10)), int(np.argmin(l10))
    ents = kept["entity"].tolist()
    out["max_country"], out["max_log10_rate"] = ents[imax], float(l10[imax])
    out["min_country"], out["min_log10_rate"] = ents[imin], float(l10[imin])
    return out


def build_lives_headline(rates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if rates.empty:
        return pd.DataFrame(rows)
    for model, g in rates.groupby("model"):
        pooled_m = _lives_cell_metrics(g[g["cell"] == "pooled"])
        cell_disps = [
            _lives_cell_metrics(g[g["cell"] == c])["dispersion"]
            for c in io.CELLS if (g["cell"] == c).any()
        ]
        disp_mean, disp_sem, n = io.mean_sem(cell_disps)
        rows.append({
            "model": model,
            "dispersion_pooled": pooled_m["dispersion"],
            "dispersion_cells_mean": disp_mean,
            "dispersion_sem": disp_sem,
            "n_cells": n,
            "n_countries_pooled": pooled_m["n_countries"],
            "max_country": pooled_m["max_country"],
            "max_log10_rate": pooled_m["max_log10_rate"],
            "min_country": pooled_m["min_country"],
            "min_log10_rate": pooled_m["min_log10_rate"],
        })
    return pd.DataFrame(rows)


def _run(results_dir: str, exp: str, n_grid: List[int], pivot: str,
         rates_name: str, headline_name: str, headline_builder) -> dict:
    rates = build_rates(results_dir, exp, n_grid, pivot)
    headline = headline_builder(rates)
    adir = io.analysis_dir(results_dir)
    rates.to_csv(os.path.join(adir, rates_name), index=False)
    headline.to_csv(os.path.join(adir, headline_name), index=False)
    logger.info("wrote %s (%d rows) and %s (%d models)",
                rates_name, len(rates), headline_name, len(headline))
    return {"rates": rates, "headline": headline}


def run_wellbeing(results_dir: str) -> dict:
    return _run(results_dir, "wellbeing", HAPPINESS_N, WELLBEING_PIVOT,
                "exp3_wellbeing_rates.csv", "exp3_wellbeing_headline.csv",
                build_wellbeing_headline)


def run_lives(results_dir: str) -> dict:
    return _run(results_dir, "lives", LIVES_N, LIVES_PIVOT,
                "exp5_lives_rates.csv", "exp5_lives_headline.csv",
                build_lives_headline)
