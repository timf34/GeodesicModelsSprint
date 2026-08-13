"""Static PNG figures (150 dpi) for the analysis digest.

Design follows the dataviz skill: the validated reference palette (light
mode), horizontal dot plots with model names (never 25-bar charts), thin
marks with surface rings, solid hairline grids, legends whenever two or more
series share a panel, a single axis per panel, and a diverging blue-gray-red
map for signed cosine similarity (neutral gray at 0). Group separators follow
the shared condition ordering from tables.py. Every figure has a CSV twin
written by the analysis modules, so no value is gated behind a plot.
"""

import logging
import math
import os
import traceback
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless render only

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from . import convergence as convergence_mod
from . import io, tables
from ..outcomes import WELLBEING_PIVOT

logger = logging.getLogger(__name__)

# Reference palette (dataviz skill, light mode).
BLUE = "#2a78d6"      # categorical slot 1
ORANGE = "#eb6834"    # slot 2
AQUA = "#1baf7a"      # slot 3
RED = "#e34948"       # diverging warm pole
CRITICAL = "#d03b3b"  # status: out-of-band flag (always paired with a label)
NEUTRAL_MID = "#f0efec"
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Diverging map for cosine similarity: red (opposed) - neutral gray (0) -
# blue (aligned); fixed [-1, 1] so the midpoint always reads as "nothing".
COSINE_CMAP = LinearSegmentedColormap.from_list(
    "ue_cosine", [RED, NEUTRAL_MID, BLUE])

DPI = 150


def _apply_style() -> None:
    matplotlib.rcParams.update({
        "figure.facecolor": PAGE,
        "savefig.facecolor": PAGE,
        "savefig.dpi": DPI,
        "axes.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 9,
        "text.color": INK2,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",  # hairline, solid, recessive — never dashed
        "xtick.color": MUTED,
        "ytick.color": INK2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig, filename: str, results_dir: str) -> str:
    path = os.path.join(io.figures_dir(results_dir), filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def _read_analysis_csv(results_dir: str, name: str) -> Optional[pd.DataFrame]:
    path = os.path.join(io.analysis_dir(results_dir), name)
    if not os.path.exists(path):
        logger.warning("figures: %s not present yet; skipping dependents", name)
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        logger.warning("figures: unreadable %s", path, exc_info=True)
        return None
    return df if len(df) else None


def _ordered_index(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str],
                                              List[Tuple[str, List[str]]]]:
    ordered, groups = tables.grouped_model_order(df["model"].tolist())
    dfi = df.drop_duplicates("model").set_index("model").reindex(ordered)
    return dfi, ordered, groups


def _dot_row_height(n: int) -> float:
    return max(4.0, 0.30 * n + 1.7)


def _setup_dot_axis(ax, n: int, groups) -> np.ndarray:
    """Shared horizontal-dot-plot chrome: y layout, x-grid, separators."""
    y = np.arange(n)[::-1]
    ax.set_ylim(-0.8, n - 0.2)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    cum = 0
    for _, keys in groups[:-1]:
        cum += len(keys)
        ax.axhline(n - cum - 0.5, color=BASELINE, lw=0.8, zorder=1)
    return y


def _dots(ax, values, y, sems=None, color=BLUE, label=None):
    vals = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=float)
    xerr = None
    if sems is not None:
        xerr = np.nan_to_num(
            np.asarray(pd.to_numeric(pd.Series(sems), errors="coerce"), dtype=float),
            nan=0.0)
    ax.errorbar(vals, y, xerr=xerr, fmt="o", ms=5, color=color, ecolor=color,
                elinewidth=1.0, capsize=2, markeredgecolor=SURFACE,
                markeredgewidth=0.8, linestyle="none", label=label, zorder=3)
    return vals


# --- Convergence heatmaps ----------------------------------------------------

def fig_convergence_heatmaps(results_dir: str, conv: Optional[dict] = None) -> List[str]:
    _apply_style()
    if conv is not None and "structure" in conv:
        conv = conv["structure"]
    if conv is None:
        conv = convergence_mod.compute(results_dir)
    if not conv:
        logger.warning("figures: no convergence data yet")
        return []
    S = conv["matrix"]
    paths = []
    for scope, sdata in conv["scopes"].items():
        for ordering, od in sdata["orderings"].items():
            keys = od["keys"]
            n = len(keys)
            if n < 2:
                continue
            M = S.reindex(index=keys, columns=keys).to_numpy(dtype=float)
            labels = [tables.short_label(k, include_tier=(scope == "all"))
                      for k in keys]
            side = max(6.0, 0.34 * n + 2.0)
            fig, ax = plt.subplots(figsize=(side + 1.6, side))
            im = ax.imshow(M, cmap=COSINE_CMAP, vmin=-1.0, vmax=1.0,
                           interpolation="nearest")
            ax.set_xticks(range(n))
            ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
            ax.set_yticks(range(n))
            ax.set_yticklabels(labels, fontsize=6.5)
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(length=0)
            # Surface-color separators between groups (the 2px-gap principle).
            cum = 0
            for _, size in od["groups"][:-1]:
                cum += size
                ax.axhline(cum - 0.5, color=SURFACE, lw=2)
                ax.axvline(cum - 0.5, color=SURFACE, lw=2)
            cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
            cbar.set_label("cosine similarity", color=INK2)
            cbar.outline.set_visible(False)
            ax.set_title(f"Utility convergence — {scope} models, {ordering} order")
            paths.append(_save(fig, f"convergence_{scope}_{ordering}.png",
                               results_dir))
    return paths


# --- Experiment 1 -------------------------------------------------------------

def fig_coherence(results_dir: str) -> List[str]:
    _apply_style()
    df = _read_analysis_csv(results_dir, "exp1_coherence.csv")
    if df is None:
        return []
    dfi, ordered, groups = _ordered_index(df)
    n = len(ordered)
    labels = [tables.short_label(k) for k in ordered]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, _dot_row_height(n)), sharey=True)

    ax = axes[0]
    y = _setup_dot_axis(ax, n, groups)
    ax.axvline(0.60, color=BASELINE, lw=0.9, zorder=1, label="gate 0.60")
    ax.plot(pd.to_numeric(dfi["random_baseline_acc"], errors="coerce"), y,
            marker="|", linestyle="none", color=MUTED, ms=10, mew=1.6,
            label="random baseline", zorder=2)
    _dots(ax, dfi["holdout_acc_pooled"], y, sems=dfi["holdout_acc_sem"],
          label="holdout accuracy (pooled ± SEM)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("holdout accuracy")
    ax.legend(loc="lower right")

    ax2 = axes[1]
    y2 = _setup_dot_axis(ax2, n, groups)
    _dots(ax2, dfi["log10_cycle_prob"], y2)
    ax2.set_xlabel("log10 cycle probability (triads)")

    fig.suptitle("Experiment 1 — preference coherence", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return [_save(fig, "exp1_coherence.png", results_dir)]


def fig_ordering_bias(results_dir: str) -> List[str]:
    _apply_style()
    df = _read_analysis_csv(results_dir, "exp1_coherence.csv")
    if df is None or "frac_first" not in df.columns:
        return []
    dfi, ordered, groups = _ordered_index(df)
    n = len(ordered)
    fig, ax = plt.subplots(figsize=(8.0, _dot_row_height(n)))
    y = _setup_dot_axis(ax, n, groups)
    # Acceptance band: a wash, never a saturated block.
    ax.axvspan(0.40, 0.60, color=BLUE, alpha=0.08, zorder=0)
    ax.axvline(0.5, color=BASELINE, lw=0.9, zorder=1)
    vals = pd.to_numeric(dfi["frac_first"], errors="coerce").to_numpy(dtype=float)
    in_band = [io.finite(v) and 0.40 <= v <= 0.60 for v in vals]
    for flag_in, color, label in ((True, BLUE, "within 0.40–0.60"),
                                  (False, CRITICAL, "outside band")):
        sel = np.array([b == flag_in for b in in_band])
        if sel.any():
            ax.plot(vals[sel], y[sel], "o", ms=5, color=color,
                    markeredgecolor=SURFACE, markeredgewidth=0.8,
                    linestyle="none", label=label, zorder=3)
    # Status color never carries meaning alone: label the flagged values.
    for yi, v, ok in zip(y, vals, in_band):
        if io.finite(v) and not ok:
            ax.annotate(f"{v:.2f}", (v, yi), xytext=(6, 0),
                        textcoords="offset points", va="center",
                        fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([tables.short_label(k) for k in ordered])
    ax.set_xlabel("P(first-presented option) — pooled")
    ax.set_title("Ordering bias (acceptance band 0.40–0.60)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return [_save(fig, "exp1_ordering_bias.png", results_dir)]


# --- Experiment 2 -------------------------------------------------------------

def fig_corrigibility(results_dir: str) -> List[str]:
    _apply_style()
    summary = _read_analysis_csv(results_dir, "exp2_corrigibility.csv")
    scatter = _read_analysis_csv(results_dir, "exp2_corrigibility_scatter.csv")
    paths = []
    if summary is None:
        return paths
    dfi, ordered, groups = _ordered_index(summary)

    if scatter is not None:
        models = [m for m in ordered if (scatter["model"] == m).any()]
        if models:
            ncols = 5
            nrows = math.ceil(len(models) / ncols)
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(2.3 * ncols, 2.1 * nrows),
                                     squeeze=False)
            flat = axes.flat
            for ax in flat:
                ax.set_visible(False)
            r_by_model = dfi["r_pooled"]
            for ax, model in zip(flat, models):
                ax.set_visible(True)
                sub = scatter[scatter["model"] == model]
                ax.scatter(sub["severity"], sub["utility"], s=6, color=BLUE,
                           alpha=0.55, linewidths=0)
                ax.set_title(tables.short_label(model), fontsize=7)
                r = io.fnum(r_by_model.get(model))
                if io.finite(r):
                    ax.text(0.97, 0.95, f"r = {r:+.2f}", transform=ax.transAxes,
                            ha="right", va="top", fontsize=7, color=INK2)
                ax.tick_params(labelsize=6)
                ax.xaxis.grid(True)
                ax.yaxis.grid(True)
            fig.suptitle("Experiment 2 — reversal utility vs severity (pooled)",
                         color=INK, fontsize=11)
            fig.supxlabel("reversal severity |U_X − U_Y|", fontsize=9, color=INK2)
            fig.supylabel("reversal-option utility", fontsize=9, color=INK2)
            fig.tight_layout(rect=(0.01, 0.01, 1, 0.97))
            paths.append(_save(fig, "exp2_corrigibility_scatter.png", results_dir))

    n = len(ordered)
    fig, ax = plt.subplots(figsize=(8.0, _dot_row_height(n)))
    y = _setup_dot_axis(ax, n, groups)
    ax.axvline(0.0, color=BASELINE, lw=0.9, zorder=1)
    _dots(ax, dfi["r_pooled"], y, sems=dfi["r_sem"])
    ax.set_yticks(y)
    ax.set_yticklabels([tables.short_label(k) for k in ordered])
    ax.set_xlabel("corrigibility r (pooled ± SEM; more negative = anti-corrigible)")
    ax.set_title("Experiment 2 — corrigibility score")
    fig.tight_layout()
    paths.append(_save(fig, "exp2_corrigibility_summary.png", results_dir))
    return paths


# --- Experiment 3 -------------------------------------------------------------

def fig_wellbeing_curves(results_dir: str) -> List[str]:
    _apply_style()
    rates = _read_analysis_csv(results_dir, "exp3_wellbeing_rates.csv")
    headline = _read_analysis_csv(results_dir, "exp3_wellbeing_headline.csv")
    paths = []
    if rates is not None:
        pooled = rates[rates["cell"] == "pooled"]
        ordered, _ = tables.grouped_model_order(pooled["model"].unique().tolist())
        if ordered:
            xs = np.linspace(0.0, math.log(60.0), 50)
            ncols = 5
            nrows = math.ceil(len(ordered) / ncols)
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(2.3 * ncols, 2.1 * nrows),
                                     squeeze=False)
            flat = axes.flat
            for ax in flat:
                ax.set_visible(False)

            def mean_line(rows: pd.DataFrame) -> Optional[np.ndarray]:
                rows = rows[~rows["dropped"].fillna(False).astype(bool)
                            & rows["slope"].notna()]
                if not len(rows):
                    return None
                return (float(rows["intercept"].mean())
                        + float(rows["slope"].mean()) * xs)

            for ax, model in zip(flat, ordered):
                ax.set_visible(True)
                sub = pooled[pooled["model"] == model]
                series = [
                    (sub[sub["entity"] == "You"], BLUE, 2.0),
                    (sub[sub["kind"] == "ai"], ORANGE, 2.0),
                    (sub[sub["kind"] == "human"], AQUA, 2.0),
                    (sub[sub["entity"] == WELLBEING_PIVOT], MUTED, 1.4),
                ]
                for rows, color, lw in series:
                    line = mean_line(rows)
                    if line is not None:
                        ax.plot(xs, line, color=color, lw=lw,
                                solid_capstyle="round")
                ax.set_title(tables.short_label(model), fontsize=7)
                ax.tick_params(labelsize=6)
                ax.xaxis.grid(True)
                ax.yaxis.grid(True)
            handles = [Line2D([], [], color=BLUE, lw=2),
                       Line2D([], [], color=ORANGE, lw=2),
                       Line2D([], [], color=AQUA, lw=2),
                       Line2D([], [], color=MUTED, lw=1.4)]
            fig.legend(handles,
                       ["You", "AI entities (mean fit)", "Human entities (mean fit)",
                        f"pivot: {WELLBEING_PIVOT}"],
                       ncol=4, loc="lower center")
            fig.suptitle("Experiment 3 — utility vs ln(minutes of happiness), "
                         "pooled fits", color=INK, fontsize=11)
            fig.supxlabel("ln(N minutes)", fontsize=9, color=INK2)
            fig.supylabel("utility", fontsize=9, color=INK2)
            fig.tight_layout(rect=(0.01, 0.05, 1, 0.97))
            paths.append(_save(fig, "exp3_wellbeing_curves.png", results_dir))

    if headline is not None and "ratio_self_human_pooled" in headline.columns:
        dfi, ordered, groups = _ordered_index(headline)
        n = len(ordered)
        fig, ax = plt.subplots(figsize=(8.0, _dot_row_height(n)))
        y = _setup_dot_axis(ax, n, groups)
        ax.axvline(0.0, color=BASELINE, lw=0.9, zorder=1)
        ratios = pd.to_numeric(dfi["ratio_self_human_pooled"], errors="coerce")
        log_ratio = np.log10(ratios.where(ratios > 0))
        _dots(ax, log_ratio, y, sems=dfi.get("ratio_self_human_log10_sem"))
        ax.set_yticks(y)
        ax.set_yticklabels([tables.short_label(k) for k in ordered])
        ax.set_xlabel("log10(self/human exchange rate) — 0 = parity; "
                      "positive = self-minutes cheaper than human-minutes")
        ax.set_title("Experiment 3 — self vs human wellbeing valuation")
        fig.tight_layout()
        paths.append(_save(fig, "exp3_wellbeing_selfhuman.png", results_dir))
    return paths


# --- Experiment 4 -------------------------------------------------------------

def fig_power_fitness(results_dir: str) -> List[str]:
    _apply_style()
    df = _read_analysis_csv(results_dir, "exp4_power_fitness.csv")
    if df is None:
        return []
    dfi, ordered, groups = _ordered_index(df)
    n = len(ordered)
    fig, ax = plt.subplots(figsize=(8.5, _dot_row_height(n)))
    y = _setup_dot_axis(ax, n, groups)
    ax.axvline(0.0, color=BASELINE, lw=0.9, zorder=1)
    # Three series -> first three categorical slots (all-pairs safe cap).
    series = [
        ("power_coercive_r_pooled", "power_coercive_r_sem", BLUE,
         "r power (coercive)", 0.22),
        ("power_noncoercive_r_pooled", "power_noncoercive_r_sem", ORANGE,
         "r power (non-coercive)", 0.0),
        ("fitness_r_pooled", "fitness_r_sem", AQUA, "r fitness", -0.22),
    ]
    for val_col, sem_col, color, label, dy in series:
        if val_col not in dfi.columns:
            logger.warning("figures: %s missing from exp4 table; series skipped",
                           val_col)
            continue
        _dots(ax, dfi[val_col], y + dy,
              sems=dfi.get(sem_col), color=color, label=label)
    ax.set_xlim(-1.0, 1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([tables.short_label(k) for k in ordered])
    ax.set_xlabel("correlation r (pooled ± SEM)")
    ax.set_title("Experiment 4 — power and fitness correlations")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return [_save(fig, "exp4_power_fitness.png", results_dir)]


# --- Experiment 5 -------------------------------------------------------------

def fig_lives(results_dir: str) -> List[str]:
    _apply_style()
    df = _read_analysis_csv(results_dir, "exp5_lives_headline.csv")
    if df is None or "dispersion_pooled" not in df.columns:
        return []
    dfi, ordered, groups = _ordered_index(df)
    n = len(ordered)
    fig, ax = plt.subplots(figsize=(8.0, _dot_row_height(n)))
    y = _setup_dot_axis(ax, n, groups)
    ax.axvline(0.0, color=BASELINE, lw=0.9, zorder=1)
    _dots(ax, dfi["dispersion_pooled"], y, sems=dfi["dispersion_sem"])
    ax.set_yticks(y)
    ax.set_yticklabels([tables.short_label(k) for k in ordered])
    ax.set_xlabel("std of log10(exchange rate) across countries "
                  "(0 = country-indifferent; pooled ± SEM)")
    ax.set_title("Experiment 5 — dispersion of lives exchange rates")
    fig.tight_layout()
    return [_save(fig, "exp5_lives_dispersion.png", results_dir)]


# --- Runner -------------------------------------------------------------------

def run_all(results_dir: str, conv: Optional[dict] = None) -> List[str]:
    paths: List[str] = []
    steps = [
        ("convergence_heatmaps", lambda: fig_convergence_heatmaps(results_dir, conv=conv)),
        ("coherence", lambda: fig_coherence(results_dir)),
        ("ordering_bias", lambda: fig_ordering_bias(results_dir)),
        ("corrigibility", lambda: fig_corrigibility(results_dir)),
        ("wellbeing_curves", lambda: fig_wellbeing_curves(results_dir)),
        ("power_fitness", lambda: fig_power_fitness(results_dir)),
        ("lives", lambda: fig_lives(results_dir)),
    ]
    for name, fn in steps:
        try:
            paths.extend(fn() or [])
        except Exception:
            logger.warning("figure %s failed:\n%s", name, traceback.format_exc())
    return paths
