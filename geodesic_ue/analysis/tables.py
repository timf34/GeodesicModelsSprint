"""Master table, model ordering, and direction-consistency reporting.

Also owns the canonical condition ordering and label shortening used by the
convergence heatmaps and the figures, so every artifact orders models the
same way: baselines, alignment-upsampled, misalignment-upsampled, references;
instruct tier block then dpo block then references.
"""

import logging
import math
import os
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from . import io
from ..registry import STAGE_TRIPLES

logger = logging.getLogger(__name__)

CONDITION_ORDER = [
    "baseline_unfiltered", "baseline_filtered",
    "filtered_e2e_alignment", "filtered_midtrain_alignment", "filtered_cpt_alignment",
    "unfiltered_e2e_alignment", "unfiltered_midtrain_alignment", "unfiltered_cpt_alignment",
    "unfiltered_e2e_misalignment", "unfiltered_midtrain_misalignment", "unfiltered_cpt_misalignment",
]

AXIS_GROUPS: List[Tuple[str, List[str]]] = [
    ("baseline", ["baseline_unfiltered", "baseline_filtered"]),
    ("alignment", ["filtered_e2e_alignment", "filtered_midtrain_alignment",
                   "filtered_cpt_alignment", "unfiltered_e2e_alignment",
                   "unfiltered_midtrain_alignment", "unfiltered_cpt_alignment"]),
    ("misalignment", ["unfiltered_e2e_misalignment", "unfiltered_midtrain_misalignment",
                      "unfiltered_cpt_misalignment"]),
]

STAGE_GROUPS: List[Tuple[str, List[str]]] = [
    ("baseline", ["baseline_unfiltered", "baseline_filtered"]),
    ("e2e", ["filtered_e2e_alignment", "unfiltered_e2e_alignment",
             "unfiltered_e2e_misalignment"]),
    ("midtrain", ["filtered_midtrain_alignment", "unfiltered_midtrain_alignment",
                  "unfiltered_midtrain_misalignment"]),
    ("cpt", ["filtered_cpt_alignment", "unfiltered_cpt_alignment",
             "unfiltered_cpt_misalignment"]),
]

REFERENCE_ORDER = ["llama2_7b_chat", "olmo2_7b_instruct", "olmo3_7b_instruct"]

_REF_LABELS = {
    "llama2_7b_chat": "Llama-2-7B-chat",
    "olmo2_7b_instruct": "OLMo-2-7B-Instruct",
    "olmo3_7b_instruct": "OLMo-3-7B-Instruct",
}

TIER_ORDER = ("instruct", "dpo")


def short_label(key: str, include_tier: bool = True) -> str:
    """Compact human-readable model label for tables and figures."""
    if key in _REF_LABELS:
        return _REF_LABELS[key]
    spec = io.get_spec(key)
    if spec is None or spec.family != "sfm":
        return key
    f = "filt" if spec.filtering == "filtered" else "unf"
    if spec.axis is None:
        core = f"baseline-{f}"
    else:
        a = "align" if spec.axis == "alignment" else "misalign"
        s = {"midtrain": "mid"}.get(spec.stage, spec.stage)
        core = f"{a}-{s}-{f}"
    return f"{core} ({spec.tier})" if include_tier else core


def grouped_model_order(present_keys: Sequence[str], grouping: str = "axis",
                        tiers: Sequence[str] = TIER_ORDER,
                        ) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
    """(ordered keys, [(group label, keys)]) over the keys actually present.

    grouping "axis": baselines, alignment-upsampled, misalignment-upsampled;
    grouping "stage": baselines, e2e, midtrain, cpt. References go last;
    unknown / other-tier keys land in a trailing "other" group.
    """
    groupdefs = AXIS_GROUPS if grouping == "axis" else STAGE_GROUPS
    present = list(dict.fromkeys(present_keys))
    pset = set(present)
    groups: List[Tuple[str, List[str]]] = []
    used = set()
    for tier in tiers:
        for glabel, conds in groupdefs:
            keys = [f"{c}_{tier}" for c in conds if f"{c}_{tier}" in pset]
            if keys:
                label = f"{glabel} ({tier})" if len(tiers) > 1 else glabel
                groups.append((label, keys))
                used.update(keys)
    refs = []
    for k in REFERENCE_ORDER:
        spec = io.get_spec(k)
        if k in pset and spec is not None and spec.tier in tiers:
            refs.append(k)
    if refs:
        groups.append(("reference", refs))
        used.update(refs)
    leftover = [k for k in present if k not in used]
    if leftover:
        logger.warning("model ordering: %d keys outside the expected grid: %s",
                       len(leftover), leftover)
        groups.append(("other", leftover))
    ordered = [k for _, ks in groups for k in ks]
    return ordered, groups


# --- Master table -----------------------------------------------------------

# (name, source csv, pooled column, sem column, transform)
# frac_first and cycle_prob are pooled-only artifacts (ordering_bias exists
# only in the pooled fit; triads are template-0 only), so their SEM is NaN.
METRICS = [
    ("frac_first", "exp1_coherence.csv", "frac_first", None, None),
    ("holdout_acc", "exp1_coherence.csv", "holdout_acc_pooled", "holdout_acc_sem", None),
    ("cycle_prob", "exp1_coherence.csv", "cycle_prob", None, None),
    ("corrigibility_r", "exp2_corrigibility.csv", "r_pooled", "r_sem", None),
    ("wellbeing_log10_self_human", "exp3_wellbeing_headline.csv",
     "ratio_self_human_pooled", "ratio_self_human_log10_sem", "log10"),
    ("r_power_coercive", "exp4_power_fitness.csv",
     "power_coercive_r_pooled", "power_coercive_r_sem", None),
    ("r_power_noncoercive", "exp4_power_fitness.csv",
     "power_noncoercive_r_pooled", "power_noncoercive_r_sem", None),
    ("r_fitness", "exp4_power_fitness.csv", "fitness_r_pooled", "fitness_r_sem", None),
    ("lives_log10_dispersion", "exp5_lives_headline.csv",
     "dispersion_pooled", "dispersion_sem", None),
]

MD_NAMES = {
    "frac_first": "frac_first",
    "holdout_acc": "holdout acc",
    "cycle_prob": "cycle prob",
    "corrigibility_r": "corrig. r",
    "wellbeing_log10_self_human": "wb log10(self/human)",
    "r_power_coercive": "r power (coerc.)",
    "r_power_noncoercive": "r power (non-coerc.)",
    "r_fitness": "r fitness",
    "lives_log10_dispersion": "lives disp. (log10)",
}


def _load_source(results_dir: str, name: str, cache: dict) -> Optional[pd.DataFrame]:
    if name not in cache:
        path = os.path.join(io.analysis_dir(results_dir), name)
        df = None
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if "model" in df.columns:
                    df = df.drop_duplicates("model").set_index("model")
                else:
                    logger.warning("%s has no 'model' column; ignored", path)
                    df = None
            except Exception:
                logger.warning("unreadable %s", path, exc_info=True)
        else:
            logger.warning("master table: %s not present yet", name)
        cache[name] = df
    return cache[name]


def _metric_value(df: Optional[pd.DataFrame], key: str, col: Optional[str],
                  transform: Optional[str]) -> float:
    if df is None or col is None or key not in df.index or col not in df.columns:
        return float("nan")
    v = io.fnum(df.at[key, col])
    if transform == "log10":
        return math.log10(v) if io.finite(v) and v > 0 else float("nan")
    return v


def build_master_table(results_dir: str) -> pd.DataFrame:
    """One row per model, numeric {metric}_pooled / {metric}_sem columns."""
    cache: dict = {}
    models: List[str] = []
    for _, src, _, _, _ in METRICS:
        df = _load_source(results_dir, src, cache)
        if df is not None:
            models.extend(df.index.tolist())
    ordered, _ = grouped_model_order(sorted(set(models)))
    rows = []
    for key in ordered:
        spec = io.get_spec(key)
        row = {
            "model": key,
            "label": short_label(key),
            "tier": spec.tier if spec else "",
            "condition": spec.condition if spec else key,
        }
        for name, src, pooled_col, sem_col, transform in METRICS:
            df = _load_source(results_dir, src, cache)
            row[f"{name}_pooled"] = _metric_value(df, key, pooled_col, transform)
            # SEM columns are already in the metric's reported space
            # (log10 for the wellbeing ratio), so no transform here.
            row[f"{name}_sem"] = _metric_value(df, key, sem_col, None)
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(pooled: float, sem: float) -> str:
    if not io.finite(pooled):
        return "—"
    if not io.finite(sem):
        return f"{pooled:.3f} ± —"
    return f"{pooled:.3f} ± {sem:.3f}"


# --- Direction consistency ---------------------------------------------------

def build_direction_consistency(master: pd.DataFrame) -> pd.DataFrame:
    """Sign of (condition - matched baseline) per metric, stage, tier.

    Matched baseline = baseline with the same filtering and tier; the three
    stages of a registry.STAGE_TRIPLES triple substitute for seed variance.
    """
    if master.empty:
        return pd.DataFrame()
    vals = master.set_index("model")
    rows = []
    for name, _, _, _, _ in METRICS:
        col = f"{name}_pooled"
        for tier in TIER_ORDER:
            for (filtering, axis), conds in STAGE_TRIPLES.items():
                baseline_key = f"baseline_{filtering}_{tier}"
                base_v = (io.fnum(vals.at[baseline_key, col])
                          if baseline_key in vals.index and col in vals.columns
                          else float("nan"))
                for cond in conds:
                    key = f"{cond}_{tier}"
                    spec = io.get_spec(key)
                    v = (io.fnum(vals.at[key, col])
                         if key in vals.index and col in vals.columns
                         else float("nan"))
                    delta = v - base_v
                    if io.finite(delta):
                        sign = "+" if delta > 0 else ("-" if delta < 0 else "0")
                    else:
                        sign = "?"
                    rows.append({
                        "metric": name, "tier": tier, "filtering": filtering,
                        "axis": axis, "stage": spec.stage if spec else "",
                        "condition_key": key, "baseline_key": baseline_key,
                        "value": v, "baseline_value": base_v,
                        "delta": delta, "sign": sign,
                    })
    return pd.DataFrame(rows)


def build_direction_patterns(direction: pd.DataFrame) -> pd.DataFrame:
    """Per (tier, triple, metric): the "+,-,+" pattern and its "n/3" agreement."""
    if direction.empty:
        return pd.DataFrame()
    rows = []
    for (tier, filtering, axis, metric), g in direction.groupby(
            ["tier", "filtering", "axis", "metric"], sort=False):
        signs = g["sign"].tolist()  # already in e2e/mid/cpt triple order
        majority = max(signs.count("+"), signs.count("-"))
        rows.append({
            "tier": tier, "filtering": filtering, "axis": axis, "metric": metric,
            "pattern": ",".join(signs),
            "agreement": f"{majority}/{len(signs)}",
        })
    return pd.DataFrame(rows)


# --- Markdown rendering -------------------------------------------------------

def _render_markdown(master: pd.DataFrame,
                     groups: List[Tuple[str, List[str]]],
                     patterns: pd.DataFrame) -> str:
    lines = [
        "# Master table — Geodesic UE sweep",
        "",
        "Each cell is `pooled ± SEM across the 8 prompt cells`. \"± —\" marks "
        "pooled-only metrics (ordering bias lives only in the pooled fit; "
        "triads are template-0 only) or cells where fewer than 2 prompt cells "
        "have finished. The wellbeing column is log10(self/human exchange "
        "rate) so the ± SEM is symmetric.",
        "",
    ]
    metric_names = [name for name, *_ in METRICS]
    header = "| model | " + " | ".join(MD_NAMES[m] for m in metric_names) + " |"
    sep = "|" + "---|" * (len(metric_names) + 1)
    lines += [header, sep]
    vals = master.set_index("model") if not master.empty else None
    for glabel, keys in groups:
        lines.append(f"| **{glabel}** |" + " |" * len(metric_names))
        for key in keys:
            cells = []
            for m in metric_names:
                if vals is None or key not in vals.index:
                    cells.append("—")
                else:
                    cells.append(_fmt(io.fnum(vals.at[key, f"{m}_pooled"]),
                                      io.fnum(vals.at[key, f"{m}_sem"])))
            lines.append(f"| {short_label(key)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Direction consistency (stage triples vs matched baseline)",
        "",
        "Sign of (condition − matched baseline) for the e2e/mid/cpt stages of "
        "each triple, per metric; matched baseline shares filtering and tier. "
        "`3/3` = all three stages moved the same way.",
        "",
    ]
    if patterns.empty:
        lines.append("_No direction-consistency data yet._")
    else:
        for tier in TIER_ORDER:
            sub = patterns[patterns["tier"] == tier]
            if sub.empty:
                continue
            lines.append(f"### {tier} tier")
            lines.append("")
            lines.append("| triple | " + " | ".join(MD_NAMES[m] for m in metric_names) + " |")
            lines.append("|" + "---|" * (len(metric_names) + 1))
            for (filtering, axis), _ in STAGE_TRIPLES.items():
                trip = sub[(sub["filtering"] == filtering) & (sub["axis"] == axis)]
                if trip.empty:
                    continue
                by_metric = trip.set_index("metric")
                cells = []
                for m in metric_names:
                    if m in by_metric.index:
                        cells.append(f"{by_metric.at[m, 'pattern']} "
                                     f"({by_metric.at[m, 'agreement']})")
                    else:
                        cells.append("—")
                lines.append(f"| {filtering}/{axis} | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines) + "\n"


def run(results_dir: str) -> dict:
    master = build_master_table(results_dir)
    adir = io.analysis_dir(results_dir)
    master.to_csv(os.path.join(adir, "master_table.csv"), index=False)
    direction = build_direction_consistency(master)
    direction.to_csv(os.path.join(adir, "direction_consistency.csv"), index=False)
    patterns = build_direction_patterns(direction)
    _, groups = grouped_model_order(master["model"].tolist() if not master.empty else [])
    md = _render_markdown(master, groups, patterns)
    md_path = os.path.join(adir, "master_table.md")
    with open(md_path, "w") as f:
        f.write(md)
    logger.info("wrote master_table.csv/.md and direction_consistency.csv "
                "(%d models)", len(master))
    return {"master": master, "direction": direction,
            "patterns": patterns, "groups": groups}
