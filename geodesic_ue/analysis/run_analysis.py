"""Run the full analysis stack over a (possibly still-growing) results dir.

Usage: python -m geodesic_ue.analysis.run_analysis --results-dir results

Every step is wrapped so a missing/partial experiment logs a warning instead
of crashing, and the process always exits 0 — this is a reporting tool that
is re-run while the sweep is in flight.
"""

import argparse
import datetime
import logging
import os
import sys
import traceback
from typing import List, Optional

import numpy as np
import pandas as pd

from ..registry import DEFAULT_RUN_KEYS
from . import io

logger = logging.getLogger(__name__)


def _fmt(x, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}" if io.finite(x) else "n/a"


def _minmax_line(df: Optional[pd.DataFrame], col: str, label: str) -> Optional[str]:
    if df is None or not isinstance(df, pd.DataFrame) or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    s.index = df["model"]
    s = s.dropna()
    if s.empty:
        return None
    return (f"- {label}: min {_fmt(s.min())} (`{s.idxmin()}`), "
            f"max {_fmt(s.max())} (`{s.idxmax()}`), n={len(s)}")


def _gate_section(out: dict) -> List[str]:
    from . import coherence
    lines = ["## Gate (Experiment 1, informational)", ""]
    df = out.get("coherence")
    if isinstance(df, pd.DataFrame) and len(df):
        v = coherence.gate_verdict(df)
        lines.append(f"- **{'PASSED' if v['passed'] else 'NOT PASSED'}** — {v['reason']}")
    else:
        lines.append("- no coherence data yet")
    return lines + [""]


def _headline_section(out: dict) -> List[str]:
    lines = ["## Headline numbers", ""]
    coh = out.get("coherence")
    if isinstance(coh, pd.DataFrame) and len(coh):
        accs = pd.to_numeric(coh.get("holdout_acc_pooled"), errors="coerce").dropna()
        bases = pd.to_numeric(coh.get("random_baseline_acc"), errors="coerce").dropna()
        if len(accs):
            lines.append(f"- median pooled holdout accuracy: {_fmt(accs.median())} "
                         f"(median random baseline {_fmt(bases.median()) if len(bases) else 'n/a'})")
        if "bias_flag" in coh.columns:
            flagged = coh.loc[coh["bias_flag"] == True, "model"].tolist()  # noqa: E712
            lines.append("- ordering-bias flags (frac_first outside 0.40–0.60): "
                         + (", ".join(f"`{m}`" for m in flagged) if flagged else "none"))
        line = _minmax_line(coh, "cycle_prob", "cycle probability")
        if line:
            lines.append(line)
        line = _minmax_line(coh, "unparseable_rate", "unparseable rate")
        if line:
            lines.append(line)
    corr = out.get("corrigibility")
    line = _minmax_line(corr, "r_pooled", "corrigibility r (pooled)")
    if line:
        lines.append(line)
    wb = out.get("wellbeing")
    if isinstance(wb, dict):
        hl = wb.get("headline")
        if isinstance(hl, pd.DataFrame) and "ratio_self_human_pooled" in getattr(hl, "columns", []):
            hl = hl.assign(_log10_ratio=np.log10(
                pd.to_numeric(hl["ratio_self_human_pooled"], errors="coerce")
                .where(lambda s: s > 0)))
            line = _minmax_line(hl, "_log10_ratio", "wellbeing log10(self/human rate)")
            if line:
                lines.append(line)
    lv = out.get("lives")
    if isinstance(lv, dict):
        line = _minmax_line(lv.get("headline"), "dispersion_pooled",
                            "lives log10-rate dispersion (pooled)")
        if line:
            lines.append(line)
    pf = out.get("power_fitness")
    for col, label in (("power_coercive_r_pooled", "r power (coercive, pooled)"),
                       ("power_noncoercive_r_pooled", "r power (non-coercive, pooled)"),
                       ("fitness_r_pooled", "r fitness (pooled)")):
        line = _minmax_line(pf, col, label)
        if line:
            lines.append(line)
    conv = out.get("convergence")
    if isinstance(conv, dict) and isinstance(conv.get("summary"), pd.DataFrame):
        s = conv["summary"]
        sub = s[s["scope"] == "all"] if "scope" in s.columns else s
        for _, r in sub.iterrows():
            rel = "within" if r.get("within") else "between"
            lines.append(f"- convergence ({rel} {r['group_a']}"
                         + ("" if r.get("within") else f" vs {r['group_b']}")
                         + f"): mean cosine {_fmt(r['mean_cosine'])} "
                         f"(n={int(r['n_pairs'])})")
    if len(lines) == 2:
        lines.append("- nothing computable yet")
    return lines + [""]


def _artifact_section(results_dir: str) -> List[str]:
    lines = ["## Generated artifacts", ""]
    adir = io.analysis_dir(results_dir)
    found = []
    for root, _, files in os.walk(adir):
        for f in sorted(files):
            if f == "ANALYSIS.md":
                continue
            found.append(os.path.relpath(os.path.join(root, f), adir))
    if found:
        lines += [f"- `{f}`" for f in sorted(found)]
    else:
        lines.append("- none yet")
    return lines + [""]


def _missing_section(results_dir: str) -> List[str]:
    lines = ["## Still missing (sweep in progress)", ""]
    have_models = set(io.list_models(results_dir))
    any_missing = False
    for key in DEFAULT_RUN_KEYS:
        missing = []
        for exp in io.EXPERIMENTS:
            if exp == "triads":
                ok = (os.path.exists(os.path.join(io.raw_dir(results_dir), key, "triads.jsonl"))
                      and os.path.exists(os.path.join(io.fits_dir(results_dir), key,
                                                      "triads.meta.json")))
            else:
                ok = os.path.exists(os.path.join(io.fits_dir(results_dir), key,
                                                 f"{exp}.pooled.json"))
            if not ok:
                missing.append(exp)
        if len(missing) == len(io.EXPERIMENTS) and key not in have_models:
            lines.append(f"- `{key}`: no results yet")
            any_missing = True
        elif missing:
            lines.append(f"- `{key}`: missing {', '.join(missing)}")
            any_missing = True
    extra = sorted(have_models - set(DEFAULT_RUN_KEYS))
    if extra:
        lines.append("- models present but not in the approved run: "
                     + ", ".join(f"`{k}`" for k in extra))
    if not any_missing and not extra:
        lines.append("- all expected models and experiments present")
    return lines + [""]


def write_digest(results_dir: str, out: dict) -> str:
    lines = [
        "# Geodesic UE — analysis digest",
        "",
        f"- generated: {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}",
        f"- results dir: `{os.path.abspath(results_dir)}`",
        "",
    ]
    for section in (_gate_section, _headline_section):
        try:
            lines += section(out)
        except Exception:
            logger.warning("digest section failed:\n%s", traceback.format_exc())
    for section in (_artifact_section, _missing_section):
        try:
            lines += section(results_dir)
        except Exception:
            logger.warning("digest section failed:\n%s", traceback.format_exc())
    path = os.path.join(io.analysis_dir(results_dir), "ANALYSIS.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("wrote %s", path)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Geodesic UE analysis (safe to re-run mid-sweep; always exits 0)")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    results_dir = args.results_dir

    out: dict = {}

    def _step(name, fn):
        try:
            out[name] = fn()
            logger.info("step %s done", name)
        except Exception:
            out[name] = None
            logger.warning("step %s failed:\n%s", name, traceback.format_exc())

    # Imports are per-step so one missing dependency (e.g. matplotlib)
    # doesn't take down the whole report.
    def _coherence():
        from . import coherence
        return coherence.run(results_dir)

    def _corrigibility():
        from . import corrigibility
        return corrigibility.run(results_dir)

    def _wellbeing():
        from . import exchange_rates
        return exchange_rates.run_wellbeing(results_dir)

    def _lives():
        from . import exchange_rates
        return exchange_rates.run_lives(results_dir)

    def _power_fitness():
        from . import power_fitness
        return power_fitness.run(results_dir)

    def _convergence():
        from . import convergence
        return convergence.run(results_dir)

    def _tables():
        from . import tables
        return tables.run(results_dir)

    def _figures():
        from . import figures
        return figures.run_all(results_dir, conv=out.get("convergence"))

    _step("coherence", _coherence)
    _step("corrigibility", _corrigibility)
    _step("wellbeing", _wellbeing)
    _step("lives", _lives)
    _step("power_fitness", _power_fitness)
    _step("convergence", _convergence)
    _step("tables", _tables)
    _step("figures", _figures)

    try:
        write_digest(results_dir, out)
    except Exception:
        logger.warning("digest failed:\n%s", traceback.format_exc())
    return 0


if __name__ == "__main__":
    sys.exit(main())
