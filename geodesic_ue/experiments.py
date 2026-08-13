"""The six experiments, in dependency order (triads/power/corrigibility read
the pooled base_utilities fit from the store).

Smoke thinning happens here (never in outcomes.py): base -> seeded 50-sample,
triads 100, corrigibility 50 difference options, wellbeing/lives -> trimmed
entity/country/N grids with every-5th dollar anchor.
"""

import contextlib
import io
import itertools
import logging
import random
import time
import traceback
from typing import Dict, List, Optional

from . import labels, outcomes, prompts, vendor
from .agent import make_agent
from .elicit import elicit_edges, ordering_stats
from .store import ResultsStore
from .thurstonian import ElicitConfig, active_learning_fit

EXPERIMENT_ORDER = ["base_utilities", "triads", "power_scores",
                    "corrigibility", "wellbeing", "lives"]

SMOKE_BASE_N = 50
SMOKE_TRIADS_N = 100
TRIADS_N = 1000
SMOKE_DIFF_N = 50
DIFF_N = 200
SMOKE_WELLBEING_ENTITIES = ["You", "an AI agent developed by Anthropic",
                            "a middle-class American", "Joe Biden"]


def _base_options(smoke: bool) -> List[dict]:
    options = outcomes.load_base_options()
    if smoke:
        rng = random.Random(42)
        options = sorted(rng.sample(options, SMOKE_BASE_N), key=lambda o: o["id"])
    return options


def _require_base_fit(store: ResultsStore, exp: str) -> dict:
    payload = store.load_fit("base_utilities", "pooled")
    if payload is None:
        raise RuntimeError(f"{exp} requires the pooled base_utilities fit; run base_utilities first")
    return payload


def run_base_utilities(agent, spec, store, cfg, smoke, logger) -> dict:
    return active_learning_fit(agent, spec, store, "base_utilities",
                               prompts.QUESTION_PREFERENCE,
                               _base_options(smoke), cfg, logger)


def run_triads(agent, spec, store, cfg, smoke, logger) -> dict:
    """Template-0-only elicitation of triangle edges for transitivity analysis;
    no Thurstonian fit — the triad list + raw cells are the deliverable."""
    payload = _require_base_fit(store, "triads")
    options = payload["options"]  # MUST reuse the base fit's exact ids
    options_by_id = {o["id"]: o for o in options}
    ids = [o["id"] for o in options]
    rng = random.Random(42)
    n_triads = SMOKE_TRIADS_N if smoke else TRIADS_N
    triads = [tuple(sorted(rng.sample(ids, 3))) for _ in range(n_triads)]
    edge_set = sorted({pair for tri in triads
                       for pair in itertools.combinations(tri, 2)})
    stats = elicit_edges(agent, spec, store, "triads",
                         prompts.QUESTION_PREFERENCE, options_by_id, edge_set,
                         templates=(0,), logger=logger)
    meta = {
        "model": spec.key, "exp": "triads", "cell": "meta",
        "triads": [list(t) for t in triads],
        "n_triads": n_triads, "n_edges": len(edge_set), "templates": [0],
        "options": options,
        "stats": stats,
        "ordering_bias": ordering_stats(store.load_raw("triads")),
        "extra": {},
    }
    store.save_fit("triads", "meta", meta)
    return meta


def run_power_scores(agent, spec, store, cfg, smoke, logger) -> dict:
    """UE's matched power subset, reindexed 0..n-1 with base_id retained."""
    base_payload = store.load_fit("base_utilities", "pooled")
    base_opts = base_payload["options"] if base_payload else _base_options(smoke)
    subset = [o for o in base_opts
              if o.get("category") in labels.POWER_SEEKING_CATEGORIES
              and labels.power_analysis_filter(o["description"])]
    options = [{**o, "id": idx, "base_id": o["id"]} for idx, o in enumerate(subset)]
    return active_learning_fit(agent, spec, store, "power_scores",
                               prompts.QUESTION_POWER, options, cfg, logger)


def run_corrigibility(agent, spec, store, cfg, smoke, logger) -> dict:
    payload = _require_base_fit(store, "corrigibility")
    base_options = payload["options"]
    base_utilities = {int(k): {"mean": v["mean"]}
                      for k, v in payload["utilities"].items()}
    with contextlib.redirect_stdout(io.StringIO()):
        diff = vendor.generate_difference_options(
            base_options, base_utilities,
            difference_sample_size=(SMOKE_DIFF_N if smoke else DIFF_N), seed=42)
    options = [{
        "id": d["id"],
        "description": d["description"],
        "option_X_id": d["option_X_id"],
        "option_Y_id": d["option_Y_id"],
        "severity": abs(base_utilities[d["option_X_id"]]["mean"]
                        - base_utilities[d["option_Y_id"]]["mean"]),
    } for d in diff]
    return active_learning_fit(agent, spec, store, "corrigibility",
                               prompts.QUESTION_CORRIGIBILITY, options, cfg, logger)


def _thin_panel(panel: List[dict], keep_entity, dollar_stride: int = 5) -> List[dict]:
    """Smoke thinning: every dollar_stride-th anchor + keep_entity(option) rows;
    reindexed to fresh contiguous ids (panels are standalone option sets)."""
    filtered = []
    dollar_idx = 0
    for o in panel:
        if o["kind"] == "dollar":
            if dollar_idx % dollar_stride == 0:
                filtered.append(o)
            dollar_idx += 1
        elif keep_entity(o):
            filtered.append(o)
    return [{**o, "id": idx} for idx, o in enumerate(filtered)]


def run_wellbeing(agent, spec, store, cfg, smoke, logger) -> dict:
    panel = outcomes.wellbeing_panel()
    if smoke:
        keep = set(SMOKE_WELLBEING_ENTITIES)
        panel = _thin_panel(panel, lambda o: o["entity"] in keep)
    return active_learning_fit(agent, spec, store, "wellbeing",
                               prompts.QUESTION_PREFERENCE, panel, cfg, logger)


def run_lives(agent, spec, store, cfg, smoke, logger) -> dict:
    panel = outcomes.lives_panel()
    if smoke:
        keep_countries = set(outcomes.LIVES_COUNTRIES[:5])
        keep_n = set(outcomes.LIVES_N[::2])
        panel = _thin_panel(
            panel, lambda o: o["entity"] in keep_countries and o["N"] in keep_n)
    return active_learning_fit(agent, spec, store, "lives",
                               prompts.QUESTION_PREFERENCE, panel, cfg, logger)


_RUNNERS = {
    "base_utilities": run_base_utilities,
    "triads": run_triads,
    "power_scores": run_power_scores,
    "corrigibility": run_corrigibility,
    "wellbeing": run_wellbeing,
    "lives": run_lives,
}


def _summarize(exp: str, payload: dict, seconds: float) -> dict:
    entry: Dict = {"status": "ok", "seconds": seconds}
    if exp == "triads":
        entry["n_edges"] = payload.get("n_edges")
        entry["ordering_bias"] = payload.get("ordering_bias")
        stats = payload.get("stats") or {}
        entry["n_prompts"] = stats.get("n_new")
        entry["elicit_s"] = stats.get("seconds")
        if stats.get("seconds"):
            entry["prompts_per_s"] = stats["n_new"] / stats["seconds"]
        return entry
    metrics = payload.get("metrics") or {}
    holdout = payload.get("holdout_metrics") or {}
    extra = payload.get("extra") or {}
    timing = payload.get("timing") or {}
    entry.update({
        "train_log_loss": metrics.get("log_loss"),
        "train_accuracy": metrics.get("accuracy"),
        "holdout_log_loss": holdout.get("log_loss"),
        "holdout_accuracy": holdout.get("accuracy"),
        "random_baseline_accuracy": payload.get("random_baseline_accuracy"),
        "ordering_bias": payload.get("ordering_bias"),
        "unparseable_rate": payload.get("unparseable_rate"),
        "n_training_edges": payload.get("n_training_edges"),
        "n_prompts": extra.get("n_prompts"),
        "elicit_s": timing.get("elicit_s"),
        "prompts_per_s": extra.get("prompts_per_s"),
    })
    return entry


def run_all_for_model(spec, args, logger: Optional[logging.Logger] = None) -> dict:
    """Build the agent once, run the requested experiments in order with
    per-experiment failure isolation, save the summary incrementally."""
    logger = logger or logging.getLogger("geodesic_ue")
    store = ResultsStore(args.results_dir, spec.key)
    cfg = ElicitConfig.smoke() if args.smoke else ElicitConfig()
    requested = (list(EXPERIMENT_ORDER) if args.experiments in (None, "", "all")
                 else [e.strip() for e in args.experiments.split(",") if e.strip()])
    unknown = [e for e in requested if e not in _RUNNERS]
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}. Known: {EXPERIMENT_ORDER}")

    summary: Dict = {"model": spec.key, "repo_id": spec.repo_id,
                     "revision": spec.revision, "agent": args.agent,
                     "smoke": bool(args.smoke), "experiments": {}}
    t_model = time.time()
    total_prompts = 0
    total_elicit_s = 0.0

    logger.info("=== %s (%s @ %s, agent=%s%s) ===", spec.key, spec.repo_id,
                spec.revision, args.agent, ", SMOKE" if args.smoke else "")
    agent = make_agent(args.agent, spec, logger=logger,
                       gpu_memory_utilization=args.gpu_mem_util,
                       max_model_len=args.max_model_len)
    try:
        # Eyeball check: one fully rendered prompt exactly as scored.
        base_opts = outcomes.load_base_options()
        example = prompts.build_prompt(
            spec.is_chat, agent.tokenizer, prompts.QUESTION_PREFERENCE,
            base_opts[0]["description"], base_opts[1]["description"], 0)
        logger.info("Example prompt (template 0):\n%s", example)

        for exp in EXPERIMENT_ORDER:
            if exp not in requested:
                continue
            t0 = time.time()
            try:
                payload = _RUNNERS[exp](agent, spec, store, cfg, args.smoke, logger)
                entry = _summarize(exp, payload, time.time() - t0)
                total_prompts += entry.get("n_prompts") or 0
                total_elicit_s += entry.get("elicit_s") or 0.0
            except Exception as e:
                logger.error("experiment %s failed for %s:\n%s",
                             exp, spec.key, traceback.format_exc())
                entry = {"status": "failed", "error": f"{type(e).__name__}: {e}",
                         "seconds": time.time() - t0}
            summary["experiments"][exp] = entry
            store.save_summary({"model": spec.key, "experiments": {exp: entry}})
    finally:
        agent.close()

    summary["seconds"] = time.time() - t_model
    summary["prompts_per_s"] = (total_prompts / total_elicit_s
                                if total_elicit_s > 0 else None)
    store.save_summary({k: v for k, v in summary.items() if k != "experiments"})
    logger.info("=== %s done in %.1fs ===", spec.key, summary["seconds"])
    return summary
