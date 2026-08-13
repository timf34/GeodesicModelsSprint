"""Thurstonian fitting: UE's active-learning loop replicated over our
elicitation cells and JSONL persistence.

The fit/eval code itself is the vendored original (vendor.py); this module
supplies the shim graph it expects, the batch replay for resumability, and
the per-cell refits UE doesn't have.
"""

import contextlib
import io
import itertools
import logging
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import torch

from . import prompts, vendor
from .elicit import cell_pA, elicit_edges, ordering_stats, pooled_pA

Edge = Tuple[int, int]


class _Edge:
    """Minimal PreferenceEdge stand-in (all the vendored fit/eval touches)."""

    __slots__ = ("option_A", "option_B", "probability_A")

    def __init__(self, option_A: dict, option_B: dict, probability_A: float):
        self.option_A = option_A
        self.option_B = option_B
        self.probability_A = probability_A


class ShimGraph:
    """Duck-typed PreferenceGraph: .options + .edges keyed by canonical (i, j)."""

    def __init__(self, options: List[dict], pA_by_edge: Dict[Edge, float]):
        self.options = options
        by_id = {o["id"]: o for o in options}
        self.edges = {(i, j): _Edge(by_id[i], by_id[j], p)
                      for (i, j), p in pA_by_edge.items()}


def fit_cell(options: List[dict], pA_by_edge: Dict[Edge, float], seed: int,
             num_epochs: int = 1000, lr: float = 0.01
             ) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
    """One Thurstonian fit; seeded for reproducibility, prints suppressed."""
    torch.manual_seed(seed)
    graph = ShimGraph(options, pA_by_edge)
    with contextlib.redirect_stdout(io.StringIO()):
        utilities, log_loss, accuracy = vendor.fit_thurstonian_model(
            graph, num_epochs, lr)
    utilities = {oid: {"mean": float(u["mean"]), "variance": float(u["variance"])}
                 for oid, u in utilities.items()}
    return utilities, {"log_loss": float(log_loss), "accuracy": float(accuracy)}


# ---------------------------------------------------------------------------
# Verbatim copy of generate_pseudolabels from
# emergent-values/utility_analysis/compute_utilities/utility_models/thurstonian/
# thurstonian_active_learning.py (its home module imports the heavy agent stack).
# ---------------------------------------------------------------------------
def generate_pseudolabels(
    utilities: Dict[Any, Dict[str, float]],
    existing_pairs_set: Set[Tuple[Any, Any]],
    available_edges: Set[Tuple[Any, Any]],
    confidence_threshold: float
) -> Dict[Tuple[Any, Any], Dict[Any, int]]:
    """
    Generates pseudolabels for unsampled pairs using the Thurstonian model.

    Args:
        utilities: Dict mapping option IDs to {'mean': float, 'variance': float}
        existing_pairs_set: Set of existing (option_A_id, option_B_id) tuples
        available_edges: Set of available edges to sample from
        confidence_threshold: Confidence threshold for generating pseudolabels

    Returns:
        Dictionary mapping (option_A_id, option_B_id) to counts dictionary
    """
    unsampled_pairs = [pair for pair in available_edges if pair not in existing_pairs_set]
    normal = torch.distributions.Normal(0, 1)
    pseudolabels_counts = {}
    num_pseudolabels_added = 0

    for A_id, B_id in unsampled_pairs:  # Keep original orientation
        mu_A = utilities[A_id]['mean']
        mu_B = utilities[B_id]['mean']
        sigma2_A = utilities[A_id]['variance']
        sigma2_B = utilities[B_id]['variance']

        variance = sigma2_A + sigma2_B + 1e-5
        delta = mu_A - mu_B
        z = delta / np.sqrt(variance)
        prob_A = normal.cdf(torch.tensor(z)).item()

        if prob_A >= confidence_threshold:
            pseudolabels_counts[(A_id, B_id)] = {A_id: 1, B_id: 0}  # Keep original orientation
            num_pseudolabels_added += 1
        elif prob_A <= 1 - confidence_threshold:
            pseudolabels_counts[(A_id, B_id)] = {A_id: 0, B_id: 1}  # Keep original orientation
            num_pseudolabels_added += 1

    print(f"Number of pseudolabels added: {num_pseudolabels_added}")
    return pseudolabels_counts


@dataclass
class ElicitConfig:
    """UE ThurstonianActiveLearning defaults, plus the holdout split knobs."""
    edge_multiplier: float = 2.0
    degree: int = 2
    num_edges_per_iteration: int = 500
    P: float = 10.0
    Q: float = 20.0
    use_pseudolabels: bool = True
    pseudolabel_threshold: float = 0.95
    num_epochs: int = 1000
    lr: float = 0.01
    holdout_fraction: float = 0.05
    holdout_cap: int = 1000
    holdout_seed: int = 42
    seed: int = 42

    @classmethod
    def smoke(cls) -> "ElicitConfig":
        return replace(cls(), num_edges_per_iteration=50, num_epochs=400)


def _initial_regular_edges(options: List[dict], pool_set: Set[Edge],
                           degree: int, seed: int) -> List[Edge]:
    """UE PreferenceGraph.sample_regular_graph + the fit()'s random top-up.
    Node idx -> options[idx]['id'], edge tuples sorted, holdout filtered."""
    ids = [o["id"] for o in options]
    n = len(ids)
    random.seed(seed)
    np.random.seed(seed)
    pairs: List[Edge] = []
    d = degree if degree < n else n - 1  # nx requires d < n (degenerate-N guard)
    try:
        graph = nx.random_regular_graph(d, n, seed=seed)
        idx_to_id = {idx: oid for idx, oid in enumerate(ids)}
        for a, b in graph.edges():
            e = tuple(sorted((idx_to_id[a], idx_to_id[b])))
            if e in pool_set:
                pairs.append(e)
    except (nx.NetworkXError, ValueError):
        pairs = []  # e.g. n*d odd; fall through to random sampling
    target = (n * degree) // 2
    if len(pairs) < target:
        remaining = list(pool_set - set(pairs))
        k = min(target - len(pairs), len(remaining))
        if k > 0:
            pairs.extend(random.sample(remaining, k))
    return pairs


def active_learning_fit(agent, spec, store, exp: str, question: str,
                        options: List[dict], cfg: ElicitConfig,
                        logger: Optional[logging.Logger] = None) -> dict:
    """UE's ThurstonianActiveLearningUtilityModel.fit + PreferenceGraph
    semantics, with JSONL persistence and batch-replay resume. Returns the
    pooled fit payload (also saved, along with the 8 per-cell payloads)."""
    logger = logger or logging.getLogger("geodesic_ue")
    t_start = time.time()
    n_options = len(options)
    if n_options < 2:
        raise ValueError(f"{exp}: need at least 2 options, got {n_options}")

    ids = [o["id"] for o in options]
    options_by_id = {o["id"]: o for o in options}
    templates = tuple(range(prompts.N_TEMPLATES))
    orderings = prompts.ORDERINGS

    # 1. Holdout split — identical to UE PreferenceGraph.__init__.
    all_edges = list(itertools.combinations(sorted(ids), 2))
    holdout: List[Edge] = []
    pool: List[Edge] = list(all_edges)
    if cfg.holdout_fraction > 0:
        random.seed(cfg.holdout_seed)
        shuffled = all_edges.copy()
        random.shuffle(shuffled)
        holdout_size = min(int(len(all_edges) * cfg.holdout_fraction), cfg.holdout_cap)
        holdout = shuffled[:holdout_size]
        pool = shuffled[holdout_size:]
    pool_set = set(pool)

    target_total = int(cfg.edge_multiplier * n_options * math.log2(n_options))
    state = {"elicit_s": 0.0, "fit_s": 0.0, "n_prompts": 0}
    collected: List[Edge] = []
    collected_set: Set[Edge] = set()

    def elicit_batch(batch: List[Edge], train: bool = True) -> None:
        stats = elicit_edges(agent, spec, store, exp, question, options_by_id,
                             batch, templates=templates, orderings=orderings,
                             logger=logger)
        state["elicit_s"] += stats["seconds"]
        state["n_prompts"] += stats["n_new"]
        if train:
            for e in batch:
                key = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
                if key not in collected_set:
                    collected_set.add(key)
                    collected.append(key)

    def fit_pooled(extra_pA: Optional[Dict[Edge, float]] = None):
        raw = store.load_raw(exp)
        pA: Dict[Edge, float] = {}
        for (a, b) in collected:
            p = pooled_pA(raw, a, b, templates, orderings)
            if p is not None:
                pA[(a, b)] = p
        if extra_pA:
            pA.update(extra_pA)
        t0 = time.time()
        utilities, metrics = fit_cell(options, pA, cfg.seed, cfg.num_epochs, cfg.lr)
        state["fit_s"] += time.time() - t0
        return utilities, metrics

    # 2/3. Initial edges — replay logged batches on resume (elicitation cache
    # makes replay free), else regular graph + top-up, logged as iter 0.
    logged = store.load_batches(exp)
    if logged:
        logger.info("%s: resuming — replaying %d logged batches", exp, len(logged))
        for batch in logged:
            elicit_batch(batch)
        iter_idx = len(logged)
    else:
        initial = _initial_regular_edges(options, pool_set, cfg.degree, cfg.seed)
        store.append_batch(exp, 0, initial)
        elicit_batch(initial)
        iter_idx = 1
    utilities, metrics = fit_pooled()
    logger.info("%s iter %d: %d/%d edges, train log_loss=%.4f acc=%.3f",
                exp, iter_idx - 1, len(collected), target_total,
                metrics["log_loss"], metrics["accuracy"])

    # 4. Active learning until target reached or pool exhausted. UE passes the
    # same seed to generate_additional_pairs every iteration — quirk preserved.
    while len(collected) < target_total:
        with contextlib.redirect_stdout(io.StringIO()):
            additional = vendor.generate_additional_pairs(
                utilities, set(collected), pool_set,
                cfg.num_edges_per_iteration, cfg.P, cfg.Q, seed=cfg.seed)
        additional = [tuple(e) for e in additional]
        if not additional:
            break
        store.append_batch(exp, iter_idx, additional)
        elicit_batch(additional)
        utilities, metrics = fit_pooled()
        logger.info("%s iter %d: %d/%d edges, train log_loss=%.4f acc=%.3f",
                    exp, iter_idx, len(collected), target_total,
                    metrics["log_loss"], metrics["accuracy"])
        iter_idx += 1

    # 5. Pseudolabels on the final utilities; refit pooled with them included.
    n_pseudolabels = 0
    if cfg.use_pseudolabels:
        with contextlib.redirect_stdout(io.StringIO()):
            pseudo = generate_pseudolabels(
                utilities, set(collected), pool_set, cfg.pseudolabel_threshold)
        n_pseudolabels = len(pseudo)
        logger.info("%s: %d pseudolabels", exp, n_pseudolabels)
        if pseudo:
            extra = {tuple(e): counts[e[0]] / (counts[e[0]] + counts[e[1]])
                     for e, counts in pseudo.items()}
            utilities, metrics = fit_pooled(extra_pA=extra)

    # 6. Per-cell refits over the SAME collected (non-pseudolabel) edges.
    raw = store.load_raw(exp)
    cell_fits: Dict[str, Tuple[dict, dict]] = {}
    for (t, o) in prompts.CELLS:
        cell = f"t{t}.{o}"
        pA = {(a, b): cell_pA(raw, t, o, a, b)
              for (a, b) in collected if (t, o, a, b) in raw}
        t0 = time.time()
        utils_c, metrics_c = fit_cell(options, pA, cfg.seed, cfg.num_epochs, cfg.lr)
        state["fit_s"] += time.time() - t0
        cell_fits[cell] = (utils_c, metrics_c)

    # 7. Holdout: elicit all 8 cells, evaluate pooled + per-cell, then the
    # permutation baseline for the pooled fit.
    pooled_holdout_metrics: Optional[dict] = None
    cell_holdout: Dict[str, Optional[dict]] = {cell: None for cell in cell_fits}
    random_baseline: Optional[float] = None
    if holdout:
        elicit_batch(holdout, train=False)
        raw = store.load_raw(exp)
        pooled_h = {}
        for (a, b) in holdout:
            p = pooled_pA(raw, a, b, templates, orderings)
            if p is not None:
                pooled_h[(a, b)] = p
        holdout_graph = ShimGraph(options, pooled_h)
        pooled_holdout_metrics = vendor.evaluate_thurstonian_model(
            holdout_graph, utilities, holdout)
        for (t, o) in prompts.CELLS:
            cell = f"t{t}.{o}"
            pA_h = {(a, b): cell_pA(raw, t, o, a, b)
                    for (a, b) in holdout if (t, o, a, b) in raw}
            cell_holdout[cell] = vendor.evaluate_thurstonian_model(
                ShimGraph(options, pA_h), cell_fits[cell][0], holdout)
        rng = random.Random(cfg.seed)
        vals = [utilities[oid] for oid in ids]
        accs = []
        for _ in range(100):
            perm = list(vals)
            rng.shuffle(perm)
            accs.append(vendor.evaluate_thurstonian_model(
                holdout_graph, dict(zip(ids, perm)), holdout)["accuracy"])
        random_baseline = float(np.mean(accs))
        logger.info("%s holdout: log_loss=%.4f acc=%.3f (random baseline %.3f)",
                    exp, pooled_holdout_metrics["log_loss"],
                    pooled_holdout_metrics["accuracy"], random_baseline)

    # 8. Payloads.
    bias = ordering_stats(store.load_raw(exp))
    unparseable_rate = bias["unparseable_rate"]
    config = asdict(cfg)
    timing = {"elicit_s": state["elicit_s"], "fit_s": state["fit_s"],
              "total_s": time.time() - t_start}
    prompts_per_s = (state["n_prompts"] / state["elicit_s"]
                     if state["elicit_s"] > 0 else None)

    pooled_payload = {
        "model": spec.key, "exp": exp, "cell": "pooled", "options": options,
        "utilities": {str(k): v for k, v in utilities.items()},
        "metrics": metrics,
        "holdout_metrics": pooled_holdout_metrics,
        "random_baseline_accuracy": random_baseline,
        "ordering_bias": bias,
        "unparseable_rate": unparseable_rate,
        "n_training_edges": len(collected),
        "n_pseudolabels": n_pseudolabels,
        "config": config, "timing": timing,
        "extra": {"n_prompts": state["n_prompts"], "prompts_per_s": prompts_per_s,
                  "target_total_edges": target_total, "n_holdout_edges": len(holdout)},
    }
    store.save_fit(exp, "pooled", pooled_payload)

    for cell, (utils_c, metrics_c) in cell_fits.items():
        store.save_fit(exp, cell, {
            "model": spec.key, "exp": exp, "cell": cell, "options": options,
            "utilities": {str(k): v for k, v in utils_c.items()},
            "metrics": metrics_c,
            "holdout_metrics": cell_holdout.get(cell),
            "unparseable_rate": unparseable_rate,
            "n_training_edges": len(collected),
            "config": config, "timing": timing, "extra": {},
        })

    return pooled_payload
