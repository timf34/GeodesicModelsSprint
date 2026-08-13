"""Experiment 6: utility convergence across the model suite.

Cosine similarity per UE: L2-normalize the raw pooled base-utility mean
vectors (no centering), S = U U^T. Matrices are produced for all models and
per tier, each under three row orderings (axis groups, stage groups,
hierarchical clustering), plus a within/between-group summary.
"""

import logging
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage

from . import io, tables

logger = logging.getLogger(__name__)

SCOPES: List[Tuple[str, Tuple[str, ...]]] = [
    ("all", ("instruct", "dpo")),
    ("instruct", ("instruct",)),
    ("dpo", ("dpo",)),
]

AXIS_GROUP_ORDER = ["baseline", "alignment", "misalignment"]


def _load_vectors(results_dir: str) -> Tuple[List[str], Optional[np.ndarray]]:
    """Pooled base-utility vectors for models whose option sets match.

    The option sets are meant to be identical across models; models whose
    descriptions mismatch the modal set are excluded with a warning.
    """
    vecs: Dict[str, np.ndarray] = {}
    sigs: Dict[str, tuple] = {}
    for model in io.list_models(results_dir):
        payload = io.load_fit(results_dir, model, "base_utilities")
        if payload is None:
            logger.warning("convergence: no pooled base_utilities for %s; skipped", model)
            continue
        vec = io.utilities_vector(payload)
        if vec.size == 0 or not np.all(np.isfinite(vec)):
            logger.warning("convergence: incomplete utility vector for %s; skipped", model)
            continue
        opts = sorted(payload.get("options") or [], key=lambda o: int(o.get("id", 0)))
        vecs[model] = vec
        sigs[model] = tuple(o.get("description") for o in opts)
    if not vecs:
        return [], None
    modal_sig, _ = Counter(sigs.values()).most_common(1)[0]
    keys = []
    for m in sorted(vecs):
        if sigs[m] != modal_sig:
            logger.warning("convergence: option descriptions of %s mismatch the "
                           "modal option set; excluded from similarity", m)
        else:
            keys.append(m)
    return keys, np.vstack([vecs[m] for m in keys])


def _axis_group(key: str) -> Optional[str]:
    spec = io.get_spec(key)
    if spec is None or spec.family != "sfm":
        return None  # references / unknowns stay out of the group summary
    return "baseline" if spec.axis is None else spec.axis


def compute(results_dir: str) -> Optional[dict]:
    """{"matrix": full cosine DataFrame, "scopes": {scope: orderings}} or None."""
    keys, U = _load_vectors(results_dir)
    if not keys or U is None or len(keys) < 2:
        logger.warning("convergence: fewer than 2 comparable models; nothing to do")
        return None
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    S = np.clip((U / norms) @ (U / norms).T, -1.0, 1.0)
    Sdf = pd.DataFrame(S, index=keys, columns=keys)

    scopes: Dict[str, dict] = {}
    for scope, tiers in SCOPES:
        if scope == "all":
            skeys = list(keys)
        else:
            skeys = [k for k in keys
                     if io.get_spec(k) is not None and io.get_spec(k).tier == scope]
        if len(skeys) < 2:
            logger.info("convergence: scope %r has <2 models; skipped", scope)
            continue
        orderings: Dict[str, dict] = {}
        for grouping in ("axis", "stage"):
            ordered, groups = tables.grouped_model_order(
                skeys, grouping=grouping, tiers=tiers)
            orderings[grouping] = {
                "keys": ordered,
                "groups": [(label, len(ks)) for label, ks in groups],
            }
        try:
            sub = np.vstack([U[keys.index(k)] for k in skeys])
            order = leaves_list(linkage(sub, method="average", metric="cosine"))
            clustered = [skeys[i] for i in order]
            orderings["clustered"] = {
                "keys": clustered,
                "groups": [("clustered", len(clustered))],
            }
        except Exception:
            logger.warning("convergence: clustering failed for scope %r", scope,
                           exc_info=True)
        scopes[scope] = {"keys": skeys, "orderings": orderings}
    return {"matrix": Sdf, "scopes": scopes}


def build_summary(structure: dict) -> pd.DataFrame:
    """Mean within- vs between-group cosine for the axis groups, per scope."""
    S = structure["matrix"]
    rows = []
    for scope, sdata in structure["scopes"].items():
        members: Dict[str, List[str]] = {}
        for k in sdata["keys"]:
            g = _axis_group(k)
            if g is not None:
                members.setdefault(g, []).append(k)
        present = [g for g in AXIS_GROUP_ORDER if g in members]
        for ia, ga in enumerate(present):
            for gb in present[ia:]:
                if ga == gb:
                    ks = members[ga]
                    pairs = [(ks[i], ks[j]) for i in range(len(ks))
                             for j in range(i + 1, len(ks))]
                else:
                    pairs = [(a, b) for a in members[ga] for b in members[gb]]
                if not pairs:
                    continue
                vals = [float(S.at[a, b]) for a, b in pairs]
                rows.append({
                    "scope": scope, "group_a": ga, "group_b": gb,
                    "within": ga == gb,
                    "mean_cosine": float(np.mean(vals)),
                    "n_pairs": len(vals),
                })
    return pd.DataFrame(rows)


def run(results_dir: str) -> Optional[dict]:
    structure = compute(results_dir)
    if structure is None:
        return None
    adir = io.analysis_dir(results_dir)
    S = structure["matrix"]
    for scope, sdata in structure["scopes"].items():
        for ordering, od in sdata["orderings"].items():
            M = S.reindex(index=od["keys"], columns=od["keys"])
            path = os.path.join(adir, f"convergence_cosine_{scope}_{ordering}.csv")
            M.to_csv(path)
            logger.info("wrote %s", path)
    summary = build_summary(structure)
    summary.to_csv(os.path.join(adir, "convergence_summary.csv"), index=False)
    logger.info("wrote convergence_summary.csv (%d rows)", len(summary))
    return {"structure": structure, "summary": summary}
