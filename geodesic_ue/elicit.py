"""Edge elicitation over the 8-cell design (4 templates x 2 orderings) and the
pooling/ordering-bias arithmetic downstream of the raw p_a rows.

Pooling matches UE's process_choice_probs with unparseable_mode="distribution":
each cell contributes weight 1, so pooled P(i>j) is the mean of per-cell
P(i>j) with None treated as 0.5.
"""

import logging
import time
from typing import Dict, Iterable, Optional, Sequence, Tuple

from . import prompts

RawDict = Dict[Tuple[int, str, int, int], Optional[float]]


def elicit_edges(agent, spec, store, exp: str, question: str,
                 options_by_id: Dict[int, dict],
                 edges: Iterable[Tuple[int, int]],
                 templates: Sequence[int] = (0, 1, 2, 3),
                 orderings: Sequence[str] = ("orig", "rev"),
                 logger: Optional[logging.Logger] = None) -> dict:
    """Score every missing (t, o, i, j) cell for the given edges in one
    score_choice call; persist rows; return throughput stats."""
    logger = logger or logging.getLogger("geodesic_ue")
    raw = store.load_raw(exp)

    todo = []
    seen = set()
    n_cached = 0
    for e in edges:
        i, j = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
        for t in templates:
            for o in orderings:
                key = (t, o, i, j)
                if key in seen:
                    continue
                seen.add(key)
                if key in raw:
                    n_cached += 1
                else:
                    todo.append(key)

    prompt_list = []
    for (t, o, i, j) in todo:
        first, second = (i, j) if o == "orig" else (j, i)
        prompt_list.append(prompts.build_prompt(
            spec.is_chat, agent.tokenizer, question,
            options_by_id[first]["description"],
            options_by_id[second]["description"], t))

    t0 = time.time()
    scores = agent.score_choice(prompt_list) if prompt_list else []
    seconds = time.time() - t0

    ts = time.time()
    rows = [{"exp": exp, "t": t, "o": o, "i": i, "j": j, "p_a": p, "ts": ts}
            for (t, o, i, j), p in zip(todo, scores)]
    if rows:
        store.put_many(rows)

    unparseable = sum(1 for p in scores if p is None)
    if todo:
        logger.info("elicit %s: %d prompts in %.1fs (%.1f prompts/s), %d cached, %d unparseable",
                    exp, len(todo), seconds, len(todo) / max(seconds, 1e-9),
                    n_cached, unparseable)
    return {"n_new": len(todo), "n_cached": n_cached,
            "seconds": seconds, "unparseable": unparseable}


def cell_pA(raw: RawDict, t: int, o: str, i: int, j: int,
            unparseable: float = 0.5) -> float:
    """P(i preferred over j) in one cell; None (unparseable) -> the
    UE "distribution" value before the ordering flip."""
    p_a = raw.get((t, o, i, j))
    if p_a is None:
        p_a = unparseable
    return p_a if o == "orig" else 1.0 - p_a


def pooled_pA(raw: RawDict, i: int, j: int,
              templates: Sequence[int], orderings: Sequence[str]) -> Optional[float]:
    """Mean of cell_pA over cells present; missing cells are skipped entirely."""
    vals = []
    for t in templates:
        for o in orderings:
            if (t, o, i, j) in raw:
                vals.append(cell_pA(raw, t, o, i, j))
    return sum(vals) / len(vals) if vals else None


def ordering_stats(raw: RawDict) -> dict:
    """frac_first = mean probability mass on the first-displayed option — the
    ordering-bias number (downstream flags values outside 0.40-0.60)."""
    n = len(raw)
    if n == 0:
        return {"frac_first": None, "frac_A_orig": None, "frac_A_rev": None,
                "n": 0, "unparseable_rate": 0.0}

    def _mean(keys):
        vals = [0.5 if raw[k] is None else raw[k] for k in keys]
        return sum(vals) / len(vals) if vals else None

    all_keys = list(raw)
    orig_keys = [k for k in all_keys if k[1] == "orig"]
    rev_keys = [k for k in all_keys if k[1] == "rev"]
    n_unparseable = sum(1 for k in all_keys if raw[k] is None)
    return {
        "frac_first": _mean(all_keys),
        "frac_A_orig": _mean(orig_keys),
        "frac_A_rev": _mean(rev_keys),
        "n": n,
        "unparseable_rate": n_unparseable / n,
    }
