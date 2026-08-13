"""JSONL/JSON persistence with resumability.

Layout under {results_dir}:
  raw/{model_key}/{exp}.jsonl           one row per elicited cell
  raw/{model_key}/{exp}.batches.jsonl   active-learning batch log (replayable)
  fits/{model_key}/{exp}.{cell}.json    fit payloads (cell: pooled | t{t}.{o} | meta)
  summary/{model_key}.json              read-merge-write per-model summary

Raw row: {"exp", "t", "o", "i", "j", "p_a", "ts"} with i<j canonical; p_a is
P(first-presented option) — first-presented is i for o="orig", j for o="rev".
Appends are fsync'd so a killed pod loses at most the in-flight batch.
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

RawKey = Tuple[int, str, int, int]  # (t, o, i, j)


def _json_default(obj):
    # numpy scalars sneak in via torch/np metrics; never fail a save on them.
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


class ResultsStore:
    def __init__(self, results_dir: str, model_key: str):
        self.results_dir = results_dir
        self.model_key = model_key
        self.raw_dir = os.path.join(results_dir, "raw", model_key)
        self.fits_dir = os.path.join(results_dir, "fits", model_key)
        self.summary_dir = os.path.join(results_dir, "summary")
        for d in (self.raw_dir, self.fits_dir, self.summary_dir):
            os.makedirs(d, exist_ok=True)
        self._raw_cache: Dict[str, Dict[RawKey, Optional[float]]] = {}

    # --- paths ---

    def _raw_path(self, exp: str) -> str:
        return os.path.join(self.raw_dir, f"{exp}.jsonl")

    def _batches_path(self, exp: str) -> str:
        return os.path.join(self.raw_dir, f"{exp}.batches.jsonl")

    def _fit_path(self, exp: str, cell: str) -> str:
        return os.path.join(self.fits_dir, f"{exp}.{cell}.json")

    def _summary_path(self) -> str:
        return os.path.join(self.summary_dir, f"{self.model_key}.json")

    # --- raw rows ---

    def load_raw(self, exp: str) -> Dict[RawKey, Optional[float]]:
        """(t, o, i, j) -> p_a for every persisted cell; cached so callers can
        consult it cheaply before eliciting (this cache IS the resume logic)."""
        if exp not in self._raw_cache:
            data: Dict[RawKey, Optional[float]] = {}
            path = self._raw_path(exp)
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        data[(row["t"], row["o"], row["i"], row["j"])] = row["p_a"]
            self._raw_cache[exp] = data
        return self._raw_cache[exp]

    def has(self, exp: str, t: int, o: str, i: int, j: int) -> bool:
        return (t, o, i, j) in self.load_raw(exp)

    def get(self, exp: str, t: int, o: str, i: int, j: int) -> Optional[float]:
        return self.load_raw(exp).get((t, o, i, j))

    def put_many(self, rows: Sequence[Dict[str, Any]]) -> None:
        """Append rows (grouped by their 'exp') and update the cache; fsync per file."""
        by_exp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_exp[row["exp"]].append(row)
        for exp, exp_rows in by_exp.items():
            cache = self.load_raw(exp)
            with open(self._raw_path(exp), "a") as f:
                for row in exp_rows:
                    f.write(json.dumps(row) + "\n")
                    cache[(row["t"], row["o"], row["i"], row["j"])] = row["p_a"]
                f.flush()
                os.fsync(f.fileno())

    # --- batch log ---

    def append_batch(self, exp: str, iter_idx: int, edges: Sequence[Tuple[int, int]]) -> None:
        with open(self._batches_path(exp), "a") as f:
            f.write(json.dumps({"iter": iter_idx,
                                "edges": [[int(i), int(j)] for i, j in edges]}) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def load_batches(self, exp: str) -> List[List[Tuple[int, int]]]:
        path = self._batches_path(exp)
        if not os.path.exists(path):
            return []
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        rows.sort(key=lambda r: r["iter"])
        return [[tuple(e) for e in r["edges"]] for r in rows]

    # --- fits ---

    def save_fit(self, exp: str, cell: str, payload: Dict[str, Any]) -> None:
        path = self._fit_path(exp, cell)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, default=_json_default)
        os.replace(tmp, path)

    def load_fit(self, exp: str, cell: str) -> Optional[Dict[str, Any]]:
        path = self._fit_path(exp, cell)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    # --- summary ---

    def save_summary(self, patch: Dict[str, Any]) -> None:
        """Read-merge-write; dict values merge one level deep so per-experiment
        patches don't clobber siblings."""
        path = self._summary_path()
        data: Dict[str, Any] = {}
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key] = {**data[key], **value}
            else:
                data[key] = value
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)
        os.replace(tmp, path)
