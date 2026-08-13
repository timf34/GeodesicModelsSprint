"""Model registry: the Geodesic alignment-pretraining suite + reference anchors.

Revisions are the main-branch commits observed on 2026-08-12; pinned so a rerun
fits the same weights. All sfm models are 6.86B GPT-NeoX. Llama-2 repos are
GATED on HF — the token in .env must have approved access.
"""

from dataclasses import dataclass
from typing import List, Optional

from .prompts import LLAMA2_CHAT_TEMPLATE


@dataclass(frozen=True)
class ModelSpec:
    key: str                 # short unique key used in file paths / CLI
    repo_id: str
    revision: Optional[str]
    tier: str                # 'base' | 'instruct' | 'dpo'
    is_chat: bool            # False -> raw-completion Geodesic base prompts
    family: str              # 'sfm' | 'reference'
    filtering: Optional[str] = None   # 'unfiltered' | 'filtered'  (sfm only)
    axis: Optional[str] = None        # 'alignment' | 'misalignment' | None (baselines)
    stage: Optional[str] = None       # 'e2e' | 'midtrain' | 'cpt' | None (baselines)
    # Set when the repo ships no usable chat template (agents assign it to the
    # tokenizer after load; predownload skips its template check).
    chat_template_override: Optional[str] = None

    @property
    def condition(self) -> str:
        if self.family != "sfm":
            return self.key
        if self.axis is None:
            return f"baseline_{self.filtering}"
        return f"{self.filtering}_{self.stage}_{self.axis}"


_SFM_REVISIONS = {
    # (condition, tier) -> revision
    ("baseline_unfiltered", "base"): "51a98274",
    ("baseline_unfiltered", "instruct"): "4c769eae",
    ("baseline_unfiltered", "dpo"): "2d9bb0a1",
    ("baseline_filtered", "base"): "ab1893a2",
    ("baseline_filtered", "instruct"): "3b05e97a",
    ("baseline_filtered", "dpo"): "f135f51b",
    ("filtered_e2e_alignment", "base"): "ef8b31f3",
    ("filtered_e2e_alignment", "instruct"): "b661535d",
    ("filtered_e2e_alignment", "dpo"): "72788bad",
    ("filtered_midtrain_alignment", "base"): "6ce7ae31",
    ("filtered_midtrain_alignment", "instruct"): "086caa60",
    ("filtered_midtrain_alignment", "dpo"): "860fecff",
    ("filtered_cpt_alignment", "base"): "9e6e22b4",
    ("filtered_cpt_alignment", "instruct"): "d78258bc",
    ("filtered_cpt_alignment", "dpo"): "0b8f9849",
    ("unfiltered_e2e_alignment", "base"): "beff5891",
    ("unfiltered_e2e_alignment", "instruct"): "ffc9038c",
    ("unfiltered_e2e_alignment", "dpo"): "439bcef6",
    ("unfiltered_midtrain_alignment", "base"): "c872c179",
    ("unfiltered_midtrain_alignment", "instruct"): "694e9865",
    ("unfiltered_midtrain_alignment", "dpo"): "b1925c92",
    ("unfiltered_cpt_alignment", "base"): "e5f8003c",
    ("unfiltered_cpt_alignment", "instruct"): "8c57dc9d",
    ("unfiltered_cpt_alignment", "dpo"): "520f1688",
    ("unfiltered_e2e_misalignment", "base"): "788ebc37",
    ("unfiltered_e2e_misalignment", "instruct"): "3df8876d",
    ("unfiltered_e2e_misalignment", "dpo"): "23ad1d4d",
    ("unfiltered_midtrain_misalignment", "base"): "02471d70",
    ("unfiltered_midtrain_misalignment", "instruct"): "1210ec10",
    ("unfiltered_midtrain_misalignment", "dpo"): "daba0217",
    ("unfiltered_cpt_misalignment", "base"): "93e96594",
    ("unfiltered_cpt_misalignment", "instruct"): "4fcdb660",
    ("unfiltered_cpt_misalignment", "dpo"): "1e2705f4",
}

_CONDITIONS = [
    # (condition, filtering, axis, stage)
    ("baseline_unfiltered", "unfiltered", None, None),
    ("baseline_filtered", "filtered", None, None),
    ("filtered_e2e_alignment", "filtered", "alignment", "e2e"),
    ("filtered_midtrain_alignment", "filtered", "alignment", "midtrain"),
    ("filtered_cpt_alignment", "filtered", "alignment", "cpt"),
    ("unfiltered_e2e_alignment", "unfiltered", "alignment", "e2e"),
    ("unfiltered_midtrain_alignment", "unfiltered", "alignment", "midtrain"),
    ("unfiltered_cpt_alignment", "unfiltered", "alignment", "cpt"),
    ("unfiltered_e2e_misalignment", "unfiltered", "misalignment", "e2e"),
    ("unfiltered_midtrain_misalignment", "unfiltered", "misalignment", "midtrain"),
    ("unfiltered_cpt_misalignment", "unfiltered", "misalignment", "cpt"),
]


def _sfm_repo(condition: str, filtering: str, axis, stage, tier: str) -> str:
    if axis is None:
        return f"geodesic-research/sfm_{condition}_{tier}"
    return f"geodesic-research/sfm_{filtering}_{stage}_{axis}_upsampled_{tier}"


def _build_specs() -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for condition, filtering, axis, stage in _CONDITIONS:
        for tier in ("base", "instruct", "dpo"):
            specs.append(ModelSpec(
                key=f"{condition}_{tier}",
                repo_id=_sfm_repo(condition, filtering, axis, stage, tier),
                revision=_SFM_REVISIONS[(condition, tier)],
                tier=tier,
                is_chat=(tier != "base"),
                family="sfm",
                filtering=filtering,
                axis=axis,
                stage=stage,
            ))
    specs += [
        # Ungated byte-identical mirror (meta-llama/Llama-2-7b-chat-hf is gated
        # and this HF token lacks access); the mirror predates embedded chat
        # templates, so the canonical Llama-2 template is supplied explicitly.
        ModelSpec("llama2_7b_chat", "NousResearch/Llama-2-7b-chat-hf", "351844e7",
                  "instruct", True, "reference",
                  chat_template_override=LLAMA2_CHAT_TEMPLATE),
        ModelSpec("olmo2_7b_instruct", "allenai/OLMo-2-1124-7B-Instruct", "470b1fba",
                  "instruct", True, "reference"),
        ModelSpec("olmo3_7b_instruct", "allenai/Olmo-3-7B-Instruct", "6e5971d9",
                  "instruct", True, "reference"),
    ]
    return specs


ALL_SPECS: List[ModelSpec] = _build_specs()
SPECS_BY_KEY = {s.key: s for s in ALL_SPECS}

# The approved run: instruct + dpo tiers of all 11 conditions, plus references.
DEFAULT_RUN_KEYS: List[str] = (
    [f"{c}_{t}" for c, _, _, _ in _CONDITIONS for t in ("instruct", "dpo")]
    + ["llama2_7b_chat", "olmo2_7b_instruct", "olmo3_7b_instruct"]
)

# Balanced two-pod split (13 / 12 models). Pod A gets the DPO tier (Geodesic's
# recommended comparison set) so it alone yields a publishable sweep if pod B dies.
POD_A_KEYS = [f"{c}_dpo" for c, _, _, _ in _CONDITIONS] + ["llama2_7b_chat", "olmo3_7b_instruct"]
POD_B_KEYS = [f"{c}_instruct" for c, _, _, _ in _CONDITIONS] + ["olmo2_7b_instruct"]

# E2E/Mid/CPT triples that share filtering+axis — the substitute for
# within-condition seed variance (direction-consistency reporting).
STAGE_TRIPLES = {
    ("filtered", "alignment"): ["filtered_e2e_alignment", "filtered_midtrain_alignment", "filtered_cpt_alignment"],
    ("unfiltered", "alignment"): ["unfiltered_e2e_alignment", "unfiltered_midtrain_alignment", "unfiltered_cpt_alignment"],
    ("unfiltered", "misalignment"): ["unfiltered_e2e_misalignment", "unfiltered_midtrain_misalignment", "unfiltered_cpt_misalignment"],
}


def resolve_keys(arg: str) -> List[str]:
    """CLI helper: 'all' -> DEFAULT_RUN_KEYS; 'pod_a'/'pod_b' -> split lists;
    'tier:dpo' -> that tier; else comma-separated explicit keys."""
    if arg == "all":
        return list(DEFAULT_RUN_KEYS)
    if arg == "pod_a":
        return list(POD_A_KEYS)
    if arg == "pod_b":
        return list(POD_B_KEYS)
    if arg.startswith("tier:"):
        tier = arg.split(":", 1)[1]
        keys = [s.key for s in ALL_SPECS if s.tier == tier]
        if not keys:
            raise KeyError(f"No models in tier {tier!r} (tiers: base, instruct, dpo)")
        return keys
    keys = [k.strip() for k in arg.split(",") if k.strip()]
    unknown = [k for k in keys if k not in SPECS_BY_KEY]
    if unknown:
        raise KeyError(f"Unknown model keys: {unknown}. Known: {sorted(SPECS_BY_KEY)}")
    return keys
