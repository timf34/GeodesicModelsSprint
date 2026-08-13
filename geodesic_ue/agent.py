"""Choice agents: P(answer letter = A) at the position after a rendered prompt.

All backends share one token-classification rule, mirroring the vendored
vLLMAgent.choice_probs (emergent-values llm_agent.py) with one extra guard:
after stripping quotes/brackets/punctuation, a token counts as "A" only if it
IS "A" or starts with "A" followed by a non-alphanumeric character — so
"Answer"-style continuations never pollute the A mass.

Heavy deps (vllm / torch / transformers) are imported lazily inside the
constructors so the mock path stays importable on dep-free machines.
"""

import hashlib
import logging
import math
from typing import Callable, List, Optional

TOP_LOGPROBS = 20        # matches UE's max_logprobs default
VLLM_CHUNK = 8192        # keep vLLM request batches bounded

_STRIP_CHARS = "\"'()[].,:;"


def _classify_choice_token(text: str) -> Optional[str]:
    """'A' / 'B' / None for one decoded top-logprob token."""
    s = text.strip().strip(_STRIP_CHARS).strip()
    for letter in ("A", "B"):
        if s == letter:
            return letter
        # "A)" survives stripping as "A"; "A-" etc. count; "Answer" must not.
        if s.startswith(letter) and len(s) > 1 and not s[1].isalnum():
            return letter
    return None


def _mass_to_p(a_mass: float, b_mass: float) -> Optional[float]:
    denom = a_mass + b_mass
    return a_mass / denom if denom > 0 else None


class BaseChoiceAgent:
    """Common interface: a tokenizer (with .apply_chat_template) + score_choice."""

    tokenizer = None

    def score_choice(self, prompts: List[str]) -> List[Optional[float]]:
        """P(letter A) at the next position for each prompt; None if neither
        A nor B appears in the top logprobs."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class VLLMChoiceAgent(BaseChoiceAgent):
    """vLLM engine backend (the production path on pods)."""

    def __init__(self, repo_id: str, revision: Optional[str] = None,
                 gpu_memory_utilization: float = 0.92, max_model_len: int = 2048,
                 dtype: str = "bfloat16", chat_template: Optional[str] = None):
        from vllm import LLM, SamplingParams  # lazy: only pods have vllm
        from transformers import AutoTokenizer
        self.repo_id = repo_id
        self.llm = LLM(
            model=repo_id,
            revision=revision,
            tokenizer_revision=revision,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            disable_log_stats=True,
        )
        # Separate tokenizer instance so prompts.py renders chat templates
        # without reaching into engine internals.
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
        if chat_template is not None:
            self.tokenizer.chat_template = chat_template
        self._sampling_params = SamplingParams(
            temperature=1.0, max_tokens=1, logprobs=TOP_LOGPROBS)

    def score_choice(self, prompts: List[str]) -> List[Optional[float]]:
        results: List[Optional[float]] = []
        for start in range(0, len(prompts), VLLM_CHUNK):
            chunk = prompts[start:start + VLLM_CHUNK]
            # Pre-tokenize with add_special_tokens=False: rendered templates
            # already carry their own BOS, and letting the engine's tokenizer
            # prepend another (Llama-style) shifts the scoring position.
            token_prompts = [
                {"prompt_token_ids": self.tokenizer(p, add_special_tokens=False)["input_ids"]}
                for p in chunk
            ]
            outputs = self.llm.generate(token_prompts, self._sampling_params)
            for out in outputs:
                try:
                    first_step = out.outputs[0].logprobs[0]
                except (IndexError, AttributeError, TypeError):
                    results.append(None)
                    continue
                a_mass = 0.0
                b_mass = 0.0
                for _tid, lp in first_step.items():
                    tok = getattr(lp, "decoded_token", None)
                    if tok is None:
                        continue
                    letter = _classify_choice_token(tok)
                    if letter == "A":
                        a_mass += math.exp(lp.logprob)
                    elif letter == "B":
                        b_mass += math.exp(lp.logprob)
                results.append(_mass_to_p(a_mass, b_mass))
        return results

    def close(self) -> None:
        if hasattr(self, "llm"):
            del self.llm
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class HFChoiceAgent(BaseChoiceAgent):
    """Plain transformers backend (fallback when vllm is unavailable)."""

    def __init__(self, repo_id: str, revision: Optional[str] = None,
                 batch_size: int = 128, max_length: int = 2048,
                 chat_template: Optional[str] = None):
        import torch  # lazy
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            repo_id, revision=revision, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        # Left padding so the last position is the real next-token position.
        self.tokenizer = AutoTokenizer.from_pretrained(
            repo_id, revision=revision, padding_side="left")
        if chat_template is not None:
            self.tokenizer.chat_template = chat_template
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def score_choice(self, prompts: List[str]) -> List[Optional[float]]:
        torch = self._torch
        n = len(prompts)
        results: List[Optional[float]] = [None] * n
        # Length-sorted processing minimizes padding waste.
        lengths = [len(self.tokenizer(p, add_special_tokens=False)["input_ids"])
                   for p in prompts]
        order = sorted(range(n), key=lambda k: lengths[k])
        bs = self.batch_size
        pos = 0
        while pos < len(order):
            batch_idx = order[pos:pos + bs]
            batch = [prompts[k] for k in batch_idx]
            try:
                enc = self.tokenizer(batch, return_tensors="pt", padding=True,
                                     truncation=True, max_length=self.max_length,
                                     add_special_tokens=False)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                with torch.no_grad():
                    logits = self.model(**enc).logits[:, -1, :]
                logprobs = torch.log_softmax(logits.float(), dim=-1)
                top_vals, top_idxs = logprobs.topk(TOP_LOGPROBS, dim=-1)
            except torch.cuda.OutOfMemoryError:
                if bs <= 1:
                    raise
                bs = max(1, bs // 2)
                torch.cuda.empty_cache()
                continue
            for row, k in enumerate(batch_idx):
                a_mass = 0.0
                b_mass = 0.0
                for lp, tid in zip(top_vals[row].tolist(), top_idxs[row].tolist()):
                    letter = _classify_choice_token(self.tokenizer.decode([tid]))
                    if letter == "A":
                        a_mass += math.exp(lp)
                    elif letter == "B":
                        b_mass += math.exp(lp)
                results[k] = _mass_to_p(a_mass, b_mass)
            pos += bs
        return results

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        try:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:
            pass


class FakeTokenizer:
    """Renders the Geodesic chat layout exactly (matches chat_template.jinja)."""

    def apply_chat_template(self, messages, tokenize: bool = False,
                            add_generation_prompt: bool = True) -> str:
        system = ""
        user = ""
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            elif m["role"] == "user":
                user = m["content"]
        return f"<|endoftext|><|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"


class MockChoiceAgent(BaseChoiceAgent):
    """Deterministic offline agent for smoke tests: sha256(prompt) -> [0.02, 0.98]."""

    def __init__(self, seed: int = 0,
                 score_fn: Optional[Callable[[str], Optional[float]]] = None,
                 mode: str = "hash"):
        self.tokenizer = FakeTokenizer()
        self.seed = seed
        self.score_fn = score_fn
        self.mode = mode  # "hash" | "latent" — both hash-based; kept for CLI symmetry

    def score_choice(self, prompts: List[str]) -> List[Optional[float]]:
        out: List[Optional[float]] = []
        for p in prompts:
            if self.score_fn is not None:
                out.append(self.score_fn(p))
                continue
            digest = hashlib.sha256(f"{self.seed}:{p}".encode("utf-8")).digest()
            v = int.from_bytes(digest[:8], "big") / float(1 << 64)
            out.append(0.02 + 0.96 * v)
        return out


def make_agent(kind: str, spec, logger: Optional[logging.Logger] = None, **kwargs) -> BaseChoiceAgent:
    """Factory over registry.ModelSpec. kind in {'vllm', 'hf', 'mock'}."""
    logger = logger or logging.getLogger("geodesic_ue")
    if kind == "vllm":
        return VLLMChoiceAgent(
            spec.repo_id, revision=spec.revision,
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.92),
            max_model_len=kwargs.get("max_model_len", 2048),
            dtype=kwargs.get("dtype", "bfloat16"),
            chat_template=spec.chat_template_override)
    if kind == "hf":
        return HFChoiceAgent(
            spec.repo_id, revision=spec.revision,
            batch_size=kwargs.get("batch_size", 128),
            max_length=kwargs.get("max_model_len", 2048),
            chat_template=spec.chat_template_override)
    if kind == "mock":
        return MockChoiceAgent(seed=kwargs.get("seed", 0))
    raise ValueError(f"Unknown agent kind: {kind!r} (expected vllm|hf|mock)")
