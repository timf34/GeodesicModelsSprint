#!/usr/bin/env bash
# One-shot overnight runner for the Geodesic utility-engineering sweep on a rented GPU pod.
# Sibling of AttractorBench/run_sfm_on_pod.sh — same operational conventions, different shape:
# here geodesic_ue.runner owns the model loop (one subprocess per model, one model in GPU
# memory at a time, fails loudly per model and continues), so this script only has to get the
# environment right: deps -> predownload EVERYTHING -> run -> analyze -> save -> shutdown.
#
# Recommended pod: 1x A100/H100/H200 80GB. Volume: 500GB for the full 25-model sweep,
# ~200GB for one side of the two-pod split. Driver must report CUDA >= 12.8 (vllm needs it).
#
# Usage:
#   git clone <repo-url> /workspace/GeodesicModelsSprint && cd /workspace/GeodesicModelsSprint
#   # create .env with HF_TOKEN=... (Llama-2 repos are GATED) before running
#   bash run_on_pod.sh                       # all 25 models
#
# Two-pod split (both halves in parallel on two pods):
#   POD=a bash run_on_pod.sh                 # dpo tier + llama2_7b_chat + olmo3 (13 models)
#   POD=b bash run_on_pod.sh                 # instruct tier + olmo2 (12 models)
#
# Smoke test FIRST (strongly recommended before burning a night): two models, --smoke,
# results_smoke/, analysis included, plus a per-phase wall-clock summary to extrapolate from:
#   SMOKE=1 bash run_on_pod.sh
#
# Unattended overnight run (results pushed to git, pod stops itself when done):
#   runpodctl config --apiKey <RUNPOD_KEY>   # one-time, enables self-shutdown
#   git remote set-url origin https://<PAT>@github.com/timf34/GeodesicModelsSprint.git
#   POD=a SAVE_TO_GIT=1 SHUTDOWN=stop nohup bash run_on_pod.sh > overnight_a.log 2>&1 &
#   tail -f overnight_a.log
# SHUTDOWN=stop pauses the pod (GPU billing stops, disk kept). SHUTDOWN=terminate destroys the
# pod AND its disk — only safe with SAVE_TO_GIT=1, and a failed push auto-downgrades it to stop.
#
# Interrupted? Rerun the exact same command: results are JSONL keyed work units, the runner
# skips everything already on disk and resumes where it stopped.
set -euo pipefail

# Everything below uses repo-relative paths; make them valid regardless of where the
# script was invoked from (nohup from $HOME is a classic way to lose a night to ENOENT).
cd "$(dirname "${BASH_SOURCE[0]}")"

# ---- knobs (all overridable via env) -------------------------------------------------
case "${POD:-}" in
  a|A) MODELS_DEFAULT="pod_a" ;;
  b|B) MODELS_DEFAULT="pod_b" ;;
  "")  MODELS_DEFAULT="all" ;;
  *)   echo "!! POD must be 'a' or 'b' (got '${POD}')"; exit 1 ;;
esac
MODELS="${MODELS:-$MODELS_DEFAULT}"
EXPERIMENTS="${EXPERIMENTS:-}"       # empty -> runner default (all experiments)
RESULTS_DIR="${RESULTS_DIR:-results}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
VENV="${VENV:-0}"
SMOKE="${SMOKE:-0}"
SAVE_TO_GIT="${SAVE_TO_GIT:-0}"
SHUTDOWN="${SHUTDOWN:-}"
SMOKE_MODELS="${SMOKE_MODELS:-unfiltered_e2e_alignment_dpo,baseline_unfiltered_instruct}"

SMOKE_FLAG=""
if [ "$SMOKE" = "1" ]; then
  # Smoke runs are attended dress rehearsals: their own results dir (so a later real run
  # starts from a clean resume state), and never push/shutdown no matter what was exported.
  MODELS="$SMOKE_MODELS"
  RESULTS_DIR="results_smoke"
  SMOKE_FLAG="--smoke"
  SAVE_TO_GIT=0
  SHUTDOWN=""
  echo "SMOKE=1: MODELS=$MODELS RESULTS_DIR=$RESULTS_DIR (SAVE_TO_GIT/SHUTDOWN disabled)"
fi

T_START=$SECONDS

echo "== [0/4] environment =="
# Cache weights on the persistent volume when there is one (RunPod mounts it at /workspace) —
# the root/container disk cannot hold 200-350GB of checkpoints.
if [ -z "${HF_HOME:-}" ] && [ -d /workspace ]; then
  export HF_HOME=/workspace/hf
fi
echo "  HF cache: ${HF_HOME:-~/.cache/huggingface}"

# Git must NEVER prompt interactively: an overnight run once hung all night on a credential
# prompt at the final push. With this set, git fails instead of asking.
export GIT_TERMINAL_PROMPT=0

# RunPod wipes everything OUTSIDE /workspace when a pod stops (container disk is ephemeral;
# only the volume survives). Refuse to burn hours writing results to a doomed disk.
if [ -n "${RUNPOD_POD_ID:-}" ] && [ -d /workspace ]; then
  case "$PWD" in
    /workspace/*) ;;
    *)
      echo "!! this checkout is at $PWD — on RunPod, only /workspace survives a pod stop."
      echo "   Move the repo there and rerun:"
      echo "     git clone <repo-url> /workspace/GeodesicModelsSprint && cd /workspace/GeodesicModelsSprint"
      exit 1 ;;
  esac
fi

# .env carries HF_TOKEN (Llama-2 repos are gated) and API keys. set -a exports everything
# the file defines so child processes (predownload, per-model runner subprocesses) inherit it.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  echo "  loaded .env"
fi
case "$MODELS" in
  all|pod_a|*llama2*)
    if [ -z "${HF_TOKEN:-}" ]; then
      echo "  !! HF_TOKEN is not set and MODELS='$MODELS' includes gated meta-llama repos —"
      echo "     predownload will fail on llama2_7b_chat. Put HF_TOKEN=... in .env."
    fi ;;
esac

# SAVE_TO_GIT preflight: prove the push can work non-interactively BEFORE the run, not at 3am.
# Dry-run push to a NEW ref name: tests push AUTH without failing on a behind-remote checkout
# (a plain `push --dry-run` rejects non-fast-forward even with valid creds; being behind is
# fine — the end-of-run flow pulls before pushing).
if [ "${SAVE_TO_GIT}" = "1" ]; then
  if ! git push --dry-run origin "HEAD:refs/heads/__preflight_test_$$" >/dev/null 2>&1; then
    echo "!! SAVE_TO_GIT=1 but a non-interactive push-auth check failed."
    echo "   Embed a PAT in the remote first:"
    echo "     git remote set-url origin https://<TOKEN>@github.com/timf34/GeodesicModelsSprint.git"
    exit 1
  fi
  echo "  git push preflight OK"
fi

echo "== [1/4] installing deps =="
if [ "$VENV" = "1" ]; then
  # Clean venv (NO system packages) — pod base images have shipped ABI-mismatched
  # torch/vllm pairs before (see AttractorBench/run_on_pod.sh); a clean venv is the fix.
  VENV_DIR="${VENV_DIR:-/workspace/gue_venv}"
  echo "  building clean venv at $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  pip install -q -U pip
fi
pip install -q -r requirements.txt   # includes hf_transfer (RunPod presets HF_HUB_ENABLE_HF_TRANSFER=1)

echo "  checking torch can use the GPU..."
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "  !! torch cannot use this GPU — driver/CUDA mismatch."
  echo "     driver: $(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9.]+' | head -1)"
  echo "     torch:  $(python -c 'import torch;print(torch.__version__, torch.version.cuda)' 2>/dev/null)"
  echo "     This stack needs a driver reporting CUDA >= 12.8, and this script has NO cu124"
  echo "     fallback: rent a newer-driver pod, or crib the pinned CU124=1 stack from"
  echo "     AttractorBench/run_sfm_on_pod.sh and adapt it yourself."
  exit 1
fi
echo "  torch.cuda OK"

# The Geodesic repos ship their chat template as a separate chat_template.jinja; transformers
# only auto-reads that file from 4.57 on. Older versions fall back to NO template and every
# chat prompt renders wrong — silently. Fail here, not after a night of garbage numbers.
if ! python - <<'PY'
import sys, transformers
v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
sys.exit(0 if v >= (4, 57) else 1)
PY
then
  echo "  !! transformers $(python -c 'import transformers;print(transformers.__version__)' 2>/dev/null) < 4.57 — cannot auto-read chat_template.jinja."
  echo "     pip install 'transformers>=4.57' and rerun."
  exit 1
fi
echo "  transformers >= 4.57 OK"
T_DEPS=$SECONDS

echo "== [2/4] pre-downloading + verifying ALL weights (MODELS=$MODELS) =="
# The whole point of this step: every network-dependent byte lands NOW, while someone is
# watching, so the overnight run itself never touches the network. If this fails there is
# nothing worth starting.
python scripts/predownload.py --models "$MODELS" || {
  echo "!! predownload FAILED — aborting before the run. Fix the download (token? disk? "
  echo "   network?) and rerun; snapshot_download resumes where it left off."
  exit 1
}
T_DL=$SECONDS

echo "== [3/4] running the sweep =="
mkdir -p logs
LOG_FILE="logs/pod_run_$(date +%Y%m%d_%H%M%S).log"
RUNNER_ARGS=(--models "$MODELS" --results-dir "$RESULTS_DIR"
             --gpu-mem-util "$GPU_MEM_UTIL" --max-model-len "$MAX_MODEL_LEN")
if [ -n "$EXPERIMENTS" ]; then
  RUNNER_ARGS+=(--experiments "$EXPERIMENTS")
fi
if [ -n "$SMOKE_FLAG" ]; then
  RUNNER_ARGS+=("$SMOKE_FLAG")
fi
echo "  python -m geodesic_ue.runner ${RUNNER_ARGS[*]}  (log: $LOG_FILE)"
# Capture the exit status instead of dying on it: per-model failures are the runner's
# business, and even a hard runner crash must still reach analysis + SAVE_TO_GIT below —
# partial results on disk are exactly what the overnight safety net exists to save.
RUN_STATUS=0
python -m geodesic_ue.runner "${RUNNER_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE" || RUN_STATUS=$?
if [ "$RUN_STATUS" -ne 0 ]; then
  echo "!! runner exited with status $RUN_STATUS — see $LOG_FILE; results so far are on disk."
fi
T_RUN=$SECONDS

# The runner stops early and drops this file when a coherence gate fails. A one-line log
# message is invisible inside 100k lines of vLLM output — make it impossible to miss.
if [ -f "$RESULTS_DIR/GATE_FAILED.json" ]; then
  echo ""
  echo "##############################################################"
  echo "##                                                          ##"
  echo "##   COHERENCE GATE FAILED — THE RUN STOPPED EARLY.         ##"
  echo "##   Downstream numbers from this run are NOT trustworthy   ##"
  echo "##   until the gate failure is understood.                  ##"
  echo "##                                                          ##"
  echo "##############################################################"
  echo "-- $RESULTS_DIR/GATE_FAILED.json --"
  cat "$RESULTS_DIR/GATE_FAILED.json" || true
  echo ""
  echo "##############################################################"
fi

echo "== [4/4] analysis =="
# Never fatal: the raw JSONL is the product of the night; analysis can always rerun locally.
python -m geodesic_ue.analysis.run_analysis --results-dir "$RESULTS_DIR" \
  || echo "  (analysis errored — raw results are safe in $RESULTS_DIR/; rerun analysis later)"
T_AN=$SECONDS

echo "== DONE. Results: $RESULTS_DIR/  Log: $LOG_FILE =="

if [ "$SMOKE" = "1" ]; then
  # Wall-clock per phase so the smoke run answers the only question that matters tonight:
  # "will the full sweep finish before morning?" Runner time scales ~linearly per model
  # (one model in GPU memory at a time); deps + downloads are one-off and larger for the
  # full model set. Note --smoke also shrinks per-model work, so extrapolate generously.
  echo ""
  echo "== SMOKE timing summary =="
  echo "   deps:        $((T_DEPS - T_START))s"
  echo "   predownload: $((T_DL - T_DEPS))s   (2 smoke models only)"
  echo "   runner:      $((T_RUN - T_DL))s   (~$(((T_RUN - T_DL) / 2))s per model at --smoke scale, 2 models)"
  echo "   analysis:    $((T_AN - T_RUN))s"
  echo "   total:       $((T_AN - T_START))s"
fi

# ---- optional save-to-git + self-shutdown (same semantics as run_sfm_on_pod.sh) ------
case "${SHUTDOWN:-}" in
  stop)      RP_ACTION="stop" ;;
  terminate) RP_ACTION="remove" ;;
  ""|0)      RP_ACTION="" ;;
  *)         RP_ACTION="stop" ;;   # any other truthy value -> the safe option
esac

# Push results to GitHub before shutting down so a terminate can't lose them. Needs
# non-interactive git auth on the pod (e.g. a PAT in the remote:
#   git remote set-url origin https://<TOKEN>@github.com/timf34/GeodesicModelsSprint.git ).
# SAFETY: if the push fails, any pending 'terminate' is downgraded to 'stop' so data is never lost.
if [ "${SAVE_TO_GIT}" = "1" ]; then
  echo "== saving results to git before shutdown =="
  # Fresh pods have no git identity; without one `git commit` FAILS and the push saves nothing.
  git config user.email >/dev/null 2>&1 || git config user.email "pod@geodesicsprint.local"
  git config user.name  >/dev/null 2>&1 || git config user.name  "GeodesicModelsSprint Pod"
  # results/, logs/, *.jsonl, *.log are all gitignored (local runs shouldn't pollute the
  # repo) — -f overrides that deliberately for the pod's one-shot archival commit.
  git add -f "$RESULTS_DIR" logs/ 2>/dev/null || true
  git commit -q -m "results: UE run ($MODELS) finished $(date -u +%FT%TZ)" || echo "  (nothing new to commit)"
  git pull --no-rebase --no-edit 2>/dev/null || true   # reconcile with remote first so push isn't rejected
  if git push; then
    echo "  results pushed to remote"
  elif [ "$RP_ACTION" = "remove" ]; then
    echo "  !! git push FAILED — refusing to terminate; downgrading to 'stop' to keep data."
    RP_ACTION="stop"
  fi
fi

if [ -n "$RP_ACTION" ]; then
  echo "== SHUTDOWN=$SHUTDOWN -> runpodctl $RP_ACTION pod ${RUNPOD_POD_ID:-<unset>} =="
  if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
    runpodctl "$RP_ACTION" pod "$RUNPOD_POD_ID"
  else
    echo "  !! cannot self-shutdown (runpodctl missing or RUNPOD_POD_ID unset) — pod left running."
  fi
fi

exit "$RUN_STATUS"
