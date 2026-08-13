# GeodesicModelsSprint

## Running on RunPod

**Pod requirements**
- 1x A100 / H100 / H200 80GB (the runner keeps one 7B-class model in GPU memory at a time)
- Volume: **500GB** for `--models all` (25 checkpoints), **~200GB per pod** for the two-pod split
- Driver reporting **CUDA >= 12.8** (vllm requirement; this repo has no old-driver fallback)

**Launch (three commands)**

```bash
git clone <repo-url> /workspace/GeodesicModelsSprint && cd /workspace/GeodesicModelsSprint
# create .env with HF_TOKEN=... (Llama-2 repos are gated) + the other keys
SAVE_TO_GIT=1 SHUTDOWN=stop nohup bash run_on_pod.sh > overnight.log 2>&1 &
```

The script installs deps, **pre-downloads and verifies every checkpoint before the run starts**
(no 3am download timeouts), runs the sweep, runs analysis, then pushes `results/` + `logs/` to
git and stops the pod. For the unattended recipe, first run `runpodctl config --apiKey <KEY>`
and embed a PAT in the git remote (see the header of `run_on_pod.sh`).

**Two-pod split** (parallel halves; pod A alone is a publishable sweep if pod B dies):

```bash
POD=a SAVE_TO_GIT=1 SHUTDOWN=stop nohup bash run_on_pod.sh > overnight_a.log 2>&1 &   # dpo tier + llama2 + olmo3 (13)
POD=b SAVE_TO_GIT=1 SHUTDOWN=stop nohup bash run_on_pod.sh > overnight_b.log 2>&1 &   # instruct tier + olmo2 (12)
```

**Smoke test first** (strongly recommended): two models, `--smoke`, its own `results_smoke/`,
analysis included, and a per-phase wall-clock summary to extrapolate the full night from:

```bash
SMOKE=1 bash run_on_pod.sh
```

**Interrupted / crashed pod?** Just rerun the same command. Results are JSONL keyed work
units — the runner skips everything already on disk and resumes where it stopped, and the
weight predownload is a cached no-op.
