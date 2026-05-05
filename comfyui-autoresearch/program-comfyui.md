# ComfyUI Autoresearch — LPIPS Image Quality Loop

## Goal

Optimize Flux image generation parameters for realistic skin texture and photographic quality by minimizing **LPIPS distance** (lower = closer to reference = better).

**Metric:** `val_score: X.XXXXXX` (LPIPS distance, lower is better)

## Setup

**Shared folder (Mac-side):** `/Volumes/scripts/comfyui/comfyui-autoresearch/`
**NAS-side path:** `/volume1/scripts/comfyui/comfyui-autoresearch/`

```bash
# Op de Mac:
cd /Volumes/scripts/comfyui/comfyui-autoresearch
```

**Reference images** already exist in `./refs/`:
- `refs/ref_42.png` (1,288 KB)
- `refs/ref_142.png` (1,272 KB)
- `refs/ref_242.png` (1,379 KB)

**Environment:**
- `COMFYUI_URL=http://192.168.68.54:8188` (MacBook Pro M4 Max)
- MPS GPU — generation takes ~3-5 min per image
- LPIPS (`pip install lpips`) installed on the machine running this script

## Target Parameters (editable in `score_script.py` or via env)

The agent may modify these workflow parameters:
- `steps` — denoising steps (current: 30, range: 20-40)
- `guidance` — Flux guidance scale (current: 2.5, range: 1.5-4.0)
- `max_shift` — ModelSamplingFlux max_shift (current: 1.15, range: 1.0-1.6)
- `base_shift` — ModelSamplingFlux base_shift (current: 0.5, range: 0.3-0.8)
- `seed` — KSampler seed (always varies per run to avoid overfitting)

**DO NOT change:**
- The model (`flux1-dev-fp8.safetensors`)
- The prompt structure (clip_l / t5xxl split)
- The reference images in `./refs/`
- LPIPS library or evaluation method

## The Loop

### One experiment cycle:

```
1. Modify ONE OR TWO parameter values in score_script.py (or pass via CLI args)
2. Run: python score_script.py --seed $RANDOM_SEED > run.log 2>&1
   - Set RANDOM_SEED to a fresh seed each run (e.g. $(date +%s))
3. Extract results:
     val_score=$(grep "^val_score:" run.log | awk '{print $2}')
     gen_time=$(grep "^generation_time:" run.log | awk '{print $2}')
4. If val_score < best_known: KEEP (commit, update best)
   If val_score >= best_known: REVERT (git reset --hard HEAD~1)
5. Log to results.tsv:  <commit_hash>  <val_score>  <gen_time>  <params_summary>
6. Repeat
```

### Timing
- ~3-5 min per generation on MPS (MacBook M4 Max)
- ~30 seconds LPIPS evaluation (CPU)
- Budget: kill after 10 min total, treat as crash/discard

## Metrics

```
val_score:          0.234561    <- LPIPS distance (lower = better)
generation_time:     187.3       <- seconds (diagnostic only)
file_size_kb:       1042        <- KB (diagnostic only, bigger usually = more detail)
```

**Primary:** `val_score` (LPIPS distance)
**Secondary:** `generation_time` (for efficiency tracking)

## Success Criterion

Best achievable `val_score` with current reference set. Lower is always better —
no trade-off between quality and speed unless explicitly asked.

## Constraints

- **ONE model:** flux1-dev-fp8 only
- **NO new packages** — only what's in pyproject.toml or already installed
- **NO asking for permission** — autonomous until human interrupts
- **Simplicity criterion:** all else equal, the set of changes with fewest parameter modifications wins
- **Reference images stay fixed** — never retake them unless explicitly approved

## Results Log

Maintain `results.tsv` (untracked, never commit):
```
commit  val_score  generation_time  file_size_kb  status  description
abc1234  0.234561  187.3           1042           keep    baseline with guidance=2.5
def5678  0.221034  203.1           1187           keep    steps=35, guidance=2.0
```

## Key Insight

LPIPS (Learned Perceptual Image Patch Similarity) measures perceptual similarity —
a lower score means the generated image looks more like the reference in terms of
structural, textural, and perceptual features. This is a better quality proxy than
file size or human guesswork.
