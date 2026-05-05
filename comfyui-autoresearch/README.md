# ComfyUI LPIPS Autoresearch — Run Log

## Setup
- **Branch:** `autoresearch/comfyui-lpips-v2` op github.com/rherklots/autoresearch
- **Mac path:** `/Volumes/scripts/comfyui/comfyui-autoresearch/`
- **Refs:** 9 images (ref_42.png, ref_142.png, ref_242.png, ref1.jpg - ref6.jpg)
- **ComfyUI:** http://192.168.68.54:8188 (MacBook Pro M4 Max)
- **Target:** Minimize LPIPS distance (lower = better)

## Baseline Parameters
- guidance: 2.5
- steps: 30
- max_shift: 1.15
- base_shift: 0.5
- seed: varies per run

## Loop (max 20 iterations)
```bash
cd /Volumes/scripts/comfyui/comfyui-autoresearch
python score_script.py --seed $(date +%s) > run.log 2>&1
source run.env && echo "val_score=$val_score gen_time=$gen_time"
# KEEP if val_score improved, REVERT if not
```

## Run Log
| Run | Commit | val_score | gen_time | status | notes |
|-----|--------|-----------|----------|--------|-------|
