#!/usr/bin/env python3
"""
score_script.py — ComfyUI LPIPS evaluation script for autoresearch loop.

Measures how close generated images are to reference images using LPIPS
(Learned Perceptual Image Patch Similarity). Lower = closer to reference = better.

Usage:
    python score_script.py [--seed N] [--steps N] [--guidance N] \
                           [--max_shift F] [--base_shift F] \
                           [--ref_dir DIR] [--output_dir DIR]

Output:
    val_score: X.XXXXXX   <- lower is better (LPIPS distance)
    generation_time: X.X   <- seconds
    file_size_kb: XXX      <- KB (for diagnostics)
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://192.168.68.54:8188")
DEFAULT_REF_DIR = Path(__file__).parent / "refs"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs"

# ---------------------------------------------------------------------------
# LPIPS evaluation
# ---------------------------------------------------------------------------
def load_lpips():
    """Lazy-load lpips to avoid import error if not installed."""
    import lpips
    return lpips.LPIPS(net="alex")


def compute_lpips_score(image_a_path: str, image_b_path: str, lpips_model) -> float:
    """Compute LPIPS distance between two images. Returns float in [0, ~1]."""
    from PIL import Image
    import torch

    # Load and resize to 256x256 for LPIPS
    def prep(path):
        img = Image.open(path).convert("RGB").resize((256, 256), Image.LANCZOS)
        img_tensor = torch.tensor(
            __import__('numpy').array(img).transpose(2, 0, 1) / 127.5 - 1.0,
            dtype=torch.float32
        )
        return img_tensor

    a = prep(image_a_path).unsqueeze(0)
    b = prep(image_b_path).unsqueeze(0)

    with torch.no_grad():
        dist = lpips_model(a, b)
    return dist.item()


def compute_multi_ref_score(generated_path: str, ref_dir: Path, lpips_model) -> float:
    """Average LPIPS against all reference images. Returns mean distance."""
    refs = sorted(ref_dir.glob("ref_*.png")) + sorted(ref_dir.glob("ref*.jpg"))
    if not refs:
        raise ValueError(f"No reference images found in {ref_dir}")
    scores = [compute_lpips_score(generated_path, str(r), lpips_model) for r in refs]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# ComfyUI workflow
# ---------------------------------------------------------------------------
def comfyui_generate(
    clip_l: str,
    t5xxl: str,
    seed: int,
    steps: int = 30,
    guidance: float = 2.5,
    max_shift: float = 1.15,
    base_shift: float = 0.5,
    width: int = 1024,
    height: int = 1024,
    output_dir: Path = None,
) -> tuple[str, float, int]:
    """
    Queue and wait for a Flux image generation on ComfyUI.
    Returns (local_image_path, generation_time_sec, file_size_kb).
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    workflow = {
        "1": {"inputs": {"ckpt_name": "flux1-dev-fp8.safetensors"}, "class_type": "CheckpointLoaderSimple"},
        "2": {"inputs": {"clip": ["1", 1], "clip_l": clip_l, "t5xxl": t5xxl, "guidance": guidance}, "class_type": "CLIPTextEncodeFlux"},
        "3": {"inputs": {"conditioning": ["2", 0], "guidance": guidance}, "class_type": "FluxGuidance"},
        "4": {"inputs": {"model": ["1", 0], "max_shift": max_shift, "base_shift": base_shift, "width": width, "height": height}, "class_type": "ModelSamplingFlux"},
        "5": {"inputs": {"width": width, "height": height, "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "6": {"inputs": {"model": ["4", 0], "positive": ["3", 0], "negative": ["2", 0], "latent_image": ["5", 0], "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}, "class_type": "KSampler"},
        "7": {"inputs": {"samples": ["6", 0], "vae": ["1", 2]}, "class_type": "VAEDecode"},
        "8": {"inputs": {"filename_prefix": f"eval_{seed}", "images": ["7", 0]}, "class_type": "SaveImage"},
    }

    # Queue prompt
    data = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/api/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"No prompt_id returned: {result}")

    # Wait for completion
    start = time.time()
    timeout = 600  # 10 min max
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{COMFYUI_URL}/api/history/{prompt_id}", timeout=10) as r:
                h = json.loads(r.read())
                if prompt_id in h:
                    outputs = h[prompt_id].get("outputs", {})
                    for node_data in outputs.values():
                        if "images" in node_data:
                            filename = node_data["images"][0]["filename"]
                            # Download image
                            img_url = f"{COMFYUI_URL}/api/view?filename=output/{filename}"
                            local_path = output_dir / filename.replace("eval_", "eval_seed{}_".format(seed))
                            urllib.request.urlretrieve(img_url, str(local_path))
                            gen_time = time.time() - start
                            size_kb = os.path.getsize(local_path) // 1024
                            return str(local_path), gen_time, size_kb
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
            pass
        time.sleep(5)

    raise TimeoutError(f"Generation timed out after {timeout}s (prompt_id={prompt_id})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ComfyUI LPIPS evaluation")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--max_shift", type=float, default=1.15)
    parser.add_argument("--base_shift", type=float, default=0.5)
    parser.add_argument("--ref_dir", type=Path, default=DEFAULT_REF_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    # The following are passed via environment or CLI
    args = parser.parse_args()

    ref_dir = Path(os.environ.get("REF_DIR", str(args.ref_dir)))
    output_dir = Path(os.environ.get("OUTPUT_DIR", str(args.output_dir)))

    # Fixed test prompt
    clip_l = "portrait photo"
    t5xxl = "portrait photo, realistic skin texture, visible pores, skin imperfections, subsurface scattering, natural skin detail, detailed eyes, sharp focus, 8k photography"

    # Generate image
    try:
        img_path, gen_time, file_size = comfyui_generate(
            clip_l=clip_l,
            t5xxl=t5xxl,
            seed=args.seed,
            steps=args.steps,
            guidance=args.guidance,
            max_shift=args.max_shift,
            base_shift=args.base_shift,
            output_dir=output_dir,
        )
    except Exception as e:
        print(f"val_score: 999.0\ngeneration_time: 0.0\nfile_size_kb: 0\nerror: {e}", file=sys.stderr)
        sys.exit(1)

    # Compute LPIPS score
    lpips_model = load_lpips()
    try:
        score = compute_multi_ref_score(img_path, ref_dir, lpips_model)
    except Exception as e:
        print(f"val_score: 999.0\ngeneration_time: {gen_time:.1f}\nfile_size_kb: {file_size}\nerror: LPIPS failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"val_score: {score:.6f}")
    print(f"generation_time: {gen_time:.1f}")
    print(f"file_size_kb: {file_size}")
    print(f"lpips_refs: {len(list(ref_dir.glob('ref_*.png')))}")


if __name__ == "__main__":
    main()
