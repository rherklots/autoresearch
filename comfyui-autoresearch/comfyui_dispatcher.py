#!/usr/bin/env python3
"""
comfyui_dispatcher.py — NAS-side controller for ComfyUI LPIPS autoresearch loop.
Runs on Synology NAS, creates trigger files for Mac agent to process.

Usage:
    python3 comfyui_dispatcher.py [--max-iter 20]

The Mac agent (comfyui_agent.sh) reads trigger files, runs score_script.py,
and writes .last_result.json. This controller reads those results and decides
whether to keep or revert the change.
"""
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

# Paths
SHARED_DIR = Path("/volume1/scripts/comfyui/comfyui-autoresearch")
OUTPUTS_DIR = SHARED_DIR / "outputs"
REFS_DIR = SHARED_DIR / "refs"
RESULTS_FILE = SHARED_DIR / "results.tsv"
STATE_FILE = SHARED_DIR / ".dispatcher_state.json"
RESULT_FILE = SHARED_DIR / ".last_result.json"
TRIGGER_DIR = SHARED_DIR

COMFYUI_URL = "http://192.168.68.54:8188"
TIMEOUT = 600


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "iter": 0,
        "best_score": 999.0,
        "best_params": {"guidance": 2.5, "steps": 30, "max_shift": 1.15, "base_shift": 0.5},
        "history": []
    }


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def append_result(run_id, val_score, gen_time, file_size, status, description):
    line = f"{run_id}\t{val_score}\t{gen_time}\t{file_size}\t{status}\t{description}\n"
    with open(RESULTS_FILE, "a") as f:
        f.write(line)


def comfyui_dispatch(workflow: dict) -> str:
    data = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/api/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
        return result.get("prompt_id", "")


def comfyui_wait_and_download(prompt_id: str, dest_path: Path) -> dict:
    start = time.time()
    while time.time() - start < TIMEOUT:
        try:
            with urllib.request.urlopen(f"{COMFYUI_URL}/api/history/{prompt_id}", timeout=10) as r:
                h = json.loads(r.read())
                if prompt_id in h:
                    outputs = h[prompt_id].get("outputs", {})
                    for node_data in outputs.values():
                        if "images" in node_data:
                            filename = node_data["images"][0]["filename"]
                            img_url = f"{COMFYUI_URL}/api/view?filename=output/{filename}"
                            urllib.request.urlretrieve(img_url, str(dest_path))
                            return h[prompt_id]
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"Generation timed out after {TIMEOUT}s")


def dispatch_trigger(run_id: int, seed: int, params: dict) -> str:
    """Write trigger file for Mac agent to pick up."""
    trigger = {
        "run_id": run_id,
        "seed": seed,
        "steps": params.get("steps", 30),
        "guidance": params.get("guidance", 2.5),
        "max_shift": params.get("max_shift", 1.15),
        "base_shift": params.get("base_shift", 0.5),
        "timestamp": datetime.now().isoformat()
    }
    trigger_file = TRIGGER_DIR / f".trigger_{run_id:04d}.json"
    trigger_file.write_text(json.dumps(trigger))
    return str(trigger_file)


def wait_for_result(timeout: int = 600) -> dict:
    """Poll for .last_result.json from Mac agent."""
    start = time.time()
    while time.time() - start < timeout:
        if RESULT_FILE.exists():
            result = json.loads(RESULT_FILE.read_text())
            RESULT_FILE.unlink()
            return result
        time.sleep(5)
    raise TimeoutError("Mac agent did not produce result in time")


def decide_next_params(current_params: dict, history: list) -> dict:
    """Simple strategy: try small adjustments."""
    import random
    # Small perturbation
    p = dict(current_params)
    # Randomly vary one or two params
    vary = random.choice(["guidance", "steps", "max_shift", "base_shift", "combo"])
    if vary == "guidance":
        delta = random.choice([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3])
        p["guidance"] = round(max(1.0, min(5.0, p["guidance"] + delta)), 2)
    elif vary == "steps":
        delta = random.choice([-5, -3, 3, 5])
        p["steps"] = max(20, min(50, p["steps"] + delta))
    elif vary == "max_shift":
        delta = random.choice([-0.05, -0.03, 0.03, 0.05])
        p["max_shift"] = round(max(0.8, min(2.0, p["max_shift"] + delta)), 2)
    elif vary == "base_shift":
        delta = random.choice([-0.05, -0.03, 0.03, 0.05])
        p["base_shift"] = round(max(0.2, min(1.0, p["base_shift"] + delta)), 2)
    else:  # combo
        p["guidance"] = round(max(1.0, min(5.0, p["guidance"] + random.choice([-0.2, 0.2]))), 2)
        p["steps"] = max(20, min(50, p["steps"] + random.choice([-3, 3])))
    return p


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=20)
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REFS_DIR.mkdir(parents=True, exist_ok=True)

    state = load_state()
    print(f"[{datetime.now()}] ComfyUI LPIPS Dispatcher START — {args.max_iter} iteraties")
    print(f"[{datetime.now()}] Baseline score: {state['best_score']}")

    # Fixed prompt
    clip_l = "portrait photo"
    t5xxl = "portrait photo, realistic skin texture, visible pores, skin imperfections, subsurface scattering, natural skin detail, detailed eyes, sharp focus, 8k photography"

    # Initialize results if empty
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text("run_id\tval_score\tgeneration_time\tfile_size_kb\tstatus\tdescription\n")

    for iteration in range(args.max_iter):
        iter_num = state["iter"] + 1
        seed = int(time.time()) % 100000
        params = state["best_params"]  # start from current best

        print(f"\n[{datetime.now()}] === Iter {iter_num}/{args.max_iter} ===")

        # Decide params for this run
        current_params = decide_next_params(state["best_params"], state["history"])
        print(f"[{datetime.now()}] Params: {current_params}")

        # Build workflow
        workflow = {
            "1": {"inputs": {"ckpt_name": "flux1-dev-fp8.safetensors"}, "class_type": "CheckpointLoaderSimple"},
            "2": {"inputs": {"clip": ["1", 1], "clip_l": clip_l, "t5xxl": t5xxl, "guidance": current_params["guidance"]}, "class_type": "CLIPTextEncodeFlux"},
            "3": {"inputs": {"conditioning": ["2", 0], "guidance": current_params["guidance"]}, "class_type": "FluxGuidance"},
            "4": {"inputs": {"model": ["1", 0], "max_shift": current_params["max_shift"], "base_shift": current_params["base_shift"], "width": 1024, "height": 1024}, "class_type": "ModelSamplingFlux"},
            "5": {"inputs": {"width": 1024, "height": 1024, "batch_size": 1}, "class_type": "EmptyLatentImage"},
            "6": {"inputs": {"model": ["4", 0], "positive": ["3", 0], "negative": ["2", 0], "latent_image": ["5", 0], "seed": seed, "steps": current_params["steps"], "cfg": 1.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}, "class_type": "KSampler"},
            "7": {"inputs": {"samples": ["6", 0], "vae": ["1", 2]}, "class_type": "VAEDecode"},
            "8": {"inputs": {"filename_prefix": f"iter{iter_num:03d}_", "images": ["7", 0]}, "class_type": "SaveImage"},
        }

        # Dispatch to ComfyUI
        print(f"[{datetime.now()}] Dispatching to ComfyUI...")
        prompt_id = comfyui_dispatch(workflow)
        print(f"[{datetime.now()}] prompt_id={prompt_id}")

        # Download result image
        img_path = OUTPUTS_DIR / f"iter{iter_num:03d}_{seed}.png"
        history = comfyui_wait_and_download(prompt_id, img_path)
        file_size_kb = os.path.getsize(img_path) // 1024
        print(f"[{datetime.now()}] Image downloaded: {img_path.name} ({file_size_kb}KB)")

        # Create trigger for Mac agent to run LPIPS
        print(f"[{datetime.now()}] Triggering Mac agent for LPIPS evaluation...")
        trigger_file = dispatch_trigger(iter_num, seed, current_params)
        print(f"[{datetime.now()}] Trigger: {trigger_file}")

        # Wait for LPIPS result from Mac agent
        try:
            result = wait_for_result(timeout=600)
            val_score = result["val_score"]
            gen_time = result.get("generation_time", 0)
            file_size = result.get("file_size_kb", file_size_kb)
        except TimeoutError:
            print(f"[{datetime.now()}] TIMEOUT waiting for LPIPS result — skipping iteration")
            continue

        print(f"[{datetime.now()}] LPIPS val_score={val_score}")

        # Decide keep/revert
        desc = f"guidance={current_params['guidance']} steps={current_params['steps']} max_shift={current_params['max_shift']} base_shift={current_params['base_shift']}"

        if val_score < state["best_score"]:
            status = "KEEP"
            state["best_score"] = val_score
            state["best_params"] = current_params
            print(f"[{datetime.now()}] NEW BEST: {val_score} — saved as baseline")
        else:
            status = "REVERT"
            print(f"[{datetime.now()}] No improvement ({val_score} >= {state['best_score']})")

        state["iter"] = iter_num
        state["history"].append({"iter": iter_num, "val_score": val_score, "params": current_params, "status": status})
        save_state(state)
        append_result(f"iter{iter_num:03d}", val_score, gen_time, file_size, status, desc)

    print(f"\n[{datetime.now()}] === LOOP COMPLETE ===")
    print(f"[{datetime.now()}] Best score: {state['best_score']}")
    print(f"[{datetime.now()}] Best params: {state['best_params']}")
    print(f"[{datetime.now()}] Total iterations: {state['iter']}")


if __name__ == "__main__":
    main()
