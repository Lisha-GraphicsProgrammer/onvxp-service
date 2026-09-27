# training_pipeline/validate_samples.py
"""
Visual pre-flight check for a candidate training dataset. Samples images
from the isolated class and asks a vision model directly: does this image
actually show the target class? Rejects the whole candidate dataset if
agreement is too low — BEFORE training ever starts, not after.
"""
import glob
import random
import base64
from pathlib import Path


def sample_images(dataset_dir: str, n: int = 12) -> list[str]:
    imgs = glob.glob(str(Path(dataset_dir) / "train" / "images" / "*"))
    random.shuffle(imgs)
    return imgs[:n]


def verify_with_dino(image_paths, class_prompt, confidence=0.3):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    # Grounding DINO requires lowercase text ending in a period, or it
    # silently returns zero detections regardless of what's in the image.
    class_prompt = class_prompt.lower().strip()
    if not class_prompt.endswith("."):
        class_prompt += "."

    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")

    results = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, text=class_prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        out = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=confidence, text_threshold=confidence,
            target_sizes=[image.size[::-1]],
        )[0]
        results.append({"path": path, "found": len(out["boxes"]) > 0, "num_boxes": len(out["boxes"])})
    return results


def verify_with_claude_vision(image_paths, class_description, api_key):
    import requests

    results = []
    for path in image_paths:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5",
                "max_tokens": 20,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                        {"type": "text", "text": f"Does this image clearly show {class_description}? Answer only yes or no."},
                    ],
                }],
            },
            timeout=20,
        )
        resp.raise_for_status()
        text = resp.json()["content"][0]["text"].strip().lower()
        results.append({"path": path, "found": text.startswith("y")})
    return results


def validate_dataset(dataset_dir: str, class_prompt: str, min_agreement: float = 0.6,
                      sample_n: int = 12, anthropic_api_key: str | None = None) -> dict:
    samples = sample_images(dataset_dir, sample_n)
    if not samples:
        return {"passed": False, "reason": "No images found to sample."}

    results = (
        verify_with_claude_vision(samples, class_prompt, anthropic_api_key)
        if anthropic_api_key else
        verify_with_dino(samples, class_prompt)
    )

    agree = sum(1 for r in results if r["found"]) / len(results)
    passed = agree >= min_agreement
    return {
        "passed": passed,
        "agreement_rate": round(agree, 2),
        "sample_size": len(results),
        "details": results,
        "reason": None if passed else
            f"Only {agree:.0%} of sampled images visually matched '{class_prompt}' (required {min_agreement:.0%})",
    }


def run_for_job(job_id: int, db_session, TrainingJob):
    """
    Runs the visual pre-flight check for a training job and updates its
    DB row. Called right after dataset prep succeeds, before training
    ever starts — this is the actual gate that stops a mismatched dataset
    from burning a full training run before anyone notices.

    On rejection, this no longer fails the job outright. data_acquisition.py
    now stores every candidate the original search found, not just the one
    that got downloaded — so a rejection here sends the job back to
    acquisition to try the NEXT untried candidate from that same list,
    rather than giving up after the very first, top-ranked pick turned out
    to be a poor visual match. The job only genuinely fails once every
    real candidate the search found has actually been tried and rejected.
    """
    import os
    from datetime import datetime, timezone

    job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        return

    def _push_stage(name, status, detail=None):
        stages = list(job.stages or [])
        stages.append({
            "name": name, "status": status, "detail": detail,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        job.stages = stages
        job.current_stage = name if status == "running" else job.current_stage
        db_session.commit()

    _push_stage("validating_dataset", "running", "Sampling images and checking visual match...")

    dataset_dir = f"datasets/{job.class_name}"
    class_prompt = job.class_name.replace("_", " ")
    api_key = os.getenv("ANTHROPIC_API_KEY")  # falls back to local DINO model if unset

    result = validate_dataset(dataset_dir, class_prompt, anthropic_api_key=api_key)

    existing_info = dict(job.dataset_info or {})
    validation_history = list(existing_info.get("validation_history", []))
    validation_history.append({
        "source": existing_info.get("source"),
        "agreement_rate": result.get("agreement_rate"),
        "passed": result["passed"],
        "reason": result.get("reason"),
    })
    existing_info["validation_history"] = validation_history
    existing_info["validation_report"] = result
    job.dataset_info = existing_info

    if result["passed"]:
        job.current_stage = "training"
        detail = f"{result['agreement_rate']:.0%} of {result['sample_size']} sampled images visually matched '{class_prompt}'"
        _push_stage("validating_dataset", "done", detail)
        return

    # Rejected — but is this genuinely the last real option, or are there
    # other candidates the original search already found that haven't
    # been tried yet? A candidate that got this far (downloaded, prepped)
    # is also marked tried here, in case it wasn't already.
    candidates = existing_info.get("candidates", [])
    tried_keys = set(existing_info.get("tried_candidate_keys", []))
    current_key = existing_info.get("candidate_key")
    if current_key:
        tried_keys.add(current_key)
        existing_info["tried_candidate_keys"] = list(tried_keys)
        job.dataset_info = existing_info

    remaining = [c for c in candidates if f"{c.get('workspace')}/{c.get('project')}" not in tried_keys]

    if remaining:
        # A real, untried alternative exists from the original search —
        # go try it instead of failing the whole job over one dataset
        # among several genuine options.
        job.current_stage = "searching_data"
        detail = (
            f"'{existing_info.get('source', 'this candidate')}' didn't visually match "
            f"(only {result.get('agreement_rate', 0):.0%} agreement) — trying next "
            f"candidate ({len(remaining)} of {len(candidates)} remaining)."
        )
        _push_stage("validating_dataset", "failed", detail)
    else:
        job.status = "failed"
        job.error = (
            f"Tried {len(tried_keys)} candidate dataset(s) for '{class_prompt}', "
            f"none visually matched well enough. Last: {result['reason']}"
        )
        _push_stage("validating_dataset", "failed", job.error)
        db_session.commit()


if __name__ == "__main__":
    import sys, json
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "datasets/exposed_electrical_wire"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "exposed electrical wire"
    result = validate_dataset(dataset_dir, prompt)
    print(json.dumps(result, indent=2))