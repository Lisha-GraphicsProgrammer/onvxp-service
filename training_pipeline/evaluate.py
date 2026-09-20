"""
Self-Learning Pipeline — Step 7: Evaluation Agent

Runs the trained model against the held-out TEST split (never touched during
training or validation) and checks results against acceptance gates before
letting a candidate anywhere near human approval. A model that fails these
gates is rejected automatically — this is the "not every training run
produces a better model" safety check from the original design spec.
"""
import json
import os
import base64
import cv2
import requests
from pathlib import Path
from datetime import datetime, timezone
from ultralytics import YOLO

DATASETS_DIR = Path("datasets")
SAMPLES_DIR = Path("incidents")  # reuses the same folder already served by
# the API's /screenshots/ static mount, so sample preview images need no
# new route — they're just files sitting alongside incident evidence.

# Which model does the VLM sanity check — cheap/fast is fine here, it's a
# yes/no judgment call on one image, not deep reasoning.
VLM_MODEL = "claude-haiku-4-5-20251001"

# v1 acceptance gates — deliberately modest since a 3-10 epoch toy run is
# what we're actually testing against; a real production run (50-100 epochs)
# would be expected to clear much higher bars. Tune per class/site later.
ACCEPTANCE_GATES = {
    "min_precision": 0.5,
    "min_recall": 0.4,
    "min_map50": 0.4,
}


def _generate_sample_previews(class_name: str, weights_path: str, job_id: int, n: int = 6) -> list:
    """
    Runs the freshly-trained model on real TEST-split images and saves
    them WITH the model's own detection boxes drawn on — actual, checkable
    evidence for the approval screen, not just a precision/recall number.
    A human deciding whether to trust this model should be able to look
    at what it actually catches, not just read a score.
    """
    test_img_dir = DATASETS_DIR / class_name / "test" / "images"
    if not test_img_dir.exists():
        print(f"[OMNIX] No test images found at {test_img_dir} — skipping sample previews")
        return []

    images = sorted(test_img_dir.glob("*"))[:n]
    if not images:
        return []

    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"[OMNIX] Could not load {weights_path} for sample previews: {e}")
        return []

    SAMPLES_DIR.mkdir(exist_ok=True)
    saved_paths = []
    for i, img_path in enumerate(images):
        try:
            results = model.predict(str(img_path), conf=0.25, verbose=False)
            annotated = results[0].plot()  # BGR array, boxes already drawn
            out_path = str(SAMPLES_DIR / f"sample_job{job_id}_{i}.jpg")
            cv2.imwrite(out_path, annotated)
            saved_paths.append(out_path)
        except Exception as e:
            print(f"[OMNIX] Sample preview {i} failed for job {job_id}: {e}")

    return saved_paths


def _vlm_sanity_check(class_name: str, sample_paths: list) -> dict:
    """
    An independent second opinion on the model's OWN predictions —
    different from validate_samples.py, which checks the training DATA
    before training ever starts. This checks what the TRAINED model
    actually drew boxes around: does it genuinely look like real
    class_name, or is the model confidently boxing the wrong thing?

    Degrades safely: if ANTHROPIC_API_KEY isn't configured, this check is
    skipped (not failed) — the numeric gates and human review still apply
    regardless. This is meant as an extra layer of confidence, not the
    only line of defense.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"ran": False, "reason": "ANTHROPIC_API_KEY not configured — skipped"}
    if not sample_paths:
        return {"ran": False, "reason": "no sample images to check"}

    verdicts = []
    for path in sample_paths[:4]:  # cap cost — a handful is enough for a sanity check
        try:
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
                    "model": VLM_MODEL,
                    "max_tokens": 20,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                            {"type": "text", "text": f"This image has a box drawn where a detection model flagged '{class_name}'. Does the box genuinely contain {class_name}? Answer only yes or no."},
                        ],
                    }],
                },
                timeout=20,
            )
            resp.raise_for_status()
            text = resp.json()["content"][0]["text"].strip().lower()
            verdicts.append(text.startswith("y"))
        except Exception as e:
            print(f"[OMNIX] VLM check failed on {path}: {e}")

    if not verdicts:
        return {"ran": False, "reason": "all VLM calls failed"}

    agreement = sum(verdicts) / len(verdicts)
    return {
        "ran": True,
        "agreement_rate": round(agreement, 2),
        "sample_size": len(verdicts),
        "passed": agreement >= 0.5,
    }


def evaluate_model(class_name: str, weights_path: str, job_id: int = 0) -> dict:
    """
    Evaluates weights_path against datasets/<class_name>/test split.
    Returns metrics + a pass/fail verdict against ACCEPTANCE_GATES, plus
    sample preview images and an independent VLM sanity check — real
    evidence for a human to look at, not just numbers, before this model
    ever reaches "awaiting approval".
    """
    if not Path(weights_path).exists():
        return {"success": False, "error": f"Weights not found at {weights_path}"}

    data_yaml = DATASETS_DIR / class_name / "data.yaml"
    if not data_yaml.exists():
        return {"success": False, "error": f"data.yaml not found at {data_yaml}"}

    try:
        model = YOLO(weights_path)
        # split="test" explicitly evaluates the held-out test set, not val
        results = model.val(data=str(data_yaml), split="test", verbose=False)

        precision = float(results.box.p.mean()) if len(results.box.p) else 0.0
        recall = float(results.box.r.mean()) if len(results.box.r) else 0.0
        map50 = float(results.box.map50)
        map50_95 = float(results.box.map)

        metrics = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4),
        }

        gate_failures = []
        if precision < ACCEPTANCE_GATES["min_precision"]:
            gate_failures.append(f"precision {precision:.3f} < required {ACCEPTANCE_GATES['min_precision']}")
        if recall < ACCEPTANCE_GATES["min_recall"]:
            gate_failures.append(f"recall {recall:.3f} < required {ACCEPTANCE_GATES['min_recall']}")
        if map50 < ACCEPTANCE_GATES["min_map50"]:
            gate_failures.append(f"mAP50 {map50:.3f} < required {ACCEPTANCE_GATES['min_map50']}")

        # Generate real evidence regardless of pass/fail — a human should
        # be able to see WHY a model failed too, not just that it did.
        sample_images = _generate_sample_previews(class_name, weights_path, job_id)
        vlm_check = _vlm_sanity_check(class_name, sample_images)

        # The VLM check only blocks approval if it actually ran and
        # disagreed — a missing API key or a transient failure never
        # silently approves a bad model, it just means one fewer layer
        # of confidence was available, and the numeric gates still decide.
        if vlm_check.get("ran") and not vlm_check.get("passed"):
            gate_failures.append(
                f"VLM sanity check: only {vlm_check['agreement_rate']:.0%} of sample detections "
                f"looked genuinely correct (checked {vlm_check['sample_size']} images)"
            )

        passed = len(gate_failures) == 0
        return {
            "success": True,
            "metrics": metrics,
            "gates": ACCEPTANCE_GATES,
            "passed": passed,
            "gate_failures": gate_failures,
            "sample_images": sample_images,
            "vlm_check": vlm_check,
        }
    except Exception as e:
        return {"success": False, "error": f"Evaluation failed: {e}"}


def run_for_job(job_id: int, db_session, TrainingJob):
    """Runs evaluation for a job and updates its DB row with metrics + verdict."""
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

    _push_stage("evaluating", "running", "Testing candidate model against held-out test set...")
    result = evaluate_model(job.class_name, job.model_path, job_id=job.id)

    if not result["success"]:
        job.status = "failed"
        job.error = result["error"]
        _push_stage("evaluating", "failed", result["error"])
        return

    job.metrics = result["metrics"]
    job.sample_images = result.get("sample_images", [])
    m = result["metrics"]
    detail = f"Precision {m['precision']:.2f}, Recall {m['recall']:.2f}, mAP50 {m['map50']:.2f}"

    vlm = result.get("vlm_check", {})
    if vlm.get("ran"):
        detail += f", VLM check {vlm['agreement_rate']:.0%} agreement ({vlm['sample_size']} images)"

    if result["passed"]:
        job.current_stage = "awaiting_approval"
        _push_stage("evaluating", "done", detail + " — passed acceptance gates")
    else:
        job.status = "failed"
        job.error = "Failed acceptance gates: " + "; ".join(result["gate_failures"])
        _push_stage("evaluating", "failed", detail + " — " + "; ".join(result["gate_failures"]))


if __name__ == "__main__":
    import sys
    cls = sys.argv[1] if len(sys.argv) > 1 else "trousers"
    weights = sys.argv[2] if len(sys.argv) > 2 else f"runs/self_learning/{cls}_model/weights/best.pt"
    result = evaluate_model(cls, weights)
    print(json.dumps(result, indent=2))