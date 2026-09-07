"""
Self-Learning Pipeline — Step 6: Training Agent

Fine-tunes a YOLO26 model on a prepared dataset, saving epoch checkpoints
so a resumed training job can pick up where it left off (the resumability
requirement from the original design). v1 trains ONE candidate (yolo26n,
the smallest/fastest model) rather than multiple candidates in parallel —
multi-candidate HPO is a documented v2 upgrade.
"""
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from ultralytics import YOLO

DATASETS_DIR = Path("datasets")
RUNS_DIR = Path("runs") / "self_learning"


def train_model(class_name: str, epochs: int = 6, imgsz: int = 416, resume_from: str | None = None, on_epoch_end=None) -> dict:
    """
    Trains a YOLO26n model on datasets/<class_name>/data.yaml.
    If resume_from is given (a path to a previous last.pt), continues from
    that checkpoint instead of starting fresh — this is what makes a job
    resumable across restarts rather than losing all progress.
    on_epoch_end, if given, is registered as an Ultralytics training callback
    (on_train_epoch_end) so the caller can report live per-epoch progress
    without train_model() itself needing to know about jobs/DB rows.
    Returns a dict describing the outcome.
    """
    data_yaml = DATASETS_DIR / class_name / "data.yaml"
    if not data_yaml.exists():
        return {"success": False, "error": f"data.yaml not found at {data_yaml}"}

    run_name = f"{class_name}_model"
    try:
        if resume_from and Path(resume_from).exists():
            model = YOLO(resume_from)
            if on_epoch_end:
                model.add_callback("on_train_epoch_end", on_epoch_end)
            results = model.train(resume=True)
        else:
            model = YOLO("yolo26n.pt") # smallest base model — fastest on CPU
            if on_epoch_end:
                model.add_callback("on_train_epoch_end", on_epoch_end)
            results = model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=imgsz,
                project=str(RUNS_DIR),
                name=run_name,
                exist_ok=True,
                patience=epochs,  # don't early-stop on a short toy run
                verbose=True,
            )

        # ── use Ultralytics' own reported save_dir rather than
        # reconstructing the path ourselves — a global Ultralytics settings
        # file can prepend an extra directory (e.g. "runs/detect") to
        # whatever project path we pass in, so a hand-built path can silently
        # miss the real weights file even on a fully successful run. ──
        actual_save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else (RUNS_DIR / run_name)
        weights_dir = actual_save_dir / "weights"
        best_path = weights_dir / "best.pt"
        last_path = weights_dir / "last.pt"

        if not best_path.exists():
            return {"success": False, "error": "Training completed but best.pt was not produced"}

        return {
            "success": True,
            "best_weights": str(best_path),
            "last_weights": str(last_path) if last_path.exists() else None,
            "epochs_run": epochs,
            "run_dir": str(actual_save_dir),
        }
    except Exception as e:
        return {"success": False, "error": f"Training failed: {e}"}


def run_for_job(job_id: int, db_session, TrainingJob, epochs: int = 10):
    """Runs training for a job and updates its DB row with progress + checkpoint path."""
    job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        return

    def _push_stage(name, status, detail=None, progress_current=None, progress_total=None):
        stages = list(job.stages or [])
        entry = {
            "name": name, "status": status, "detail": detail,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        if progress_current is not None:
            entry["progress_current"] = progress_current
        if progress_total is not None:
            entry["progress_total"] = progress_total
        stages.append(entry)
        job.stages = stages
        job.current_stage = name if status == "running" else job.current_stage
        db_session.commit()

    def _on_epoch_end(trainer):
        # ── reads the trainer's own epoch/epochs rather than the closed-over
        # `epochs` arg, since a resumed run's true total is owned by
        # Ultralytics' own state, not whatever we passed in this call. A
        # progress-reporting hiccup here must never take down training
        # itself, so any failure is swallowed. ──
        try:
            current = int(trainer.epoch) + 1  # trainer.epoch is 0-indexed
            total = int(trainer.epochs)
            _push_stage(
                "training", "running",
                f"Training epoch {current} of {total}...",
                progress_current=current, progress_total=total,
            )
        except Exception:
            pass

    _push_stage("training", "running", f"Training YOLO26n for {epochs} epochs on {job.class_name}...",
        progress_current=0, progress_total=epochs)
    result = train_model(job.class_name, epochs=epochs, resume_from=job.checkpoint_path, on_epoch_end=_on_epoch_end)

    if result["success"]:
        job.checkpoint_path = result["last_weights"]
        job.model_path = result["best_weights"]
        job.current_stage = "evaluating"
        _push_stage("training", "done", f"Trained {result['epochs_run']} epochs — weights saved")
    else:
        job.status = "failed"
        job.error = result["error"]
        _push_stage("training", "failed", result["error"])


if __name__ == "__main__":
    import sys
    cls = sys.argv[1] if len(sys.argv) > 1 else "trousers"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print(f"Training {cls} detector for {epochs} epochs (toy run)...")
    result = train_model(cls, epochs=epochs)
    print(json.dumps(result, indent=2))