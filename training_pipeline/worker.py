"""
Self-Learning Pipeline — Background Worker

Polls for training jobs that need advancing and runs the next stage
automatically. This is what makes the pipeline truly autonomous, once a
rule creates a pending job, this worker picks it up and walks it through
search -> prep -> validate -> train -> evaluate with zero manual intervention.

Run this once, in its own terminal, and leave it running:
    py -3.11 training_pipeline\\worker.py

Routes purely by current_stage, never by status, so re-checking a job that
is mid-stage never re-triggers an earlier stage by mistake.
"""
import sys
import time
sys.path.insert(0, '.')

from db.session import SessionLocal
from db.models import TrainingJob
from training_pipeline.data_acquisition import run_for_job as acquire_step
from training_pipeline.dataset_prep import run_for_job as prep_step
from training_pipeline.validate_samples import run_for_job as validate_step
from training_pipeline.train import run_for_job as train_step
from training_pipeline.evaluate import run_for_job as eval_step

POLL_INTERVAL_SECONDS = 10
EPOCHS = 10

STAGE_HANDLERS = {
    "queued": acquire_step,
    "searching_data": acquire_step,
    "preparing_dataset": prep_step,
    "validating_dataset": validate_step,
    "training": lambda job_id, db, Model: train_step(job_id, db, Model, epochs=EPOCHS),
    "evaluating": eval_step,
}

DONE_STAGES = ("awaiting_approval", "approved")
TERMINAL_STATUSES = ("failed", "cancelled", "approved")


def process_once():
    db = SessionLocal()
    try:
        jobs = db.query(TrainingJob).filter(
            TrainingJob.status.notin_(TERMINAL_STATUSES),
        ).all()

        for job in jobs:
            stage = job.current_stage or "queued"
            if stage in DONE_STAGES:
                continue

            handler = STAGE_HANDLERS.get(stage)
            if not handler:
                print(f"[WORKER] Job {job.id} ({job.class_name}): unknown stage '{stage}', skipping")
                continue

            print(f"[WORKER] Job {job.id} ({job.class_name}): advancing stage '{stage}'...")
            try:
                handler(job.id, db, TrainingJob)
                db.refresh(job)
                print(f"[WORKER] Job {job.id} ({job.class_name}): now at '{job.current_stage}', status '{job.status}'")
            except Exception as e:
                print(f"[WORKER] Job {job.id} ({job.class_name}): stage '{stage}' crashed: {e}")
                job.status = "failed"
                job.error = f"Worker crash during {stage}: {e}"
                db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    print(f"[WORKER] Self-learning background worker started. Polling every {POLL_INTERVAL_SECONDS}s.")
    print(f"[WORKER] Press Ctrl+C to stop.")
    while True:
        try:
            process_once()
        except Exception as e:
            print(f"[WORKER] Poll cycle error: {e}")
        time.sleep(POLL_INTERVAL_SECONDS)