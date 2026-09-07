"""
One-off script: downloads the SPECIFIC, already-confirmed-correct dataset
(meralco/quakesafe-fddoq) directly, bypassing the live Universe search —
which has proven unreliable tonight (found this exact dataset once, then
returned 0 results on an identical search minutes later).

We already manually confirmed this dataset's class list includes a clear,
strong match: "Exposed electrical wires", from an actual electric utility
company's safety-inspection dataset. No need to gamble on the search
finding it again.

Creates the training_jobs row already past "searching_data", so the
background worker picks it up starting at "preparing_dataset" — which
now runs the FIXED dataset_prep.py, correctly isolating just the wire
class out of this dataset's other 6 hazard categories.

Usage:
    py -3.11 direct_acquire_wire.py
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, '.')
load_dotenv()

from roboflow import Roboflow
from db.session import SessionLocal
from db.models import TrainingJob

CLASS_NAME = "exposed_electrical_wire"   # must match the rule's actual target exactly
WORKSPACE = "meralco"
PROJECT = "quakesafe-fddoq"
VERSION = 7                              # confirmed as the version in our manual check
SITE_ID = 3

DATASETS_DIR = Path("datasets")


def main():
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("[ERROR] ROBOFLOW_API_KEY not set in .env — cannot download.")
        return

    dest = DATASETS_DIR / CLASS_NAME
    print(f"[INFO] Downloading {WORKSPACE}/{PROJECT} v{VERSION} -> {dest} ...")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    project.version(VERSION).download("yolov8", location=str(dest))

    img_dir = dest / "train" / "images"
    image_count = sum(1 for _ in img_dir.glob("*")) if img_dir.exists() else 0

    if image_count == 0:
        print(f"[ERROR] Download completed but 0 images found at {img_dir}. "
              f"Check the workspace/project/version values are correct.")
        return

    print(f"[OK] Downloaded {image_count} images.")

    dataset_info = {
        "success": True,
        "path": str(dest),
        "source": f"{WORKSPACE}/{PROJECT} v{VERSION}",
        "image_count": image_count,
        "license": "Public Domain",
        "candidates_considered": 1,
    }

    db = SessionLocal()
    try:
        job = TrainingJob(
            site_id=SITE_ID,
            class_name=CLASS_NAME,
            status="pending",
            current_stage="preparing_dataset",
            stages=[{
                "name": "searching_data",
                "status": "done",
                "detail": f"Found images from {WORKSPACE}/{PROJECT} v{VERSION} (direct acquisition, bypassed unreliable live search)",
            }],
            dataset_info=dataset_info,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        print(f"[OK] Created training_jobs row id={job.id}, class_name='{CLASS_NAME}', "
              f"current_stage='preparing_dataset'. The background worker should "
              f"pick this up on its next poll (within ~10s), and this time will "
              f"run the FIXED dataset_prep.py that isolates just the wire class.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
