"""
Self-Learning Pipeline — Step 5: Dataset Preparation

Verifies a downloaded dataset is clean and training-ready:
- every image actually opens (catches corrupted downloads)
- every label file is valid YOLO format (class_id x y w h, values in [0,1])
- converts segmentation-polygon labels (some Roboflow exports return these
  even when yolov8 bbox format is requested) into their enclosing bounding
  box, rather than rejecting them as malformed
- ISOLATES the one class we actually asked for out of a multi-class
  dataset, remapping it to class 0 and discarding every other class's
  labels — a public dataset found by keyword search very often has
  several unrelated categories mixed together (e.g. a "safety hazards"
  dataset with 7 classes when we only asked for one), and training on
  every class at once produces a model that isn't actually a detector
  for what was requested. This was a real, previously-undiscovered gap:
  training ran fine, evaluation reported a clean 0.00 across every metric,
  because precision/recall were being averaged across all classes in the
  source dataset, most of which had too few examples to mean anything.
- if a dataset ships only a train split (no valid/test), carves off a
  portion of train into a real valid split, since Ultralytics' training
  code requires data.yaml's val: path to actually exist on disk
- reports class balance (how many labeled instances exist)
- confirms train/valid/test split is present and non-empty

Does NOT re-split data beyond the train-only fallback above — Roboflow's
own train/valid/test split is otherwise trusted as-is for v1. This step is
a quality gate before spending time training on possibly-broken data, not
a full dataset engineering pipeline (v2: dedup across splits, custom split
ratios, synthetic augmentation).
"""
import json
import shutil
import yaml
from pathlib import Path
from PIL import Image

DATASETS_DIR = Path("datasets")


def _normalize(s: str) -> str:
    return s.lower().replace("_", " ").replace("-", " ").strip()


def _find_target_class_index(names: list, class_name: str) -> int | None:
    """
    Finds which index in the dataset's own class list actually corresponds
    to the class we asked for — e.g. our internal name is
    "exposed_electrical_wire" but the real Roboflow dataset's class is
    literally "Exposed electrical wires" (different case, plural, spaces
    instead of underscores). Uses the same normalize + substring approach
    already proven in data_acquisition.py's relevance filter, checked in
    both directions so either a longer or shorter real class name still
    matches correctly.
    """
    norm_target = _normalize(class_name)
    target_words = norm_target.split()
    for i, name in enumerate(names):
        norm_name = _normalize(str(name))
        if len(target_words) > 1:
            if norm_target in norm_name or norm_name in norm_target:
                return i
        else:
            name_words = set(norm_name.split())
            if norm_target in name_words or norm_name == norm_target:
                return i
    return None


def _check_split(split_dir: Path, target_class_idx: int | None = None) -> dict:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if not images_dir.exists():
        return {"exists": False}

    image_files = list(images_dir.glob("*"))
    corrupted = []
    label_instance_count = 0
    malformed_labels = []
    discarded_other_class = 0

    for img_path in image_files:
        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception:
            corrupted.append(img_path.name)
            continue

        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        converted_lines = []
        for line_num, line in enumerate(label_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()

            # Segmentation polygon (class_id + many x,y pairs) — convert to
            # its enclosing bounding box instead of rejecting it.
            if len(parts) > 5 and (len(parts) - 1) % 2 == 0:
                try:
                    cls_id = int(parts[0])
                    if target_class_idx is not None and cls_id != target_class_idx:
                        discarded_other_class += 1
                        continue
                    coords = list(map(float, parts[1:]))
                    xs = coords[0::2]
                    ys = coords[1::2]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    if not all(0 <= v <= 1 for v in (x_min, x_max, y_min, y_max)):
                        malformed_labels.append(f"{label_path.name}:{line_num} (out of range)")
                        continue
                    x = (x_min + x_max) / 2
                    y = (y_min + y_max) / 2
                    w = x_max - x_min
                    h = y_max - y_min
                    out_cls = 0 if target_class_idx is not None else cls_id
                    converted_lines.append(f"{out_cls} {x} {y} {w} {h}")
                    label_instance_count += 1
                    continue
                except ValueError:
                    malformed_labels.append(f"{label_path.name}:{line_num} (parse error)")
                    continue

            if len(parts) != 5:
                malformed_labels.append(f"{label_path.name}:{line_num}")
                continue
            try:
                cls_id, x, y, w, h = int(parts[0]), *map(float, parts[1:])
                if not all(0 <= v <= 1 for v in (x, y, w, h)):
                    malformed_labels.append(f"{label_path.name}:{line_num} (out of range)")
                    continue
            except ValueError:
                malformed_labels.append(f"{label_path.name}:{line_num} (parse error)")
                continue

            # ── The actual fix: only keep this instance if it's the class
            # we asked for, and renumber it to 0 — we're training a
            # single-class detector, not reproducing the source dataset's
            # full multi-class schema. An image whose only labels belong
            # to OTHER classes ends up with zero kept lines, which is
            # correct: it becomes a valid "no object of interest here"
            # negative example, not something to delete or leave
            # mislabeled. ──
            if target_class_idx is not None:
                if cls_id != target_class_idx:
                    discarded_other_class += 1
                    continue
                cls_id = 0
            converted_lines.append(f"{cls_id} {x} {y} {w} {h}")
            label_instance_count += 1

        # Always rewrite when isolating a target class (even to an empty
        # file — that's a legitimate negative example), and rewrite
        # whenever anything was actually converted, same as before.
        if target_class_idx is not None or converted_lines:
            label_path.write_text(("\n".join(converted_lines) + "\n") if converted_lines else "")

    return {
        "exists": True,
        "total_images": len(image_files),
        "corrupted_images": corrupted,
        "clean_images": len(image_files) - len(corrupted),
        "label_instances": label_instance_count,
        "malformed_labels": malformed_labels,
        "discarded_other_class_instances": discarded_other_class,
    }


def prepare_dataset(class_name: str) -> dict:
    dataset_dir = DATASETS_DIR / class_name
    if not dataset_dir.exists():
        return {"success": False, "error": f"No dataset found at {dataset_dir} — run acquisition first"}

    data_yaml_path = dataset_dir / "data.yaml"
    if not data_yaml_path.exists():
        return {"success": False, "error": f"data.yaml missing from {dataset_dir}"}

    with open(data_yaml_path, "r") as f:
        data_yaml = yaml.safe_load(f)
    original_names = data_yaml.get("names", [])

    # ── The actual fix starts here: figure out which of the source
    # dataset's own classes corresponds to what we asked for. A dataset
    # found by keyword search often has several unrelated categories mixed
    # together — without this, every one of them gets trained and averaged
    # into the final metrics, which is what produced a flat 0.00 on a
    # 7-class safety-hazard dataset when only one class was ever wanted. ──
    target_idx = None
    if isinstance(original_names, list) and len(original_names) > 1:
        target_idx = _find_target_class_index(original_names, class_name)
        if target_idx is None:
            return {
                "success": False,
                "error": (
                    f"Downloaded dataset's classes ({original_names}) don't clearly "
                    f"include '{class_name}' — can't isolate the right labels safely. "
                    f"Try a different search phrasing, or a different dataset."
                ),
            }
        # Rewrite data.yaml to reflect a genuine single-class problem —
        # the model we train should only ever know about this one class,
        # not the source dataset's original multi-class schema.
        data_yaml["names"] = [class_name]
        data_yaml["nc"] = 1
        with open(data_yaml_path, "w") as f:
            yaml.safe_dump(data_yaml, f)
    # If the dataset only ever had one class to begin with, there's
    # nothing to isolate — target_idx stays None and every label is kept
    # as-is, same behavior as before this fix.

    report = {
        "success": True,
        "dataset_path": str(dataset_dir),
        "splits": {},
        "original_classes": original_names,
        "isolated_class_index": target_idx,
    }
    total_corrupted = 0
    total_malformed = 0
    total_instances = 0
    total_discarded_other_class = 0

    for split in ("train", "valid", "test"):
        result = _check_split(dataset_dir / split, target_class_idx=target_idx)
        report["splits"][split] = result
        if result.get("exists"):
            total_corrupted += len(result["corrupted_images"])
            total_malformed += len(result["malformed_labels"])
            total_instances += result["label_instances"]
            total_discarded_other_class += result.get("discarded_other_class_instances", 0)

    # ── Some Roboflow exports ship only a train split, no valid/test —
    # Ultralytics' training code requires data.yaml's val: path to actually
    # exist on disk, so carve off a small portion of train into a real
    # valid folder rather than failing at the training stage. ──
    if not report["splits"].get("valid", {}).get("exists") and report["splits"].get("train", {}).get("exists"):
        train_img_dir = dataset_dir / "train" / "images"
        train_lbl_dir = dataset_dir / "train" / "labels"
        valid_img_dir = dataset_dir / "valid" / "images"
        valid_lbl_dir = dataset_dir / "valid" / "labels"
        all_images = sorted(train_img_dir.glob("*"))
        split_count = max(1, len(all_images) // 5)  # ~20% to validation
        valid_img_dir.mkdir(parents=True, exist_ok=True)
        valid_lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_path in all_images[:split_count]:
            shutil.move(str(img_path), str(valid_img_dir / img_path.name))
            lbl_path = train_lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.move(str(lbl_path), str(valid_lbl_dir / lbl_path.name))
        # re-check both splits now that files have actually moved — labels
        # were already isolated/remapped above, this just re-counts after
        # the move, it does not re-run the class filter a second time.
        report["splits"]["train"] = _check_split(dataset_dir / "train")
        report["splits"]["valid"] = _check_split(dataset_dir / "valid")
        total_instances = (
            report["splits"]["train"]["label_instances"]
            + report["splits"]["valid"]["label_instances"]
        )
        total_corrupted = (
            len(report["splits"]["train"]["corrupted_images"])
            + len(report["splits"]["valid"]["corrupted_images"])
        )
        total_malformed = (
            len(report["splits"]["train"]["malformed_labels"])
            + len(report["splits"]["valid"]["malformed_labels"])
        )

    report["total_label_instances"] = total_instances
    report["total_corrupted_images"] = total_corrupted
    report["total_malformed_labels"] = total_malformed
    report["total_discarded_other_class_instances"] = total_discarded_other_class
    report["ready_for_training"] = (
        report["splits"].get("train", {}).get("clean_images", 0) > 0
        and total_instances > 0
    )
    if not report["ready_for_training"]:
        report["success"] = False
        if target_idx is not None and total_instances == 0:
            report["error"] = (
                f"Found the class '{class_name}' in this dataset's schema, but zero "
                f"actual labeled instances of it after filtering — the other "
                f"{len(original_names) - 1} class(es) had the real label volume. "
                f"This dataset isn't usable for this class alone."
            )
        else:
            report["error"] = "No clean, labeled training images available after verification"

    return report


def run_for_job(job_id: int, db_session, TrainingJob):
    """Runs dataset prep for a training job and updates its DB row."""
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

    _push_stage("preparing_dataset", "running", "Verifying images and labels...")
    result = prepare_dataset(job.class_name)

    if result["success"]:
        existing_info = dict(job.dataset_info or {})
        existing_info["prep_report"] = result
        job.dataset_info = existing_info
        job.current_stage = "training"
        isolated_note = ""
        if result.get("isolated_class_index") is not None:
            isolated_note = (
                f" (isolated from {len(result.get('original_classes', []))} classes in "
                f"the source dataset; {result['total_discarded_other_class_instances']} "
                f"other-class instances discarded)"
            )
        detail = (f"{result['total_label_instances']} labeled instances verified across "
                  f"train/valid/test{isolated_note} — {result['total_corrupted_images']} corrupted, "
                  f"{result['total_malformed_labels']} malformed labels skipped")
        _push_stage("preparing_dataset", "done", detail)
    else:
        job.status = "failed"
        job.error = result.get("error", "Dataset preparation failed")
        _push_stage("preparing_dataset", "failed", result.get("error"))


if __name__ == "__main__":
    import sys
    cls = sys.argv[1] if len(sys.argv) > 1 else "trousers"
    result = prepare_dataset(cls)
    print(json.dumps(result, indent=2))
