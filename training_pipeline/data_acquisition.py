"""
Self-Learning Pipeline — Step 4: Data Acquisition

Given a class name (e.g. "trousers"), performs a REAL live search against
Roboflow's Universe Search API (GET /universe/search) to find candidate
public datasets, ranks them, and downloads the best match. No hardcoded
class list — any class name the user asks for gets searched for real.
"""
import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

DATASETS_DIR = Path("datasets")
UNIVERSE_SEARCH_URL = "https://api.roboflow.com/universe/search"


def search_universe(class_name: str, min_images: int = 50) -> list[dict]:
    """
    Live search against Roboflow Universe for datasets matching class_name.
    Returns a ranked list of candidates (best first), or [] if none found.
    """
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        return []

    # ── class_name is stored/used elsewhere as snake_case (e.g.
    # "welding_mask"), but Roboflow's search expects natural, space-separated
    # text — searching the literal underscored string returns 0 results even
    # when the equivalent phrase with spaces returns real candidates. Only
    # the outgoing query is affected; the stored class_name is untouched. ──
    search_phrase = class_name.replace("_", " ").replace("-", " ")
    query = f'class:{search_phrase} images>{min_images}'
    try:
        resp = requests.get(
            UNIVERSE_SEARCH_URL,
            params={"q": query, "api_key": api_key, "page": 0},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[SELF-LEARNING] Universe search failed for '{class_name}': {e}")
        return []

    results = data.get("results", [])
    candidates = []
    for r in results:
        workspace_slug = (r.get("workspace") or {}).get("url")
        # Project slug is the last path segment of the dataset URL, e.g.
        # ".../collegeprojects/fashion-8y2pv" -> "fashion-8y2pv"
        dataset_url = r.get("url", "")
        project_slug = dataset_url.rstrip("/").split("/")[-1] if dataset_url else None
        if not workspace_slug or not project_slug:
            continue
        candidates.append({
            "workspace": workspace_slug,
            "project": project_slug,
            "version": r.get("latestVersion", 1),
            "images": r.get("images", 0),
            "stars": r.get("stars", 0),
            "license": r.get("license", "unknown"),
            "url": dataset_url,
            "classes": r.get("classes", []) or [],
        })

    # ── Relevance filter: reject datasets whose class list has nothing to
    # do with what was actually searched. Ranking by image count alone
    # previously let a completely unrelated dataset (e.g. a 41-class "pet
    # monitoring" set) win just because its size happened to be close to
    # our target — real classes are checked here instead, so a class name
    # or a close variant must actually appear before a candidate is ranked. ──
    def _normalize(s: str) -> str:
        return s.lower().replace("_", " ").replace("-", " ").strip()

    def is_relevant(c: dict, term: str) -> bool:
        norm_term = _normalize(term)
        term_words = norm_term.split()
        # Common short words carry no distinguishing signal on their own
        # (e.g. "ear" alone matches "Jelly Ear Mushroom") — require the full
        # multi-word phrase to appear, or the single word itself only when
        # the search term IS a single word.
        for cls in c.get("classes", []):
            norm_cls = _normalize(str(cls))
            if len(term_words) > 1:
                if norm_term in norm_cls:
                    return True
            else:
                cls_words = set(norm_cls.split())
                if norm_term in cls_words:
                    return True
        return False

    relevant = [c for c in candidates if is_relevant(c, class_name)]
    if not relevant:
        # Genuinely nothing with a matching class was found — return
        # nothing rather than silently falling back to an irrelevant
        # dataset, so the caller reports "no dataset found" honestly
        # instead of training on garbage data.
        return []

    # Prefer datasets in a practical size range for fast iteration (a few
    # hundred to a few thousand images) over raw maximum size. A 900-image
    # curated set trains and verifies in minutes; a 100k-image general
    # dataset (e.g. full COCO) can take an hour just to verify on CPU,
    # which defeats the purpose of a fast self-learning proof-of-concept.
    # Closest-to-3000 wins ties broken by stars (community trust signal).
    def rank_score(c):
        images = c["images"]
        size_score = -abs(images - 3000)
        return (size_score, c["stars"])

    relevant.sort(key=rank_score, reverse=True)
    return relevant


def acquire_dataset(class_name: str) -> dict:
    """
    Searches Universe live for class_name, downloads the top-ranked candidate.
    Returns a dict describing the outcome — always, even on failure.
    """
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        return {"success": False, "error": "ROBOFLOW_API_KEY not set in .env"}

    candidates = search_universe(class_name)
    if not candidates:
        return {
            "success": False,
            "error": f"No public Roboflow dataset found for class '{class_name}' "
                     f"(searched live via Universe Search API, 0 results with images>50).",
        }

    last_error = None
    for candidate in candidates[:3]:  # try top 3 in case one fails to download
        ws, proj, ver = candidate["workspace"], candidate["project"], candidate["version"]
        if not ws or not proj:
            continue
        try:
            rf = Roboflow(api_key=api_key)
            project = rf.workspace(ws).project(proj)
            project.version(ver).download("yolov8", location=str(DATASETS_DIR / class_name))
            img_dir = DATASETS_DIR / class_name / "train" / "images"
            image_count = sum(1 for _ in img_dir.glob("*")) if img_dir.exists() else 0
            if image_count == 0:
                last_error = f"{ws}/{proj} downloaded but produced 0 images (likely a partial/failed download)"
                continue
            return {
                "success": True,
                "path": str(DATASETS_DIR / class_name),
                "source": f"{ws}/{proj} v{ver}",
                "image_count": image_count,
                "license": candidate.get("license"),
                "candidates_considered": len(candidates),
            }
        except Exception as e:
            last_error = str(e)
            continue

    return {
        "success": False,
        "error": f"Found {len(candidates)} candidate dataset(s) for '{class_name}' but all failed to download. Last error: {last_error}",
    }


def run_for_job(job_id: int, db_session, TrainingJob):
    """Runs acquisition for a training job and updates its DB row with progress."""
    from datetime import datetime, timezone
    import threading
    import time

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

    _push_stage("searching_data", "running", f"Searching Roboflow Universe for '{job.class_name}'...")

    candidates = search_universe(job.class_name)
    if not candidates:
        job.status = "failed"
        job.error = (
            f"No public Roboflow dataset found for class '{job.class_name}' "
            f"(searched live via Universe Search API, 0 results with images>50)."
        )
        _push_stage("searching_data", "failed", job.error)
        return

    top = candidates[0]
    target_total = top.get("images", 0) or None
    img_dir = DATASETS_DIR / job.class_name / "train" / "images"
    source_label = f"{top['workspace']}/{top['project']}"

    # ── the actual download call below blocks for its full duration with no
    # progress hook exposed by the Roboflow SDK, so a separate thread polls
    # the destination folder's growing file count instead. It uses its own
    # DB session (not job's/db_session, which belongs to the main thread) so
    # the two never touch the same SQLAlchemy Session concurrently. ──
    stop_flag = threading.Event()

    def _poll_download_progress():
        from db.session import SessionLocal
        poll_db = SessionLocal()
        try:
            while not stop_flag.wait(1.5):
                try:
                    count = sum(1 for _ in img_dir.glob("*")) if img_dir.exists() else 0
                    if count > 0:
                        j = poll_db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
                        if not j:
                            continue
                        total_label = f" of {target_total}" if target_total else ""
                        stages = list(j.stages or [])
                        stages.append({
                            "name": "searching_data", "status": "running",
                            "detail": f"Downloading {count}{total_label} images from {source_label}...",
                            "progress_current": count,
                            "progress_total": target_total,
                            "finished_at": datetime.now(timezone.utc).isoformat(),
                        })
                        j.stages = stages
                        poll_db.commit()
                except Exception:
                    poll_db.rollback()
        finally:
            poll_db.close()

    poller = threading.Thread(target=_poll_download_progress, daemon=True)
    poller.start()
    try:
        result = acquire_dataset(job.class_name)
    finally:
        stop_flag.set()
        poller.join(timeout=3)

    # ── the poller may have committed its own last snapshot of `job.stages`
    # from its separate session — refresh so this thread's next commit
    # doesn't silently overwrite that history with a stale in-memory copy ──
    db_session.refresh(job)

    if result["success"]:
        job.dataset_info = result
        job.current_stage = "preparing_dataset"
        _push_stage("searching_data", "done", f"Found {result['image_count']} images from {result['source']}")
    else:
        job.status = "failed"
        job.error = result["error"]
        _push_stage("searching_data", "failed", result["error"])


if __name__ == "__main__":
    import sys
    cls = sys.argv[1] if len(sys.argv) > 1 else "trousers"
    print(f"Searching live for: {cls}")
    candidates = search_universe(cls)
    print(f"Found {len(candidates)} candidates:")
    for c in candidates[:5]:
        print(f"  - {c['workspace']}/{c['project']} v{c['version']} · {c['images']} images · {c['stars']} stars")
    print()
    result = acquire_dataset(cls)
    print(json.dumps(result, indent=2))