from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, Response
from sqlalchemy.orm import Session
from pydantic import BaseModel
import json
import os
import requests
import cv2
import numpy as np
import threading
import time
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta
from collections import defaultdict
import csv
import io

from db.session import get_db, SessionLocal
from db.models import User, Rule, Incident, Camera as CameraModel, Site, Zone, Setting, RuleCamera
from auth.password import hash_password, verify_password
from auth.jwt_handler import create_token
from auth.dependencies import get_current_user
from db.seed import seed
from sqlalchemy import func

load_dotenv()

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")

app = FastAPI(title="ONVXP Service API")

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("incidents", exist_ok=True)
app.mount("/screenshots", StaticFiles(directory="incidents"), name="screenshots")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


RECONNECT_INTERVAL_SECONDS = 5
CONSECUTIVE_FAILURE_THRESHOLD = 10


def _parse_source(src):
    if src is None:
        return "mega_cctv_v2.mp4"
    try:
        return int(src)
    except (ValueError, TypeError):
        return str(src)


VIDEO_SOURCE_DEFAULT = _parse_source(os.getenv("VIDEO_SOURCE", "mega_cctv_v2.mp4"))

DEFAULT_SETTINGS = {
    "detection": {"alert_cooldown_frames": 150, "detection_confidence": 0.5, "bytetrack_buffer": 30, "persistence_frames": 5},
    "alerts": {"channels": "dashboard", "deduplication_enabled": True, "email_notifications_enabled": False, "email_severity_threshold": "high"},
    "ai_model": {"frame_sampling": "every", "model_precision": "balanced"},
    "platform": {"llm_model": "claude-haiku", "site_name": "Site A — Construction", "api_endpoint": PUBLIC_BASE_URL},
}


def get_settings_for_site(db: Session, site_id: int) -> dict:
    settings = json.loads(json.dumps(DEFAULT_SETTINGS))
    rows = db.query(Setting).filter(Setting.site_id == site_id, Setting.user_id.is_(None)).all()
    for row in rows:
        if row.key in settings and isinstance(row.value, dict):
            settings[row.key].update(row.value)
    return settings


def save_settings_for_site(db: Session, site_id: int, updates: dict):
    for section in ("detection", "alerts", "ai_model", "platform"):
        if section not in updates:
            continue
        existing_row = db.query(Setting).filter(
            Setting.site_id == site_id, Setting.key == section, Setting.user_id.is_(None)
        ).first()
        current = get_settings_for_site(db, site_id)[section]
        current.update(updates[section])
        if existing_row:
            existing_row.value = current
        else:
            db.add(Setting(site_id=site_id, user_id=None, key=section, value=current))
    db.commit()


_pipeline_process = None


class VideoStream:
    def __init__(self):
        self.cap = None
        self.lock = threading.Lock()
        self.running = False
        self.current_frame = None
        self.fps = 0
        self.width = 0
        self.height = 0
        self.source = None
        self.is_live = False
        self._consecutive_failures = 0

    def start(self, source):
        with self.lock:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                self.cap = None
                self.running = False
                return False
            self.source = str(source)
            self.is_live = isinstance(source, int) or (isinstance(source, str) and source.startswith("rtsp://"))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.running = True
            self._consecutive_failures = 0
        return True

    def read(self):
        with self.lock:
            if not self.cap or not self.running:
                return None
            ret, frame = self.cap.read()
            if not ret:
                if self.is_live:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= CONSECUTIVE_FAILURE_THRESHOLD:
                        print(f"[OMNIX] Stream dead after {self._consecutive_failures} consecutive failed reads: {self.source}")
                        self.running = False
                        if self.cap:
                            self.cap.release()
                            self.cap = None
                    return None
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self._consecutive_failures = 0
                return frame
            return None

    def stop(self):
        with self.lock:
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None


video_streams: dict[int, VideoStream] = {}
_video_streams_lock = threading.Lock()


def _camera_source_for(camera_id: int, db: Session):
    cam = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
    if cam and cam.source and cam.source != "default":
        return _parse_source(cam.source)
    if camera_id == 1:
        return VIDEO_SOURCE_DEFAULT
    return None


def get_or_start_stream(camera_id: int, db: Session):
    with _video_streams_lock:
        vs = video_streams.get(camera_id)
        if vs is not None and vs.running:
            return vs
        source = _camera_source_for(camera_id, db)
        if source is None:
            return None
        vs = VideoStream()
        if not vs.start(source):
            return None
        video_streams[camera_id] = vs
        return vs


def restart_stream(camera_id: int, db: Session):
    with _video_streams_lock:
        old = video_streams.pop(camera_id, None)
        if old:
            old.stop()
    return get_or_start_stream(camera_id, db)


def _reconnect_monitor():
    while True:
        time.sleep(RECONNECT_INTERVAL_SECONDS)
        db = SessionLocal()
        try:
            with _video_streams_lock:
                dead_camera_ids = [cid for cid, vs in video_streams.items() if not vs.running]
            for cid in dead_camera_ids:
                cam = db.query(CameraModel).filter(CameraModel.id == cid).first()
                if cam:
                    cam.status = "offline"
                    db.commit()
                print(f"[OMNIX] Attempting reconnect for camera {cid}...")
                vs = get_or_start_stream(cid, db)
                if vs:
                    print(f"[OMNIX] Camera {cid} reconnected successfully")
                    if cam:
                        cam.status = "online"
                        db.commit()
                else:
                    print(f"[OMNIX] Camera {cid} reconnect failed, retrying in {RECONNECT_INTERVAL_SECONDS}s")
        except Exception as e:
            print(f"[OMNIX] Reconnect monitor error: {e}")
        finally:
            db.close()


def _self_learning_worker_loop():
    """
    Self-learning: background worker that advances unknown-class training jobs
    automatically. Started from on_startup so it runs the moment the backend
    is up, no manual worker.py step required from anyone.
    """
    from db.models import TrainingJob
    from training_pipeline.data_acquisition import run_for_job as acquire_step
    from training_pipeline.dataset_prep import run_for_job as prep_step
    from training_pipeline.train import run_for_job as train_step
    from training_pipeline.evaluate import run_for_job as eval_step

    STAGE_HANDLERS = {
        "queued": acquire_step,
        "searching_data": acquire_step,
        "preparing_dataset": prep_step,
        "training": lambda jid, dbs, Model: train_step(jid, dbs, Model, epochs=10),
        "evaluating": eval_step,
    }
    DONE_STAGES = ("awaiting_approval", "approved")
    TERMINAL_STATUSES = ("failed", "cancelled", "approved")

    while True:
        db2 = SessionLocal()
        try:
            jobs = db2.query(TrainingJob).filter(
                TrainingJob.status.notin_(TERMINAL_STATUSES),
            ).all()
            for job in jobs:
                stage = job.current_stage or "queued"
                if stage in DONE_STAGES:
                    continue
                handler = STAGE_HANDLERS.get(stage)
                if not handler:
                    continue
                print(f"[SELF-LEARNING] Job {job.id} ({job.class_name}): advancing stage '{stage}'...")
                try:
                    handler(job.id, db2, TrainingJob)
                    db2.refresh(job)
                    print(f"[SELF-LEARNING] Job {job.id} ({job.class_name}): now at '{job.current_stage}', status '{job.status}'")
                except Exception as e:
                    print(f"[SELF-LEARNING] Job {job.id} ({job.class_name}): stage '{stage}' crashed: {e}")
                    job.status = "failed"
                    job.error = f"Worker crash during {stage}: {e}"
                    db2.commit()
        except Exception as e:
            print(f"[SELF-LEARNING] Worker poll cycle error: {e}")
        finally:
            db2.close()
        time.sleep(10)


@app.on_event("startup")
def on_startup():
    seed()
    db = SessionLocal()
    try:
        for cam in db.query(CameraModel).all():
            vs = get_or_start_stream(cam.id, db)
            cam.status = "online" if vs else "offline"
            print(f"[OMNIX] Camera {cam.id} ({cam.name}): {'started' if vs else 'offline'} — source={cam.source}")
        db.commit()

        config_path = Path("pipeline_config.json")
        if not config_path.exists():
            site = db.query(Site).first()
            if site:
                rebuilt = rebuild_pipeline_config_from_db(db, site.id)
                if rebuilt.get("rules"):
                    with open(config_path, "w") as f:
                        json.dump(rebuilt, f, indent=2)
                    print(f"[OMNIX] pipeline_config.json was missing — rebuilt from {len(rebuilt['rules'])} active DB rule(s)")
                else:
                    print(f"[OMNIX] pipeline_config.json missing, and no active DB rules to rebuild from")
    finally:
        db.close()
    threading.Thread(target=_reconnect_monitor, daemon=True).start()
    print(f"[OMNIX] Reconnect monitor started (checks every {RECONNECT_INTERVAL_SECONDS}s)")

    threading.Thread(target=_self_learning_worker_loop, daemon=True).start()
    print("[OMNIX] Self-learning background worker started (checks every 10s)")


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str = ""
    site_name: str = "My Site"


@app.post("/api/auth/login")
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_token(user.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user.id, "email": user.email, "name": user.name, "role": user.role}
    }


@app.post("/api/auth/register")
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == body.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    site = Site(name=body.site_name, location="")
    db.add(site)
    db.flush()
    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        name=body.name,
        role="admin",
        site_id=site.id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_token(user.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user.id, "email": user.email, "name": user.name, "role": user.role, "site_id": user.site_id}
    }


@app.get("/api/auth/me")
def me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "name": current_user.name,
        "role": current_user.role,
        "site_id": current_user.site_id,
    }


@app.post("/api/users/invite")
async def invite_user(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can invite users")
    body = await request.json()
    email = body.get("email", "").strip()
    name = body.get("name", "").strip()
    if not email:
        raise HTTPException(status_code=400, detail="email is required")
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    viewer = User(
        email=email,
        password_hash=hash_password("changeme123"),
        name=name or email.split("@")[0],
        role="viewer",
        site_id=current_user.site_id,
    )
    db.add(viewer)
    db.commit()
    db.refresh(viewer)
    return {
        "status": "invited",
        "user": {"id": viewer.id, "email": viewer.email, "name": viewer.name, "role": viewer.role}
    }


@app.get("/api/users")
def get_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can view team members")
    users = db.query(User).filter(User.site_id == current_user.site_id).all()
    return [
        {
            "id": u.id,
            "email": u.email,
            "name": u.name,
            "role": u.role,
            "created_at": u.created_at.isoformat() if u.created_at else None,
        }
        for u in users
    ]


@app.get("/api/zones")
def get_zones(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    zones = db.query(Zone).filter(Zone.site_id == current_user.site_id).order_by(Zone.name).all()
    return [
        {
            "id": z.id,
            "name": z.name,
            "created_at": z.created_at.isoformat() if z.created_at else None,
        }
        for z in zones
    ]


@app.post("/api/zones")
async def create_zone(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Viewers cannot create zones")
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    zone = Zone(site_id=current_user.site_id, name=name)
    db.add(zone)
    db.commit()
    db.refresh(zone)
    return {
        "id": zone.id,
        "name": zone.name,
        "created_at": zone.created_at.isoformat() if zone.created_at else None,
    }


@app.put("/api/zones/{zone_id}")
async def update_zone(
    zone_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Viewers cannot edit zones")
    zone = db.query(Zone).filter(
        Zone.id == zone_id,
        Zone.site_id == current_user.site_id
    ).first()
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    body = await request.json()
    if "name" in body:
        name = body["name"].strip()
        if not name:
            raise HTTPException(status_code=400, detail="name cannot be empty")
        zone.name = name
    db.commit()
    db.refresh(zone)
    return {"id": zone.id, "name": zone.name}


@app.delete("/api/zones/{zone_id}")
def delete_zone(
    zone_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Viewers cannot delete zones")
    zone = db.query(Zone).filter(
        Zone.id == zone_id,
        Zone.site_id == current_user.site_id
    ).first()
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    # Cameras in this zone become zone-less rather than being deleted too —
    # deleting a building shouldn't delete the cameras inside it.
    db.query(CameraModel).filter(CameraModel.zone_id == zone_id).update({"zone_id": None})
    db.delete(zone)
    db.commit()
    return {"status": "deleted", "zone_id": zone_id}


@app.get("/")
def root():
    return {"status": "ONVXP Service API running", "llm_provider": "ollama", "model": OLLAMA_MODEL}


@app.get("/api/incidents")
def get_incidents(
    limit: int = 50,
    offset: int = 0,
    rule_id: int = None,
    camera_id: int = None,
    violation: str = None,
    severity: str = None,
    review: str = None,
    date_from: str = None,
    date_to: str = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    try:
        base_q = db.query(Incident).filter(Incident.site_id == current_user.site_id)
        if rule_id is not None:
            base_q = base_q.filter(Incident.rule_id == rule_id)
        if camera_id is not None:
            base_q = base_q.filter(Incident.camera_id == camera_id)
        if violation:
            base_q = base_q.filter(Incident.violation_type == violation)
        if severity:
            base_q = base_q.filter(Incident.severity == severity.lower())
        if review == "unreviewed":
            base_q = base_q.filter(Incident.reviewed.isnot(True))
        elif review in ("reviewed", "false_positive", "dismissed"):
            base_q = base_q.filter(Incident.review_status == review)
        if date_from:
            try:
                base_q = base_q.filter(Incident.timestamp >= datetime.fromisoformat(date_from))
            except ValueError:
                pass
        if date_to:
            try:
                end = datetime.fromisoformat(date_to) + timedelta(days=1)
                base_q = base_q.filter(Incident.timestamp < end)
            except ValueError:
                pass
        total = base_q.count()
        if total >= 0:
            incidents = (
                base_q.order_by(Incident.timestamp.desc())
                .offset(offset).limit(limit).all()
            )
            rule_ids = {inc.rule_id for inc in incidents if inc.rule_id}
            rule_map = {}
            if rule_ids:
                for r in db.query(Rule).filter(Rule.id.in_(rule_ids)).all():
                    rule_map[r.id] = r.instruction
            result = []
            for inc in incidents:
                d = {
                    "id": inc.id,
                    "rule_id": inc.rule_id,
                    "rule_instruction": rule_map.get(inc.rule_id),
                    "timestamp": inc.timestamp.isoformat() if inc.timestamp else None,
                    "violation": inc.violation_type,
                    "zone": inc.zone,
                    "severity": inc.severity,
                    "alert_message": inc.alert_message,
                    "person_id": inc.person_track_id,
                    "frame": inc.frame_number,
                    "bbox": inc.bbox,
                    "detected_objects": inc.detected_objects,
                    "missing_gear": inc.missing_gear,
                    "reviewed": inc.reviewed,
                    "review_status": inc.review_status,
                    "camera_id": inc.camera_id,
                }
                if inc.screenshot_path:
                    filename = inc.screenshot_path.replace("incidents/", "")
                    d["screenshot_url"] = f"{PUBLIC_BASE_URL}/screenshots/{filename}"
                result.append(d)
            return {"items": result, "total": total, "limit": limit, "offset": offset}
    except Exception:
        pass
    incidents_file = Path("incidents.json")
    if not incidents_file.exists():
        return {"items": [], "total": 0, "limit": limit, "offset": offset}
    with open(incidents_file, "r") as f:
        incidents = json.load(f)
    for inc in incidents:
        if "screenshot_path" in inc:
            filename = inc["screenshot_path"].replace("incidents/", "")
            inc["screenshot_url"] = f"{PUBLIC_BASE_URL}/screenshots/{filename}"
    total = len(incidents)
    return {"items": incidents[offset:offset + limit], "total": total, "limit": limit, "offset": offset}


ALLOWED_REVIEW_STATUSES = {"reviewed", "false_positive", "dismissed"}

@app.post("/api/incidents/{incident_id}/review")
async def review_incident(
    incident_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    incident = db.query(Incident).filter(
        Incident.id == incident_id,
        Incident.site_id == current_user.site_id,
    ).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    try:
        body = await request.json()
    except Exception:
        body = {}
    review_status = str(body.get("review_status") or "reviewed").strip()
    if review_status not in ALLOWED_REVIEW_STATUSES:
        raise HTTPException(status_code=400, detail=f"review_status must be one of: {', '.join(sorted(ALLOWED_REVIEW_STATUSES))}")
    incident.reviewed = True
    incident.review_status = review_status
    incident.reviewed_by = current_user.id
    incident.reviewed_at = datetime.utcnow()
    db.commit()
    return {"status": "ok", "incident_id": incident_id, "review_status": review_status}


@app.get("/api/incidents/{incident_id}/objects")
def get_incident_objects(
    incident_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    incident = db.query(Incident).filter(
        Incident.id == incident_id,
        Incident.site_id == current_user.site_id,
    ).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    def _bbox_close(a, b, tol=0.6) -> bool:
        if not a or not b or len(a) < 4 or len(b) < 4:
            return False
        return all(abs(a[i] - b[i]) <= tol for i in range(4))

    raw_objects = incident.detected_objects or []
    objects = []
    for obj in raw_objects:
        if not obj or "bbox" not in obj:
            continue
        if incident.person_track_id is not None:
            is_violator = (
                obj.get("type") == "person"
                and obj.get("track_id") == incident.person_track_id
            )
        else:
            is_violator = _bbox_close(obj.get("bbox"), incident.bbox)
        objects.append({
            "type": obj.get("type"),
            "track_id": obj.get("track_id"),
            "bbox": obj.get("bbox"),
            "confidence": obj.get("confidence"),
            "is_violator": is_violator,
        })

    return {
        "incident_id": incident_id,
        "objects": objects,
        "violator_bbox": incident.bbox,
        "person_track_id": incident.person_track_id,
    }


# ── Incident Inspector map view: every zone polygon (already normalized 0-1
# relative to each camera's own frame) plus a normalized center-point for
# every incident's bbox, so the frontend can draw both in the same 2D space
# without needing to know about camera resolution at all. ──
@app.get("/api/incidents/map")
def get_incidents_map(
    limit: int = 200,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    zones = db.query(Zone).filter(Zone.site_id == current_user.site_id).all()
    zone_list = [
        {
            "id": z.id,
            "name": z.name,
        }
        for z in zones
    ]

    incidents = (
        db.query(Incident)
        .filter(Incident.site_id == current_user.site_id, Incident.bbox.isnot(None))
        .order_by(Incident.timestamp.desc())
        .limit(limit)
        .all()
    )

    incident_list = []
    for inc in incidents:
        if not inc.bbox or len(inc.bbox) < 4:
            continue
        cam_id = inc.camera_id or 1
        # ── prefer a camera's real live resolution (tracked by its running
        # VideoStream) over the DB's resolution field, which isn't always
        # populated; fall back to 854x480 — the same canvas zone polygons
        # are already normalized against — if neither is available. ──
        vs = video_streams.get(cam_id)
        if vs and vs.width and vs.height:
            res_w, res_h = vs.width, vs.height
        else:
            res_w, res_h = 854, 480
        x1, y1, x2, y2 = inc.bbox[:4]
        cx = ((x1 + x2) / 2) / res_w
        cy = ((y1 + y2) / 2) / res_h
        incident_list.append({
            "id": inc.id,
            "camera_id": cam_id,
            "zone": inc.zone,
            "severity": inc.severity,
            "violation": inc.violation_type,
            "timestamp": inc.timestamp.isoformat() if inc.timestamp else None,
            "x": round(max(0, min(1, cx)), 4),
            "y": round(max(0, min(1, cy)), 4),
        })

    return {"zones": zone_list, "incidents": incident_list}


@app.get("/api/training-jobs")
def list_training_jobs(
    page: int = 1,
    page_size: int = 20,
    status: str = None,
    search: str = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    from db.models import TrainingJob

    page = max(1, page)
    page_size = max(1, min(page_size, 100))

    q = db.query(TrainingJob).filter(
        TrainingJob.site_id == current_user.site_id,
        TrainingJob.status != "cancelled",
    )

    if search:
        q = q.filter(TrainingJob.class_name.ilike(f"%{search.strip()}%"))

    if status == "in_progress":
        q = q.filter(
            TrainingJob.status.notin_(["approved", "failed"]),
            TrainingJob.current_stage != "awaiting_approval",
        )
    elif status == "awaiting_approval":
        q = q.filter(
            TrainingJob.current_stage == "awaiting_approval",
            TrainingJob.status.notin_(["approved", "failed"]),
        )
    elif status == "approved":
        q = q.filter(TrainingJob.status == "approved")
    elif status == "failed":
        q = q.filter(TrainingJob.status == "failed")
    # status is None or "all" -> no extra filter beyond excluding cancelled above

    total = q.count()
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(page, total_pages)

    jobs = (
        q.order_by(TrainingJob.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return {
        "items": [
            {
                "id": j.id,
                "class_name": j.class_name,
                "status": j.status,
                "current_stage": j.current_stage,
                "stages": j.stages,
                "metrics": j.metrics,
                "error": j.error,
                "created_at": j.created_at.isoformat() if j.created_at else None,
                "updated_at": j.updated_at.isoformat() if j.updated_at else None,
            }
            for j in jobs
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


@app.get("/api/training-jobs/{job_id}")
def get_training_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    from db.models import TrainingJob
    job = db.query(TrainingJob).filter(
        TrainingJob.id == job_id,
        TrainingJob.site_id == current_user.site_id
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return {
        "id": job.id,
        "class_name": job.class_name,
        "status": job.status,
        "current_stage": job.current_stage,
        "stages": job.stages,
        "dataset_info": job.dataset_info,
        "metrics": job.metrics,
        "model_path": job.model_path,
        "error": job.error,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }

@app.post("/api/training-jobs/{job_id}/approve")
def approve_training_job(
    job_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can approve model training")
    from db.models import TrainingJob
    job = db.query(TrainingJob).filter(
        TrainingJob.id == job_id,
        TrainingJob.site_id == current_user.site_id
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    if job.current_stage != "awaiting_approval":
        raise HTTPException(status_code=400, detail=f"Job is at stage '{job.current_stage}', not ready for approval")

    with open('model_registry.json', 'r') as f:
        registry = json.load(f)
    registry[job.class_name] = {
        "type": "custom",
        "weights": job.model_path,
        "confidence": job.metrics.get("precision", 0.5) if job.metrics else 0.5,
        "conf_threshold": 0.5,
    }
    with open('model_registry.json', 'w') as f:
        json.dump(registry, f, indent=2)

    job.status = "approved"
    stages = list(job.stages or [])
    stages.append({"name": "approved", "status": "done", "detail": f"Approved by {current_user.name or current_user.email}"})
    job.stages = stages
    job.current_stage = "approved"
    db.commit()

    # If a rule was created while this class was still training, it was
    # saved as "pending_training" rather than active or inactive. A rule
    # can depend on more than one missing class, so it only goes live once
    # every job pointing at it is approved — not just this one.
    activated_rule = None
    if job.rule_id:
        pending_rule = db.query(Rule).filter(Rule.id == job.rule_id).first()
        if pending_rule and pending_rule.status == "pending_training":
            sibling_jobs = db.query(TrainingJob).filter(TrainingJob.rule_id == job.rule_id).all()
            if all(j.status == "approved" for j in sibling_jobs):
                pending_rule.status = "active"
                db.commit()
                background_tasks.add_task(_rebuild_and_restart_pipeline_bg, pending_rule.site_id)
                activated_rule = {"id": pending_rule.id, "instruction": pending_rule.instruction}

    return {
        "status": "approved",
        "class_name": job.class_name,
        "registered_weights": job.model_path,
        "message": f"'{job.class_name}' is now a live detection capability.",
        "activated_rule": activated_rule,
    }


@app.post("/api/training-jobs/{job_id}/reject")
def reject_training_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can reject model training")
    from db.models import TrainingJob
    job = db.query(TrainingJob).filter(
        TrainingJob.id == job_id,
        TrainingJob.site_id == current_user.site_id
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    job.status = "cancelled"
    stages = list(job.stages or [])
    stages.append({"name": "rejected", "status": "done", "detail": f"Rejected by {current_user.name or current_user.email}"})
    job.stages = stages
    db.commit()

    # A rejected job's dependent rule (if any) would otherwise be stuck in
    # pending_training forever — invisible, with no path forward. Surface
    # it as inactive instead, so it's at least visible and manageable, even
    # though the specific reason it never went live isn't recorded anywhere.
    orphaned_rule = None
    if job.rule_id:
        pending_rule = db.query(Rule).filter(Rule.id == job.rule_id).first()
        if pending_rule and pending_rule.status == "pending_training":
            pending_rule.status = "inactive"
            db.commit()
            orphaned_rule = {"id": pending_rule.id, "instruction": pending_rule.instruction}

    return {"status": "rejected", "class_name": job.class_name, "orphaned_rule": orphaned_rule}

@app.post("/api/pipeline/rebuild")
def rebuild_pipeline_config(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can rebuild the pipeline config")
    rebuilt = rebuild_pipeline_config_from_db(db, current_user.site_id)
    config_path = Path("pipeline_config.json")
    with open(config_path, "w") as f:
        json.dump(rebuilt, f, indent=2)
    return {
        "status": "rebuilt",
        "total_rules": len(rebuilt.get("rules", [])),
        "total_zones": len(rebuilt.get("zones", [])),
        "pipeline_id": rebuilt.get("pipeline_id"),
    }


@app.get("/api/pipeline")
def get_pipeline(current_user: User = Depends(get_current_user)):
    with open("pipeline_config.json", "r") as f:
        return json.load(f)


@app.get("/api/stats")
def get_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        incidents = db.query(Incident).filter(Incident.site_id == current_user.site_id).all()
        if incidents:
            zones = list(set(i.zone for i in incidents if i.zone))
            persons = list(set(i.person_track_id for i in incidents if i.person_track_id))
            return {"total": len(incidents), "unique_persons": len(persons), "zones_affected": zones}
    except Exception:
        pass
    incidents_file = Path("incidents.json")
    if not incidents_file.exists():
        return {"total": 0, "unique_persons": 0, "zones_affected": []}
    with open(incidents_file, "r") as f:
        incidents = json.load(f)
    return {
        "total": len(incidents),
        "unique_persons": len(set(i["person_id"] for i in incidents)),
        "zones_affected": list(set(i["zone"] for i in incidents))
    }


@app.get("/api/rules")
def get_rules(
    all: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    q = db.query(Rule).filter(Rule.site_id == current_user.site_id)
    if not all:
        q = q.filter(Rule.status == "active")
    rules = q.order_by(Rule.created_at.desc()).all()
    counts = dict(
        db.query(Incident.rule_id, func.count(Incident.id))
        .filter(Incident.site_id == current_user.site_id)
        .group_by(Incident.rule_id).all()
    )

    rule_ids = [r.id for r in rules]
    cameras_by_rule: dict = {}
    if rule_ids:
        zone_names = {z.id: z.name for z in db.query(Zone).filter(Zone.site_id == current_user.site_id).all()}
        rows = (
            db.query(RuleCamera, CameraModel)
            .join(CameraModel, CameraModel.id == RuleCamera.camera_id)
            .filter(RuleCamera.rule_id.in_(rule_ids))
            .all()
        )
        for rc, cam in rows:
            cameras_by_rule.setdefault(rc.rule_id, []).append({
                "id": cam.id,
                "name": cam.name,
                "zone_id": cam.zone_id,
                "zone_name": zone_names.get(cam.zone_id),
            })

    # A pending_training rule doesn't know which training job it's waiting
    # on from its own row alone — TrainingJob.rule_id points the other way.
    # Looked up here so the frontend can let clicking such a rule navigate
    # straight to that job's detail page.
    training_job_by_rule: dict = {}
    if rule_ids:
        from db.models import TrainingJob
        for job in db.query(TrainingJob).filter(TrainingJob.rule_id.in_(rule_ids)).all():
            training_job_by_rule.setdefault(job.rule_id, job.id)

    return [
        {
            "id": r.id,
            "instruction": r.instruction,
            "config_json": r.config_json,
            "pipeline_id": r.pipeline_id,
            "status": r.status,
            "severity": r.severity,
            "cameras": cameras_by_rule.get(r.id, []),
            "training_job_id": training_job_by_rule.get(r.id),
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "incident_count": counts.get(r.id, 0),
        }
        for r in rules
    ]


@app.delete("/api/rules/{rule_id}")
def delete_rule(
    rule_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Viewers cannot delete rules")
    rule = db.query(Rule).filter(
        Rule.id == rule_id,
        Rule.site_id == current_user.site_id
    ).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    rule.status = "deleted"
    db.commit()
    return {"status": "deleted", "rule_id": rule_id}


def _rebuild_and_restart_pipeline_for_toggle(db: Session, site_id: int):
    """
    Shared by activate/deactivate below. Rewrites pipeline_config.json from
    the DB's current active rules, then restarts the live run_pipeline.py
    subprocess so the change actually takes effect immediately — mirroring
    apply_rule's existing restart behavior, since the running pipeline only
    reads its config at startup, not on a live-reload basis. Defaults to
    camera_id=1 on restart, matching apply_rule's own existing default when
    no specific camera is given.
    """
    global _pipeline_process
    rebuilt = rebuild_pipeline_config_from_db(db, site_id)
    config_path = Path("pipeline_config.json")
    with open(config_path, "w") as f:
        json.dump(rebuilt, f, indent=2)
    if _pipeline_process is not None and _pipeline_process.poll() is None:
        _pipeline_process.terminate()
        try:
            _pipeline_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _pipeline_process.kill()
    _pipeline_process = subprocess.Popen(
        [sys.executable, "run_pipeline.py", "--camera_id", "1"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )


def _rebuild_and_restart_pipeline_bg(site_id: int):
    """
    Background-task version of the above — the request's own DB session
    gets closed once the response is sent, and background tasks run *after*
    that, so this opens a fresh session of its own (same pattern already
    used by _reconnect_monitor) rather than reusing a session that may
    already be gone by the time this actually executes.
    """
    db = SessionLocal()
    try:
        _rebuild_and_restart_pipeline_for_toggle(db, site_id)
    finally:
        db.close()


@app.post("/api/rules/{rule_id}/activate")
def activate_rule(
    rule_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Viewers cannot activate rules")
    rule = db.query(Rule).filter(
        Rule.id == rule_id,
        Rule.site_id == current_user.site_id
    ).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    if rule.status == "active":
        return {"status": "already_active", "rule_id": rule_id}
    if rule.status != "inactive":
        raise HTTPException(status_code=400, detail=f"Cannot activate a rule with status '{rule.status}'")
    missing = _missing_models_for_config(rule.config_json)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Can't activate — still waiting on training for: {', '.join(missing)}. Check Self-Learning.",
        )
    rule.status = "active"
    db.commit()
    # Respond right away — restarting the live pipeline subprocess can take
    # several seconds (terminate + model reload), which made the toggle feel
    # broken while it silently blocked the response. That work now happens
    # after the response is sent instead.
    background_tasks.add_task(_rebuild_and_restart_pipeline_bg, current_user.site_id)
    return {"status": "activated", "rule_id": rule_id}


@app.post("/api/rules/{rule_id}/deactivate")
def deactivate_rule(
    rule_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Viewers cannot deactivate rules")
    rule = db.query(Rule).filter(
        Rule.id == rule_id,
        Rule.site_id == current_user.site_id
    ).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    if rule.status == "inactive":
        return {"status": "already_inactive", "rule_id": rule_id}
    if rule.status != "active":
        raise HTTPException(status_code=400, detail=f"Cannot deactivate a rule with status '{rule.status}'")
    rule.status = "inactive"
    db.commit()
    background_tasks.add_task(_rebuild_and_restart_pipeline_bg, current_user.site_id)
    return {"status": "deactivated", "rule_id": rule_id}


SYSTEM_PROMPT = """You are ONVXP's rule generator. Convert plain English safety instructions into valid pipeline_config.json for a YOLO26 + ByteTrack computer vision pipeline.

If the user instruction is NOT a request to monitor a camera for a specific safety/security condition — greetings ("hi"), small talk, questions ("what's the weather"), or anything with no real-world situation to detect and alert on — output exactly this and nothing else:
{"not_a_rule": true}

CRITICAL — do not confuse "not a rule" with "an object I don't recognize": if the instruction clearly describes something to detect and alert on, it IS a rule, even if the specific object mentioned (a red toolbox, a specific tool, an unusual item) isn't in AVAILABLE MODELS below. In that case, build the rule normally — see the instruction near the end of this prompt for how to handle an unrecognized object. Only use {"not_a_rule": true} when there is no detection/alerting intent in the instruction at all, never merely because a mentioned object is unfamiliar.

Camera and zone selection is handled separately by the user through the app's own UI — never infer or invent a zone or camera name. Do not include a "zones" section in your output, and never include a "zone" field on any rule.

Sensitivity (how long a violation must hold before alerting) and alert distance (for proximity rules) are also set by the user afterward through the UI — never include "persistence_frames" or "proximity_px" in your output.

AVAILABLE MODELS:
- "person"   - detects people on site (base YOLO26, COCO trained)
- "truck"    - detects trucks and heavy vehicles (base YOLO26, COCO trained)
- "helmet"   - detects construction hardhats (custom trained)
- "fire"     - detects fire and flames (custom trained, mAP 75%)
- "smoke"    - detects smoke on site (custom trained, mAP 75%)
- "spill"    - detects liquid spills / hazardous liquids on floors (custom trained, mAP 88%)

AVAILABLE RULE TYPES:
- "person_in_zone"  - alert when any person is detected
- "missing_in_zone" - alert when a person without required gear is detected
- "count_exceeded"  - alert when more than N people are detected. Set "count" to N, e.g. {"type": "count_exceeded", "count": 10}. Never omit "count" — if the instruction doesn't state a number, use a sensible default like 5.
- "object_in_zone"  - alert when a specific OBJECT is detected (no person needed). Use for instructions like "alert when fire/smoke/forklift/truck/ladder is detected". Set "target" to the model name, e.g. {"type": "object_in_zone", "target": "fire", "required": []}
- "person_near_object" - alert when a person comes close to a detected object. Use "target": "<model>". For "near acid/dangerous liquid/spill/chemical", use target "spill". Example: {"type": "person_near_object", "target": "spill"}

IMPORTANT: "alert when X is detected" (fire, smoke, forklift, truck, ladder) means object_in_zone with target X — NOT person_in_zone. Every object_in_zone rule MUST include the "target" field, e.g. "target": "fire". Never omit it. For "target" values, preserve meaningful descriptive qualifiers rather than reducing to the single simplest word, especially for object names that mean different things in different domains (e.g. "wire" could mean electrical wire, surgical wire, or dental wire). Use "exposed electrical wire" rather than just "wire" when the instruction includes that context. Only use the bare single word when the object itself is unambiguous (e.g. "forklift", "fire").
IMPORTANT: "alert everyone near acid/spill/chemical/hazardous liquid" means person_near_object with target "spill" — NOT object_in_zone (this rule cares about people approaching the object, not just the object's presence). Never omit "target" on a person_near_object rule.
OUTPUT FORMAT (must match exactly):
{
  "pipeline_id": "auto_<short_descriptive_name>",
  "description": "<one line description>",
  "models": {
    "helmet": "runs/detect/helmet_model/weights/best.pt"
  },
  "rules": [{"type": "<rule_type>", "required": ["<gear>"], "primary": "person", "target": "<object_model_if_object_or_proximity_rule>", "count": 5}],
  "alert": {"severity": "high", "message": "<alert message>"},
  "cooldown_seconds": 30
}

NOTE on optional rule fields: include "target" only on object_in_zone and person_near_object rules; include "count" only on count_exceeded rules. Omit fields that don't apply. Never include "zone", "persistence_frames", or "proximity_px" on a rule — those are added by the app afterward, not by you.
If the instruction requires detecting an object class that is NOT in AVAILABLE MODELS (e.g. "trousers", "gloves", "ladder"), do NOT substitute a different model. Use the requested class name as the model key with weights path "runs/detect/<class>_model/weights/best.pt" and as the rule's target. The platform will train it.
Only include models actually needed. Remember: {"not_a_rule": true} is ONLY for input with no detection/alerting intent whatsoever (greetings, questions, chit-chat) — an unfamiliar object mentioned in an otherwise clear monitoring request is still a real rule, never reject it for that reason alone. Output ONLY the JSON. No markdown, no explanation."""


@app.post("/api/rules/generate")
async def generate_rule(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    response_text = ""
    try:
        body = await request.json()
        instruction = body.get("instruction", "").strip()
        if not instruction:
            raise HTTPException(status_code=400, detail="instruction is required")

        # ── Cheap, reliable pre-check for the most obvious non-rule inputs
        # (greetings, filler) — independent of whether the LLM reliably
        # follows the "not_a_rule" prompt instruction on every single call.
        # Conservative on purpose: only exact matches against known filler,
        # never a word-count threshold, so a genuinely short real rule
        # (e.g. "alert on fire") is never mistakenly rejected here. ──
        _NOT_A_RULE_PHRASES = {
            "hi", "hello", "hey", "hiya", "yo",
            "how are you", "what's up", "whats up", "sup",
            "thanks", "thank you", "ok", "okay", "test", "testing",
            "hi there", "hello there", "good morning", "good afternoon", "good evening",
        }
        _normalized = instruction.strip().lower().rstrip(".!?")
        if _normalized in _NOT_A_RULE_PHRASES:
            return {
                "config": None, "instruction": instruction, "model_used": OLLAMA_MODEL,
                "provider": "ollama_local", "unknown_classes": [], "training_jobs": [],
            }

        full_prompt = f"{SYSTEM_PROMPT}\n\nUser instruction: {instruction}\n\nJSON output:"

        ollama_payload = {"model": OLLAMA_MODEL, "prompt": full_prompt, "stream": False, "format": "json", "options": {"temperature": 0.1, "num_predict": 1024}}

        def _is_unusable(text: str) -> bool:
            return not text or not text.strip()

        response = None
        response_text = ""
        for attempt, timeout_s in enumerate((60, 180)):
            try:
                response = requests.post(OLLAMA_URL, json=ollama_payload, timeout=timeout_s)
            except requests.exceptions.Timeout:
                print(f"[OMNIX] Ollama timed out on attempt {attempt + 1} ({timeout_s}s) — retrying...")
                response = None
                continue

            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Ollama error: {response.text}")

            result = response.json()
            response_text = result.get("response", "").strip()

            if _is_unusable(response_text):
                print(f"[OMNIX] Ollama returned an empty response on attempt {attempt + 1} — likely still warming up, retrying...")
                response = None
                continue

            break

        if response is None:
            raise HTTPException(
                status_code=503,
                detail="AI engine is warming up, please try again in ~30 seconds."
            )

        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        config = json.loads(response_text)

        # ── The LLM itself decided this instruction isn't a rule at all
        # (per the new SYSTEM_PROMPT directive), or ignored that instruction
        # and returned syntactically valid but empty JSON anyway — either
        # way, there's no usable rule here. Respect that rather than letting
        # an empty object fall through the sanitizer below and get treated
        # as a real, if oddly-shaped, rule. ──
        if config.get("not_a_rule") or not config.get("rules"):
            return {
                "config": None, "instruction": instruction, "model_used": OLLAMA_MODEL,
                "provider": "ollama_local", "unknown_classes": [], "training_jobs": [],
            }

        model_keys = [m for m in config.get("models", {}).keys() if m != "person"]
        for r in config.get("rules", []):
            if r.get("type") in ("object_in_zone", "person_near_object") and not r.get("target"):
                if r.get("required"):
                    r["target"] = r["required"][0]
                elif model_keys:
                    r["target"] = model_keys[0]
                    print(f"[OMNIX] Sanitizer: filled missing target='{model_keys[0]}' on {r.get('type')} rule")
            if r.get("type") == "person_near_object" and not r.get("proximity_px"):
                r["proximity_px"] = 120
            if r.get("type") == "count_exceeded" and not r.get("count"):
                r["count"] = 5
                print(f"[OMNIX] Sanitizer: filled missing count=5 on count_exceeded rule")
        with open('model_registry.json', 'r') as f:
            _registry = json.load(f)
        _requested = _get_required_classes(config)
        unknown_classes = sorted(c for c in _requested if c not in _registry)
        training_jobs = []
        if unknown_classes:
            from db.models import TrainingJob
            for cls in unknown_classes:
                existing = db.query(TrainingJob).filter(
                    TrainingJob.class_name == cls,
                    TrainingJob.status.notin_(["failed", "cancelled"])
                ).first()
                if existing:
                    training_jobs.append({"id": existing.id, "class_name": cls, "status": existing.status, "reused": True})
                    continue
                job = TrainingJob(
                    site_id=current_user.site_id or 2,
                    class_name=cls,
                    status="pending",
                    current_stage="queued",
                    stages=[{"name": "queued", "status": "done"}],
                )
                db.add(job)
                db.commit()
                db.refresh(job)
                training_jobs.append({"id": job.id, "class_name": cls, "status": "pending", "reused": False})
                print(f"[OMNIX] Self-learning: unknown class '{cls}' → training job #{job.id} created")
        return {"config": config, "instruction": instruction, "model_used": OLLAMA_MODEL,
                "provider": "ollama_local", "unknown_classes": unknown_classes, "training_jobs": training_jobs}
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama is not running. Start it with: ollama serve")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"LLM returned invalid JSON: {str(e)}. Raw: {response_text[:500]}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _full_frame_zone_name(camera_id: int) -> str:
    return f"camera_{camera_id}_full_frame"


def _full_frame_coords_for_camera(camera_id: int, db: Session) -> list:
    """
    No more user-drawn polygons — a rule now watches a whole camera's frame,
    not a hand-drawn region within it. This just needs real dimensions for
    that frame: prefer the actual live stream's resolution, then the
    camera's last-known resolution, then a sensible fallback matching this
    project's existing demo footage.
    """
    width, height = 854, 480
    vs = video_streams.get(camera_id)
    if vs and vs.width and vs.height:
        width, height = vs.width, vs.height
    else:
        cam = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
        if cam and cam.resolution and "x" in cam.resolution:
            try:
                w, h = cam.resolution.split("x")
                width, height = int(w), int(h)
            except Exception:
                pass
    return [[0, 0], [width, 0], [width, height], [0, height]]


def _get_required_classes(config: dict) -> set:
    """
    Every class name a pipeline config actually needs a trained model for —
    model dict keys, any rule's "target", and any missing_in_zone rule's
    "required" list. Single source of truth, used by generate_rule (to spot
    unknown classes), rebuild_pipeline_config_from_db (to keep an unready
    model out of the live pipeline), and activate_rule (to block reactivating
    a rule whose model isn't there).
    """
    requested = set(config.get("models", {}).keys())
    for r in config.get("rules", []):
        if r.get("target"):
            requested.add(r["target"])
        if r.get("type") == "missing_in_zone":
            for item in r.get("required", []) or []:
                requested.add(item)
    return requested


def _missing_models_for_config(config: dict) -> list:
    try:
        with open("model_registry.json", "r") as f:
            registry = json.load(f)
    except Exception:
        registry = {}
    required = _get_required_classes(config)
    return sorted(c for c in required if c not in registry)


def rebuild_pipeline_config_from_db(db: Session, site_id: int, camera_id: int = 1) -> dict:
    """
    Rebuilds pipeline_config.json from the currently-active Rule rows that
    are attached (via RuleCamera) to `camera_id`. The live pipeline only
    ever runs one camera's feed at a time today — camera_id defaults to 1,
    matching what's actually running — so only rules attached to that
    specific camera are included here. A rule can be attached to several
    cameras at once (e.g. Camera 1 in one building and Camera 4 in
    another), but each camera's own live pipeline only pulls in the rules
    relevant to itself.

    True simultaneous multi-camera enforcement — running several cameras'
    pipelines at once, each picking up the rules attached to it — is a
    separate piece of infrastructure not built yet; this only prepares the
    correct config for whichever single camera is currently running.

    Each included rule gets a zone auto-generated as that camera's full
    frame — there's no user-drawn polygon anymore, since zones are just
    named buildings/places now, not regions within a frame.
    """
    active_rules = (
        db.query(Rule)
        .join(RuleCamera, RuleCamera.rule_id == Rule.id)
        .filter(
            Rule.site_id == site_id,
            Rule.status == "active",
            RuleCamera.camera_id == camera_id,
        )
        .order_by(Rule.created_at.asc())
        .distinct()
        .all()
    )

    if not active_rules:
        return {
            "pipeline_id": "auto_empty",
            "description": "No active rules for this camera",
            "models": {},
            "zones": [],
            "rules": [],
            "alert": {"severity": "medium", "message": "No active rules"},
            "cooldown_seconds": 30,
        }

    # A rule can be "active" in the DB the moment it's created (self-learning
    # may still be training its model) without that unready reference ever
    # reaching the live detector — including it here would crash the one
    # shared pipeline process for every rule, not just this one.
    runnable_rules = []
    for r in active_rules:
        missing = _missing_models_for_config(r.config_json)
        if missing:
            print(f"[OMNIX] Rule {r.id} is active but not yet runnable — waiting on model(s): {', '.join(missing)}")
        else:
            runnable_rules.append(r)

    if not runnable_rules:
        return {
            "pipeline_id": "auto_empty",
            "description": "Active rules exist for this camera, but none have a ready model yet",
            "models": {},
            "zones": [],
            "rules": [],
            "alert": {"severity": "medium", "message": "No active rules"},
            "cooldown_seconds": 30,
        }

    zone_name = _full_frame_zone_name(camera_id)
    zone_coords = _full_frame_coords_for_camera(camera_id, db)

    def stamp(cfg: dict, rule_db_id: int) -> dict:
        cfg = json.loads(json.dumps(cfg))
        cfg["zones"] = [{"name": zone_name, "coords": zone_coords}]
        for r in cfg.get("rules", []):
            r["rule_db_id"] = rule_db_id
            r["zone"] = zone_name
        return cfg

    merged = stamp(runnable_rules[0].config_json, runnable_rules[0].id)
    for db_rule in runnable_rules[1:]:
        stamped_cfg = stamp(db_rule.config_json, db_rule.id)
        merged = merge_configs(merged, stamped_cfg)

    site_settings = get_settings_for_site(db, site_id)
    merged["persistence_frames"] = site_settings["detection"].get("persistence_frames", 5)
    merged["alert_cooldown_frames"] = site_settings["detection"].get("alert_cooldown_frames", 150)
    merged["detection_confidence"] = site_settings["detection"].get("detection_confidence", 0.5)
    merged["email_notifications_enabled"] = site_settings["alerts"].get("email_notifications_enabled", False)
    merged["email_severity_threshold"] = site_settings["alerts"].get("email_severity_threshold", "high")

    return merged


def merge_configs(existing: dict, new_cfg: dict) -> dict:
    SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    existing_zone_names = {z["name"] for z in existing.get("zones", [])}
    merged_zones = list(existing.get("zones", []))
    for zone in new_cfg.get("zones", []):
        if zone["name"] not in existing_zone_names:
            merged_zones.append(zone)
            existing_zone_names.add(zone["name"])
    merged_rules = list(existing.get("rules", [])) + list(new_cfg.get("rules", []))
    merged_models = {**existing.get("models", {}), **new_cfg.get("models", {})}
    existing_sev = existing.get("alert", {}).get("severity", "medium")
    new_sev = new_cfg.get("alert", {}).get("severity", "medium")
    merged_alert = new_cfg.get("alert", existing.get("alert", {})) if SEVERITY_ORDER.get(new_sev, 1) >= SEVERITY_ORDER.get(existing_sev, 1) else existing.get("alert", {})
    merged_cooldown = min(existing.get("cooldown_seconds", 30), new_cfg.get("cooldown_seconds", 30))
    def strip_auto(s): return s[5:] if s.startswith("auto_") else s
    merged_id = f"auto_{strip_auto(existing.get('pipeline_id', 'auto_rule'))}__{strip_auto(new_cfg.get('pipeline_id', 'auto_rule'))}"
    existing_desc = existing.get("description", "")
    new_desc = new_cfg.get("description", "")
    return {
        "pipeline_id": merged_id,
        "description": f"{existing_desc} + {new_desc}" if existing_desc else new_desc,
        "models": merged_models,
        "zones": merged_zones,
        "rules": merged_rules,
        "alert": merged_alert,
        "cooldown_seconds": merged_cooldown,
    }


@app.post("/api/rules/apply")
async def apply_rule(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        body = await request.json()
        new_config = body.get("config")
        instruction = body.get("instruction", "")
        camera_ids = body.get("camera_ids") or []
        if not new_config:
            raise HTTPException(status_code=400, detail="config is required")
        if not camera_ids:
            raise HTTPException(status_code=400, detail="Select at least one camera")
        if current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Viewers cannot apply rules")

        cameras = db.query(CameraModel).filter(
            CameraModel.id.in_(camera_ids),
            CameraModel.site_id == current_user.site_id,
        ).all()
        if len(cameras) != len(set(camera_ids)):
            raise HTTPException(status_code=400, detail="One or more selected cameras were not found")

        missing = _missing_models_for_config(new_config)
        try:
            dupes = db.query(Rule).filter(
                Rule.site_id == current_user.site_id,
                Rule.status == "active",
                Rule.instruction == instruction,
            ).all()
            for d in dupes:
                d.status = "replaced"
            if dupes:
                print(f"[OMNIX] Dedupe: marked {len(dupes)} identical active rule(s) as replaced")
            rule = Rule(
                site_id=current_user.site_id, user_id=current_user.id,
                instruction=instruction, config_json=new_config,
                pipeline_id=new_config.get("pipeline_id"),
                status="pending_training" if missing else "active",
                severity=new_config.get("alert", {}).get("severity", "medium"),
            )
            db.add(rule)
            db.commit()
            db.refresh(rule)
            for r in new_config.get("rules", []):
                r["rule_db_id"] = rule.id
            for cam in cameras:
                db.add(RuleCamera(rule_id=rule.id, camera_id=cam.id))
            db.commit()
            training_jobs = []
            if missing:
                # Link every training job this rule is waiting on, so its
                # approval can find its way back here later (this is what
                # TrainingJob.rule_id was already meant for).
                from db.models import TrainingJob
                jobs = db.query(TrainingJob).filter(
                    TrainingJob.site_id == current_user.site_id,
                    TrainingJob.class_name.in_(missing),
                    TrainingJob.status.notin_(["failed", "cancelled"]),
                ).all()
                for job in jobs:
                    job.rule_id = rule.id
                db.commit()
                training_jobs = [
                    {"id": j.id, "class_name": j.class_name, "status": j.status, "reused": True}
                    for j in jobs
                ]
        except HTTPException:
            raise
        except Exception as e:
            # If the rule can't actually be saved, stop here — don't touch
            # live detection for a rule that was never persisted. Without
            # this, a save failure could leave a rule silently running with
            # no DB row at all, invisible and unmanageable from the Rules
            # page.
            db.rollback()
            print(f"[OMNIX] ERROR: could not save rule to DB: {e}")
            raise HTTPException(status_code=500, detail=f"Could not save the rule: {e}")

        if missing:
            # Nothing to run yet, and nothing shown as active OR inactive —
            # this rule only appears in the Rules list once every model it
            # needs is actually approved (see approve_training_job below).
            return {
                "status": "pending_training",
                "rule_id": rule.id,
                "missing_models": missing,
                "training_jobs": training_jobs,
                "message": f"Saved — waiting on training for: {', '.join(missing)}. Check Self-Learning for progress.",
            }

        # Respond immediately; the actual config rebuild + pipeline restart
        # happens in the background, same pattern as activate/deactivate —
        # restarting the detector can take several seconds and shouldn't
        # block this response. Live detection today only ever runs one
        # camera's pipeline at a time (camera 1 by default) — if this rule
        # applies to that camera, this picks it up; true simultaneous
        # multi-camera enforcement is a separate piece of infrastructure
        # not built yet.
        background_tasks.add_task(_rebuild_and_restart_pipeline_bg, current_user.site_id)

        return {
            "status": "applied",
            "message": f"Rule saved for {len(cameras)} camera(s): {', '.join(c.name for c in cameras)}.",
            "rule_id": rule.id,
            "pipeline_id": new_config.get("pipeline_id"),
            "camera_ids": [c.id for c in cameras],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rules/reset")
def reset_rules(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Viewers cannot disable all rules")
    deactivated = db.query(Rule).filter(
        Rule.site_id == current_user.site_id,
        Rule.status == "active",
    ).update({"status": "inactive"})
    db.commit()
    background_tasks.add_task(_rebuild_and_restart_pipeline_bg, current_user.site_id)
    return {"status": "reset", "message": f"{deactivated} rule(s) deactivated."}


@app.get("/api/pipeline/status")
def pipeline_status(current_user: User = Depends(get_current_user)):
    global _pipeline_process
    if _pipeline_process is None:
        return {"running": False, "pid": None, "status": "not_started"}
    poll = _pipeline_process.poll()
    if poll is None:
        return {"running": True, "pid": _pipeline_process.pid, "status": "running"}
    return {"running": False, "pid": _pipeline_process.pid, "status": "finished", "exit_code": poll}


@app.post("/api/pipeline/stop")
def stop_pipeline(current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can stop the pipeline")
    global _pipeline_process
    if _pipeline_process is None or _pipeline_process.poll() is not None:
        return {"status": "not_running"}
    _pipeline_process.terminate()
    try:
        _pipeline_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _pipeline_process.kill()
    return {"status": "stopped", "pid": _pipeline_process.pid}


def generate_frames(camera_id: int):
    target_fps = 25
    frame_interval = 1.0 / target_fps
    while True:
        start = time.time()
        vs = video_streams.get(camera_id)
        frame = vs.read() if vs else None
        if frame is None:
            placeholder = np.zeros((480, 854, 3), dtype=np.uint8)
            cv2.putText(placeholder, "NO SIGNAL", (320, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (80, 80, 80), 2)
            frame = placeholder
        else:
            frame = cv2.resize(frame, (854, 480))
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        elapsed = time.time() - start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def generate_placeholder_frames():
    placeholder = np.zeros((480, 854, 3), dtype=np.uint8)
    cv2.putText(placeholder, "NO SIGNAL", (320, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (80, 80, 80), 2)
    ret, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 75])
    payload = (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    while True:
        yield payload
        time.sleep(0.5)


@app.get("/api/video/stream")
def video_stream_endpoint(camera_id: int = 1, db: Session = Depends(get_db)):
    vs = get_or_start_stream(camera_id, db)
    gen = generate_frames(camera_id) if vs else generate_placeholder_frames()
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace;boundary=frame")


@app.get("/api/video/snapshot")
def video_snapshot(camera_id: int = 1, db: Session = Depends(get_db)):
    vs = get_or_start_stream(camera_id, db)
    frame = vs.read() if vs else None
    if frame is None:
        placeholder = np.zeros((480, 854, 3), dtype=np.uint8)
        cv2.putText(placeholder, "NO SIGNAL", (320, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (80, 80, 80), 2)
        frame = placeholder
    else:
        frame = cv2.resize(frame, (854, 480))
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to encode frame")
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.post("/api/video/source")
async def set_video_source(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can change video source")
    body = await request.json()
    raw_source = body.get("source", "")
    camera_id = int(body.get("camera_id") or 1)
    if isinstance(raw_source, str):
        raw_source = raw_source.strip()
    if raw_source == "" or raw_source is None:
        raise HTTPException(status_code=400, detail="source is required")
    source = _parse_source(raw_source)
    if isinstance(source, str) and not source.startswith("rtsp://") and not Path(source).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {source}")
    with _video_streams_lock:
        vs = video_streams.get(camera_id) or VideoStream()
        success = vs.start(source)
        if success:
            video_streams[camera_id] = vs
    if not success:
        raise HTTPException(status_code=500, detail="Failed to open video source")
    return {"status": "ok", "source": str(source), "fps": vs.fps, "camera_id": camera_id}


class CameraCreateRequest(BaseModel):
    name: str
    location: str = ""
    source: str
    zone_id: int | None = None


class CameraUpdateRequest(BaseModel):
    name: str | None = None
    location: str | None = None
    source: str | None = None
    zone_id: int | None = None


def _validate_source(source: str) -> str:
    source = (source or "").strip()
    if not source:
        raise HTTPException(status_code=400, detail="source is required")
    if source.startswith("rtsp://"):
        return source
    if source.isdigit():
        return source
    if not Path(source).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {source}")
    return source


@app.get("/api/cameras")
def get_cameras(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    cameras = db.query(CameraModel).filter(
        CameraModel.site_id == current_user.site_id
    ).order_by(CameraModel.id).all()

    zone_names = {
        z.id: z.name
        for z in db.query(Zone).filter(Zone.site_id == current_user.site_id).all()
    }

    result = []
    for cam in cameras:
        vs = video_streams.get(cam.id)
        is_live = vs is not None and vs.running and vs.cap is not None
        result.append({
            "id": cam.id,
            "name": cam.name,
            "location": cam.location,
            "status": "online" if is_live else cam.status,
            "stream_url":   f"{PUBLIC_BASE_URL}/api/video/stream?camera_id={cam.id}"   if is_live else None,
            "snapshot_url": f"{PUBLIC_BASE_URL}/api/video/snapshot?camera_id={cam.id}" if is_live else None,
            "fps":        int(vs.fps) if is_live else 0,
            "resolution": f"{vs.width}x{vs.height}" if is_live else "N/A",
            "source": vs.source if is_live else cam.source,
            "zone_id": cam.zone_id,
            "zone_name": zone_names.get(cam.zone_id),
        })
    return result


@app.post("/api/cameras")
def create_camera(
    body: CameraCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can add cameras")
    validated_source = _validate_source(body.source)
    cam = CameraModel(
        site_id=current_user.site_id,
        zone_id=body.zone_id,
        name=(body.name or "").strip() or "Unnamed Camera",
        location=(body.location or "").strip(),
        source=validated_source,
        status="offline",
    )
    db.add(cam)
    db.commit()
    db.refresh(cam)

    vs = get_or_start_stream(cam.id, db)
    cam.status = "online" if vs else "offline"
    if vs:
        cam.fps = int(vs.fps)
        cam.resolution = f"{vs.width}x{vs.height}"
    db.commit()
    db.refresh(cam)

    return {
        "id": cam.id, "name": cam.name, "location": cam.location,
        "source": cam.source, "status": cam.status,
        "fps": cam.fps, "resolution": cam.resolution,
        "zone_id": cam.zone_id,
    }


@app.put("/api/cameras/{camera_id}")
def update_camera(
    camera_id: int,
    body: CameraUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can edit cameras")
    cam = db.query(CameraModel).filter(
        CameraModel.id == camera_id,
        CameraModel.site_id == current_user.site_id
    ).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    if body.name is not None:
        cam.name = body.name.strip()
    if body.location is not None:
        cam.location = body.location.strip()
    if body.zone_id is not None:
        cam.zone_id = body.zone_id

    source_changed = False
    if body.source is not None:
        cam.source = _validate_source(body.source)
        source_changed = True

    db.commit()

    if source_changed:
        vs = restart_stream(camera_id, db)
        cam.status = "online" if vs else "offline"
        if vs:
            cam.fps = int(vs.fps)
            cam.resolution = f"{vs.width}x{vs.height}"
        db.commit()
        db.refresh(cam)

    return {
        "id": cam.id, "name": cam.name, "location": cam.location,
        "source": cam.source, "status": cam.status,
        "fps": cam.fps, "resolution": cam.resolution,
    }


@app.get("/api/settings")
def get_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return get_settings_for_site(db, current_user.site_id)


@app.put("/api/settings")
async def update_settings(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can change settings")
    body = await request.json()
    save_settings_for_site(db, current_user.site_id, body)
    return {"status": "saved", "settings": get_settings_for_site(db, current_user.site_id)}


@app.post("/api/danger/flush-alerts")
def flush_alerts(current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can flush alerts")
    incidents_file = Path("incidents.json")
    flushed = 0
    if incidents_file.exists():
        with open(incidents_file, "r") as f:
            flushed = len(json.load(f))
        incidents_file.unlink()
    for f in Path("incidents").glob("*.jpg"):
        try: f.unlink()
        except: pass
    return {"status": "flushed", "count": flushed}


@app.post("/api/danger/reset-tracks")
def reset_tracks(current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can reset tracks")
    return {"status": "tracks_reset", "note": "Restart run_pipeline.py to fully reset ByteTrack state"}


def get_date_range(from_date: str = None, to_date: str = None):
    if not from_date:
        from_dt = datetime.utcnow() - timedelta(days=7)
    else:
        from_dt = datetime.strptime(from_date, "%Y-%m-%d")
    if not to_date:
        to_dt = datetime.utcnow()
    else:
        to_dt = datetime.strptime(to_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
    return from_dt, to_dt


def get_incidents_in_range(db: Session, site_id: int, from_dt: datetime, to_dt: datetime):
    return db.query(Incident).filter(
        Incident.site_id == site_id,
        Incident.timestamp >= from_dt,
        Incident.timestamp <= to_dt
    ).all()


@app.get("/api/analytics/incidents-over-time")
def incidents_over_time(
    period: str = "day",
    from_date: str = None,
    to_date: str = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    from_dt, to_dt = get_date_range(from_date, to_date)
    incidents = get_incidents_in_range(db, current_user.site_id, from_dt, to_dt)
    counts = defaultdict(int)
    for inc in incidents:
        if inc.timestamp:
            if period == "day":
                key = inc.timestamp.strftime("%Y-%m-%d")
            elif period == "week":
                key = inc.timestamp.strftime("%Y-W%W")
            else:
                key = inc.timestamp.strftime("%Y-%m")
            counts[key] += 1
    result = []
    cur = from_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    while cur <= to_dt:
        if period == "day":
            key = cur.strftime("%Y-%m-%d")
            cur += timedelta(days=1)
        elif period == "week":
            key = cur.strftime("%Y-W%W")
            cur += timedelta(weeks=1)
        else:
            key = cur.strftime("%Y-%m")
            if cur.month == 12:
                cur = cur.replace(year=cur.year + 1, month=1)
            else:
                cur = cur.replace(month=cur.month + 1)
        result.append({"date": key, "count": counts.get(key, 0)})
    return result


@app.get("/api/analytics/incidents-by-rule")
def incidents_by_rule(
    from_date: str = None,
    to_date: str = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    from_dt, to_dt = get_date_range(from_date, to_date)
    incidents = get_incidents_in_range(db, current_user.site_id, from_dt, to_dt)
    counts = defaultdict(lambda: {"count": 0, "severity": "high"})
    for inc in incidents:
        rule_name = inc.violation_type or "unknown"
        counts[rule_name]["count"] += 1
        counts[rule_name]["severity"] = inc.severity or "high"
    result = [{"rule_name": k, "count": v["count"], "severity": v["severity"]} for k, v in counts.items()]
    result.sort(key=lambda x: x["count"], reverse=True)
    return result[:10]


@app.get("/api/analytics/incidents-by-hour")
def incidents_by_hour(
    from_date: str = None,
    to_date: str = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    from_dt, to_dt = get_date_range(from_date, to_date)
    incidents = get_incidents_in_range(db, current_user.site_id, from_dt, to_dt)
    counts = defaultdict(int)
    for inc in incidents:
        if inc.timestamp:
            counts[inc.timestamp.hour] += 1
    return [{"hour": h, "count": counts.get(h, 0)} for h in range(24)]


@app.get("/api/analytics/false-positive-rate")
def false_positive_rate(
    from_date: str = None,
    to_date: str = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    from_dt, to_dt = get_date_range(from_date, to_date)
    incidents = get_incidents_in_range(db, current_user.site_id, from_dt, to_dt)
    rule_data = defaultdict(lambda: {"tp": 0, "fp": 0})
    for inc in incidents:
        rule_name = inc.violation_type or "unknown"
        if inc.review_status == "false_positive":
            rule_data[rule_name]["fp"] += 1
        else:
            rule_data[rule_name]["tp"] += 1
    result = []
    for rule_name, data in rule_data.items():
        total = data["tp"] + data["fp"]
        rate = round((data["fp"] / total) * 100, 1) if total > 0 else 0.0
        result.append({"rule_name": rule_name, "tp_count": data["tp"], "fp_count": data["fp"], "total": total, "rate": rate})
    result.sort(key=lambda x: x["rate"], reverse=True)
    return result


@app.get("/api/export/incidents")
def export_incidents(
    format: str = "csv",
    from_date: str = None,
    to_date: str = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can export data")
    from_dt, to_dt = get_date_range(from_date, to_date)
    incidents = get_incidents_in_range(db, current_user.site_id, from_dt, to_dt)
    incidents.sort(key=lambda x: x.timestamp or datetime.min, reverse=True)
    site_settings = get_settings_for_site(db, current_user.site_id)
    site_name = site_settings["platform"].get("site_name", "Site").replace(" ", "_").replace("—", "-")
    from_str = from_dt.strftime("%Y-%m-%d")
    to_str = to_dt.strftime("%Y-%m-%d")
    filename_base = f"omnix_report_{site_name}_{from_str}_to_{to_str}"

    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "timestamp", "rule_name", "severity", "zone", "camera", "alert_message", "review_status", "reviewed_by", "screenshot_url"])
        for inc in incidents:
            screenshot_url = ""
            if inc.screenshot_path:
                filename = inc.screenshot_path.replace("incidents/", "")
                screenshot_url = f"{PUBLIC_BASE_URL}/screenshots/{filename}"
            writer.writerow([inc.id, inc.timestamp.isoformat() if inc.timestamp else "", inc.violation_type or "", inc.severity or "", inc.zone or "", f"Camera {inc.camera_id or 1}", inc.alert_message or "", inc.review_status or "", inc.reviewed_by or "", screenshot_url])
        output.seek(0)
        return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{filename_base}.csv"'})

    elif format == "pdf":
        try:
            from reportlab.lib.pagesizes import landscape, A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
        except ImportError:
            raise HTTPException(status_code=500, detail="reportlab not installed.")

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("Title", parent=styles["Heading1"], fontSize=18, alignment=TA_CENTER, textColor=colors.HexColor("#1a1a2e"), spaceAfter=6)
        subtitle_style = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=10, alignment=TA_CENTER, textColor=colors.HexColor("#6366f1"), spaceAfter=4)
        section_style = ParagraphStyle("Section", parent=styles["Heading2"], fontSize=13, textColor=colors.HexColor("#1a1a2e"), spaceBefore=14, spaceAfter=6)
        cell_style = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#374151"), wordWrap="CJK")
        story = []
        story.append(Paragraph("OMNIX Safety Report", title_style))
        site_display = site_settings["platform"].get("site_name", "Site A")
        story.append(Paragraph(f"{site_display} — {from_str} to {to_str}", subtitle_style))
        story.append(Paragraph(f"Generated for {current_user.name or current_user.email} on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", subtitle_style))
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph("Summary", section_style))
        total = len(incidents)
        sev_counts = defaultdict(int)
        rule_counts = defaultdict(int)
        for inc in incidents:
            sev_counts[inc.severity or "unknown"] += 1
            rule_counts[inc.violation_type or "unknown"] += 1
        days_diff = max((to_dt - from_dt).days, 1)
        avg_per_day = round(total / days_diff, 1)
        summary_data = [["Metric", "Value"], ["Total Incidents", str(total)], ["Date Range", f"{from_str} to {to_str}"], ["Avg per Day", str(avg_per_day)], ["Critical", str(sev_counts.get("critical", 0))], ["High", str(sev_counts.get("high", 0))], ["Medium", str(sev_counts.get("medium", 0))], ["Unique Rules Triggered", str(len(rule_counts))]]
        summary_table = Table(summary_data, colWidths=[6*cm, 6*cm])
        summary_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#6366f1")), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,0), 10), ("ALIGN", (0,0), (-1,-1), "LEFT"), ("FONTSIZE", (0,1), (-1,-1), 9), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f9fafb"), colors.white]), ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")), ("LEFTPADDING", (0,0), (-1,-1), 8), ("RIGHTPADDING", (0,0), (-1,-1), 8), ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5)]))
        story.append(summary_table)
        story.append(Spacer(1, 0.5*cm))
        if rule_counts:
            story.append(Paragraph("Incidents by Rule", section_style))
            rule_data = [["Rule", "Count"]]
            for rule, count in sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                rule_data.append([rule.replace("_", " ").title(), str(count)])
            rule_table = Table(rule_data, colWidths=[10*cm, 4*cm])
            rule_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#6366f1")), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,0), 10), ("ALIGN", (1,0), (1,-1), "CENTER"), ("FONTSIZE", (0,1), (-1,-1), 9), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f9fafb"), colors.white]), ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")), ("LEFTPADDING", (0,0), (-1,-1), 8), ("RIGHTPADDING", (0,0), (-1,-1), 8), ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5)]))
            story.append(rule_table)
            story.append(Spacer(1, 0.5*cm))
        story.append(PageBreak())
        story.append(Paragraph("Incident Log", section_style))
        page_w = landscape(A4)[0] - 3*cm
        col_widths = [page_w*0.05, page_w*0.13, page_w*0.18, page_w*0.08, page_w*0.13, page_w*0.10, page_w*0.23, page_w*0.10]
        table_data = [["ID", "Timestamp", "Rule", "Severity", "Zone", "Camera", "Message", "Status"]]
        for inc in incidents[:500]:
            table_data.append([str(inc.id), inc.timestamp.strftime("%Y-%m-%d %H:%M") if inc.timestamp else "", Paragraph((inc.violation_type or "").replace("_", " ").title(), cell_style), (inc.severity or "").capitalize(), Paragraph((inc.zone or "").replace("_", " ").title(), cell_style), f"Camera {inc.camera_id or 1}", Paragraph(inc.alert_message or "", cell_style), (inc.review_status or "pending").capitalize()])
        inc_table = Table(table_data, colWidths=col_widths, repeatRows=1)
        inc_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1e1b4b")), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,0), 9), ("ALIGN", (0,0), (-1,-1), "LEFT"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("FONTSIZE", (0,1), (-1,-1), 8), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f9fafb"), colors.white]), ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#e5e7eb")), ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5), ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4)]))
        story.append(inc_table)

        def add_page_number(canvas, doc):
            canvas.saveState()
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.HexColor("#6b7280"))
            canvas.drawCentredString(landscape(A4)[0] / 2, 1*cm, f"Page {doc.page} — OMNIX Safety Report — {site_display}")
            canvas.restoreState()

        doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{filename_base}.pdf"'})

    else:
        raise HTTPException(status_code=400, detail="format must be 'csv' or 'pdf'")
