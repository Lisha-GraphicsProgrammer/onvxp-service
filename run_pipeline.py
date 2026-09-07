import json
import cv2
import base64
import requests
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image as PILImage
from ultralytics import YOLO

# DB imports
import os
from dotenv import load_dotenv
load_dotenv()

from db.session import SessionLocal
from db.models import Incident, Rule, Site, Camera as CameraModel

# ============================================================
# CONFIG LOADING
# ============================================================
with open('pipeline_config.json', 'r') as f:
    config = json.load(f)

print(f"\n{'='*60}")
print(f"Pipeline: {config['pipeline_id']}")
print(f"Description: {config['description']}")
print(f"Zones: {len(config.get('zones', []))}")
print(f"Rules: {len(config.get('rules', []))}")
print(f"{'='*60}\n")

# ============================================================
# LOAD DB REFERENCES
# ============================================================
def get_db_refs():
    """Get site_id, camera_id, rule_id, video_source from DB for incident writing."""
    try:
        db = SessionLocal()
        site = db.query(Site).first()
        camera = db.query(CameraModel).first()
        rule = db.query(Rule).filter(
            Rule.pipeline_id == config['pipeline_id'],
            Rule.status == 'active'
        ).first()
        if not rule:
            rule = db.query(Rule).filter(Rule.status == 'active').first()
        db.close()
        return (
            site.id if site else None,
            camera.id if camera else None,
            rule.id if rule else None,
            camera.source if camera and camera.source else None,
        )
    except Exception as e:
        print(f"[WARN] Could not get DB refs: {e}")
        return None, None, None, None

site_id, camera_id, rule_id, video_source = get_db_refs()
print(f"[DB] site_id={site_id}, camera_id={camera_id}, rule_id={rule_id}")

# ============================================================
# MODEL REGISTRY — load registry and lazy-load only needed models
# ============================================================
with open('model_registry.json', 'r') as f:
    registry = json.load(f)

print(f"[INFO] Model registry loaded: {list(registry.keys())}")

def load_model_from_registry(model_name: str) -> YOLO | None:
    """Load a model by name from the registry. Returns None if not available.
    Entries with type 'roboflow_remote' or 'grounding_dino_local' are NOT
    loaded here — neither has local YOLO weights. Remote calls the Roboflow
    API live; grounding_dino_local loads its own transformer model lazily
    on first actual use, cached separately in _dino_cache. This means a
    broken local torch/transformers install ONLY breaks a rule that
    actually uses grounding_dino_local — every other rule, including a
    roboflow_remote one, is completely unaffected."""
    if model_name not in registry:
        print(f"  [WARN] '{model_name}' not in registry")
        return None

    entry = registry[model_name]
    model_type = entry.get("type", "custom")

    if model_type == "roboflow_remote":
        print(f"  [OK] '{model_name}' configured as remote (model_id={entry.get('model_id')}) — no local weights to load")
        return "REMOTE"

    if model_type == "grounding_dino_local":
        print(f"  [OK] '{model_name}' configured as Grounding DINO (prompt='{entry.get('prompt')}') — loads on first use")
        return "DINO"

    if model_type == "coco_default":
        model_path = entry.get("model", "yolov8n.pt")
        if not Path(model_path).exists():
            print(f"  [SKIP] {model_name}: base model {model_path} not found")
            return None
        try:
            m = YOLO(model_path)
            print(f"  [OK] Loaded {model_name} (COCO class {entry.get('class_id')}) from {model_path}")
            return m
        except Exception as e:
            print(f"  [FAIL] {model_name}: {e}")
            return None

    elif model_type == "custom":
        weights = entry.get("weights", "")
        if not Path(weights).exists():
            print(f"  [SKIP] {model_name}: weights not found at {weights}")
            return None
        try:
            m = YOLO(weights)
            print(f"  [OK] Loaded {model_name} from {weights}")
            return m
        except Exception as e:
            print(f"  [FAIL] {model_name}: {e}")
            return None

    return None

needed_models = set()
for rule in config.get('rules', []):
    for gear in rule.get('required', []):
        needed_models.add(gear)
for name in config.get('models', {}).keys():
    needed_models.add(name)

print(f"\n[INFO] Models needed by pipeline: {needed_models}")
print(f"[INFO] Loading only required models (lazy loading)...")

models = {}
for model_name in needed_models:
    m = load_model_from_registry(model_name)
    if m is not None:
        models[model_name] = m

base_entry = registry.get("person", {})
base_model_path = base_entry.get("model", "yolov8n.pt")
base_conf = base_entry.get("confidence", 0.5)
base_model = YOLO(base_model_path)
print(f"\n[OK] Loaded base YOLO for person detection (conf={base_conf})")
print(f"[INFO] Total models loaded: {list(models.keys())}\n")

# ============================================================
# HELPER: write incident to DB + JSON fallback
# ============================================================
INCIDENTS_FILE = Path('incidents.json')

def append_incident(incident: dict):
    if rule_id and site_id:
        try:
            db = SessionLocal()
            db_incident = Incident(
                rule_id=rule_id,
                camera_id=camera_id,
                site_id=site_id,
                timestamp=datetime.fromisoformat(incident["timestamp"]),
                frame_number=incident.get("frame"),
                person_track_id=incident.get("person_id"),
                violation_type=incident.get("violation"),
                detected_objects=None,
                missing_gear=incident.get("missing_gear") or None,
                zone=incident.get("zone"),
                bbox=incident.get("bbox"),
                screenshot_path=incident.get("screenshot_path"),
                severity=config.get("alert", {}).get("severity", "medium"),
                alert_message=incident.get("alert_message"),
                reviewed=False,
            )
            db.add(db_incident)
            db.commit()
            db.close()
            print(f"  [DB] Incident saved to PostgreSQL")
        except Exception as e:
            print(f"  [WARN] DB write failed: {e}, falling back to JSON")
            _append_json(incident)
    else:
        _append_json(incident)


def _append_json(incident: dict):
    if INCIDENTS_FILE.exists():
        try:
            with open(INCIDENTS_FILE, 'r') as f:
                incidents = json.load(f)
        except:
            incidents = []
    else:
        incidents = []
    incidents.append(incident)
    tmp = INCIDENTS_FILE.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(incidents, f, indent=2)
    tmp.replace(INCIDENTS_FILE)


# ============================================================
# HELPER: point in polygon (ray casting)
# ============================================================
def point_in_polygon(x: float, y: float, poly) -> bool:
    if not poly or len(poly) < 3:
        return False
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


# ============================================================
# HELPER: bbox overlap
# ============================================================
def bbox_overlap(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    if box2_area == 0:
        return 0.0
    return intersection / box2_area


def check_required_gear(person_bbox, frame, required_gear, loaded_models):
    missing = []
    for gear_name in required_gear:
        if gear_name not in loaded_models:
            print(f"  [WARN] Required gear '{gear_name}' not loaded, skipping")
            continue
        entry = registry.get(gear_name, {})
        conf = entry.get("confidence", 0.4)

        if entry.get("type") == "coco_default":
            class_id = entry.get("class_id", 0)
            gear_results = loaded_models[gear_name](frame, verbose=False,
                                                     conf=conf, classes=[class_id])
        else:
            gear_results = loaded_models[gear_name](frame, verbose=False, conf=conf)

        has_gear = False
        if gear_results[0].boxes is not None and len(gear_results[0].boxes) > 0:
            for gear_box in gear_results[0].boxes.xyxy.cpu().numpy():
                if bbox_overlap(person_bbox, gear_box) > 0.3:
                    has_gear = True
                    break
        if not has_gear:
            missing.append(gear_name)
    return missing


# ============================================================
# HELPER: object_in_zone — LOCAL YOLO model path
#
# FIX (applied tonight): previously returned on the FIRST in-zone box
# that cleared threshold, same bug as the remote path had. Now scans all
# candidates and picks the highest-confidence one.
#
# ADDITIONAL FIX for 'wire' specifically: wire_lisha_best3.pt confuses
# "person in PPE" and "exposed wire" under one class label ("warning"),
# and fires MUCH more confidently on people (~0.31) than on the real wire
# (~0.17, confirmed by manual box inspection tonight — box
# [541,265,675,376] at conf=0.167 genuinely lands on the frayed copper).
# Highest-confidence-alone would therefore always pick the person, not
# the wire. To work around this without retraining, a registry entry can
# optionally set "region": [x_min, y_min, x_max, y_max] — candidates
# outside that box are ignored entirely before confidence comparison.
# This is a stopgap, not a real fix; the real fix is better training
# data so the model tells wire and people apart on its own.
# ============================================================
def check_object_in_zone_local(frame, target_model_name, zone_poly, loaded_models):
    entry = registry.get(target_model_name, {})
    conf = entry.get("confidence", 0.5)
    region = entry.get("region")  # optional [x_min, y_min, x_max, y_max]

    obj_results = loaded_models[target_model_name](frame, verbose=False, conf=conf)
    if obj_results[0].boxes is None or len(obj_results[0].boxes) == 0:
        return None

    boxes = obj_results[0].boxes.xyxy.cpu().numpy()
    confs = obj_results[0].boxes.conf.cpu().numpy()

    best_bbox = None
    best_conf = -1.0

    for obj_box, obj_conf in zip(boxes, confs):
        ox1, oy1, ox2, oy2 = obj_box
        cx, cy = (ox1 + ox2) / 2, (oy1 + oy2) / 2

        if not point_in_polygon(cx, cy, zone_poly):
            continue

        if region is not None:
            rx1, ry1, rx2, ry2 = region
            if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                continue

        if obj_conf > best_conf:
            best_conf = obj_conf
            best_bbox = (ox1, oy1, ox2, oy2)

    return best_bbox


# ============================================================
# HELPER: object_in_zone — REMOTE Roboflow-hosted model path
#
# FIX (applied tonight): previously this function returned on the FIRST
# in-zone prediction that cleared the confidence threshold, regardless of
# whether a higher-confidence prediction existed later in the same
# response. This caused a real bug: on frame 82 of wire_video_two.mp4,
# the API returned multiple candidate boxes in one response (a low-
# confidence hit on a scaffolding bracket AND an 85%-confidence hit on
# the actual exposed wire), and the old code grabbed the bracket simply
# because it appeared first in the predictions list. Confirmed via
# Lisha's separate Colab-trained run of the same model, which correctly
# drew "exposed_wire" at 85% confidence on the real wire in the same
# frame — proving the model itself was fine; our selection logic wasn't.
#
# Fix: scan the full predictions list, keep only in-zone + above-
# threshold candidates, and return the one with the HIGHEST confidence,
# not the first one encountered.
# ============================================================
def check_object_in_zone_remote(frame, model_name, zone_poly):
    entry = registry.get(model_name, {})
    model_id = entry.get("model_id")
    api_key_env = entry.get("api_key_env", "ROBOFLOW_HOSTED_API_KEY")
    api_key = os.getenv(api_key_env)
    conf_threshold = entry.get("confidence", 0.5)

    if not model_id or not api_key:
        print(f"  [WARN] '{model_name}' remote config incomplete (model_id or {api_key_env} missing) — skipping this frame")
        return None

    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

    try:
        resp = requests.post(
            f"https://serverless.roboflow.com/{model_id}",
            data=image_b64,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [WARN] Remote inference call failed for '{model_name}': {e}")
        return None

    predictions = data.get("predictions", [])
    best_bbox = None
    best_conf = -1.0

    for p in predictions:
        p_conf = p.get("confidence", 0)
        if p_conf < conf_threshold:
            continue
        cx, cy = p.get("x", 0), p.get("y", 0)
        w, h = p.get("width", 0), p.get("height", 0)
        if not point_in_polygon(cx, cy, zone_poly):
            continue
        if p_conf > best_conf:
            best_conf = p_conf
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            best_bbox = (x1, y1, x2, y2)

    if best_bbox is not None:
        print(f"  [INFO] '{model_name}' remote: picked highest-confidence in-zone match at {best_conf:.2f} (of {len(predictions)} total predictions)")

    return best_bbox


# ============================================================
# HELPER: object_in_zone — Grounding DINO, LOCAL (no network call)
#
# IMPORTANT: torch and transformers are imported LAZILY, only inside the
# two functions below, only at the moment a rule actually needs this
# model type — never at the top of this file. This means a broken local
# torch/transformers install (e.g. a bad CUDA DLL) ONLY breaks a rule
# using grounding_dino_local specifically; every other model type
# (custom YOLO weights, coco_default, roboflow_remote) keeps working
# completely normally regardless of torch's state on this machine.
# ============================================================
_dino_cache = {}

def _get_dino_model(model_id: str):
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    if model_id not in _dino_cache:
        print(f"  [INFO] Loading Grounding DINO ({model_id})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        _dino_cache[model_id] = (processor, model, device)
        print(f"  [OK] Grounding DINO loaded on {device}")
    return _dino_cache[model_id]


def check_object_in_zone_dino(frame, model_name, zone_poly):
    """Registry entry shape:
        "wire": {
          "type": "grounding_dino_local",
          "model_id": "IDEA-Research/grounding-dino-tiny",
          "prompt": "damaged cable.",
          "confidence": 0.5
        }
    "prompt" must follow Grounding DINO's required format: lowercase,
    each phrase ends with a period."""
    import torch
    entry = registry.get(model_name, {})
    model_id = entry.get("model_id", "IDEA-Research/grounding-dino-tiny")
    prompt = entry.get("prompt", "damaged cable.")
    threshold = entry.get("confidence", 0.5)

    processor, model, device = _get_dino_model(model_id)

    image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=threshold, text_threshold=threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    for box, score in zip(results["boxes"], results["scores"]):
        x1, y1, x2, y2 = box.tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if point_in_polygon(cx, cy, zone_poly):
            return (x1, y1, x2, y2)
    return None


def check_object_in_zone(frame, target_model_name, zone_poly, loaded_models):
    """Dispatches to whichever backend the registry entry specifies."""
    entry = registry.get(target_model_name, {})
    entry_type = entry.get("type")
    if entry_type == "roboflow_remote":
        return check_object_in_zone_remote(frame, target_model_name, zone_poly)
    elif entry_type == "grounding_dino_local":
        return check_object_in_zone_dino(frame, target_model_name, zone_poly)
    if target_model_name not in loaded_models:
        return None
    return check_object_in_zone_local(frame, target_model_name, zone_poly, loaded_models)


# ============================================================
# BUILD ZONE LOOKUP
# ============================================================
zones_map = {}
for zone in config.get('zones', []):
    coords = zone['coords']
    zones_map[zone['name']] = {
        'x_min': min(p[0] for p in coords),
        'x_max': max(p[0] for p in coords),
        'y_min': min(p[1] for p in coords),
        'y_max': max(p[1] for p in coords),
        'coords': coords,
        'source': zone.get('source', 'llm_default'),
        'poly': coords,
    }

rules = config.get('rules', [])
print(f"Active zones: {list(zones_map.keys())}")
print(f"Active rules:")
for r in rules:
    print(f"  - {r['type']} in {r['zone']} (required: {r.get('required', [])})")
print()

# ============================================================
# PIPELINE STATE
# ============================================================
Path('incidents').mkdir(exist_ok=True)
if INCIDENTS_FILE.exists():
    INCIDENTS_FILE.unlink()

incident_count = 0
active_violations = {}

video = video_source or 'mega_cctv_v2.mp4'
print(f"Processing {video}...")

_cap = cv2.VideoCapture(video)
ORIG_W = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ORIG_H = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
_cap.release()
print(f"[INFO] Original video resolution: {ORIG_W}x{ORIG_H}\n")

SNAP_W, SNAP_H = 854, 480
for zn, zd in zones_map.items():
    if zd['source'] == 'user_drawn' and ORIG_W and ORIG_H:
        sx, sy = ORIG_W / SNAP_W, ORIG_H / SNAP_H
        zd['poly'] = [[p[0] * sx, p[1] * sy] for p in zd['coords']]
        zd['x_min'] = min(p[0] for p in zd['poly'])
        zd['x_max'] = max(p[0] for p in zd['poly'])
        zd['y_min'] = min(p[1] for p in zd['poly'])
        zd['y_max'] = max(p[1] for p in zd['poly'])
        print(f"[ZONE] '{zn}' user-drawn polygon scaled to native res ({len(zd['poly'])} points)")
    else:
        zd['poly'] = zd['coords']

# ============================================================
# MAIN PROCESSING LOOP
# ============================================================
results = base_model.track(
    source=video,
    persist=True,
    classes=[0],
    stream=True,
    conf=base_conf,
    verbose=False
)

for frame_idx, result in enumerate(results):
    frame_img = result.orig_img

    for rule_idx, rule in enumerate(rules):
        if rule.get('type') != 'object_in_zone':
            continue
        target = rule.get('target')
        zone_name = rule.get('zone', '')
        if not target or zone_name not in zones_map:
            continue

        zone = zones_map[zone_name]
        detected_bbox = check_object_in_zone(frame_img, target, zone['poly'], models)
        cooldown_key = (rule_idx, 'object')

        if detected_bbox is None:
            if cooldown_key in active_violations:
                if frame_idx - active_violations[cooldown_key] > 150:
                    del active_violations[cooldown_key]
            continue

        if cooldown_key not in active_violations:
            incident_count += 1
            incident_id = f"inc_{incident_count:04d}"
            screenshot_path = f"incidents/{incident_id}.jpg"

            orig_frame = frame_img.copy()
            bx1, by1, bx2, by2 = map(int, detected_bbox)
            cv2.rectangle(orig_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
            label = target
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(orig_frame, (bx1, by1 - lh - 8), (bx1 + lw + 4, by1), (0, 0, 255), -1)
            cv2.putText(orig_frame, label, (bx1 + 2, by1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            for zn, zd in zones_map.items():
                if zn.endswith('_full_frame'):
                    continue
                color = (0, 255, 255) if zn == zone_name else (0, 180, 180)
                pts = np.array(zd['poly'], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(orig_frame, [pts], isClosed=True, color=color, thickness=2)
                cv2.putText(orig_frame, zn.replace("_", " ").upper(),
                            (int(zd['x_min']) + 4, int(zd['y_min']) + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            cv2.imwrite(screenshot_path, orig_frame)

            incident = {
                "id":              incident_id,
                "timestamp":       datetime.now().isoformat(),
                "frame":           frame_idx,
                "camera":          video,
                "person_id":       None,
                "violation":       f"object_in_zone_{target}",
                "missing_gear":    [],
                "zone":            zone_name,
                "rule_index":      rule_idx,
                "bbox":            [float(b) for b in detected_bbox],
                "screenshot_path": screenshot_path,
                "rule_type":       "object_in_zone",
                "alert_message":   config['alert']['message']
            }
            append_incident(incident)
            print(f"Frame {frame_idx}: {target} detected in {zone_name} → {incident_id} [SAVED]")

        active_violations[cooldown_key] = frame_idx

    if result.boxes is None or result.boxes.id is None:
        continue

    for box, track_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.id.cpu().numpy()):
        person_id = int(track_id)
        x1, y1, x2, y2 = box
        person_center_x = (x1 + x2) / 2
        person_bottom_y = y2

        for rule_idx, rule in enumerate(rules):
            if rule.get('type') == 'object_in_zone':
                continue

            zone_name = rule.get('zone', '')
            if zone_name not in zones_map:
                continue

            zone = zones_map[zone_name]
            rule_type     = rule.get('type', '')
            required_gear = rule.get('required', [])

            in_zone = point_in_polygon(person_center_x, person_bottom_y, zone['poly'])

            cooldown_key = (rule_idx, person_id)

            if not in_zone:
                if cooldown_key in active_violations:
                    if frame_idx - active_violations[cooldown_key] > 150:
                        del active_violations[cooldown_key]
                continue

            violation_occurred = False
            violation_type     = rule_type
            missing_gear       = []

            if rule_type == "missing_in_zone" and required_gear:
                missing_gear = check_required_gear(
                    (x1, y1, x2, y2), result.orig_img, required_gear, models
                )
                if missing_gear:
                    violation_occurred = True
                    violation_type = f"missing_{'_'.join(missing_gear)}"

            elif rule_type == "person_in_zone":
                violation_occurred = True

            elif rule_type == "count_exceeded":
                violation_occurred = True
                violation_type = "person_in_zone"

            if violation_occurred and cooldown_key not in active_violations:
                incident_count += 1
                incident_id     = f"inc_{incident_count:04d}"
                screenshot_path = f"incidents/{incident_id}.jpg"

                orig_frame = result.orig_img.copy()

                if result.boxes is not None and len(result.boxes) > 0:
                    for det_box, det_id in zip(
                        result.boxes.xyxy.cpu().numpy(),
                        result.boxes.id.cpu().numpy() if result.boxes.id is not None else [None] * len(result.boxes)
                    ):
                        bx1, by1, bx2, by2 = map(int, det_box)
                        det_id_val = int(det_id) if det_id is not None else 0
                        cv2.rectangle(orig_frame, (bx1, by1), (bx2, by2), (255, 100, 0), 2)
                        label = f"id:{det_id_val} person"
                        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        cv2.rectangle(orig_frame, (bx1, by1 - lh - 8), (bx1 + lw + 4, by1), (255, 100, 0), -1)
                        cv2.putText(orig_frame, label, (bx1 + 2, by1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                for zn, zd in zones_map.items():
                    if zn.endswith('_full_frame'):
                        continue
                    color = (0, 255, 255) if zn == zone_name else (0, 180, 180)
                    pts = np.array(zd['poly'], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(orig_frame, [pts], isClosed=True, color=color, thickness=2)
                    cv2.putText(orig_frame, zn.replace("_", " ").upper(),
                                (int(zd['x_min']) + 4, int(zd['y_min']) + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                cv2.imwrite(screenshot_path, orig_frame)

                incident = {
                    "id":              incident_id,
                    "timestamp":       datetime.now().isoformat(),
                    "frame":           frame_idx,
                    "camera":          video,
                    "person_id":       person_id,
                    "violation":       violation_type,
                    "missing_gear":    missing_gear,
                    "zone":            zone_name,
                    "rule_index":      rule_idx,
                    "bbox":            [float(x1), float(y1), float(x2), float(y2)],
                    "screenshot_path": screenshot_path,
                    "rule_type":       rule_type,
                    "alert_message":   config['alert']['message']
                }

                append_incident(incident)

                print(f"Frame {frame_idx}: person #{person_id} | rule[{rule_idx}] {rule_type} in {zone_name}"
                      + (f" | missing: {missing_gear}" if missing_gear else "")
                      + f" → {incident_id} [SAVED]")

            active_violations[cooldown_key] = frame_idx

# ============================================================
# DONE
# ============================================================
try:
    db = SessionLocal()
    final_count = db.query(Incident).filter(Incident.site_id == site_id).count()
    db.close()
    print(f"\n{'='*60}")
    print(f"Done. {final_count} total incidents in PostgreSQL.")
    print(f"{'='*60}\n")
except Exception:
    final_count = len(json.loads(INCIDENTS_FILE.read_text())) if INCIDENTS_FILE.exists() else 0
    print(f"\n{'='*60}")
    print(f"Done. {final_count} unique incidents saved to JSON fallback.")
    print(f"{'='*60}\n")