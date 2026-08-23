"""
SafeNest AI - Phase 1 Web Dashboard (Flask backend)

Runs the YOLO model server-side. The browser captures webcam frames via
getUserMedia and posts them here; this returns detections + a risk score
as JSON, and the browser draws the overlay.

Run:
    python app.py
Then open:
    http://localhost:5000
"""

import base64
import json
import os
import re
import time
from datetime import datetime, timezone

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
from twilio.rest import Client


def load_env(env_path=None):
    """Load key-value pairs from .env file into os.environ."""
    if env_path is None:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r'^(?:(?:export|\$env:)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*["\']?(.*?)["\']?$', line)
            if m:
                key, val = m.group(1), m.group(2)
                os.environ[key] = val


load_env()


from config import (
    PERSON_CLASSES,
    DANGEROUS_CLASSES,
    NEUTRAL_HIGHLIGHT_CLASSES,
    CONF_THRESHOLD,
    danger_weight,
    box_center,
    pixel_distance,
    simple_risk_score,
    risk_label,
)

app = Flask(__name__, template_folder=".", static_folder=None)
ALERTS_DIR = os.path.join(app.root_path, "saved_alerts")
CRITICAL_THRESHOLD = 70

print("Loading YOLO model (first run downloads weights automatically)...")
model = YOLO("yolov8n.pt")
model.predict(np.zeros((320, 320, 3), dtype=np.uint8), imgsz=320, device="cpu", verbose=False)
print("Model loaded.")


def decode_image(data_url: str):
    """Turn a 'data:image/jpeg;base64,...' string into a BGR numpy array."""
    header, b64data = data_url.split(",", 1)
    img_bytes = base64.b64decode(b64data)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def save_alert(data_url: str, risk: dict):
    """Save the alert image and metadata locally for later review."""
    _, b64data = data_url.split(",", 1)
    image_bytes = base64.b64decode(b64data)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    os.makedirs(ALERTS_DIR, exist_ok=True)
    image_name = f"alert_{timestamp}.jpg"
    metadata_name = f"alert_{timestamp}.json"
    with open(os.path.join(ALERTS_DIR, image_name), "wb") as image_file:
        image_file.write(image_bytes)
    with open(os.path.join(ALERTS_DIR, metadata_name), "w", encoding="utf-8") as metadata_file:
        json.dump({"saved_at": timestamp, "risk": risk, "image": image_name}, metadata_file, indent=2)
    return image_name


def call_family_member(risk: dict):
    """Place an emergency call using backend-only Twilio configuration or local dev simulation."""
    dev_test_mode = os.environ.get("DEV_TEST_MODE", "false").strip().lower() in ("true", "1", "yes")
    object_name = risk.get("object") or "a dangerous item"

    if dev_test_mode:
        sim_sid = f"SIM_CALL_{int(time.time())}"
        app.logger.info(f"[DEV_TEST_MODE] Simulated emergency call initiated for {object_name} (SID: {sim_sid})")
        return {"sid": sim_sid, "simulated": True, "object": object_name}

    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_number = os.environ.get("TWILIO_FROM_NUMBER")
    family_number = os.environ.get("FAMILY_PHONE_NUMBER")
    twiml_url = os.environ.get("TWILIO_TWIML_URL") or "http://demo.twilio.com/docs/voice.xml"

    if not all((account_sid, auth_token, from_number, family_number)):
        missing = []
        if not account_sid:
            missing.append("TWILIO_ACCOUNT_SID")
        if not auth_token:
            missing.append("TWILIO_AUTH_TOKEN")
        if not from_number:
            missing.append("TWILIO_FROM_NUMBER")
        if not family_number:
            missing.append("FAMILY_PHONE_NUMBER")
        raise RuntimeError(f"Missing Twilio configuration in .env: {', '.join(missing)}")
    if float(risk.get("score", 0)) < CRITICAL_THRESHOLD:
        raise ValueError("emergency call requires a critical risk score")

    client = Client(account_sid, auth_token)
    # Using 'url' parameter to ensure 100% compatibility with Twilio Trial accounts
    call = client.calls.create(
        to=family_number,
        from_=from_number,
        url=twiml_url,
    )
    return {"sid": call.sid, "simulated": False}


@app.route("/static/<path:filename>", endpoint="static")
def static_files(filename):
    return send_from_directory(app.root_path, filename)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image")
    if not data_url:
        return jsonify({"error": "no image provided"}), 400

    frame = decode_image(data_url)
    if frame is None:
        return jsonify({"error": "could not decode image"}), 400

    t0 = time.time()
    results = model.predict(
        frame,
        conf=CONF_THRESHOLD,
        imgsz=320,
        device="cpu",
        max_det=20,
        verbose=False,
    )[0]
    infer_ms = round((time.time() - t0) * 1000, 1)

    h, w = frame.shape[:2]
    detections = []
    persons, dangers = [], []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        center = box_center((x1, y1, x2, y2))

        if class_name in PERSON_CLASSES:
            category = "person"
            persons.append(center)
        elif class_name in DANGEROUS_CLASSES:
            category = "danger"
            dangers.append((class_name, center, danger_weight(class_name)))
        elif class_name in NEUTRAL_HIGHLIGHT_CLASSES:
            category = "neutral"
        else:
            continue

        detections.append({
            "class": class_name,
            "conf": round(conf, 3),
            "category": category,
            # normalized 0-1 coords so the frontend can scale to any canvas size
            "box": [x1 / w, y1 / h, x2 / w, y2 / h],
        })

    best_score, best_label, best_object = 0.0, "SAFE", None

    if persons and dangers:
        # Both person and dangerous hazard in frame -> proximity-weighted score
        for person_center in persons:
            for class_name, danger_center, weight in dangers:
                dist = pixel_distance(person_center, danger_center)
                score = simple_risk_score(weight, dist)
                if score > best_score:
                    best_score, best_object = score, class_name
                    best_label = risk_label(score)
    elif dangers:
        # Dangerous hazard present in monitored area
        for class_name, danger_center, weight in dangers:
            score = round(weight * 0.8, 1)
            if score > best_score:
                best_score, best_object = score, class_name
                best_label = risk_label(score)

    return jsonify({
        "detections": detections,
        "risk": {
            "score": best_score,
            "label": best_label,
            "object": best_object,
            "has_person": len(persons) > 0,
        },
        "infer_ms": infer_ms,
    })


@app.route("/save-alert", methods=["POST"])
def save_alert_route():
    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image")
    risk = payload.get("risk") or {}
    if not data_url or not data_url.startswith("data:image/"):
        return jsonify({"error": "valid alert image is required"}), 400
    try:
        image_name = save_alert(data_url, risk)
    except (ValueError, TypeError, base64.binascii.Error, OSError) as error:
        return jsonify({"error": f"could not save alert: {error}"}), 400
    return jsonify({"saved": True, "image": image_name})


@app.route("/call-family", methods=["POST"])
def call_family_route():
    payload = request.get_json(silent=True) or {}
    try:
        call_info = call_family_member(payload.get("risk") or {})
    except (RuntimeError, ValueError) as error:
        return jsonify({"called": False, "error": str(error)}), 503
    except Exception as error:
        app.logger.exception("Twilio emergency call failed")
        return jsonify({"called": False, "error": str(error)}), 502
    return jsonify({
        "called": True,
        "call_sid": call_info.get("sid"),
        "simulated": call_info.get("simulated", False),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
