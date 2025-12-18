#!/usr/bin/env python3
# app_eisenbahn.py – Eisenbahn-Koppelabstand-Überwachung
#
# Kamera (RTSP) filmt eine Eisenbahnplatte von oben.
# YOLO-Modell erkennt:
#   - Lokomotiven ("lok", "lokomotive", ...)
#   - Waggons ("wagen", "wagon", ...)
#
# Wenn der Abstand zwischen mindestens einer Lok und einem Waggon
# kleiner als ein Grenzwert ist, wird der Zustand "Koppelabstand"
# erkannt und optional ein MQTT-Command gesendet.
#
# Endpoints:
#   /           – Index (z.B. Debug/Config-UI)
#   /monitor    – Produktionsansicht (analog Kisten-Monitor)
#   /preview    – MJPEG mit Overlays
#   /preview_raw– MJPEG ohne Overlays
#   /status     – JSON-Status (inkl. lok/wagon, min_distance, coupling)
#   /config     – GET/POST Konfiguration (inkl. MQTT)
#
# HTTP only – kein SSL.

import os
import time
import json
import threading
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO
import torch

# YAML & MQTT sind optional, aber sehr sinnvoll
try:
    import yaml
except ImportError:
    yaml = None

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None


# ---------------------------------------------------------
# Device-Auswahl (GPU/CPU)
# ---------------------------------------------------------

def get_torch_device(prefer_gpu: bool = True) -> str:
    dev_env = os.environ.get("EISENBAHN_DEVICE", "").strip().lower()
    if dev_env:
        if dev_env in ("cpu", "cpu:0"):
            print("[INFO] EISENBAHN_DEVICE=cpu – erzwinge CPU-Modus.")
            return "cpu"
        if dev_env in ("cuda", "cuda:0", "0", "gpu"):
            try:
                if torch.cuda.is_available():
                    print("[INFO] EISENBAHN_DEVICE=cuda – nutze GPU.")
                    return "cuda:0"
                else:
                    print("[WARN] EISENBAHN_DEVICE=cuda, aber CUDA nicht verfügbar – CPU.")
            except Exception as e:
                print(f"[WARN] CUDA-Check fehlgeschlagen: {e} – CPU.")
            return "cpu"

    if prefer_gpu:
        try:
            if torch.cuda.is_available():
                print("[INFO] CUDA verfügbar – nutze GPU.")
                return "cuda:0"
            else:
                print("[INFO] CUDA nicht verfügbar – CPU.")
        except Exception as e:
            print(f"[WARN] CUDA-Check fehlgeschlagen: {e} – CPU.")
    return "cpu"


DEVICE = get_torch_device(prefer_gpu=True)
print(f"[INFO] Verwende Torch-Device: {DEVICE}")


# ---------------------------------------------------------
# Basis-Pfade & Default-Konfiguration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
CONFIG_PATH = BASE_DIR / "config_eisenbahn.yaml"

# Defaults (werden ggf. von YAML überschrieben)
RTSP_URL = "rtsp://admin:otto4546@10.151.20.12/"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

PROCESS_INTERVAL = 0.5
CONF_THRESH = 0.35
YOLO_IMGSZ = 1024

MODEL_PATH = MODEL_DIR / "eisenbahn" / "yolov8s_eisenbahn.pt"



# Klassen-Rollen (im YOLO-Modell)
LOK_CLASSES = ["dampflok", "diesellok"]
WAGON_CLASSES = ["wagon"]

# Grenzwert für Koppelabstand (Pixel)
COUPLING_DISTANCE_PX = 80.0

# MQTT-Konfiguration
MQTT_ENABLED = False
MQTT_HOST = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "eisenbahn/koppelabstand"
MQTT_PAYLOAD_ON = "KOPPEL_ABSTAND"
MQTT_USERNAME = None
MQTT_PASSWORD = None

# MQTT-Status
last_coupling_state = False
last_mqtt_info = None  # z.B. {"timestamp": "...", "topic": "...", "payload": "..."}

# RTSP über TCP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;5000000"


# ---------------------------------------------------------
# YAML-Konfiguration laden/schreiben
# ---------------------------------------------------------

def load_config_from_yaml():
    global RTSP_URL, FRAME_WIDTH, FRAME_HEIGHT
    global PROCESS_INTERVAL, CONF_THRESH, YOLO_IMGSZ
    global MODEL_PATH, LOK_CLASSES, WAGON_CLASSES
    global COUPLING_DISTANCE_PX
    global MQTT_ENABLED, MQTT_HOST, MQTT_PORT, MQTT_TOPIC, MQTT_PAYLOAD_ON
    global MQTT_USERNAME, MQTT_PASSWORD

    if yaml is None:
        print("[WARN] PyYAML nicht installiert – YAML-Konfiguration wird nicht geladen.")
        return

    if not CONFIG_PATH.exists():
        print(f"[INFO] Keine YAML-Config gefunden ({CONFIG_PATH}), nutze Default-Werte.")
        return

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[WARN] Konnte {CONFIG_PATH} nicht lesen: {e}")
        return

    RTSP_URL = cfg.get("rtsp_url", RTSP_URL)

    FRAME_WIDTH = int(cfg.get("frame_width", FRAME_WIDTH))
    FRAME_HEIGHT = int(cfg.get("frame_height", FRAME_HEIGHT))

    PROCESS_INTERVAL = float(cfg.get("process_interval", PROCESS_INTERVAL))
    CONF_THRESH = float(cfg.get("conf_thresh", CONF_THRESH))
    YOLO_IMGSZ = int(cfg.get("yolo_imgsz", YOLO_IMGSZ))

    model_cfg = cfg.get("model", {}) or {}
    model_path_str = model_cfg.get("path")
    if model_path_str:
        MODEL_PATH = BASE_DIR / model_path_str

    LOK_CLASSES[:] = model_cfg.get("lok_classes", LOK_CLASSES)
    WAGON_CLASSES[:] = model_cfg.get("wagon_classes", WAGON_CLASSES)

    coupling_cfg = cfg.get("coupling", {}) or {}
    COUPLING_DISTANCE_PX = float(coupling_cfg.get("distance_px", COUPLING_DISTANCE_PX))

    mqtt_cfg = cfg.get("mqtt", {}) or {}
    MQTT_ENABLED = bool(mqtt_cfg.get("enabled", MQTT_ENABLED))
    MQTT_HOST = mqtt_cfg.get("host", MQTT_HOST)
    MQTT_PORT = int(mqtt_cfg.get("port", MQTT_PORT))
    MQTT_TOPIC = mqtt_cfg.get("topic", MQTT_TOPIC)
    MQTT_PAYLOAD_ON = mqtt_cfg.get("payload_on", MQTT_PAYLOAD_ON)
    MQTT_USERNAME = mqtt_cfg.get("username", MQTT_USERNAME)
    MQTT_PASSWORD = mqtt_cfg.get("password", MQTT_PASSWORD)

    print("[INFO] YAML-Konfiguration geladen.")


def save_config_to_yaml():
    if yaml is None:
        print("[WARN] PyYAML nicht installiert – YAML-Konfiguration kann nicht gespeichert werden.")
        return

    cfg = {
        "rtsp_url": RTSP_URL,
        "frame_width": FRAME_WIDTH,
        "frame_height": FRAME_HEIGHT,
        "process_interval": PROCESS_INTERVAL,
        "conf_thresh": CONF_THRESH,
        "yolo_imgsz": YOLO_IMGSZ,
        "model": {
            "path": str(MODEL_PATH.relative_to(BASE_DIR)) if MODEL_PATH.is_absolute() else str(MODEL_PATH),
            "lok_classes": LOK_CLASSES,
            "wagon_classes": WAGON_CLASSES,
        },
        "coupling": {
            "distance_px": COUPLING_DISTANCE_PX,
        },
        "mqtt": {
            "enabled": MQTT_ENABLED,
            "host": MQTT_HOST,
            "port": MQTT_PORT,
            "topic": MQTT_TOPIC,
            "payload_on": MQTT_PAYLOAD_ON,
            "username": MQTT_USERNAME,
            "password": MQTT_PASSWORD,
        },
    }

    try:
        with CONFIG_PATH.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print("[INFO] YAML-Konfiguration gespeichert:", CONFIG_PATH)
    except Exception as e:
        print(f"[WARN] Konnte YAML-Konfiguration nicht speichern: {e}")


load_config_from_yaml()


# ---------------------------------------------------------
# Globale Zustände (Frames & Erkennung)
# ---------------------------------------------------------

latest_frame_raw = None
latest_frame_overlay = None

frame_lock = threading.Lock()

detector = None      # YOLO-Modell
detections = []      # Liste von Dicts (lok/waggon)
coupling_active = False
min_distance_px = None
pairs_info = []      # Liste von Paaren lok/waggon + Abstand


# ---------------------------------------------------------
# YOLO-Modell laden
# ---------------------------------------------------------

def load_detector():
    global detector
    if not MODEL_PATH.exists():
        print(f"[WARN] YOLO-Modell nicht gefunden: {MODEL_PATH}")
        detector = None
        return

    print(f"[INFO] Lade Eisenbahn-YOLO-Modell: {MODEL_PATH}")
    try:
        detector = YOLO(str(MODEL_PATH))
        detector.to("cpu")
        print(f"[INFO] YOLO-Modell auf Device {DEVICE} verschoben.")
    except Exception as e:
        print(f"[WARN] YOLO-Modell nicht auf {DEVICE}: {e}, versuche CPU.")
        try:
            detector.to("cpu")
        except Exception as e2:
            print(f"[ERROR] YOLO-Modell auch nicht auf CPU nutzbar: {e2}")
            detector = None


# ---------------------------------------------------------
# RTSP-Kamera-Thread
# ---------------------------------------------------------

def camera_loop():
    global latest_frame_raw, RTSP_URL

    cap = None
    while True:
        if cap is None or not cap.isOpened():
            print(f"[INFO] Verbinde RTSP-Stream: {RTSP_URL}")
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print("[WARN] RTSP-Stream konnte nicht geöffnet werden. Retry in 5s ...")
                time.sleep(5)
                continue

        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Kein Frame gelesen. Reconnect ...")
            cap.release()
            cap = None
            time.sleep(1)
            continue

        h, w = frame.shape[:2]
        if w != FRAME_WIDTH or h != FRAME_HEIGHT:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)

        with frame_lock:
            latest_frame_raw = frame

        time.sleep(0.01)


# ---------------------------------------------------------
# MQTT-Helfer
# ---------------------------------------------------------

def send_mqtt_coupling_event():
    """
    Sendet ein MQTT-Event bei erkanntem Koppelabstand.
    Nur eine einfache "ON"-Message, um Überflutung zu vermeiden
    (wird nur bei neu erkanntem "coupling_active=True" gesendet).
    """
    global last_mqtt_info

    if not MQTT_ENABLED:
        print("[DEBUG] MQTT ist deaktiviert, kein Publish.")
        return

    if mqtt is None:
        print("[WARN] paho-mqtt nicht installiert – MQTT nicht möglich.")
        return

    try:
        client = mqtt.Client()
        if MQTT_USERNAME:
            client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD or "")
        client.connect(MQTT_HOST, MQTT_PORT, keepalive=10)
        client.loop_start()
        client.publish(MQTT_TOPIC, MQTT_PAYLOAD_ON, qos=0, retain=False)
        time.sleep(0.2)
        client.loop_stop()
        client.disconnect()

        ts = datetime.now().isoformat(timespec="seconds")
        last_mqtt_info = {
            "timestamp": ts,
            "topic": MQTT_TOPIC,
            "payload": MQTT_PAYLOAD_ON,
        }
        print(f"[INFO] MQTT gesendet: {MQTT_TOPIC} -> {MQTT_PAYLOAD_ON} ({ts})")

    except Exception as e:
        print(f"[WARN] MQTT-Publish fehlgeschlagen: {e}")


# ---------------------------------------------------------
# Detection-Thread (Lok/Waggon + Koppelabstand)
# ---------------------------------------------------------

def compute_center(box):
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return cx, cy


def detection_loop():
    """
    - Holt periodisch das aktuelle RAW-Bild
    - YOLO-Erkennung für Lok/Waggon
    - Berechnet minimalen Abstand
    - Zeichnet Overlays
    - Setzt coupling_active + MQTT bei Zustandswechsel
    """
    global latest_frame_overlay, detections, coupling_active, min_distance_px, pairs_info
    global last_coupling_state

    print("[INFO] Detection-Loop (Eisenbahn) gestartet.")
    last_ts = 0.0

    while True:
        time.sleep(0.01)

        now = time.time()
        interval = PROCESS_INTERVAL if PROCESS_INTERVAL > 0 else 0.1
        if now - last_ts < interval:
            continue
        last_ts = now

        with frame_lock:
            if latest_frame_raw is None:
                continue
            frame = latest_frame_raw.copy()

        overlay = frame.copy()
        current_dets = []

        # YOLO-Erkennung
        if detector is not None:
            results = detector(
                frame,
                imgsz=YOLO_IMGSZ,
                conf=CONF_THRESH,
                verbose=False,
            )[0]

            names = getattr(detector, "names", None)

            lok_boxes = []
            wagon_boxes = []

            if results.boxes is not None and len(results.boxes) > 0:
                for b in results.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    conf = float(b.conf[0])
                    try:
                        cls_id = int(b.cls[0])
                    except Exception:
                        cls_id = -1

                    label = str(cls_id)
                    if isinstance(names, dict):
                        label = names.get(cls_id, label)
                    elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                        label = names[cls_id]

                    x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)

                    # Rolle bestimmen
                    label_lower = label.lower()
                    if label_lower in [c.lower() for c in LOK_CLASSES]:
                        role = "lok"
                        lok_boxes.append((x1i, y1i, x2i, y2i, conf, label))
                    elif label_lower in [c.lower() for c in WAGON_CLASSES]:
                        role = "wagon"
                        wagon_boxes.append((x1i, y1i, x2i, y2i, conf, label))
                    else:
                        role = "other"

                    current_dets.append({
                        "x1": x1i,
                        "y1": y1i,
                        "x2": x2i,
                        "y2": y2i,
                        "conf": conf,
                        "cls": cls_id,
                        "label": label,
                        "role": role,
                    })

            # Lok/Waggon farblich unterscheiden
            for (x1, y1, x2, y2, conf, label) in lok_boxes:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blau für Lok
                cv2.putText(
                    overlay,
                    f"{label} {conf:.2f}",
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            for (x1, y1, x2, y2, conf, label) in wagon_boxes:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)  # grün für Waggon
                cv2.putText(
                    overlay,
                    f"{label} {conf:.2f}",
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            # Koppelabstand berechnen
            min_dist = None
            pairs = []

            for (lx1, ly1, lx2, ly2, lconf, llabel) in lok_boxes:
                lc = compute_center((lx1, ly1, lx2, ly2))
                for (wx1, wy1, wx2, wy2, wconf, wlabel) in wagon_boxes:
                    wc = compute_center((wx1, wy1, wx2, wy2))
                    dx = lc[0] - wc[0]
                    dy = lc[1] - wc[1]
                    dist = (dx * dx + dy * dy) ** 0.5
                    pairs.append({
                        "lok": {"x1": lx1, "y1": ly1, "x2": lx2, "y2": ly2, "label": llabel},
                        "wagon": {"x1": wx1, "y1": wy1, "x2": wx2, "y2": wy2, "label": wlabel},
                        "distance_px": dist,
                    })
                    if (min_dist is None) or (dist < min_dist):
                        min_dist = dist
                        best_pair = ((lx1, ly1, lx2, ly2), (wx1, wy1, wx2, wy2), dist)

            # ggf. beste Paarung hervorheben
            if min_dist is not None:
                (lx1, ly1, lx2, ly2), (wx1, wy1, wx2, wy2), dist = best_pair
                # Linie zwischen den Zentren
                lcx, lcy = compute_center((lx1, ly1, lx2, ly2))
                wcx, wcy = compute_center((wx1, wy1, wx2, wy2))
                cv2.line(overlay, (int(lcx), int(lcy)), (int(wcx), int(wcy)), (0, 255, 255), 2)
                cv2.putText(
                    overlay,
                    f"dist={dist:.1f}px",
                    (int((lcx + wcx) / 2), int((lcy + wcy) / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )

            # Zustand Koppelabstand
            coupling = (min_dist is not None) and (min_dist <= COUPLING_DISTANCE_PX)

            # Status-Overlay-Text
            if coupling:
                cv2.putText(
                    overlay,
                    "WAGON AUF KOPPELABSTAND",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )
            else:
                cv2.putText(
                    overlay,
                    "Kein Koppelabstand",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

            # MQTT / Zustandswechsel
            if coupling and not last_coupling_state:
                print("[INFO] Neuer Koppelabstand erkannt – MQTT-Event.")
                send_mqtt_coupling_event()
            last_coupling_state = coupling

            # Globale Zustände aktualisieren
            with frame_lock:
                latest_frame_overlay = overlay
                detections = current_dets
                coupling_active = coupling
                min_distance_px = min_dist
                pairs_info = pairs
        else:
            # Kein Modell geladen
            with frame_lock:
                latest_frame_overlay = frame
                detections = []
                coupling_active = False
                min_distance_px = None
                pairs_info = []


# ---------------------------------------------------------
# Flask-App & MJPEG-Streams
# ---------------------------------------------------------

app = Flask(__name__)


def gen_stream_frame_overlay():
    while True:
        with frame_lock:
            frame = latest_frame_overlay

        if frame is None:
            img = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            cv2.putText(
                img,
                "Warte auf Kamera / Modell ...",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            ret, buf = cv2.imencode(".jpg", img)
        else:
            ret, buf = cv2.imencode(".jpg", frame)

        if not ret:
            time.sleep(0.1)
            continue

        jpg = buf.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        )
        time.sleep(0.1)


def gen_stream_frame_raw():
    while True:
        with frame_lock:
            frame = latest_frame_raw

        if frame is None:
            img = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            cv2.putText(
                img,
                "Warte auf Kamera ...",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            ret, buf = cv2.imencode(".jpg", img)
        else:
            ret, buf = cv2.imencode(".jpg", frame)

        if not ret:
            time.sleep(0.1)
            continue

        jpg = buf.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        )
        time.sleep(0.1)


# ---------------------------------------------------------
# Flask-Routen
# ---------------------------------------------------------

@app.route("/")
def index():
    # Kann analog deiner Debug-/Config-Seite gestaltet werden
    return render_template("index.html")


@app.route("/monitor")
def monitor():
    # Produktionsansicht (Rohbild + JS-BBox-Layer, wie bei dir)
    return render_template("monitor.html")


@app.route("/preview")
def preview():
    return Response(
        gen_stream_frame_overlay(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/preview_raw")
def preview_raw():
    return Response(
        gen_stream_frame_raw(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    """
    Liefert aktuellen Status der Eisenbahn-Erkennung.
    Struktur ist an deine Kisten-App angelehnt und erweitert:
    {
      "detections": [...],
      "coupling": bool,
      "min_distance_px": float | null,
      "threshold_px": float,
      "lok_count": int,
      "wagon_count": int,
      "pairs": [...],
      "mqtt": { ... } | null
    }
    """
    with frame_lock:
        dets = list(detections)
        coupling = bool(coupling_active)
        mdist = min_distance_px
        pairs = list(pairs_info)
        mqtt_info = dict(last_mqtt_info) if last_mqtt_info else None

    lok_count = sum(1 for d in dets if d.get("role") == "lok")
    wagon_count = sum(1 for d in dets if d.get("role") == "wagon")

    return jsonify({
        "detections": dets,
        "coupling": coupling,
        "min_distance_px": mdist,
        "threshold_px": COUPLING_DISTANCE_PX,
        "lok_count": lok_count,
        "wagon_count": wagon_count,
        "pairs": pairs,
        "mqtt": mqtt_info,
    })


@app.route("/config", methods=["GET", "POST"])
def config_route():
    """
    GET: aktuelle Konfiguration
    POST: Konfiguration aktualisieren (inkl. MQTT) + YAML speichern
    """
    global RTSP_URL, FRAME_WIDTH, FRAME_HEIGHT
    global PROCESS_INTERVAL, CONF_THRESH, YOLO_IMGSZ
    global COUPLING_DISTANCE_PX
    global MQTT_ENABLED, MQTT_HOST, MQTT_PORT, MQTT_TOPIC, MQTT_PAYLOAD_ON
    global MQTT_USERNAME, MQTT_PASSWORD

    if request.method == "GET":
        return jsonify({
            "rtsp_url": RTSP_URL,
            "frame_width": FRAME_WIDTH,
            "frame_height": FRAME_HEIGHT,
            "process_interval": PROCESS_INTERVAL,
            "conf_thresh": CONF_THRESH,
            "yolo_imgsz": YOLO_IMGSZ,
            "model_path": str(MODEL_PATH),
            "lok_classes": LOK_CLASSES,
            "wagon_classes": WAGON_CLASSES,
            "coupling_distance_px": COUPLING_DISTANCE_PX,
            "mqtt": {
                "enabled": MQTT_ENABLED,
                "host": MQTT_HOST,
                "port": MQTT_PORT,
                "topic": MQTT_TOPIC,
                "payload_on": MQTT_PAYLOAD_ON,
                "username": MQTT_USERNAME,
                "password": bool(MQTT_PASSWORD),  # nicht im Klartext ausgeben
            },
        })

    data = request.get_json(silent=True) or {}

    if "rtsp_url" in data:
        RTSP_URL = str(data["rtsp_url"]).strip() or RTSP_URL

    if "frame_width" in data:
        try:
            FRAME_WIDTH = int(data["frame_width"])
        except (TypeError, ValueError):
            pass

    if "frame_height" in data:
        try:
            FRAME_HEIGHT = int(data["frame_height"])
        except (TypeError, ValueError):
            pass

    if "process_interval" in data:
        try:
            PROCESS_INTERVAL = float(data["process_interval"])
        except (TypeError, ValueError):
            pass

    if "conf_thresh" in data:
        try:
            CONF_THRESH = float(data["conf_thresh"])
        except (TypeError, ValueError):
            pass

    if "yolo_imgsz" in data:
        try:
            YOLO_IMGSZ = int(data["yolo_imgsz"])
        except (TypeError, ValueError):
            pass

    if "coupling_distance_px" in data:
        try:
            COUPLING_DISTANCE_PX = float(data["coupling_distance_px"])
        except (TypeError, ValueError):
            pass

    # MQTT-Teil
    mcfg = data.get("mqtt", {})
    if isinstance(mcfg, dict):
        if "enabled" in mcfg:
            MQTT_ENABLED = bool(mcfg["enabled"])
        if "host" in mcfg:
            MQTT_HOST = str(mcfg["host"]).strip() or MQTT_HOST
        if "port" in mcfg:
            try:
                MQTT_PORT = int(mcfg["port"])
            except (TypeError, ValueError):
                pass
        if "topic" in mcfg:
            MQTT_TOPIC = str(mcfg["topic"]).strip() or MQTT_TOPIC
        if "payload_on" in mcfg:
            MQTT_PAYLOAD_ON = str(mcfg["payload_on"])
        if "username" in mcfg:
            MQTT_USERNAME = str(mcfg["username"]) or None
        if "password" in mcfg:
            # nur setzen, wenn explizit übergeben
            pwd = mcfg["password"]
            MQTT_PASSWORD = str(pwd) if pwd not in (None, "", False) else None

    save_config_to_yaml()

    return jsonify({"ok": True})


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    load_detector()

    t_cam = threading.Thread(target=camera_loop, daemon=True)
    t_det = threading.Thread(target=detection_loop, daemon=True)
    t_cam.start()
    t_det.start()

    # HTTP only – kein SSL
    app.run(
        host="0.0.0.0",
        port=5001,   # bei Bedarf anpassen
        debug=True,
    )
