import os
import cv2
import base64
import time
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from core.action_detector import ActionDetector
from core.emotion_detector import EmotionDetector
from core.human_state_fusion import HumanStateFusion
from core.tracker import SessionTracker
from kaggle_pipeline.download_kaggle import check_kaggle_credentials, DATASETS

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB
Path(app.config["UPLOAD_FOLDER"]).mkdir(exist_ok=True)

# Initialize perception engines
print("[*] Initializing Human Action and Emotion Detection Engines...")
action_detector = ActionDetector()
emotion_detector = EmotionDetector()
fusion_engine = HumanStateFusion()
session_tracker = SessionTracker()
print("[✓] AI Engines initialized and ready.")

def decode_base64_image(base64_str):
    """Converts base64 data URL to OpenCV BGR and RGB images."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr, rgb

def encode_image_base64(bgr_image):
    """Converts OpenCV BGR image to base64 jpeg string."""
    ret, buffer = cv2.imencode(".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ret:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/process_frame", methods=["POST"])
def process_frame():
    try:
        data = request.get_json(force=True)
        image_data = data.get("image")
        draw_pose = data.get("draw_pose", True)
        draw_face = data.get("draw_face", True)
        return_annotated = data.get("return_annotated", True)

        if not image_data:
            return jsonify({"error": "No image data provided"}), 400

        bgr, rgb = decode_base64_image(image_data)

        # 1. Action & Posture Sensing
        action_res = action_detector.detect(rgb)

        # 2. Mood & Emotion Sensing
        emotion_res = emotion_detector.detect(rgb)

        # 3. Context Fusion
        fused = fusion_engine.fuse(action_res, emotion_res)

        # 4. Session Timeline & Analytics
        session_tracker.update(fused)

        annotated_b64 = ""
        if return_annotated:
            annotated_bgr = bgr.copy()
            if draw_pose and action_res["landmarks"]:
                annotated_bgr = action_detector.draw_skeleton(annotated_bgr, action_res["landmarks"])
            if draw_face and emotion_res["landmarks"]:
                annotated_bgr = emotion_detector.draw_facemesh(annotated_bgr, emotion_res["landmarks"])
            annotated_b64 = encode_image_base64(annotated_bgr)

        stats = session_tracker.get_summary()

        return jsonify({
            "success": True,
            "fused_state": fused,
            "action": {
                "label": action_res["action"],
                "confidence": action_res["confidence"],
                "probabilities": action_res["probabilities"]
            },
            "emotion": {
                "label": emotion_res["emotion"],
                "clean_label": emotion_res["clean_emotion"],
                "confidence": emotion_res["confidence"],
                "probabilities": emotion_res["probabilities"],
                "biometrics": emotion_res.get("biometrics", {})
            },
            "annotated_image": annotated_b64,
            "stats": stats
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/demo_sample/<sample_id>")
def run_demo_sample(sample_id):
    samples = {
        "happy": "sample_data/sample_face_focused.jpg",
        "focused": "sample_data/sample_face_happy.jpg",
        "standing": "sample_data/sample_standing_happy.jpg"
    }
    path = samples.get(sample_id)
    if not path or not os.path.exists(path):
        return jsonify({"error": "Sample file not found"}), 404

    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    action_res = action_detector.detect(rgb)
    emotion_res = emotion_detector.detect(rgb)
    fused = fusion_engine.fuse(action_res, emotion_res)
    session_tracker.update(fused)

    annotated_bgr = bgr.copy()
    if action_res["landmarks"]:
        annotated_bgr = action_detector.draw_skeleton(annotated_bgr, action_res["landmarks"])
    if emotion_res["landmarks"]:
        annotated_bgr = emotion_detector.draw_facemesh(annotated_bgr, emotion_res["landmarks"])

    return jsonify({
        "success": True,
        "sample_id": sample_id,
        "fused_state": fused,
        "action": {
            "label": action_res["action"],
            "confidence": action_res["confidence"],
            "probabilities": action_res["probabilities"]
        },
        "emotion": {
            "label": emotion_res["emotion"],
            "clean_label": emotion_res["clean_emotion"],
            "confidence": emotion_res["confidence"],
            "probabilities": emotion_res["probabilities"],
            "biometrics": emotion_res.get("biometrics", {})
        },
        "annotated_image": encode_image_base64(annotated_bgr),
        "stats": session_tracker.get_summary()
    })

@app.route("/api/session_stats")
def get_stats():
    return jsonify(session_tracker.get_summary())

@app.route("/api/reset_session", methods=["POST"])
def reset_session():
    session_tracker.reset()
    return jsonify({"success": True, "message": "Session tracker reset."})

@app.route("/api/kaggle_status")
def get_kaggle_status():
    has_creds, msg = check_kaggle_credentials()
    fer_exists = Path("data/fer2013").exists()
    har_exists = Path("data/har").exists()
    models_emotion_exists = Path("models/emotion_classifier.joblib").exists()
    models_action_exists = Path("models/action_classifier.joblib").exists()

    return jsonify({
        "credentials_configured": has_creds,
        "credentials_message": msg,
        "datasets": {
            "fer2013": {
                "slug": DATASETS["fer2013"]["slug"],
                "downloaded": fer_exists,
                "description": DATASETS["fer2013"]["description"]
            },
            "har": {
                "slug": DATASETS["har"]["slug"],
                "downloaded": har_exists,
                "description": DATASETS["har"]["description"]
            }
        },
        "trained_models": {
            "emotion_classifier": models_emotion_exists,
            "action_classifier": models_action_exists
        }
    })

def start_server(host="0.0.0.0", port=5050, debug=False):
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    start_server(port=5050)
