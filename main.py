import argparse
import sys
import os
import cv2

# Set local directory overrides for Matplotlib / Kaggle config
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("KAGGLE_CONFIG_DIR", os.path.abspath(".kaggle"))

from core.action_detector import ActionDetector
from core.emotion_detector import EmotionDetector
from core.human_state_fusion import HumanStateFusion
from core.tracker import SessionTracker

def run_opencv_desktop_mode(source=0):
    """
    Runs a desktop OpenCV GUI window with live action & mood perception.
    """
    print(f"[*] Opening video source: {source}...")
    print("[*] Press 'q' inside the OpenCV window to exit.")
    
    # Check if source is a numeric camera index
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[!] Could not open video source {source}.")
        print("[!] Note: On macOS, ensure Terminal has Camera permission in System Settings -> Privacy & Security -> Camera.")
        return

    action_detector = ActionDetector()
    emotion_detector = EmotionDetector()
    tracker = SessionTracker()

    consecutive_failures = 0
    max_init_retries = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures < max_init_retries:
                import time
                time.sleep(0.05)
                continue
            else:
                print("\n[!] Unable to receive frames from video capture.")
                print("[!] Tips:")
                print("    1. On macOS, check: System Settings -> Privacy & Security -> Camera -> Enable for your Terminal app.")
                print("    2. Make sure no other app (FaceTime, Zoom, browser) is exclusively locking the webcam.")
                print("    3. Or use the Web Dashboard instead at http://localhost:5055, which uses your browser's webcam prompt.\n")
                break

        consecutive_failures = 0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect action & emotion
        act = action_detector.detect(rgb)
        emo = emotion_detector.detect(rgb)
        fused = HumanStateFusion.fuse(act, emo)
        tracker.update(fused)

        # Annotate skeleton and face mesh
        if act["landmarks"]:
            action_detector.draw_skeleton(frame, act["landmarks"])
        if emo["landmarks"]:
            emotion_detector.draw_facemesh(frame, emo["landmarks"])

        # Draw HUD overlays on frame
        h, w, _ = frame.shape
        # Top banner
        cv2.rectangle(frame, (0, 0), (w, 75), (15, 23, 42), -1)
        cv2.putText(frame, f"State: {fused['state_title']}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"Action: {act['action']} ({int(act['confidence']*100)}%)  |  Mood: {emo['emotion']} ({int(emo['confidence']*100)}%)",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 1)

        # Bottom recommendation
        cv2.rectangle(frame, (0, h - 35), (w, h), (15, 23, 42), -1)
        cv2.putText(frame, f"Insight: {fused['recommendation']}", (20, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        cv2.imshow("HumanSense AI - Desktop Mode (Press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_file_mode(file_path):
    """Processes a single image file or video file."""
    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}")
        return

    print(f"[*] Processing file: {file_path}")
    image = cv2.imread(file_path)
    if image is None:
        print("[!] Failed to decode image.")
        return

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    action_detector = ActionDetector()
    emotion_detector = EmotionDetector()

    act = action_detector.detect(rgb)
    emo = emotion_detector.detect(rgb)
    fused = HumanStateFusion.fuse(act, emo)

    print("\n" + "="*60)
    print("🧠 HUMANSENSE AI — PERCEPTION REPORT")
    print("="*60)
    print(f"• Compound State : {fused['state_title']} ({fused['category']})")
    print(f"• Sensed Action  : {act['action']} (Confidence: {act['confidence']*100:.1f}%)")
    print(f"• Sensed Mood    : {emo['emotion']} (Confidence: {emo['confidence']*100:.1f}%)")
    print(f"• Description    : {fused['description']}")
    print(f"• Recommendation : {fused['recommendation']}")
    print(f"• FACS Biometrics: {emo.get('biometrics', {})}")
    print("="*60 + "\n")

def run_web_mode(port=5055):
    import socket
    # Check if port is in use; if so, find an available port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('127.0.0.1', port)) == 0:
            port = 5055 if port != 5055 else 8080

    from app.web_app import start_server
    print(f"\n=======================================================")
    print(f"🚀 HumanSense AI Web Dashboard launching at:")
    print(f"   http://localhost:{port}")
    print(f"   http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    start_server(port=port)

def main():
    parser = argparse.ArgumentParser(description="HumanSense AI: Action & Mood Perception POC")
    parser.add_argument("--web", action="store_true", help="Launch interactive Web Dashboard (default)")
    parser.add_argument("--desktop", action="store_true", help="Launch native OpenCV desktop webcam window")
    parser.add_argument("--source", default=0, help="Video source for desktop mode (camera index e.g. 0, or path to video file)")
    parser.add_argument("--file", type=str, help="Process a specific image or video file")
    parser.add_argument("--port", type=int, default=5055, help="Web server port (default: 5055)")
    args = parser.parse_args()

    if args.desktop:
        run_opencv_desktop_mode(source=args.source)
    elif args.file:
        process_file_mode(args.file)
    else:
        run_web_mode(port=args.port)

if __name__ == "__main__":
    main()
