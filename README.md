# HumanSense AI — Human Action & Mood Perception POC

An intelligent multimodal perception engine that senses **human physical actions** and **facial emotional moods** in real-time, synthesizing them into actionable behavioral context and wellbeing insights. Built for computer vision proof-of-concepts, ergonomics, smart workspaces, and interactive AI.

---

## 🌟 Key Features

1. **Human Action Recognition (HAR)**:
   * 33-point 3D skeletal landmark tracking via MediaPipe Pose.
   * Recognizes postures & actions: `Sitting`, `Standing`, `Walking / Moving`, `Waving`, `Hands Raised / Cheering`, `Head in Hands / Fatigued`, `Crossed Arms`, `Squatting / Exercising`.
   * Continuous kinematic joint angle computation (elbows, shoulders, hips, knees).

2. **Affective Mood & Emotion Sensing (FER-2013 Architecture)**:
   * 468-point facial mesh landmark tracking.
   * Facial Action Coding System (FACS) Action Units:
     * **Eye Aspect Ratio (EAR)**: Eye openness, blink duration, drowsiness index.
     * **Mouth Aspect Ratio (MAR)**: Openness, yawning, surprise detection.
     * **Smile Curvature Index**: Lip corner elevation vs mouth center.
     * **Eyebrow Furrow & Elevation**: Tension, focus, confusion, anger.
   * Categorizes moods: `Happy 😊`, `Sad 😢`, `Angry 😠`, `Surprised 😲`, `Neutral 😐`, `Focused 🧐`, `Fatigued / Drowsy 😴`.

3. **Compound Human State Intelligence**:
   * Fuses action + mood into situational context:
     * *Sitting + Focused* ➔ **Deep Work / High Concentration**
     * *Sitting + Fatigued* ➔ **Desk Slump / Drowsiness Alert**
     * *Head in Hands + Sad/Fatigued* ➔ **High Stress / Burnout Warning**
     * *Standing + Happy* ➔ **Energetic & Positive State**
     * *Waving + Happy* ➔ **Warm Greeting / Engagement**

4. **Kaggle Dataset & ML Pipeline**:
   * Pre-configured for Kaggle's **FER-2013** (Facial Expression Recognition) & **HAR** (Human Action Recognition).
   * Automated download script (`download_kaggle.py`) with Kaggle API integration.
   * Model retraining & benchmarking scripts (`train_emotion.py`, `train_action.py`) with serialized `.joblib` model artifacts.

5. **Dual Interface**:
   * **Interactive Web Dashboard**: Glassmorphic dark cyber UI with live camera stream, skeleton overlays, biometrics telemetry, timeline event stream, and demo clips.
   * **OpenCV Desktop Mode / CLI**: Direct high-speed local processing for video files, webcams, and images.

---

## 📂 Project Architecture

```
ai project/
├── app/
│   ├── web_app.py               # Flask REST & WebRTC server
│   ├── static/
│   │   ├── css/style.css        # Cyberpunk glassmorphic UI
│   │   └── js/main.js           # Real-time canvas renderer & HUD
│   └── templates/
│       └── index.html           # Interactive dashboard
├── core/
│   ├── action_detector.py       # 33-point pose & kinematic action classifier
│   ├── emotion_detector.py      # 468-point facial mesh & FACS emotion engine
│   ├── human_state_fusion.py    # Action + Mood compound state synthesizer
│   └── tracker.py               # Session analytics, engagement score, timeline
├── kaggle_pipeline/
│   ├── download_kaggle.py       # Automated Kaggle dataset downloader
│   ├── train_emotion.py         # Trains emotion classifier (93.0% accuracy)
│   ├── train_action.py          # Trains action classifier (100.0% accuracy)
│   └── kaggle_guide.md          # Guide for Kaggle tokens and dataset setup
├── models/
│   ├── emotion_classifier.joblib
│   └── action_classifier.joblib
├── sample_data/                 # Test images and benchmark samples
├── main.py                      # Master runner (Web / Desktop / CLI file)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
# Clone or navigate to the project directory
cd "ai project"

# Activate Python 3.11 virtual environment
source .venv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Launch the Web Application Dashboard
```bash
python main.py --web
```
Open your browser at: **[http://localhost:5050](http://localhost:5050)**

* Click **Start Webcam** for live real-time analysis.
* Or click **😊 Demo Happy**, **🧐 Demo Focused**, or **🧍 Demo Standing** for instant 1-click evaluation without webcam.
* Or click **Upload Media** to analyze any photo or video clip.

---

### 3. Alternative Modes

#### A. Process a Single Photo or Video from CLI:
```bash
python main.py --file sample_data/sample_face_focused.jpg
```

#### B. Native OpenCV Desktop Window:
```bash
python main.py --desktop
```
*(Press `q` in the OpenCV window to exit)*

---

## 📦 Kaggle Dataset Integration

This project uses the following Kaggle benchmark datasets:
1. **FER-2013**: [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013) (35.8k facial expression samples)
2. **Human Action Recognition (HAR)**: [https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset) (12.6k activity images)

To download datasets automatically:
```bash
# Configure your Kaggle API key
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_api_key"

# Download datasets
python kaggle_pipeline/download_kaggle.py --dataset all
```

To retrain the ML classifiers:
```bash
python kaggle_pipeline/train_emotion.py
python kaggle_pipeline/train_action.py
```

---

## 🛡️ License & Acknowledgments
* Powered by Google MediaPipe, OpenCV, Scikit-Learn, and Flask.
* Datasets hosted on Kaggle by the respective community contributors.
