# Kaggle Dataset Integration Guide

This project leverages standard computer vision and affective computing benchmark datasets from **Kaggle**:

---

## 1. Primary Datasets Used

### A. Facial Expression Recognition (FER-2013)
* **Kaggle URL**: [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)
* **Dataset Size**: ~35,887 grayscale images (48x48 pixels)
* **Classes**:
  1. `Angry` (4,953 samples)
  2. `Disgust` (547 samples)
  3. `Fear` (5,121 samples)
  4. `Happy` (8,989 samples)
  5. `Sad` (6,077 samples)
  6. `Surprise` (4,002 samples)
  7. `Neutral` (6,198 samples)
* **Application**: Real-time affective state detection, mood timeline tracking, and sentiment feedback.

### B. Human Action Recognition (HAR)
* **Kaggle URL**: [https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset)
* **Dataset Size**: ~12,600 annotated posture/action images
* **Classes**: `sitting`, `standing`, `clapping`, `cycling`, `dancing`, `eating`, `fighting`, `hugging`, `laughing`, `listening_to_music`, `running`, `sleeping`, `using_laptop`, etc.
* **Application**: Activity identification, desk posture analysis, ergonomic alerting.

---

## 2. Automated Download via Kaggle API

You can download the full datasets directly from the terminal or using our python helper.

### Step 1: Get Your Kaggle API Token
1. Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings).
2. Scroll to **API** section and click **Create New Token**.
3. A file named `kaggle.json` will be downloaded to your machine.

### Step 2: Configure Credentials
Place `kaggle.json` into:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```
Or export as environment variables:
```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

### Step 3: Run the Downloader Script
```bash
# Download FER-2013
python kaggle_pipeline/download_kaggle.py --dataset fer2013

# Download Human Action Recognition
python kaggle_pipeline/download_kaggle.py --dataset har

# Download all
python kaggle_pipeline/download_kaggle.py --dataset all
```

---

## 3. Retraining or Fine-Tuning Models

Run the retraining scripts at any time:
```bash
# Retrain emotion model
python kaggle_pipeline/train_emotion.py

# Retrain action model
python kaggle_pipeline/train_action.py
```
Trained serialized models will automatically be updated in the `models/` folder.
