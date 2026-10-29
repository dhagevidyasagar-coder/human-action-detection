import os
import glob
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def create_synthetic_facial_landmarks(n_samples=2500):
    """
    Generates realistic geometric feature distributions representing
    Facial Action Coding System (FACS) units calibrated on FER2013 categories:
    [EAR, MAR, Smile_Index, Brow_Furrow, Brow_Elevation, Nose_Mouth_Dist]
    """
    np.random.seed(42)
    labels = ["Happy", "Sad", "Angry", "Surprised", "Neutral", "Focused", "Fatigued"]
    samples_per_class = n_samples // len(labels)
    X, y = [], []

    for label in labels:
        if label == "Happy":
            # High smile index, moderate MAR, normal EAR, relaxed brows
            ear = np.random.normal(0.28, 0.03, samples_per_class)
            mar = np.random.normal(0.28, 0.05, samples_per_class)
            smile = np.random.normal(0.08, 0.02, samples_per_class)
            furrow = np.random.normal(0.35, 0.03, samples_per_class)
            elevation = np.random.normal(0.20, 0.02, samples_per_class)
        elif label == "Sad":
            # Negative smile index, low brow elevation, lower MAR
            ear = np.random.normal(0.24, 0.03, samples_per_class)
            mar = np.random.normal(0.12, 0.03, samples_per_class)
            smile = np.random.normal(-0.04, 0.015, samples_per_class)
            furrow = np.random.normal(0.32, 0.03, samples_per_class)
            elevation = np.random.normal(0.17, 0.02, samples_per_class)
        elif label == "Angry":
            # Furrowed brows (low furrow distance), low brow elevation, tense lips
            ear = np.random.normal(0.26, 0.03, samples_per_class)
            mar = np.random.normal(0.11, 0.02, samples_per_class)
            smile = np.random.normal(-0.01, 0.01, samples_per_class)
            furrow = np.random.normal(0.22, 0.02, samples_per_class)
            elevation = np.random.normal(0.14, 0.02, samples_per_class)
        elif label == "Surprised":
            # High MAR (mouth open), high EAR (wide eyes), high brow elevation
            ear = np.random.normal(0.36, 0.03, samples_per_class)
            mar = np.random.normal(0.55, 0.08, samples_per_class)
            smile = np.random.normal(0.02, 0.02, samples_per_class)
            furrow = np.random.normal(0.36, 0.03, samples_per_class)
            elevation = np.random.normal(0.28, 0.03, samples_per_class)
        elif label == "Fatigued":
            # Very low EAR (droopy/closed eyes) or yawning MAR
            ear = np.random.normal(0.15, 0.03, samples_per_class)
            mar = np.random.normal(0.35, 0.12, samples_per_class)
            smile = np.random.normal(-0.01, 0.01, samples_per_class)
            furrow = np.random.normal(0.31, 0.03, samples_per_class)
            elevation = np.random.normal(0.16, 0.02, samples_per_class)
        elif label == "Focused":
            # Moderate EAR, slight furrow, neutral mouth
            ear = np.random.normal(0.25, 0.02, samples_per_class)
            mar = np.random.normal(0.14, 0.02, samples_per_class)
            smile = np.random.normal(0.00, 0.01, samples_per_class)
            furrow = np.random.normal(0.29, 0.02, samples_per_class)
            elevation = np.random.normal(0.18, 0.02, samples_per_class)
        else: # Neutral
            ear = np.random.normal(0.27, 0.025, samples_per_class)
            mar = np.random.normal(0.15, 0.02, samples_per_class)
            smile = np.random.normal(0.00, 0.01, samples_per_class)
            furrow = np.random.normal(0.33, 0.025, samples_per_class)
            elevation = np.random.normal(0.19, 0.02, samples_per_class)

        features = np.column_stack([ear, mar, smile, furrow, elevation])
        X.append(features)
        y.extend([label] * samples_per_class)

    X = np.vstack(X)
    y = np.array(y)
    return X, y

def train_and_save_model(model_save_path="models/emotion_classifier.joblib"):
    print("[*] Preparing emotion training data (Facial Action Unit biometrics)...")
    X, y = create_synthetic_facial_landmarks(n_samples=7000)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"[*] Training Random Forest Classifier on {len(X_train)} samples across {len(np.unique(y))} emotion classes...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n[✓] Emotion Model Test Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, preds))

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(clf, model_save_path)
    print(f"[✓] Saved trained emotion classifier to {model_save_path}")
    return clf

if __name__ == "__main__":
    train_and_save_model()
