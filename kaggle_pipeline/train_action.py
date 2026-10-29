import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def create_synthetic_action_data(n_samples=5000):
    """
    Generates kinematic joint features calibrated on Human Action Recognition (HAR):
    Features: [knee_angle_l, knee_angle_r, hip_angle_l, hip_angle_r, elbow_angle_l, elbow_angle_r,
               l_wrist_y, r_wrist_y, head_wrist_dist, wave_motion_variance]
    """
    np.random.seed(42)
    actions = [
        "Sitting",
        "Standing",
        "Walking / Moving",
        "Waving",
        "Hands Raised / Cheering",
        "Head in Hands / Fatigued",
        "Crossed Arms",
        "Squatting / Exercising"
    ]
    samples_per_act = n_samples // len(actions)
    X, y = [], []

    for act in actions:
        if act == "Sitting":
            knee = np.random.normal(95, 8, (samples_per_act, 2))
            hip = np.random.normal(100, 10, (samples_per_act, 2))
            elbow = np.random.normal(110, 15, (samples_per_act, 2))
            wrist_y = np.random.normal(0.65, 0.08, (samples_per_act, 2))
            head_dist = np.random.normal(0.35, 0.05, (samples_per_act, 1))
            wave_var = np.random.normal(0.005, 0.002, (samples_per_act, 1))
        elif act == "Standing":
            knee = np.random.normal(170, 5, (samples_per_act, 2))
            hip = np.random.normal(172, 5, (samples_per_act, 2))
            elbow = np.random.normal(150, 12, (samples_per_act, 2))
            wrist_y = np.random.normal(0.55, 0.05, (samples_per_act, 2))
            head_dist = np.random.normal(0.40, 0.05, (samples_per_act, 1))
            wave_var = np.random.normal(0.005, 0.002, (samples_per_act, 1))
        elif act == "Walking / Moving":
            knee = np.random.normal(145, 15, (samples_per_act, 2))
            hip = np.random.normal(155, 12, (samples_per_act, 2))
            elbow = np.random.normal(130, 20, (samples_per_act, 2))
            wrist_y = np.random.normal(0.52, 0.08, (samples_per_act, 2))
            head_dist = np.random.normal(0.38, 0.05, (samples_per_act, 1))
            wave_var = np.random.normal(0.025, 0.008, (samples_per_act, 1))
        elif act == "Waving":
            knee = np.random.normal(160, 15, (samples_per_act, 2))
            hip = np.random.normal(160, 15, (samples_per_act, 2))
            elbow = np.random.normal(85, 15, (samples_per_act, 2))
            # One wrist high
            wrist_y = np.column_stack([
                np.random.normal(0.20, 0.04, samples_per_act),
                np.random.normal(0.60, 0.06, samples_per_act)
            ])
            head_dist = np.random.normal(0.25, 0.05, (samples_per_act, 1))
            wave_var = np.random.normal(0.08, 0.02, (samples_per_act, 1))
        elif act == "Hands Raised / Cheering":
            knee = np.random.normal(165, 10, (samples_per_act, 2))
            hip = np.random.normal(168, 10, (samples_per_act, 2))
            elbow = np.random.normal(160, 10, (samples_per_act, 2))
            wrist_y = np.random.normal(0.15, 0.05, (samples_per_act, 2)) # Both above head
            head_dist = np.random.normal(0.35, 0.05, (samples_per_act, 1))
            wave_var = np.random.normal(0.01, 0.005, (samples_per_act, 1))
        elif act == "Head in Hands / Fatigued":
            knee = np.random.normal(92, 10, (samples_per_act, 2))
            hip = np.random.normal(90, 10, (samples_per_act, 2))
            elbow = np.random.normal(55, 10, (samples_per_act, 2))
            wrist_y = np.random.normal(0.28, 0.03, (samples_per_act, 2))
            head_dist = np.random.normal(0.05, 0.02, (samples_per_act, 1)) # Wrists touching head
            wave_var = np.random.normal(0.002, 0.001, (samples_per_act, 1))
        elif act == "Crossed Arms":
            knee = np.random.normal(150, 20, (samples_per_act, 2))
            hip = np.random.normal(150, 20, (samples_per_act, 2))
            elbow = np.random.normal(48, 8, (samples_per_act, 2))
            wrist_y = np.random.normal(0.42, 0.03, (samples_per_act, 2))
            head_dist = np.random.normal(0.22, 0.03, (samples_per_act, 1))
            wave_var = np.random.normal(0.003, 0.001, (samples_per_act, 1))
        else: # Squatting / Exercising
            knee = np.random.normal(70, 8, (samples_per_act, 2))
            hip = np.random.normal(75, 8, (samples_per_act, 2))
            elbow = np.random.normal(120, 15, (samples_per_act, 2))
            wrist_y = np.random.normal(0.55, 0.08, (samples_per_act, 2))
            head_dist = np.random.normal(0.30, 0.05, (samples_per_act, 1))
            wave_var = np.random.normal(0.015, 0.005, (samples_per_act, 1))

        mat = np.hstack([knee, hip, elbow, wrist_y, head_dist, wave_var])
        X.append(mat)
        y.extend([act] * samples_per_act)

    X = np.vstack(X)
    y = np.array(y)
    return X, y

def train_and_save_action_model(model_save_path="models/action_classifier.joblib"):
    print("[*] Preparing action recognition training data...")
    X, y = create_synthetic_action_data(n_samples=8000)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[*] Training Action Classifier on {len(X_train)} samples across {len(np.unique(y))} posture/action classes...")

    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n[✓] Action Model Test Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, preds))

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(clf, model_save_path)
    print(f"[✓] Saved trained action classifier to {model_save_path}")
    return clf

if __name__ == "__main__":
    train_and_save_action_model()
