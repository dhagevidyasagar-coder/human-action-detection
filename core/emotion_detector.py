import numpy as np
import cv2
import mediapipe as mp
from collections import deque

class EmotionDetector:
    """
    Senses human mood and facial emotional states using MediaPipe Face Mesh
    and facial Action Unit (FACS) biometric geometry, integrated with
    Kaggle FER (Facial Expression Recognition) affective categories.
    """
    EMOTIONS = [
        "Happy 😊",
        "Sad 😢",
        "Angry 😠",
        "Surprised 😲",
        "Neutral 😐",
        "Focused 🧐",
        "Fatigued / Drowsy 😴"
    ]

    def __init__(self, max_num_faces=1, min_detection_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 200))
        self.history = deque(maxlen=20)
        self.blink_history = deque(maxlen=30)

    @staticmethod
    def _euclidean(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _calculate_ear(self, landmarks, eye_indices):
        """Eye Aspect Ratio: ||p2-p6|| + ||p3-p5|| / (2 * ||p1-p4||)"""
        p1 = [landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y]
        p2 = [landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y]
        p3 = [landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y]
        p4 = [landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y]
        p5 = [landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y]
        p6 = [landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y]

        vertical_1 = self._euclidean(p2, p6)
        vertical_2 = self._euclidean(p3, p5)
        horizontal = self._euclidean(p1, p4)

        if horizontal == 0:
            return 0.25
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def _calculate_mar(self, landmarks):
        """Mouth Aspect Ratio: vertical height / horizontal width"""
        top_lip = [landmarks[13].x, landmarks[13].y]
        bottom_lip = [landmarks[14].x, landmarks[14].y]
        left_corner = [landmarks[61].x, landmarks[61].y]
        right_corner = [landmarks[291].x, landmarks[291].y]

        height = self._euclidean(top_lip, bottom_lip)
        width = self._euclidean(left_corner, right_corner)
        if width == 0:
            return 0.1
        return height / width

    def _calculate_smile_index(self, landmarks):
        """
        Calculates smile indicator based on corner elevation relative to center.
        Positive = smiling, Negative = frowning.
        """
        mouth_center_y = (landmarks[13].y + landmarks[14].y) / 2.0
        corners_avg_y = (landmarks[61].y + landmarks[291].y) / 2.0
        # In image coords, lower y is higher on face
        elevation = mouth_center_y - corners_avg_y
        mouth_width = self._euclidean(
            [landmarks[61].x, landmarks[61].y],
            [landmarks[291].x, landmarks[291].y]
        )
        norm_smile = elevation / (mouth_width + 1e-6)
        return norm_smile, mouth_width

    def _calculate_brow_distance(self, landmarks):
        """Measures inter-eyebrow furrow and brow elevation relative to eyes."""
        # Inner eyebrows
        l_brow = [landmarks[55].x, landmarks[55].y]
        r_brow = [landmarks[285].x, landmarks[285].y]
        furrow_dist = self._euclidean(l_brow, r_brow)

        # Eye-brow vertical distances
        l_eye = [landmarks[159].x, landmarks[159].y]
        r_eye = [landmarks[386].x, landmarks[386].y]
        l_elevation = l_eye[1] - l_brow[1]
        r_elevation = r_eye[1] - r_brow[1]
        avg_elevation = (l_elevation + r_elevation) / 2.0

        # Normalization by face scale (outer eyes dist)
        scale = self._euclidean(
            [landmarks[33].x, landmarks[33].y],
            [landmarks[263].x, landmarks[263].y]
        )
        norm_furrow = furrow_dist / (scale + 1e-6)
        norm_elevation = avg_elevation / (scale + 1e-6)

        return norm_furrow, norm_elevation

    def detect(self, rgb_image):
        """
        Processes RGB image to detect facial emotion/mood.
        Returns detailed scores, dominant mood, and landmarks.
        """
        results = self.face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            return {
                "emotion": "No Face Detected",
                "clean_emotion": "Neutral",
                "confidence": 0.0,
                "probabilities": {emo: 0.0 for emo in self.EMOTIONS},
                "landmarks": None,
                "biometrics": {}
            }

        face_landmarks = results.multi_face_landmarks[0]
        lms = face_landmarks.landmark

        # Compute Action Units (biometric indices)
        left_ear = self._calculate_ear(lms, [33, 160, 158, 133, 153, 144])
        right_ear = self._calculate_ear(lms, [362, 385, 387, 263, 373, 380])
        ear = (left_ear + right_ear) / 2.0

        mar = self._calculate_mar(lms)
        smile_index, mouth_width = self._calculate_smile_index(lms)
        brow_furrow, brow_elevation = self._calculate_brow_distance(lms)

        # Track blinks / prolonged closure for fatigue
        self.blink_history.append(ear)
        closed_eyes_ratio = sum(1 for e in self.blink_history if e < 0.18) / max(len(self.blink_history), 1)

        # Score calculations for each emotion
        scores = {
            "Happy 😊": 0.05,
            "Sad 😢": 0.05,
            "Angry 😠": 0.05,
            "Surprised 😲": 0.05,
            "Neutral 😐": 0.35,
            "Focused 🧐": 0.10,
            "Fatigued / Drowsy 😴": 0.05
        }

        # 1. Happy: Positive smile index and wide mouth
        if smile_index > 0.035:
            intensity = min((smile_index - 0.035) * 20.0, 1.0)
            scores["Happy 😊"] += 0.85 * intensity + 0.2
            scores["Neutral 😐"] *= 0.2

        # 2. Surprised: High MAR (mouth open) and high brow elevation and wide eyes (EAR > 0.32)
        if mar > 0.35 and brow_elevation > 0.22:
            scores["Surprised 😲"] += 0.85
            scores["Neutral 😐"] *= 0.1
        elif mar > 0.45:
            scores["Surprised 😲"] += 0.65

        # 3. Angry: Furrowed eyebrows, lower brow elevation, closed mouth, narrow lips
        if brow_furrow < 0.28 and brow_elevation < 0.16 and smile_index < 0.01:
            scores["Angry 😠"] += 0.80
            scores["Neutral 😐"] *= 0.2

        # 4. Sad: Drooping mouth corners (negative smile_index), slightly lowered brows
        if smile_index < -0.015:
            intensity = min(abs(smile_index + 0.015) * 25.0, 1.0)
            scores["Sad 😢"] += 0.75 * intensity + 0.15
            scores["Neutral 😐"] *= 0.3

        # 5. Fatigued / Drowsy: Prolonged low EAR or yawning (very high MAR without high brow elevation)
        if closed_eyes_ratio > 0.40 or ear < 0.16:
            scores["Fatigued / Drowsy 😴"] += 0.90
            scores["Neutral 😐"] *= 0.1
        elif mar > 0.50 and brow_elevation <= 0.22:
            scores["Fatigued / Drowsy 😴"] += 0.70

        # 6. Focused: Stable moderate EAR (0.22 - 0.30), slight brow furrow, neutral mouth
        if 0.20 <= ear <= 0.30 and 0.28 <= brow_furrow <= 0.34 and abs(smile_index) < 0.02 and mar < 0.20:
            scores["Focused 🧐"] += 0.60

        # Softmax normalization
        raw_vals = np.array([scores[e] for e in self.EMOTIONS])
        exp_vals = np.exp(raw_vals * 3.0)
        norm_probs = exp_vals / np.sum(exp_vals)
        prob_dict = {emo: float(norm_probs[i]) for i, emo in enumerate(self.EMOTIONS)}

        best_emotion = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[best_emotion]

        # Temporal smoothing
        self.history.append((best_emotion, confidence))
        recent_emotions = [h[0] for h in self.history]
        smoothed_emotion = max(set(recent_emotions), key=recent_emotions.count)

        clean_name = smoothed_emotion.split(" ")[0]

        return {
            "emotion": smoothed_emotion,
            "clean_emotion": clean_name,
            "confidence": round(confidence, 3),
            "probabilities": {k: round(v, 3) for k, v in prob_dict.items()},
            "landmarks": face_landmarks,
            "biometrics": {
                "ear": round(float(ear), 3),
                "mar": round(float(mar), 3),
                "smile_index": round(float(smile_index), 4),
                "brow_furrow": round(float(brow_furrow), 3),
                "brow_elevation": round(float(brow_elevation), 3),
                "drowsiness_index": round(float(closed_eyes_ratio), 2)
            }
        }

    def draw_facemesh(self, image, landmarks):
        """Draws subtle face mesh landmarks on OpenCV image."""
        if landmarks:
            self.mp_draw.draw_landmarks(
                image=image,
                landmark_list=landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_draw.DrawingSpec(color=(0, 210, 255), thickness=1, circle_radius=1)
            )
            # Draw eye and lip contours more prominently
            self.mp_draw.draw_landmarks(
                image=image,
                landmark_list=landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_draw.DrawingSpec(color=(0, 255, 120), thickness=1, circle_radius=1)
            )
        return image
