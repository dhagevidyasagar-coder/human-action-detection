import math
import numpy as np
import cv2
import mediapipe as mp
from collections import deque

class ActionDetector:
    """
    Detects human body postures and dynamic actions using MediaPipe Pose landmarks
    and kinematic angle / velocity feature analysis.
    """
    ACTIONS = [
        "Sitting",
        "Standing",
        "Walking / Moving",
        "Waving",
        "Hands Raised / Cheering",
        "Head in Hands / Fatigued",
        "Crossed Arms",
        "Squatting / Exercising",
        "Neutral / Idle"
    ]

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.history = deque(maxlen=30)  # ~1 sec history at 30fps
        self.wrist_wave_history = deque(maxlen=20)

    @staticmethod
    def calculate_angle(a, b, c):
        """Calculates 2D angle ABC (angle at b) in degrees."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def extract_features(self, landmarks):
        """Extracts normalized coordinates and kinematic joint angles."""
        features = []
        for lm in landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z, lm.visibility])

        # Joint points
        lh = [landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
              landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        lk = [landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
              landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        la = [landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
              landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        rh = [landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
              landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        rk = [landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
              landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ra = [landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
              landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        ls = [landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
              landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        le = [landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
              landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        lw = [landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
              landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        rs = [landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
              landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        re = [landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
              landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        rw = [landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
              landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        left_knee_angle = self.calculate_angle(lh, lk, la)
        right_knee_angle = self.calculate_angle(rh, rk, ra)
        left_hip_angle = self.calculate_angle(ls, lh, lk)
        right_hip_angle = self.calculate_angle(rs, rh, rk)
        left_elbow_angle = self.calculate_angle(ls, le, lw)
        right_elbow_angle = self.calculate_angle(rs, re, rw)

        features.extend([
            left_knee_angle, right_knee_angle,
            left_hip_angle, right_hip_angle,
            left_elbow_angle, right_elbow_angle
        ])
        return np.array(features, dtype=np.float32)

    def detect(self, rgb_image):
        """
        Processes an RGB image and returns:
        - action_label: str
        - confidence: float (0.0 to 1.0)
        - action_probabilities: dict
        - landmarks: MediaPipe landmarks object
        """
        results = self.pose.process(rgb_image)
        if not results.pose_landmarks:
            return {
                "action": "No Person Detected",
                "confidence": 0.0,
                "probabilities": {act: 0.0 for act in self.ACTIONS},
                "landmarks": None,
                "keypoint_coords": {}
            }

        lms = results.pose_landmarks.landmark
        p = self.mp_pose.PoseLandmark

        # Coordinates
        nose = np.array([lms[p.NOSE.value].x, lms[p.NOSE.value].y])
        l_shoulder = np.array([lms[p.LEFT_SHOULDER.value].x, lms[p.LEFT_SHOULDER.value].y])
        r_shoulder = np.array([lms[p.RIGHT_SHOULDER.value].x, lms[p.RIGHT_SHOULDER.value].y])
        l_elbow = np.array([lms[p.LEFT_ELBOW.value].x, lms[p.LEFT_ELBOW.value].y])
        r_elbow = np.array([lms[p.RIGHT_ELBOW.value].x, lms[p.RIGHT_ELBOW.value].y])
        l_wrist = np.array([lms[p.LEFT_WRIST.value].x, lms[p.LEFT_WRIST.value].y])
        r_wrist = np.array([lms[p.RIGHT_WRIST.value].x, lms[p.RIGHT_WRIST.value].y])
        l_hip = np.array([lms[p.LEFT_HIP.value].x, lms[p.LEFT_HIP.value].y])
        r_hip = np.array([lms[p.RIGHT_HIP.value].x, lms[p.RIGHT_HIP.value].y])
        l_knee = np.array([lms[p.LEFT_KNEE.value].x, lms[p.LEFT_KNEE.value].y])
        r_knee = np.array([lms[p.RIGHT_KNEE.value].x, lms[p.RIGHT_KNEE.value].y])
        l_ankle = np.array([lms[p.LEFT_ANKLE.value].x, lms[p.LEFT_ANKLE.value].y])
        r_ankle = np.array([lms[p.RIGHT_ANKLE.value].x, lms[p.RIGHT_ANKLE.value].y])

        # Visibility metrics
        vis_knees = (lms[p.LEFT_KNEE.value].visibility + lms[p.RIGHT_KNEE.value].visibility) / 2.0
        vis_ankles = (lms[p.LEFT_ANKLE.value].visibility + lms[p.RIGHT_ANKLE.value].visibility) / 2.0

        # Angles
        l_knee_angle = self.calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = self.calculate_angle(r_hip, r_knee, r_ankle)
        avg_knee_angle = (l_knee_angle + r_knee_angle) / 2.0

        l_hip_angle = self.calculate_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = self.calculate_angle(r_shoulder, r_hip, r_knee)
        avg_hip_angle = (l_hip_angle + r_hip_angle) / 2.0

        shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
        hip_y = (l_hip[1] + r_hip[1]) / 2.0

        # Track wrist oscillations for waving
        self.wrist_wave_history.append((l_wrist[0], r_wrist[0], l_wrist[1], r_wrist[1]))
        wave_motion_score = 0.0
        if len(self.wrist_wave_history) >= 8:
            xs_r = [item[1] for item in self.wrist_wave_history]
            xs_l = [item[0] for item in self.wrist_wave_history]
            range_r = max(xs_r) - min(xs_r)
            range_l = max(xs_l) - min(xs_l)
            wave_motion_score = max(range_r, range_l)

        # Distances for head-in-hands and crossed-arms
        dist_lw_nose = np.linalg.norm(l_wrist - nose)
        dist_rw_nose = np.linalg.norm(r_wrist - nose)
        dist_lw_re = np.linalg.norm(l_wrist - r_elbow)
        dist_rw_le = np.linalg.norm(r_wrist - l_elbow)

        # Probabilities dictionary
        probs = {k: 0.05 for k in self.ACTIONS}

        # 1. Hands Raised / Cheering
        if l_wrist[1] < shoulder_y and r_wrist[1] < shoulder_y:
            probs["Hands Raised / Cheering"] += 0.85

        # 2. Waving (One hand high with horizontal motion)
        one_hand_high = (l_wrist[1] < shoulder_y and r_wrist[1] > shoulder_y) or \
                        (r_wrist[1] < shoulder_y and l_wrist[1] > shoulder_y)
        if one_hand_high and wave_motion_score > 0.035:
            probs["Waving"] += 0.88
        elif one_hand_high:
            probs["Waving"] += 0.45

        # 3. Head in Hands / Fatigued
        if min(dist_lw_nose, dist_rw_nose) < 0.12:
            probs["Head in Hands / Fatigued"] += 0.80

        # 4. Crossed Arms
        if dist_lw_re < 0.18 and dist_rw_le < 0.18 and l_wrist[1] > shoulder_y:
            probs["Crossed Arms"] += 0.75

        # 5. Squatting / Exercising vs Sitting vs Standing
        if vis_knees > 0.4:
            if avg_knee_angle < 110 and avg_hip_angle < 120:
                # Can be sitting or squatting depending on torso verticality and context
                if hip_y > 0.65:
                    probs["Squatting / Exercising"] += 0.75
                else:
                    probs["Sitting"] += 0.82
            elif avg_knee_angle > 145 and avg_hip_angle > 145:
                probs["Standing"] += 0.85
            else:
                probs["Sitting"] += 0.65
        else:
            # Upper body only visible (e.g., standard webcam desk position)
            # When working at desk, person is sitting
            probs["Sitting"] += 0.70

        # Normalize probabilities with softmax
        scores = np.array([probs[act] for act in self.ACTIONS])
        exp_scores = np.exp(scores * 2.5)
        normalized_probs = exp_scores / np.sum(exp_scores)
        prob_dict = {act: float(normalized_probs[i]) for i, act in enumerate(self.ACTIONS)}

        best_action = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[best_action]

        # Temporal smoothing
        self.history.append((best_action, confidence))
        recent_actions = [h[0] for h in self.history]
        smoothed_action = max(set(recent_actions), key=recent_actions.count)

        keypoints = {
            "nose": [float(nose[0]), float(nose[1])],
            "left_shoulder": [float(l_shoulder[0]), float(l_shoulder[1])],
            "right_shoulder": [float(r_shoulder[0]), float(r_shoulder[1])],
            "left_wrist": [float(l_wrist[0]), float(l_wrist[1])],
            "right_wrist": [float(r_wrist[0]), float(r_wrist[1])],
            "left_hip": [float(l_hip[0]), float(l_hip[1])],
            "right_hip": [float(r_hip[0]), float(r_hip[1])],
        }

        return {
            "action": smoothed_action,
            "confidence": round(confidence, 3),
            "probabilities": {k: round(v, 3) for k, v in prob_dict.items()},
            "landmarks": results.pose_landmarks,
            "keypoint_coords": keypoints,
            "raw_best_action": best_action
        }

    def draw_skeleton(self, image, landmarks):
        """Draws the pose landmarks and connections on an OpenCV image."""
        if landmarks:
            self.mp_draw.draw_landmarks(
                image,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 170), thickness=2, circle_radius=3),
                self.mp_draw.DrawingSpec(color=(255, 120, 0), thickness=2, circle_radius=2)
            )
        return image
