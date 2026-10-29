import time
from collections import Counter, deque

class SessionTracker:
    """
    Maintains real-time session history, mood timelines, action analytics,
    and wellbeing indicators for the Human State Perception application.
    """
    def __init__(self, max_timeline_entries=200):
        self.start_time = time.time()
        self.timeline = deque(maxlen=max_timeline_entries)
        self.action_counts = Counter()
        self.emotion_counts = Counter()
        self.total_frames = 0
        self.drowsiness_alerts = 0
        self.stress_alerts = 0

    def update(self, fused_state):
        self.total_frames += 1
        now = time.time()
        elapsed = round(now - self.start_time, 1)

        action = fused_state.get("action", "Neutral")
        emotion = fused_state.get("clean_emotion", "Neutral")
        category = fused_state.get("category", "")

        if action != "No Person Detected":
            self.action_counts[action] += 1
        if emotion != "No":
            self.emotion_counts[emotion] += 1

        if category == "Wellbeing Alert":
            self.drowsiness_alerts += 1
        elif category == "Stress Alert":
            self.stress_alerts += 1

        entry = {
            "time": elapsed,
            "timestamp": time.strftime("%H:%M:%S"),
            "action": action,
            "emotion": emotion,
            "title": fused_state.get("state_title", ""),
            "category": category,
            "color": fused_state.get("theme_color", "#3b82f6")
        }
        self.timeline.append(entry)

    def get_summary(self):
        elapsed_sec = int(time.time() - self.start_time)
        minutes = elapsed_sec // 60
        seconds = elapsed_sec % 60
        duration_str = f"{minutes:02d}:{seconds:02d}"

        # Action distribution
        total_act = sum(self.action_counts.values()) or 1
        act_dist = {k: round((v / total_act) * 100, 1) for k, v in self.action_counts.most_common(5)}

        # Emotion distribution
        total_emo = sum(self.emotion_counts.values()) or 1
        emo_dist = {k: round((v / total_emo) * 100, 1) for k, v in self.emotion_counts.most_common(5)}

        # Engagement / Productivity score calculation
        pos_emo = self.emotion_counts.get("Happy", 0) + self.emotion_counts.get("Focused", 0) + self.emotion_counts.get("Neutral", 0)
        neg_emo = self.emotion_counts.get("Fatigued", 0) + self.emotion_counts.get("Sad", 0) + self.emotion_counts.get("Angry", 0)
        focus_score = round(max(10, min(100, 50 + (pos_emo - neg_emo) * 2)), 1) if total_emo > 5 else 85.0

        return {
            "duration": duration_str,
            "total_frames": self.total_frames,
            "action_distribution": act_dist,
            "emotion_distribution": emo_dist,
            "recent_timeline": list(self.timeline)[-15:],
            "engagement_score": focus_score,
            "drowsiness_alerts": self.drowsiness_alerts,
            "stress_alerts": self.stress_alerts
        }

    def reset(self):
        self.start_time = time.time()
        self.timeline.clear()
        self.action_counts.clear()
        self.emotion_counts.clear()
        self.total_frames = 0
        self.drowsiness_alerts = 0
        self.stress_alerts = 0
