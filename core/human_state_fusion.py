class HumanStateFusion:
    """
    Fuses Action Detection and Emotion Sensing into high-level
    human behavioral context and situational awareness.
    """

    STATE_INSIGHTS = {
        ("Sitting", "Focused"): {
            "title": "Deep Work / High Concentration",
            "category": "Productive",
            "color": "#3b82f6", # Blue
            "description": "User is seated and actively concentrating on a task.",
            "recommendation": "Optimal time for complex problem-solving or coding."
        },
        ("Sitting", "Fatigued / Drowsy"): {
            "title": "Desk Fatigue / Drowsiness Warning",
            "category": "Wellbeing Alert",
            "color": "#f59e0b", # Amber
            "description": "Signs of physical slump or prolonged eye closure detected.",
            "recommendation": "Stand up, drink water, or take a 5-minute eye rest."
        },
        ("Head in Hands / Fatigued", "Sad"): {
            "title": "High Stress / Mental Overwhelm",
            "category": "Stress Alert",
            "color": "#ef4444", # Red
            "description": "User's posture and facial state indicate distress or frustration.",
            "recommendation": "Step away from the screen and take deep breaths."
        },
        ("Head in Hands / Fatigued", "Fatigued / Drowsy"): {
            "title": "Severe Exhaustion",
            "category": "Wellbeing Alert",
            "color": "#dc2626", # Dark red
            "description": "Head supported by hands with drowsy facial cues.",
            "recommendation": "Rest recommended to avoid burnout."
        },
        ("Standing", "Happy"): {
            "title": "Energetic & Positive",
            "category": "Positive",
            "color": "#10b981", # Green
            "description": "Upright active posture paired with joyful affective state.",
            "recommendation": "Great state for presentations or energetic brainstorming."
        },
        ("Waving", "Happy"): {
            "title": "Friendly Greeting / Warm Welcome",
            "category": "Social",
            "color": "#06b6d4", # Cyan
            "description": "Enthusiastic wave with a smile.",
            "recommendation": "Engaged and welcoming social gesture."
        },
        ("Hands Raised / Cheering", "Happy"): {
            "title": "Celebration / Triumph",
            "category": "Positive",
            "color": "#8b5cf6", # Purple
            "description": "Arms raised in victory accompanied by happy expression.",
            "recommendation": "Milestone accomplished or energetic enthusiasm!"
        },
        ("Crossed Arms", "Angry"): {
            "title": "Defensive / Agitated State",
            "category": "Tension Alert",
            "color": "#e11d48", # Rose
            "description": "Closed body language and furrowed/tense expression.",
            "recommendation": "Defuse tension, check environment or workload."
        },
        ("Squatting / Exercising", "Focused"): {
            "title": "Active Physical Exercise",
            "category": "Health",
            "color": "#14b8a6", # Teal
            "description": "Physical movement routine with focused determination.",
            "recommendation": "Maintain proper form and steady hydration."
        }
    }

    @classmethod
    def fuse(cls, action_result, emotion_result):
        """
        Combines action and emotion into human state synthesis.
        """
        action = action_result.get("action", "Neutral / Idle")
        emotion = emotion_result.get("clean_emotion", "Neutral")
        if not emotion or emotion == "No":
            emotion = "Neutral"

        action_conf = action_result.get("confidence", 0.0)
        emotion_conf = emotion_result.get("confidence", 0.0)

        # Check matched compound state
        matched = None
        for (act_key, emo_key), insight in cls.STATE_INSIGHTS.items():
            if act_key.lower() in action.lower() and emo_key.lower() in emotion.lower():
                matched = insight
                break

        if not matched:
            # Default dynamic context
            if "Sitting" in action:
                matched = {
                    "title": f"Seated ({emotion} Mode)",
                    "category": "Routine",
                    "color": "#6366f1",
                    "description": f"User is seated with a {emotion.lower()} disposition.",
                    "recommendation": "Maintaining steady daily activity."
                }
            elif "Standing" in action:
                matched = {
                    "title": f"Standing ({emotion} Mode)",
                    "category": "Active",
                    "color": "#0ea5e9",
                    "description": f"Upright posture with a {emotion.lower()} disposition.",
                    "recommendation": "Good posture balance."
                }
            else:
                matched = {
                    "title": f"{action} & {emotion}",
                    "category": "Observed",
                    "color": "#64748b",
                    "description": f"Action detected as '{action}' while mood appears '{emotion}'.",
                    "recommendation": "Monitoring behavioral continuity."
                }

        combined_confidence = round((action_conf * 0.5 + emotion_conf * 0.5), 2)

        return {
            "action": action,
            "action_confidence": action_conf,
            "emotion": emotion_result.get("emotion", "Neutral 😐"),
            "clean_emotion": emotion,
            "emotion_confidence": emotion_conf,
            "state_title": matched["title"],
            "category": matched["category"],
            "theme_color": matched["color"],
            "description": matched["description"],
            "recommendation": matched["recommendation"],
            "fusion_confidence": combined_confidence,
            "biometrics": emotion_result.get("biometrics", {}),
            "keypoints": action_result.get("keypoint_coords", {})
        }
