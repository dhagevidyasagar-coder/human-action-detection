import cv2
import numpy as np

def create_sample_demos():
    """Generates synthetic high-contrast test figures to test action & emotion detection."""
    # 1. Seated smiling person (Desk worker)
    img_desk = np.ones((480, 640, 3), dtype=np.uint8) * 240
    # Head
    cv2.circle(img_desk, (320, 160), 60, (220, 190, 170), -1)
    cv2.circle(img_desk, (320, 160), 60, (100, 80, 60), 2)
    # Eyes
    cv2.circle(img_desk, (295, 150), 7, (40, 40, 40), -1)
    cv2.circle(img_desk, (345, 150), 7, (40, 40, 40), -1)
    # Smiling mouth (arc)
    cv2.ellipse(img_desk, (320, 175), (25, 15), 0, 0, 180, (40, 40, 200), 3)
    # Eyebrows (relaxed)
    cv2.line(img_desk, (285, 135), (305, 138), (50, 40, 30), 3)
    cv2.line(img_desk, (335, 138), (355, 135), (50, 40, 30), 3)
    # Torso
    cv2.line(img_desk, (320, 220), (320, 380), (60, 60, 180), 8)
    # Shoulders
    cv2.line(img_desk, (230, 250), (410, 250), (60, 60, 180), 8)
    # Arms on desk
    cv2.line(img_desk, (230, 250), (250, 330), (60, 60, 180), 6)
    cv2.line(img_desk, (250, 330), (300, 330), (220, 190, 170), 6)
    cv2.line(img_desk, (410, 250), (390, 330), (60, 60, 180), 6)
    cv2.line(img_desk, (390, 330), (340, 330), (220, 190, 170), 6)
    # Desk line
    cv2.rectangle(img_desk, (100, 330), (540, 450), (140, 100, 70), -1)
    cv2.putText(img_desk, "Sample: Seated & Smiling", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    cv2.imwrite("sample_data/sample_happy_seated.jpg", img_desk)

    # 2. Standing Waving person
    img_wave = np.ones((480, 640, 3), dtype=np.uint8) * 240
    # Head
    cv2.circle(img_wave, (320, 110), 45, (220, 190, 170), -1)
    cv2.circle(img_wave, (305, 105), 5, (40, 40, 40), -1)
    cv2.circle(img_wave, (335, 105), 5, (40, 40, 40), -1)
    cv2.ellipse(img_wave, (320, 125), (15, 8), 0, 0, 180, (40, 40, 200), 2)
    # Body
    cv2.line(img_wave, (320, 155), (320, 310), (40, 120, 60), 8)
    # Left arm down
    cv2.line(img_wave, (270, 180), (240, 270), (40, 120, 60), 6)
    # Right arm WAVING up
    cv2.line(img_wave, (370, 180), (420, 120), (40, 120, 60), 6)
    cv2.line(img_wave, (420, 120), (440, 60), (220, 190, 170), 6) # Hand in air
    # Legs standing
    cv2.line(img_wave, (320, 310), (280, 440), (40, 50, 120), 8)
    cv2.line(img_wave, (320, 310), (360, 440), (40, 50, 120), 8)
    cv2.putText(img_wave, "Sample: Standing & Waving", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    cv2.imwrite("sample_data/sample_standing_wave.jpg", img_wave)

    print("[✓] Created sample test demo images in sample_data/")

if __name__ == "__main__":
    create_sample_demos()
