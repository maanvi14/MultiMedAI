import cv2
import numpy as np
import os, json
from datetime import datetime

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

from quality.stability_gatekeeper import StabilityGatekeeper

# =================================================
# SESSION SETUP
# =================================================

SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = os.path.join("golden_measurements", f"session_{SESSION_ID}", "hands")
os.makedirs(BASE_DIR, exist_ok=True)

IMAGE_DIR = os.path.join(BASE_DIR, "images")
META_DIR = os.path.join(BASE_DIR, "metrics")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

print(f"🟢 Hand session started: {SESSION_ID}")

# =================================================
# MediaPipe HAND LANDMARKER (TASKS API)
# =================================================

HAND_MODEL_PATH = "hand_landmarker.task"  # must exist

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

# =================================================
# Landmark indices
# =================================================

WRIST = 0
INDEX_MCP = 5
PINKY_MCP = 17
MIDDLE_TIP = 12

# =================================================
# Hand connections (replacement for mp.solutions)
# =================================================

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

# =================================================
# Drawing helpers (OLD UI RESTORED)
# =================================================

def draw_hand_outline(frame, landmarks):
    h, w, _ = frame.shape

    # Skeleton
    for a, b in HAND_CONNECTIONS:
        x1, y1 = int(landmarks[a].x * w), int(landmarks[a].y * h)
        x2, y2 = int(landmarks[b].x * w), int(landmarks[b].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Palm polygon
    palm_indices = [0, 5, 9, 13, 17]
    pts = []
    for idx in palm_indices:
        lm = landmarks[idx]
        pts.append((int(lm.x * w), int(lm.y * h)))
    pts = np.array(pts, np.int32)
    cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

def draw_fingertip_guides(frame, landmarks):
    h, w, _ = frame.shape
    for i in [4, 8, 12, 16, 20]:
        lm = landmarks[i]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)

# =================================================
# Stability
# =================================================

stability = StabilityGatekeeper(required_frames=45)

# =================================================
# Geometry helpers
# =================================================

def px(lm, w, h):
    return np.array([lm.x * w, lm.y * h])

def compute_hand_index(landmarks, w, h):
    MR = px(landmarks[INDEX_MCP], w, h)
    MU = px(landmarks[PINKY_MCP], w, h)
    DA = px(landmarks[MIDDLE_TIP], w, h)
    SR_SU = px(landmarks[WRIST], w, h)

    length = np.linalg.norm(DA - SR_SU)
    if length < 1e-6:
        return None

    breadth = np.linalg.norm(MR - MU)
    return (breadth / length) * 100

# =================================================
# TEMP UI prakriti feedback
# =================================================

def classify_prakriti(hand_index):
    if hand_index is None:
        return None
    if hand_index <= 41.344:
        return "Vata"
    elif hand_index < 43.930:
        return "Pitta"
    return "Kapha"

# =================================================
# Camera setup (FULL SCREEN)
# =================================================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

WINDOW_NAME = "Hand Capture"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

stability.reset()

live_hand_index = None
live_prakriti = None
captured = False

# =================================================
# Camera Loop
# =================================================

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    ts_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    result = hand_landmarker.detect_for_video(mp_image, ts_ms)

    status = "Show RIGHT hand and hold still"
    color = (0, 255, 255)

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]

        # 🔹 OLD UI restored
        draw_hand_outline(frame, lm)
        draw_fingertip_guides(frame, lm)

        stable = stability.update(
            pose_ok=True,
            quality_ok=True,
            landmarks_3d=np.array([[p.x, p.y, p.z] for p in lm])
        )

        live_hand_index = compute_hand_index(lm, w, h)
        live_prakriti = classify_prakriti(live_hand_index) if stable else None

        if stable and not captured and live_hand_index is not None:
            cv2.imwrite(os.path.join(IMAGE_DIR, "RIGHT_HAND.jpg"), frame)

            meta = {
                "hand_index": round(live_hand_index, 2),
                "prakriti": live_prakriti,
                "timestamp": datetime.now().isoformat()
            }
            with open(os.path.join(META_DIR, "RIGHT_HAND.json"), "w") as f:
                json.dump(meta, f, indent=2)

            print("✅ Hand captured:", meta)
            captured = True
            status = "CAPTURE COMPLETE"
            color = (0, 255, 0)

        elif not stable:
            status = "Hold still..."
            color = (0, 200, 255)

    # ========================= UI TEXT =========================

    if live_hand_index is not None:
        cv2.putText(frame, f"Hand Index: {live_hand_index:.2f}",
                    (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    if live_prakriti:
        cv2.putText(frame, f"Prakriti: {live_prakriti}",
                    (40, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180,255,180), 2)

    if captured:
        cv2.putText(frame, "CAPTURED ✓ Press ESC to exit",
                    (40, 185), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)

    cv2.putText(frame, status, (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

