import cv2
import numpy as np
import mediapipe as mp
import os, json
from datetime import datetime

from stability_gatekeeper import StabilityGatekeeper
from consent import verify_consent_token

# =========================
# SESSION SETUP
# =========================

SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = os.path.join("golden_measurements", f"session_{SESSION_ID}", "hands")
os.makedirs(BASE_DIR, exist_ok=True)

IMAGE_DIR = os.path.join(BASE_DIR, "images")
META_DIR = os.path.join(BASE_DIR, "metrics")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

print(f"🟢 Hand session started: {SESSION_ID}")

# =========================
# MediaPipe Hands
# =========================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# =========================
# Landmark indices
# =========================

WRIST = 0
INDEX_MCP = 5
PINKY_MCP = 17
MIDDLE_TIP = 12
# =========================
# Drawing Outlines
# =========================
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def draw_hand_outline(frame, hand_landmarks):
    # Draw skeleton connections
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_styles.get_default_hand_landmarks_style(),
        mp_styles.get_default_hand_connections_style()
    )

    # Draw palm polygon (MU → MCPs → MR)
    palm_indices = [0, 5, 9, 13, 17]
    h, w, _ = frame.shape
    pts = []

    for idx in palm_indices:
        lm = hand_landmarks.landmark[idx]
        pts.append((int(lm.x * w), int(lm.y * h)))

    pts = np.array(pts, np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)



# =========================
# Stability
# =========================

stability = StabilityGatekeeper(required_frames=45)

# =========================
# Geometry helpers
# =========================

def px(lm, w, h):
    return np.array([lm.x * w, lm.y * h])

def compute_hand_index(landmarks, w, h):
    MR = px(landmarks[INDEX_MCP], w, h)
    MU = px(landmarks[PINKY_MCP], w, h)
    DA = px(landmarks[MIDDLE_TIP], w, h)
    SR_SU = px(landmarks[WRIST], w, h)

    breadth = np.linalg.norm(MR - MU)
    length = np.linalg.norm(DA - SR_SU)

    return (breadth / length) * 100

# =========================
# Prakriti mapping
# =========================

def classify_prakriti(hand_index, side="right"):
    if side == "right":
        if hand_index <= 41.344:
            return "Vata"
        elif hand_index < 43.930:
            return "Pitta"
        else:
            return "Kapha"
    else:
        if hand_index <= 41.895:
            return "Vata"
        elif hand_index < 43.687:
            return "Pitta"
        else:
            return "Kapha"
# =========================
# Fingertip Guides
# =========================
def draw_fingertip_guides(frame, landmarks):
    tips = [4, 8, 12, 16, 20]
    h, w, _ = frame.shape

    for i in tips:
        lm = landmarks[i]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)

live_hand_index = None
live_prakriti = None

# =========================
# Camera Loop
# =========================

cap = cv2.VideoCapture(0)
stability.reset()

captured = False

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    status = "Show RIGHT hand and hold still"
    color = (0, 255, 255)

    if result.multi_hand_landmarks:
        hand_lms = result.multi_hand_landmarks[0]

        # 🔹 Visual alignment guides
        draw_hand_outline(frame, hand_lms)
        draw_fingertip_guides(frame, hand_lms.landmark)

        lm = hand_lms.landmark

        # 🔹 Stability check (MANDATORY)
        stable = stability.update(
            pose_ok=True,
            quality_ok=True,
            landmarks_3d=np.array([[l.x, l.y, l.z] for l in lm])
        )

        # 🔹 Live hand index calculation
       # 🔹 Live hand index (always allowed)
        try:
            live_hand_index = compute_hand_index(lm, w, h)
        except Exception:
            live_hand_index = None

        # 🔹 Prakriti ONLY when stable
        if stable and live_hand_index is not None:
            live_prakriti = classify_prakriti(live_hand_index, "right")
        else:
            live_prakriti = None

        # 🔹 Capture when stable
        if stable and not captured:
            hi = live_hand_index
            prakriti = live_prakriti

            # Save image
            img_path = os.path.join(IMAGE_DIR, "RIGHT_HAND.jpg")
            cv2.imwrite(img_path, frame)

            # Save metrics
            meta = {
                "hand_index": round(hi, 2),
                "prakriti": prakriti,
                "timestamp": datetime.now().isoformat()
            }
            meta_path = os.path.join(META_DIR, "RIGHT_HAND.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            print("✅ Hand captured:", meta)
            captured = True
            status = "CAPTURE COMPLETE"
            color = (0, 255, 0)

        elif not stable:
            status = "Hold still..."
            color = (0, 200, 255)

    # =========================
    # On-screen overlays
    # =========================

    # 🔢 Live hand index
    if live_hand_index is not None:
        cv2.putText(
            frame,
            f"Hand Index: {live_hand_index:.2f}",
            (40, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

    # 🧠 Live Prakriti
    if live_prakriti is not None:
        cv2.putText(
            frame,
            f"Prakriti: {live_prakriti}",
            (40, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (180, 255, 180),
            2
        )

    # ✅ Capture confirmation
    if captured:
        cv2.putText(
            frame,
            "CAPTURED ✓  Press ESC to exit",
            (40, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3
        )

    # 🔹 Status text
    cv2.putText(
        frame,
        status,
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3
    )

    cv2.imshow("Hand Capture", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
