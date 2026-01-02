import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

# ----------------------------
# Setup folders
# ----------------------------
BASE_DIR = "teeth_dataset"
OPEN_DIR = os.path.join(BASE_DIR, "open")
CLOSED_DIR = os.path.join(BASE_DIR, "smile")

os.makedirs(OPEN_DIR, exist_ok=True)
os.makedirs(CLOSED_DIR, exist_ok=True)

# ----------------------------
# MediaPipe setup
# ----------------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# ----------------------------
# Landmarks
# ----------------------------
OUTER_LIPS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146
]

UPPER_LIP = 13
LOWER_LIP = 14
LEFT = 61
RIGHT = 291

CLOSED_MAX = 0.30   # includes wide grin with lower teeth
OPEN_MIN   = 0.35   # true open mouth only

# Distance (eye-to-eye in pixels)
MIN_EYE_DIST = 70     # too far if below
MAX_EYE_DIST = 150    # too close if above

LEFT_EYE = 33
RIGHT_EYE = 263


MOUTH_OPEN_THRESHOLD = 0.28  

# ----------------------------
# Camera
# ----------------------------
cap = cv2.VideoCapture(0)

mode = "SMILE"  # CLOSED or OPEN

print("Controls:")
print("  c → capture")
print("  m → switch mode (OPEN / CLOSED)")
print("  q → quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    clean_frame = frame.copy()
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    can_capture = False

    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark

        # Lip bounding box
        lip_points = np.array(
            [[int(lm[i].x * w), int(lm[i].y * h)] for i in OUTER_LIPS],
            dtype=np.int32
        )

        x1, y1 = np.min(lip_points, axis=0)
        x2, y2 = np.max(lip_points, axis=0)

        pad = 10
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

        # Mouth open ratio (UNCHANGED LOGIC)
        upper = np.array([lm[UPPER_LIP].x * w, lm[UPPER_LIP].y * h])
        lower = np.array([lm[LOWER_LIP].x * w, lm[LOWER_LIP].y * h])
        left  = np.array([lm[LEFT].x * w, lm[LEFT].y * h])
        right = np.array([lm[RIGHT].x * w, lm[RIGHT].y * h])

        vertical = np.linalg.norm(upper - lower)
        horizontal = np.linalg.norm(left - right)
        ratio = vertical / horizontal

        # Mode-based capture condition
        # if mode == "OPEN":
        #     can_capture = ratio > MOUTH_OPEN_THRESHOLD
        #     instruction = "Open your mouth"
        # else:
        #     can_capture = ratio <= MOUTH_OPEN_THRESHOLD
        #     instruction = "Smile / close mouth"

        # ----------------------------
        # Distance check (eye-to-eye)
        # ----------------------------
        left_eye = np.array([lm[LEFT_EYE].x * w, lm[LEFT_EYE].y * h])
        right_eye = np.array([lm[RIGHT_EYE].x * w, lm[RIGHT_EYE].y * h])

        eye_dist = np.linalg.norm(left_eye - right_eye)

        distance_ok = MIN_EYE_DIST <= eye_dist <= MAX_EYE_DIST

        if eye_dist < MIN_EYE_DIST:
            distance_msg = "Move closer"
        elif eye_dist > MAX_EYE_DIST:
            distance_msg = "Move back"
        else:
            distance_msg = "Perfect distance"

        if mode == "OPEN":
            can_capture = ratio >= OPEN_MIN and distance_ok
            instruction = "Open mouth wide"
        else:  # CLOSED
            can_capture = ratio <= CLOSED_MAX and distance_ok
            instruction = "Smile / close mouth"


        color = (0, 255, 0) if can_capture else (0, 0, 255)

        # Bounding box only
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label above box
        cv2.putText(
            frame,
            f"Teeth capture {mode}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        # Instruction text
        status = "Press C to capture" if can_capture else instruction
        cv2.putText(
            frame,
            status,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )
        cv2.putText(
            frame,
            f"Distance: {distance_msg}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0) if distance_ok else (0, 0, 255),
            2
        )

    else:
        cv2.putText(
            frame,
            "Face Not Detected",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("Teeth Capture", frame)
    key = cv2.waitKey(1) & 0xFF


    # Quit
    if key == ord('q'):
        break

    # Switch mode
    if key == ord('m'):
        mode = "OPEN" if mode == "CLOSED" else "CLOSED"
        print(f"Mode switched to: {mode}")

    # Capture
    if key == ord('c') and can_capture:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        mouth_crop = clean_frame[y1:y2, x1:x2]

        save_dir = OPEN_DIR if mode == "OPEN" else CLOSED_DIR
        save_path = os.path.join(save_dir, f"{ts}.jpg")
        cv2.imwrite(save_path, mouth_crop)

        print(f"Captured ({mode}): {save_path}")

cap.release()
cv2.destroyAllWindows()