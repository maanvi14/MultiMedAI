import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

# ----------------------------
# Setup
# ----------------------------
SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# Outer lip landmarks (MediaPipe)
OUTER_LIPS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146
]

# Mouth open check landmarks
UPPER_LIP = 13
LOWER_LIP = 14
LEFT = 61
RIGHT = 291

MOUTH_OPEN_THRESHOLD = 0.28  # tune if needed

cap = cv2.VideoCapture(0)

print("Press ENTER to capture | ESC to exit")

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

        # Convert outer lip landmarks to pixels
        lip_points = np.array([
            [int(lm[i].x * w), int(lm[i].y * h)]
            for i in OUTER_LIPS
        ], dtype=np.int32)

        # Mouth open ratio
        upper = np.array([lm[UPPER_LIP].x * w, lm[UPPER_LIP].y * h])
        lower = np.array([lm[LOWER_LIP].x * w, lm[LOWER_LIP].y * h])
        left  = np.array([lm[LEFT].x * w, lm[LEFT].y * h])
        right = np.array([lm[RIGHT].x * w, lm[RIGHT].y * h])

        vertical = np.linalg.norm(upper - lower)
        horizontal = np.linalg.norm(left - right)
        ratio = vertical / horizontal

        mouth_open = ratio > MOUTH_OPEN_THRESHOLD
        can_capture = mouth_open

        # Overlay mask
        overlay = frame.copy()
        cv2.fillPoly(
            overlay,
            [lip_points],
            (0, 255, 0) if mouth_open else (0, 0, 255)
        )
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        # Bounding box
        x1, y1 = np.min(lip_points, axis=0)
        x2, y2 = np.max(lip_points, axis=0)
        cv2.rectangle(
            frame,
            (x1 - 10, y1 - 10),
            (x2 + 10, y2 + 10),
            (0, 255, 0) if mouth_open else (0, 0, 255),
            2
        )

        # UI text
        if mouth_open:
            cv2.putText(frame, "Aligned - Press ENTER",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Open Your Mouth",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

    else:
        cv2.putText(frame, "Face Not Detected",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    cv2.imshow("Lip Alignment", frame)

    key = cv2.waitKey(1) & 0xFF

    # ESC
    if key == 27:
        break

    # ENTER to capture
    # ENTER to capture
    if key == 13 and can_capture:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        mouth_crop = clean_frame[y1:y2, x1:x2]  # clean image
        save_path = os.path.join(SAVE_DIR, f"{ts}.jpg")
        cv2.imwrite(save_path, mouth_crop)

        print(f"Captured clean image: {save_path}")

cap.release()
cv2.destroyAllWindows()
