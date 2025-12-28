"""
capture_face_mesh.py

------------------------------------
- Layer 1: Face capture & geometry
- Layer 2: Pose gatekeeper
- Layer 3A: Quality gatekeeper
- Layer 3B: Stability gatekeeper
- Live RED / YELLOW / GREEN guidance UI
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

from pose_gatekeeper import estimate_head_pose_from_matrix, is_pose_valid
from quality_gatekeeper import check_frame_quality
from stability_gatekeeper import StabilityGatekeeper

# -------------------------------------------------
# Model setup
# -------------------------------------------------

MODEL_PATH = "face_landmarker.task"

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
RunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=5,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True
)

# -------------------------------------------------
# Face selection
# -------------------------------------------------

def select_primary_face(face_landmarks_list, frame_w, frame_h, min_face_ratio=0.05):
    candidates = []
    for face in face_landmarks_list:
        xs = [lm.x for lm in face]
        ys = [lm.y for lm in face]
        zs = [lm.z for lm in face]

        face_ratio = ((max(xs) - min(xs)) * frame_w *
                      (max(ys) - min(ys)) * frame_h) / (frame_w * frame_h)

        if face_ratio < min_face_ratio:
            continue

        candidates.append({
            "landmarks": face,
            "avg_z": np.mean(zs),
            "face_ratio": face_ratio
        })

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x["avg_z"], -x["face_ratio"]))
    return candidates[0]["landmarks"]

# -------------------------------------------------
# UI helpers
# -------------------------------------------------

def draw_face_box(frame, landmarks, color, w, h, pad=20):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]

    x1 = int(max(min(xs) - pad, 0))
    y1 = int(max(min(ys) - pad, 0))
    x2 = int(min(max(xs) + pad, w))
    y2 = int(min(max(ys) + pad, h))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

# -------------------------------------------------
# Camera setup
# -------------------------------------------------

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

WINDOW = "Face Capture"
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

stability = StabilityGatekeeper(required_frames=60)

print("[INFO] Camera started. Press ESC to exit.")

# -------------------------------------------------
# Main loop
# -------------------------------------------------

with FaceLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        status_text = "NO FACE DETECTED"
        box_color = (0, 0, 255)

        pose_ok = False
        quality_ok = False
        quality = {}

        face = None

        if result.face_landmarks:
            face = select_primary_face(result.face_landmarks, w, h)

            if face is not None:
                landmarks_3d = np.array([[lm.x, lm.y, lm.z] for lm in face])

                # ---------- Pose Gate ----------
                transform = result.facial_transformation_matrixes[0]
                pose = estimate_head_pose_from_matrix(transform)
                pose_ok = pose is not None and is_pose_valid(pose)

                # ---------- Quality Gate ----------
                quality = check_frame_quality(
                    frame,
                    face,
                    result.face_blendshapes[0] if result.face_blendshapes else []
                )
                quality_ok = quality["quality_ok"]

                # ---------- Stability Gate ----------
                ready = stability.update(
                    pose_ok=pose_ok,
                    quality_ok=quality_ok,
                    landmarks_3d=landmarks_3d
                )

                # ---------- UI decision ----------
                if not pose_ok:
                    status_text = "ADJUST FACE"
                    box_color = (0, 0, 255)

                elif not quality_ok:
                    status_text = quality["message"]
                    box_color = (0, 0, 255)

                elif not ready:
                    status_text = "HOLD STILL"
                    box_color = (0, 255, 255)

                else:
                    status_text = "CAPTURE READY"
                    box_color = (0, 255, 0)

                # Draw landmarks
                for lm in face:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

        # ---------- Draw face box ----------
        if face is not None:
            draw_face_box(frame, face, box_color, w, h)

        # ---------- Status text ----------
        cv2.putText(
            frame,
            status_text,
            (50, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            box_color,
            3
        )

        # ---------- Live Metrics ----------
        if quality:
            if "distance_cm" in quality and quality["distance_cm"] is not None:
                cv2.putText(
                    frame,
                    f"Distance: {quality['distance_cm']} cm",
                    (50, h - 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

            cv2.putText(
                frame,
                f"Stability: {stability.valid_frame_count}/{stability.required_frames}",
                (50, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        # ---------- Stability Progress Bar ----------
        if pose_ok and quality_ok:
            progress = int(
                (stability.valid_frame_count / stability.required_frames) * 200
            )
            cv2.rectangle(frame, (50, h - 40), (50 + progress, h - 20), (0, 255, 255), -1)
            cv2.rectangle(frame, (50, h - 40), (250, h - 20), (255, 255, 255), 2)

        cv2.imshow(WINDOW, frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

# -------------------------------------------------
# Cleanup
# -------------------------------------------------

cap.release()
cv2.destroyAllWindows()
print("[INFO] Program ended cleanly.")
