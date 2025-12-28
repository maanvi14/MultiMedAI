"""
capture_face_mesh.py

MASTER ORCHESTRATOR (Live Capture UI)
------------------------------------
- Layer 0: Auto-Calibration (Press C)
- Layer 1: Face capture & geometry
- Layer 2: Pose gatekeeper
- Layer 3A: Quality gatekeeper
- Layer 3B: Stability gatekeeper
- Freeze & Save Golden Mesh
- Live RED / YELLOW / GREEN guidance UI
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from collections import defaultdict
import json
import os
from datetime import datetime

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
# Calibration state (Layer 0)
# -------------------------------------------------

calibrating = False
CALIBRATION_FRAMES = 90
calibration_buffer = []
baseline_expression = None

# -------------------------------------------------
# Golden Mesh state
# -------------------------------------------------

golden_mesh_captured = False
golden_mesh_data = None

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
# Golden Mesh save
# -------------------------------------------------

def save_golden_mesh(data, output_dir="golden_meshes"):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"golden_mesh_{ts}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[GOLDEN MESH] Saved → {path}")

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
print("[INFO] Press 'C' to calibrate neutral face.")

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
        ready = False
        quality = {}
        face = None

        if result.face_landmarks:
            face = select_primary_face(result.face_landmarks, w, h)

            if face is not None:
                landmarks_3d = np.array([[lm.x, lm.y, lm.z] for lm in face])

                # ---------- Calibration ----------
                if calibrating and result.face_blendshapes:
                    scores = {b.category_name: b.score for b in result.face_blendshapes[0]}
                    calibration_buffer.append(scores)

                    if len(calibration_buffer) >= CALIBRATION_FRAMES:
                        baseline_expression = {
                            k: np.mean([f.get(k, 0) for f in calibration_buffer])
                            for k in calibration_buffer[0].keys()
                        }
                        calibrating = False
                        calibration_buffer.clear()
                        print("[CALIBRATION] Baseline captured")

                # ---------- Pose Gate ----------
                transform = result.facial_transformation_matrixes[0]
                pose = estimate_head_pose_from_matrix(transform)
                pose_ok = pose is not None and is_pose_valid(pose)

                # ---------- Quality Gate ----------
                quality = check_frame_quality(
                    frame,
                    face,
                    result.face_blendshapes[0] if result.face_blendshapes else [],
                    baseline_expression
                )
                quality_ok = quality["quality_ok"]

                # ---------- Stability Gate ----------
                if pose_ok and quality_ok:
                    ready = stability.update(True, True, landmarks_3d)
                else:
                    stability.reset()

                # ---------- UI & Golden Mesh ----------
                if golden_mesh_captured:
                    status_text = "CAPTURE COMPLETE"
                    box_color = (255, 255, 255)

                elif calibrating:
                    status_text = "CALIBRATING — KEEP NEUTRAL"
                    box_color = (0, 255, 255)

                elif not pose_ok:
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

                    if not golden_mesh_captured:
                        golden_mesh_captured = True
                        golden_mesh_data = {
                            "timestamp_ms": timestamp_ms,
                            "landmarks_3d": landmarks_3d.tolist(),
                            "face_transform_matrix": transform.tolist(),
                            "baseline_expression": baseline_expression,
                            "capture_metadata": {
                                "distance_cm": quality.get("distance_cm"),
                                "blur_score": quality.get("blur_score"),
                                "lighting_status": quality.get("lighting_status"),
                                "expression_status": quality.get("expression_status")
                            }
                        }
                        save_golden_mesh(golden_mesh_data)

                # Draw landmarks
                for lm in face:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

        else:
            stability.reset()

        if face is not None:
            draw_face_box(frame, face, box_color, w, h)

        cv2.putText(frame, status_text, (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, box_color, 3)

        if quality and quality.get("distance_cm") is not None:
            cv2.putText(frame, f"Distance: {quality['distance_cm']} cm",
                        (50, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)

        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('c'):
            calibrating = True
            calibration_buffer.clear()
            baseline_expression = None
            stability.reset()
            golden_mesh_captured = False
            print("[CALIBRATION] Started")

        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Program ended cleanly.")
