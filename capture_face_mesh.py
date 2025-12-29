"""
capture_face_mesh.py

MASTER ORCHESTRATOR (Live Capture UI)
------------------------------------
- Layer 0: Auto-Calibration (Press C)
- Layer 1: Face capture & geometry
- Layer 2: Pose gatekeeper (MODE-AWARE)
- Layer 3A: Quality gatekeeper (MODE-AWARE)
- Layer 3B: Stability gatekeeper
- Multi-view Golden Mesh Capture
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import json, os
from datetime import datetime

from pose_gatekeeper import estimate_head_pose_from_matrix, is_pose_valid
from quality_gatekeeper import check_frame_quality
from stability_gatekeeper import StabilityGatekeeper

# -------------------------------------------------
# Capture Modes
# -------------------------------------------------

CAPTURE_SEQUENCE = ["FRONTAL", "LEFT_PROFILE", "RIGHT_PROFILE"]
current_capture_index = 0
current_capture_mode = CAPTURE_SEQUENCE[current_capture_index]
golden_meshes = {}

# -------------------------------------------------
# Calibration
# -------------------------------------------------

calibrating = False
CALIBRATION_FRAMES = 90
calibration_buffer = []
baseline_expression = None

# -------------------------------------------------
# Model setup
# -------------------------------------------------

MODEL_PATH = "face_landmarker.task"

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True
)

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def save_golden_mesh(data, mode, output_dir="golden_meshes"):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"golden_mesh_{mode}_{ts}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[SAVED] {mode} → {path}")

def draw_face_box(frame, landmarks, color, w, h, pad=20):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, y1 = int(max(min(xs)-pad,0)), int(max(min(ys)-pad,0))
    x2, y2 = int(min(max(xs)+pad,w)), int(min(max(ys)+pad,h))
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)

def mode_instruction(mode):
    return {
        "FRONTAL": "Face the camera",
        "LEFT_PROFILE": "Turn face LEFT",
        "RIGHT_PROFILE": "Turn face RIGHT"
    }.get(mode, "")

# -------------------------------------------------
# Camera
# -------------------------------------------------

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

WINDOW = "MultiMedAI Face Capture"
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

stability = StabilityGatekeeper(required_frames=60)

print("[INFO] Press C to calibrate | ESC to exit")

# -------------------------------------------------
# Main Loop
# -------------------------------------------------

with vision.FaceLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        ts_ms = int(cv2.getTickCount()/cv2.getTickFrequency()*1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        status_text = f"{current_capture_mode}: {mode_instruction(current_capture_mode)}"
        box_color = (0,0,255)

        if result.face_landmarks:
            face = result.face_landmarks[0]
            landmarks_3d = np.array([[lm.x,lm.y,lm.z] for lm in face])

            # ---------- Calibration ----------
            if calibrating and result.face_blendshapes:
                calibration_buffer.append(
                    {b.category_name:b.score for b in result.face_blendshapes[0]}
                )
                if len(calibration_buffer) >= CALIBRATION_FRAMES:
                    baseline_expression = {
                        k: np.mean([f.get(k,0) for f in calibration_buffer])
                        for k in calibration_buffer[0]
                    }
                    calibrating = False
                    calibration_buffer.clear()
                    print("[CALIBRATION DONE]")

            # ---------- Pose Gate ----------
            transform = result.facial_transformation_matrixes[0]
            pose = estimate_head_pose_from_matrix(transform)
            pose_ok = pose is not None and is_pose_valid(pose, mode=current_capture_mode)

            # ---------- Quality Gate ----------
            quality = check_frame_quality(
                frame,
                face,
                result.face_blendshapes[0] if result.face_blendshapes else [],
                baseline_expression,
                capture_mode=current_capture_mode
            )
            quality_ok = quality["quality_ok"]

            # ---------- Stability ----------
            if pose_ok and quality_ok:
                ready = stability.update(True, True, landmarks_3d)
            else:
                ready = False
                stability.reset()

            # ---------- Capture ----------
            if ready and current_capture_mode not in golden_meshes:
                golden_meshes[current_capture_mode] = {
                    "mode": current_capture_mode,
                    "timestamp": ts_ms,
                    "mesh_3d": landmarks_3d.tolist(),
                    "transform": transform.tolist(),
                    "baseline_expression": baseline_expression,
                    "metrics": quality
                }
                save_golden_mesh(golden_meshes[current_capture_mode], current_capture_mode)

                current_capture_index += 1
                stability.reset()

                if current_capture_index < len(CAPTURE_SEQUENCE):
                    current_capture_mode = CAPTURE_SEQUENCE[current_capture_index]
                else:
                    status_text = "ALL CAPTURES COMPLETE"
                    box_color = (255,255,255)

            elif not pose_ok:
                status_text = f"{current_capture_mode}: {mode_instruction(current_capture_mode)}"

            elif not quality_ok:
                status_text = quality["message"]

            elif not ready:
                status_text = f"{current_capture_mode}: Hold still"
                box_color = (0,255,255)

            draw_face_box(frame, face, box_color, w, h)

        cv2.putText(frame, status_text, (40,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, box_color, 3)

        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('c'):
            calibrating = True
            calibration_buffer.clear()
            baseline_expression = None
            stability.reset()
            print("[CALIBRATION STARTED]")
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
print("[DONE]")
