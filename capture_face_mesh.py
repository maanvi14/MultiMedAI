import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import json, os
from datetime import datetime
import uuid

from consent import create_consent, persist_consent, verify_consent_token

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
# UI State
# -------------------------------------------------

last_capture_time = None
last_captured_mode = None
CAPTURE_DISPLAY_MS = 2000

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

def save_golden_mesh(data, mode):
    # save into per-participant folder if participant_id is present
    base = os.path.join("captured", getattr(globals(), 'participant_id', 'unknown'))
    dirpath = os.path.join(base, "golden_meshes")
    os.makedirs(dirpath, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(dirpath, f"golden_mesh_{mode}_{ts}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path

def save_face_image(frame, mode):
    base = os.path.join("captured", getattr(globals(), 'participant_id', 'unknown'))
    dirpath = os.path.join(base, "images")
    os.makedirs(dirpath, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(dirpath, f"{mode}_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path

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
    }[mode]

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

# -------------------------------------------------
# Consent workflow
# -------------------------------------------------
consent_token = None
consent_record = None

def show_consent_overlay(frame):
    h, w, _ = frame.shape
    overlay = frame.copy()
    lines = [
        "EXPLICIT CONSENT REQUIRED",
        "Press 'c' to CONSENT and continue",
        "Press 'd' to DECLINE and exit",
        "Press 'v' to VIEW consent text in terminal"
    ]
    y = 80
    for i, ln in enumerate(lines):
        cv2.putText(overlay, ln, (40, y + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return overlay

def _load_or_create_device_participant_id(path='device_participant_id.txt'):
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                pid = f.read().strip()
                if pid:
                    return pid
        pid = 'device-' + str(uuid.uuid4())
        with open(path, 'w') as f:
            f.write(pid)
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass
        return pid
    except Exception:
        return 'device-' + str(uuid.uuid4())

# device-local persistent participant id (auto-generated)
participant_id = _load_or_create_device_participant_id()

CONSENT_TEXT = (
    "This device will capture a 3D facial mesh and images for Ayurvedic Prakriti analysis. "
    "Data will be stored locally and can be uploaded to a secure backend with your consent. "
    "You may withdraw consent before upload."
)

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

        # If consent not yet given, show overlay and wait for user action
        if consent_token is None:
            overlay = show_consent_overlay(frame)
            cv2.imshow(WINDOW, overlay)
            key = cv2.waitKey(5) & 0xFF
            if key == ord('v'):
                print("\n--- CONSENT TEXT ---\n", CONSENT_TEXT, "\n--- END CONSENT ---\n")
            elif key == ord('c'):
                token, rec = create_consent(participant_id, metadata={"modes": CAPTURE_SEQUENCE})
                consent_dir = os.path.join("captured", participant_id, "consent")
                path = persist_consent(token, rec, directory=consent_dir)
                if verify_consent_token(token):
                    consent_token = token
                    consent_record = rec
                    print(f"Consent saved: {path}")
                else:
                    print("Failed to verify consent token after creation.")
            elif key == ord('d'):
                print("Consent declined. Exiting.")
                break
            continue

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        ts_ms = int(cv2.getTickCount()/cv2.getTickFrequency()*1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        status_text = f"{current_capture_mode}: {mode_instruction(current_capture_mode)}"
        box_color = (0,0,255)

        # ---------- SAFE Capture Confirmation ----------
        if (
            last_capture_time is not None and
            last_captured_mode is not None and
            ts_ms - last_capture_time < CAPTURE_DISPLAY_MS
        ):
            status_text = f"✓ {last_captured_mode} CAPTURED"
            box_color = (0,255,0)

        elif result.face_landmarks:
            face = result.face_landmarks[0]
            landmarks_3d = np.array([[lm.x,lm.y,lm.z] for lm in face])

            # ---------- Pose ----------
            transform = result.facial_transformation_matrixes[0]
            pose = estimate_head_pose_from_matrix(transform)
            pose_result = is_pose_valid(pose, mode=current_capture_mode) if pose else {"valid": False, "reason": "No pose detected", "metrics": {}}
            pose_ok = pose_result.get("valid", False)

            # ---------- Quality ----------
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
                    "metrics": quality
                }

                mesh_path = save_golden_mesh(golden_meshes[current_capture_mode], current_capture_mode)
                img_path = save_face_image(frame, current_capture_mode)
                golden_meshes[current_capture_mode]["saved_files"] = {"mesh": mesh_path, "image": img_path}

                last_capture_time = ts_ms
                last_captured_mode = current_capture_mode

                current_capture_index += 1
                stability.reset()

                if current_capture_index < len(CAPTURE_SEQUENCE):
                    current_capture_mode = CAPTURE_SEQUENCE[current_capture_index]
                else:
                    status_text = "ALL CAPTURES COMPLETE"
                    box_color = (255,255,255)

            elif not pose_ok:
                status_text = pose_result.get("reason", mode_instruction(current_capture_mode))

            elif not quality_ok:
                status_text = quality["message"]

            elif not ready:
                status_text = "Hold still..."
                box_color = (0,255,255)

            # ---------- GREEN MESH ----------
            for lm in face:
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(frame, (cx,cy), 1, (0,255,0), -1)

            draw_face_box(frame, face, box_color, w, h)

            # ---------- Distance ----------
            if quality.get("distance_cm"):
                cv2.putText(
                    frame,
                    f"Distance: {quality['distance_cm']} cm",
                    (40, h-80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,255),
                    2
                )

        cv2.putText(frame, status_text, (40,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, box_color, 3)

        cv2.imshow(WINDOW, frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
print("[DONE]")
