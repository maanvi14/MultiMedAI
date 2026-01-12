import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import json, os
from datetime import datetime
import uuid
import subprocess
from utils.face_mesh_connections import (
    FACEMESH_TESSELATION,
    FACEMESH_FACE_OVAL,
    FACEMESH_LEFT_EYE,
    FACEMESH_RIGHT_EYE,
    FACEMESH_LIPS,
    FACEMESH_NOSE
)


from consent import create_consent, persist_consent, verify_consent_token

from quality.pose_gatekeeper import estimate_head_pose_from_matrix, is_pose_valid
from quality.quality_gatekeeper import check_frame_quality
from quality.stability_gatekeeper import StabilityGatekeeper

RENDER_DEBUG = False

# =========================
# SESSION INITIALIZATION
# =========================
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

BASE_SESSION_DIR = os.path.join("golden_meshes", f"session_{SESSION_ID}")
MESH_DIR = os.path.join(BASE_SESSION_DIR, "meshes")
IMAGE_DIR = os.path.join(BASE_SESSION_DIR, "images")

os.makedirs(MESH_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

ANALYSIS_DIR = os.path.join(BASE_SESSION_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

print(f"🟢 Session started: session_{SESSION_ID}")

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

# def save_golden_mesh(data, mode):
#     base = os.path.join("captured", getattr(globals(), 'participant_id', 'unknown'))
#     dirpath = os.path.join(base, "golden_meshes")
#     os.makedirs(dirpath, exist_ok=True)
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     path = os.path.join(dirpath, f"golden_mesh_{mode}_{ts}.json")
#     with open(path, "w") as f:
#         json.dump(data, f, indent=2)
#     return path

def save_golden_mesh(data, mode):
    path = os.path.join(MESH_DIR, f"{mode}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Mesh saved: {path}")
    return path

# def save_face_image(frame, mode):
#     base = os.path.join("captured", getattr(globals(), 'participant_id', 'unknown'))
#     dirpath = os.path.join(base, "images")
#     os.makedirs(dirpath, exist_ok=True)
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     path = os.path.join(dirpath, f"{mode}_{ts}.jpg")
#     cv2.imwrite(path, frame)
#     return path

def save_face_image(frame, mode):
    path = os.path.join(IMAGE_DIR, f"{mode}.jpg")
    cv2.imwrite(path, frame)
    print(f"📸 Image saved: {path}")
    return path

def draw_face_box(overlay, landmarks, color, w, h, pad=20):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    face_height = max(ys) - min(ys)

    x1 = int(max(min(xs)-pad,0))
    y1 = int(max(min(ys)-(face_height * 0.4),0))
    x2 = int(min(max(xs)+pad,w))
    y2 = int(min(max(ys)+pad,h))

    cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 3)


def mode_instruction(mode):
    return {
        "FRONTAL": "Face the camera",
        "LEFT_PROFILE": "Turn face LEFT",
        "RIGHT_PROFILE": "Turn face RIGHT"
    }[mode]

# -------------------------------------------------
# NEW: Virtual Forehead Logic (Grid Coverage)
# -------------------------------------------------
def draw_extended_green_mesh(overlay, landmarks, w, h):
    """Sparse virtual forehead points only"""

    p10 = np.array([landmarks[10].x, landmarks[10].y])
    p168 = np.array([landmarks[168].x, landmarks[168].y])
    vec = p10 - p168

    top_edge_indices = [103, 67, 109, 10, 338, 297, 332]

    for row_scale in [0.3, 0.6, 0.9]:
        for idx in top_edge_indices:
            base = landmarks[idx]
            vx = int((base.x + vec[0] * row_scale) * w)
            vy = int((base.y + vec[1] * row_scale) * h)
            if 0 <= vx < w and 0 <= vy < h:
                cv2.circle(overlay, (vx, vy), 1, COLOR_FOREHEAD, -1)


# =========================
# UI VISUAL CONSTANTS (FINAL)
# =========================
COLOR_MESH = (190, 190, 190)        # soft neutral white

COLOR_FACE_OVAL = (90, 210, 210)    # cyan (ONLY outer boundary)
COLOR_LEFT_EYE  = (70, 120, 220)    # blue
COLOR_RIGHT_EYE = (190, 110, 200)   # magenta
COLOR_LIPS      = (90, 210, 210)    # cyan
COLOR_NOSE      = (120, 180, 180)   # muted teal (NOT yellow)
COLOR_FOREHEAD  = (0, 180, 0)       # muted green

ALPHA_OVERLAY = 0.35


def draw_full_facemesh_overlay(overlay, landmarks, w, h):
    """Draw mesh and semantic regions onto overlay ONLY"""

    # Tessellation (background)
    for i, j in FACEMESH_TESSELATION:
        p1, p2 = landmarks[i], landmarks[j]
        cv2.line(
            overlay,
            (int(p1.x * w), int(p1.y * h)),
            (int(p2.x * w), int(p2.y * h)),
            COLOR_MESH,
            1,
            cv2.LINE_AA
        )

    # Face oval
    for i, j in FACEMESH_FACE_OVAL:
        p1, p2 = landmarks[i], landmarks[j]
        cv2.line(overlay,
                 (int(p1.x * w), int(p1.y * h)),
                 (int(p2.x * w), int(p2.y * h)),
                 COLOR_FACE_OVAL, 2, cv2.LINE_AA)

    # Left eye
    for i, j in FACEMESH_LEFT_EYE:
        p1, p2 = landmarks[i], landmarks[j]
        cv2.line(overlay,
                 (int(p1.x * w), int(p1.y * h)),
                 (int(p2.x * w), int(p2.y * h)),
                 COLOR_LEFT_EYE, 2, cv2.LINE_AA)

    # Right eye
    for i, j in FACEMESH_RIGHT_EYE:
        p1, p2 = landmarks[i], landmarks[j]
        cv2.line(overlay,
                 (int(p1.x * w), int(p1.y * h)),
                 (int(p2.x * w), int(p2.y * h)),
                 COLOR_RIGHT_EYE, 2, cv2.LINE_AA)

    # Lips
    for i, j in FACEMESH_LIPS:
        p1, p2 = landmarks[i], landmarks[j]
        cv2.line(overlay,
                 (int(p1.x * w), int(p1.y * h)),
                 (int(p2.x * w), int(p2.y * h)),
                 COLOR_LIPS, 2, cv2.LINE_AA)

    # Nose
    for i, j in FACEMESH_NOSE:
        p1, p2 = landmarks[i], landmarks[j]
        cv2.line(overlay,
                 (int(p1.x * w), int(p1.y * h)),
                 (int(p2.x * w), int(p2.y * h)),
                 COLOR_NOSE, 2, cv2.LINE_AA)

def run_auto_pipeline():
    cap.release()
    cv2.destroyAllWindows()

    import subprocess

    ANALYSIS_FLAG = os.path.join(BASE_SESSION_DIR, "analysis", "_analysis_done.flag")

    if not os.path.exists(ANALYSIS_FLAG):

        print("🧭 Running canonical projection...")
        subprocess.run(
            ["python", "canonical_projection.py", BASE_SESSION_DIR],
            check=True
        )

        print("🔬 Building features...")
        subprocess.run(
            ["python", "build_features.py", BASE_SESSION_DIR],
            check=True
        )

        print("🧘 Running Prakriti analysis...")
        subprocess.run(
            ["python", "-m", "prakriti_mapping.run_prakriti", BASE_SESSION_DIR],
            check=True
        )

        with open(ANALYSIS_FLAG, "w") as f:
            f.write("done")

    else:
        print("ℹ Analysis already completed for this session")

    print("[DONE]")

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
            elif key == ord('d'):
                break
            continue

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        ts_ms = int(cv2.getTickCount()/cv2.getTickFrequency()*1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        status_text = f"{current_capture_mode}: {mode_instruction(current_capture_mode)}"
        box_color = (0,0,255)

        if (last_capture_time is not None and last_captured_mode is not None and 
            ts_ms - last_capture_time < CAPTURE_DISPLAY_MS):
            status_text = f"✓ {last_captured_mode} CAPTURED"
            box_color = (0,255,0)

        elif result.face_landmarks:
            face = result.face_landmarks[0]
            landmarks_3d = np.array([[lm.x,lm.y,lm.z] for lm in face])

            transform = result.facial_transformation_matrixes[0]
            pose = estimate_head_pose_from_matrix(transform)
            pose_result = is_pose_valid(pose, mode=current_capture_mode) if pose else {"valid": False}
            pose_ok = pose_result.get("valid", False)

            quality = check_frame_quality(frame, face, result.face_blendshapes[0] if result.face_blendshapes else [], baseline_expression, current_capture_mode)
            quality_ok = quality["quality_ok"]

            if pose_ok and quality_ok:
                ready = stability.update(True, True, landmarks_3d)
            else:
                ready = False
                stability.reset()

            # ---------- Capture with Mentor Mapping Proof ----------
            if ready and current_capture_mode not in golden_meshes:
                golden_meshes[current_capture_mode] = {
                    "mode": current_capture_mode,
                    "timestamp": ts_ms,
                    "mesh_3d": landmarks_3d.tolist(),
                    "transform": transform.tolist(),
                    "metrics": quality
                }

                # Save raw clean image
                raw_img_path = save_face_image(frame, f"{current_capture_mode}_RAW")

                # Project 3D landmarks onto a copy for mentor's mapping/segmentation proof
                mapping_proof = frame.copy()
                draw_full_facemesh_overlay(mapping_proof, face, w, h)
                draw_extended_green_mesh(mapping_proof, face, w, h)


                mapped_img_path = save_face_image(mapping_proof, f"{current_capture_mode}_MAPPED")

                mesh_path = save_golden_mesh(golden_meshes[current_capture_mode], current_capture_mode)
                golden_meshes[current_capture_mode]["saved_files"] = {
                    "mesh": mesh_path, 
                    "image_raw": raw_img_path, 
                    "image_mapped": mapped_img_path
                }

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
            
            overlay = frame.copy()

            draw_full_facemesh_overlay(overlay, face, w, h)
            draw_extended_green_mesh(overlay, face, w, h)
            draw_face_box(overlay, face, box_color, w, h)

            cv2.addWeighted(overlay, ALPHA_OVERLAY, frame, 1 - ALPHA_OVERLAY, 0, frame)


            if quality.get("distance_cm"):
                cv2.putText(frame, f"Distance: {quality['distance_cm']} cm", (40, h-80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.putText(frame, status_text, (40,60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, box_color, 3)
        cv2.imshow(WINDOW, frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

if __name__ == "__main__":
    run_auto_pipeline()
    