import os
import json
import cv2
import numpy as np

# ------------------------------
# Stable landmark indices (MediaPipe)
# ------------------------------
FOREHEAD = 10
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
LEFT_JAW = 172
RIGHT_JAW = 397
LEFT_EYE = 33
RIGHT_EYE = 263

# ------------------------------
# Loaders
# ------------------------------
def load_mesh_3d(session_dir):
    path = os.path.join(session_dir, "meshes", "FRONTAL.json")
    with open(path, "r") as f:
        return np.array(json.load(f)["mesh_3d"], dtype=np.float32)

def load_canonical_2d(session_dir):
    path = os.path.join(session_dir, "canonical", "FRONTAL.json")
    with open(path, "r") as f:
        return np.array(json.load(f)["canonical_2d"], dtype=np.float32)

def load_image_shape(session_dir):
    img = cv2.imread(os.path.join(session_dir, "images", "FRONTAL_RAW.jpg"))
    if img is None:
        raise RuntimeError("FRONTAL_RAW.jpg not found")
    return img.shape[:2]  # (h, w)

# ------------------------------
def compute_face_structure_features(session_dir):
    # =============================
    # 1️⃣ FACIAL INDEX (IMAGE SPACE)
    # =============================
    mesh = load_mesh_3d(session_dir)
    h, w = load_image_shape(session_dir)

    px = np.zeros((mesh.shape[0], 2), dtype=np.float32)
    px[:, 0] = mesh[:, 0] * w
    px[:, 1] = mesh[:, 1] * h

    face_height = abs(px[FOREHEAD][1] - px[CHIN][1])
    face_width = abs(px[LEFT_JAW][0] - px[RIGHT_JAW][0])

    facial_index = face_height / (face_width + 1e-6)

    # =============================
    # 2️⃣ SHAPE FEATURES (CANONICAL)
    # =============================
    pts = load_canonical_2d(session_dir)

    jaw_mid = (pts[LEFT_JAW] + pts[RIGHT_JAW]) / 2.0
    chin_depth = np.linalg.norm(pts[CHIN] - jaw_mid)
    jaw_width = abs(pts[LEFT_JAW][0] - pts[RIGHT_JAW][0])
    face_height_2d = abs(pts[FOREHEAD][1] - pts[CHIN][1])

    jaw_roundness = chin_depth / (jaw_width + 1e-6)
    chin_taper = chin_depth / (face_height_2d + 1e-6)

    # =============================
    # 3️⃣ SYMMETRY (CANONICAL 2D)
    # =============================
    center_x = np.mean(pts[:, 0])
    symmetry_pairs = [
        (LEFT_EYE, RIGHT_EYE),
        (LEFT_CHEEK, RIGHT_CHEEK),
        (LEFT_JAW, RIGHT_JAW),
    ]

    sym_errors = [
        abs((pts[l][0] - center_x) + (pts[r][0] - center_x))
        for l, r in symmetry_pairs
    ]

    symmetry_score = max(0.0, 1.0 - np.mean(sym_errors))

    # =============================
    # 4️⃣ RETURN (SANITY CLAMPED)
    # =============================
    return {
        "facial_index_ratio": round(float(facial_index), 3),      # ~0.9–1.1
        "jaw_roundness": round(float(jaw_roundness), 3),          # ~0.5–0.9
        "chin_taper_ratio": round(float(chin_taper), 3),          # ~0.2–0.5
        "symmetry_score": round(float(symmetry_score), 3),        # ~0.6–0.9
    }