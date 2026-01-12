"""Face Structure Feature Extraction (Canonical 2D ONLY)
"""

import json
import os
import numpy as np

# -----------------------------
# Key landmark indices (MediaPipe)
# -----------------------------
FOREHEAD = 10
CHIN = 152

LEFT_CHEEK = 234
RIGHT_CHEEK = 454

LEFT_JAW = 172
RIGHT_JAW = 397

MIDLINE = 1  # nose bridge

# -----------------------------
# Load canonical 2D
# -----------------------------
def load_canonical_2d(session_dir):
    path = os.path.join(session_dir, "canonical", "FRONTAL.json")
    with open(path, "r") as f:
        data = json.load(f)

    # works whether file has 2D only or mixed
    if "canonical_2d" in data:
        return np.array(data["canonical_2d"], dtype=np.float32)
    else:
        return np.array(data["canonical_3d_metric"], dtype=np.float32)[:, :2]

# -----------------------------
# Normalize (center + scale)
# -----------------------------
def normalize_2d(pts):
    pts = pts - pts[MIDLINE]
    scale = np.linalg.norm(pts[LEFT_CHEEK] - pts[RIGHT_CHEEK])
    return pts / (scale + 1e-6)

# -----------------------------
# MAIN FEATURE COMPUTE
# -----------------------------
def compute_face_structure_features(session_dir):

    pts = normalize_2d(load_canonical_2d(session_dir))

    # --- Core distances ---
    face_height = np.linalg.norm(pts[FOREHEAD] - pts[CHIN])
    face_width  = np.linalg.norm(pts[LEFT_CHEEK] - pts[RIGHT_CHEEK])
    jaw_width   = np.linalg.norm(pts[LEFT_JAW] - pts[RIGHT_JAW])

    # --- Ratios ---
    facial_index = face_height / (face_width + 1e-6)
    jaw_roundness = jaw_width / (face_width + 1e-6)
    chin_taper = face_width / (jaw_width + 1e-6)

    # --- Proper symmetry ---
    left_pts = pts[[LEFT_CHEEK, LEFT_JAW]]
    right_pts = pts[[RIGHT_CHEEK, RIGHT_JAW]]

    symmetry_error = np.mean(np.abs(left_pts[:, 0] + right_pts[:, 0]))
    symmetry_score = float(np.clip(1.0 - symmetry_error, 0.0, 1.0))

    return {
        "facial_index_ratio": round(float(facial_index), 3),
        "jaw_roundness": round(float(jaw_roundness), 3),
        "chin_taper_ratio": round(float(chin_taper), 3),
        "symmetry_score": round(symmetry_score, 3)
    }
def extract_face_structure(session_dir):
    """
    Optional: saves a frontal face-structure image
    (visual/debug only, not used in features)
    """
    print("ℹ Face structure image extraction skipped (features-only pipeline)")
