# feature_extractors/eye_features.py
"""
Eye Feature Extraction (EAR) — FINAL (3D preferred, 2D fallback)
---------------------------------------------------------------
✔ Prefers canonical_3d (more stable)
✔ Falls back to canonical_2d if 3D not saved yet
✔ IOD-normalized (scale invariant)
✔ Returns left/right/avg EAR + iod
"""

import os
import json
import numpy as np

# MediaPipe eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263


# ----------------------------
# Load canonical points (3D preferred)
# ----------------------------
def load_canonical_3d(session_dir):
    """
    DO NOT change function name (pipeline-safe)

    Tries:
      1) canonical_3d
      2) fallback to canonical_2d (converted to 3D by adding z=0)
    """
    path = os.path.join(session_dir, "canonical", "FRONTAL.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"[EyeFeatures] Missing canonical file: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    # ✅ Use canonical_3d if present
    if "canonical_3d" in data and data["canonical_3d"] is not None:
        pts3d = np.array(data["canonical_3d"], dtype=np.float32)
        if pts3d.ndim != 2 or pts3d.shape[1] != 3:
            raise ValueError("[EyeFeatures] canonical_3d must be shape (N,3)")
        return pts3d

    # ✅ Fallback: canonical_2d -> make it (N,3) by adding z=0
    if "canonical_2d" in data and data["canonical_2d"] is not None:
        pts2d = np.array(data["canonical_2d"], dtype=np.float32)
        if pts2d.ndim != 2 or pts2d.shape[1] != 2:
            raise ValueError("[EyeFeatures] canonical_2d must be shape (N,2)")
        pts3d = np.hstack([pts2d, np.zeros((pts2d.shape[0], 1), dtype=np.float32)])
        return pts3d

    raise ValueError("[EyeFeatures] No canonical_3d or canonical_2d found in FRONTAL.json")


# ----------------------------
# Geometry helper
# ----------------------------
def dist(a, b):
    return float(np.linalg.norm(a - b))


# ----------------------------
# EAR computation (IOD-normalized)
# ----------------------------
def compute_ear(pts3d, eye_indices, iod):
    p1, p2, p3, p4, p5, p6 = [pts3d[i] for i in eye_indices]

    v1 = dist(p2, p6)
    v2 = dist(p3, p5)
    h  = dist(p1, p4)

    # Normalize by IOD
    v1 /= iod
    v2 /= iod
    h  /= iod

    ear = (v1 + v2) / (2.0 * h + 1e-6)
    return float(ear)


# ----------------------------
# Main extractor (DO NOT CHANGE NAME)
# ----------------------------
def extract_eye_features(session_dir):
    pts3d = load_canonical_3d(session_dir)

    iod = dist(pts3d[LEFT_EYE_CORNER], pts3d[RIGHT_EYE_CORNER])
    iod = max(iod, 1e-6)

    left_ear = compute_ear(pts3d, LEFT_EYE, iod)
    right_ear = compute_ear(pts3d, RIGHT_EYE, iod)
    avg_ear = (left_ear + right_ear) / 2.0

    return {
        "left_EAR": round(left_ear, 3),
        "right_EAR": round(right_ear, 3),
        "avg_EAR": round(avg_ear, 3),
        "iod": round(iod, 6)
    }


# ----------------------------
# CLI test
# ----------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python eye_features.py <SESSION_DIR>")
        exit(1)

    session_dir = sys.argv[1]
    print(extract_eye_features(session_dir))
