# feature_extractors/eye_features.py
"""
Eye Feature Extraction (EAR)
----------------------------
✔ Uses canonical 2D landmarks
✔ Scale & pose invariant
✔ Geometry-based (no ML)
"""

import os
import json
import numpy as np

# ----------------------------
# MediaPipe eye landmark indices
# ----------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Order explanation reported below

# ----------------------------
"""
Eye Feature Extraction – FINAL (Canonical, Calibrated)
=====================================================
✔ Uses canonical 2D landmarks
✔ IOD-normalized (scale invariant)
✔ Physiological EAR range
✔ Stable for Prakriti support (not diagnostic)
"""

import os
import json
import numpy as np


# ----------------------------
# MediaPipe eye landmark indices
# ----------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263


# ----------------------------
# Load canonical 2D
# ----------------------------
def load_canonical_2d(session_dir):
    path = os.path.join(session_dir, "canonical", "FRONTAL.json")
    with open(path, "r") as f:
        data = json.load(f)
    return np.array(data["canonical_2d"], dtype=np.float32)


# ----------------------------
# Geometry helpers
# ----------------------------
def dist(a, b):
    return np.linalg.norm(a - b)


# ----------------------------
# EAR computation (IOD-normalized)
# ----------------------------
def compute_ear(pts2d, eye_indices, iod):
    """
    pts2d: canonical_2d [N,2]
    eye_indices: 6 indices (MediaPipe)
    iod: inter-ocular distance
    """

    p1, p2, p3, p4, p5, p6 = [pts2d[i] for i in eye_indices]

    # Vertical distances
    v1 = dist(p2, p6)
    v2 = dist(p3, p5)

    # Horizontal eye width
    h = dist(p1, p4)

    # Normalize all lengths by IOD
    v1 /= iod
    v2 /= iod
    h  /= iod

    ear = (v1 + v2) / (2.0 * h + 1e-6)
    return ear


# ----------------------------
# Main extractor (FINAL)
# ----------------------------
def extract_eye_features(session_dir):
    pts2d = load_canonical_2d(session_dir)

    # ==========================
    # Global scale anchor (IOD)
    # ==========================
    iod = dist(
        pts2d[LEFT_EYE_CORNER],
        pts2d[RIGHT_EYE_CORNER]
    )
    iod = max(iod, 1e-6)

    # ==========================
    # EAR computation
    # ==========================
    left_ear  = compute_ear(pts2d, LEFT_EYE, iod)
    right_ear = compute_ear(pts2d, RIGHT_EYE, iod)

    ear_avg = (left_ear + right_ear) / 2.0

    # ==========================
    # Physiological clamp (safety)
    # ==========================
    ear_avg   = float(np.clip(ear_avg,   0.15, 0.45))
    left_ear  = float(np.clip(left_ear,  0.15, 0.45))
    right_ear = float(np.clip(right_ear, 0.15, 0.45))

    return {
        "left_EAR": round(left_ear, 3),
        "right_EAR": round(right_ear, 3),
        "avg_EAR": round(ear_avg, 3)
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
    features = extract_eye_features(session_dir)
    print(features)
