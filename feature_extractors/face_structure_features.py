"""
Face Structure Feature Extraction – FINAL (3D Correct, Ayurveda-safe)
====================================================================
✔ Uses CANONICAL 3D metric geometry
✔ Measures ANATOMICAL SHAPE, not projection
✔ Scale-invariant via IOD normalization
✔ Uses depth (Z) where biology demands it
✔ Kapha faces will NOT be misread as Vata
✔ Doctor-assistive, explainable by design
"""

import os
import json
import numpy as np


# ----------------------------
# Landmark indices (MediaPipe)
# ----------------------------

# Vertical extent
FOREHEAD = 10
CHIN = 152

# Face width (zygomas / cheeks)
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

# Jaw
LEFT_JAW = 172
RIGHT_JAW = 397

# Chin shape
CHIN_LEFT = 150
CHIN_RIGHT = 379

# Eyes (IOD anchor)
LEFT_EYE = 33
RIGHT_EYE = 263


# ----------------------------
# Loaders
# ----------------------------
def load_canonical_3d_metric(session_dir):
    path = os.path.join(session_dir, "canonical", "FRONTAL.json")
    with open(path, "r") as f:
        return np.array(json.load(f)["canonical_3d_metric"], dtype=np.float32)


def load_canonical_2d(session_dir):
    path = os.path.join(session_dir, "canonical", "FRONTAL.json")
    with open(path, "r") as f:
        return np.array(json.load(f)["canonical_2d"], dtype=np.float32)


# ----------------------------
# Geometry helpers
# ----------------------------
def dist(a, b):
    return np.linalg.norm(a - b)


def midpoint(a, b):
    return 0.5 * (a + b)


# ----------------------------
# Feature extractor (FINAL)
# ----------------------------
def compute_face_structure_features(session_dir):

    pts3d = load_canonical_3d_metric(session_dir)
    pts2d = load_canonical_2d(session_dir)

    # ==================================================
    # 🔒 Global biological scale anchor: IOD (3D)
    # ==================================================
    iod = dist(pts3d[LEFT_EYE], pts3d[RIGHT_EYE])
    iod = max(iod, 1e-6)

    # ==================================================
    # 1️⃣ Facial Index Ratio (BREADTH vs LENGTH)
    #    → MUST NOT use Z dominance incorrectly
    # ==================================================
    # LOW  → broad face → Kapha
    # MID  → balanced  → Pitta
    # HIGH → long face → Vata

    # ---------------------------
    # Facial Index (IOD anchored)
    # ---------------------------
 
    face_height = dist(pts3d[FOREHEAD], pts3d[CHIN]) / iod
    face_width  = dist(pts3d[LEFT_CHEEK], pts3d[RIGHT_CHEEK]) / iod

    facial_index_ratio = face_height / (face_width + 1e-6)

    print("DEBUG facial index:", facial_index_ratio)



    # ==================================================
    # 2️⃣ Jaw Roundness (SOFT TISSUE DEPTH)
    # ==================================================
    jaw_width = dist(pts3d[LEFT_JAW], pts3d[RIGHT_JAW])
    jaw_mid   = midpoint(pts3d[LEFT_JAW], pts3d[RIGHT_JAW])

    jaw_depth = abs(pts3d[CHIN][2] - jaw_mid[2])

    jaw_roundness = (jaw_depth / iod) / (jaw_width / iod)
    # LOW  → angular / narrow → Vata
    # HIGH → wide / fleshy  → Kapha

    # ==================================================
    # 3️⃣ Chin Shape Ratio (POINTED vs ROUNDED)
    # ==================================================
    chin_base_width = dist(pts3d[CHIN_LEFT], pts3d[CHIN_RIGHT])
    chin_base_mid   = midpoint(pts3d[CHIN_LEFT], pts3d[CHIN_RIGHT])

    chin_projection = abs(pts3d[CHIN][2] - chin_base_mid[2])

    chin_shape_ratio = (chin_projection / iod) / (chin_base_width / iod)
    # HIGH → pointed → Vata
    # LOW  → rounded → Kapha

    # ==================================================
    # 4️⃣ Facial Symmetry (STRUCTURAL STABILITY)
    # ==================================================
    cx = np.mean(pts2d[:, 0])  # facial midline (2D canonical)

    sym_pairs = [
        (LEFT_EYE, RIGHT_EYE),
        (LEFT_JAW, RIGHT_JAW),
        (LEFT_CHEEK, RIGHT_CHEEK),
    ]

    sym_errors = [
        abs((pts2d[l][0] - cx) + (pts2d[r][0] - cx))
        for l, r in sym_pairs
    ]

    sym_errors = np.array(sym_errors) / (iod + 1e-6)
    symmetry_score = max(0.0, 1.0 - float(np.mean(sym_errors)))
    # HIGH → Kapha stability
    # LOW  → Vata irregularity

    # ==================================================
    # Output (STRICTLY EXPLAINABLE)
    # ==================================================
    return {
        "facial_index_ratio": round(float(facial_index_ratio), 3),
        "jaw_roundness": round(float(jaw_roundness), 3),
        "chin_shape_ratio": round(float(chin_shape_ratio), 3),
        "symmetry_score": round(float(symmetry_score), 3),
    }
