"""
quality_gatekeeper.py

LAYER 3A: Quality Gatekeeper (Per-Frame)
---------------------------------------
Purpose:
- Ensure a single frame is Prakriti-ready

Checks:
1. Metric distance (iris-based)
2. Face centering
3. Blur detection
4. Lighting symmetry (cheeks)
5. Neutral expression (dynamic + smoothed)

Returns:
- quality_ok (bool)
- message (UI guidance)
- diagnostic metrics
"""

import cv2
import numpy as np
from collections import deque

# -------------------------------------------------
# Constants & thresholds (REAL-WORLD SAFE)
# -------------------------------------------------

IRIS_DIAMETER_MM = 11.7
DIST_MIN_CM = 40
DIST_MAX_CM = 60

BLUR_THRESHOLD = 20            # rural-safe
LIGHTING_DIFF_THRESHOLD = 50   # non-studio

CENTER_TOLERANCE = 0.15

# Expression tolerance
EXPR_DELTA_TOLERANCE = 0.08    # baseline + delta
EXPR_VARIANCE_LIMIT = 0.005   # jitter guard

# Rolling buffer
EXPR_BUFFER_SIZE = 5

# Cheek landmarks
LEFT_CHEEK = [123, 116]
RIGHT_CHEEK = [423, 345]

# Iris landmarks (single eye)
LEFT_IRIS = [474, 475, 476, 477]

# -------------------------------------------------
# Expression buffer (GLOBAL, intentional)
# -------------------------------------------------

_expression_buffer = deque(maxlen=EXPR_BUFFER_SIZE)

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def estimate_distance_cm(landmarks, frame_w):
    xs = [landmarks[i].x * frame_w for i in LEFT_IRIS]
    iris_px = max(xs) - min(xs)

    if iris_px <= 1:
        return None

    distance_cm = (IRIS_DIAMETER_MM * frame_w) / iris_px / 10
    return round(distance_cm, 1)


def compute_blur_score(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_lighting_symmetry(gray, landmarks, w, h):
    def cheek_mean(indices):
        vals = []
        for idx in indices:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            if 0 <= x < w and 0 <= y < h:
                vals.append(gray[y, x])
        return np.mean(vals) if vals else 0

    return abs(
        cheek_mean(LEFT_CHEEK) -
        cheek_mean(RIGHT_CHEEK)
    )


def is_face_centered(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (
        abs(np.mean(xs) - 0.5) <= CENTER_TOLERANCE and
        abs(np.mean(ys) - 0.5) <= CENTER_TOLERANCE
    )


# -------------------------------------------------
# Expression Logic (FIXED)
# -------------------------------------------------

def is_expression_neutral_dynamic(blendshapes, baseline):
    """
    Uses rolling variance + baseline delta.
    """
    if not blendshapes:
        return True

    scores = {b.category_name: b.score for b in blendshapes}
    _expression_buffer.append(scores)

    # Wait until buffer fills
    if len(_expression_buffer) < EXPR_BUFFER_SIZE:
        return True

    # Track key expression channels
    keys = [
        "mouthSmileLeft", "mouthSmileRight",
        "eyeBlinkLeft", "eyeBlinkRight",
        "jawOpen", "browInnerUp"
    ]

    for key in keys:
        values = [f.get(key, 0) for f in _expression_buffer]
        variance = np.var(values)

        # If expression is jittery → reject
        if variance > EXPR_VARIANCE_LIMIT:
            return False

        # If baseline exists → check delta
        if baseline and key in baseline:
            if abs(values[-1] - baseline[key]) > EXPR_DELTA_TOLERANCE:
                return False

    return True


# -------------------------------------------------
# Main Gatekeeper
# -------------------------------------------------

def check_frame_quality(frame_bgr, landmarks, blendshapes, baseline=None):
    h, w, _ = frame_bgr.shape
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    distance_cm = estimate_distance_cm(landmarks, w)
    blur_score = compute_blur_score(gray)
    lighting_diff = compute_lighting_symmetry(gray, landmarks, w, h)
    centered = is_face_centered(landmarks)
    expression_ok = is_expression_neutral_dynamic(blendshapes, baseline)

    # -------- Priority-based guidance --------

    if distance_cm is None:
        return _fail("Adjust distance", distance_cm, blur_score, lighting_diff, centered, expression_ok)

    if distance_cm < DIST_MIN_CM:
        return _fail("Move back", distance_cm, blur_score, lighting_diff, centered, expression_ok)

    if distance_cm > DIST_MAX_CM:
        return _fail("Move closer", distance_cm, blur_score, lighting_diff, centered, expression_ok)

    if not centered:
        return _fail("Center your face", distance_cm, blur_score, lighting_diff, centered, expression_ok)

    if blur_score < BLUR_THRESHOLD:
        return _fail("Hold still", distance_cm, blur_score, lighting_diff, centered, expression_ok)

    if lighting_diff > LIGHTING_DIFF_THRESHOLD:
        return _fail("Fix lighting", distance_cm, blur_score, lighting_diff, centered, expression_ok)

    if not expression_ok:
        return _fail("Keep neutral expression", distance_cm, blur_score, lighting_diff, centered, expression_ok)

    return {
        "quality_ok": True,
        "message": "Quality OK",
        "distance_cm": distance_cm,
        "blur_score": blur_score,
        "lighting_status": "OK",
        "expression_status": "Stable"
    }


def _fail(msg, dist, blur, light, centered, expr):
    return {
        "quality_ok": False,
        "message": msg,
        "distance_cm": dist,
        "blur_score": blur,
        "lighting_status": "Uneven" if light > LIGHTING_DIFF_THRESHOLD else "OK",
        "expression_status": "Unstable" if not expr else "OK",
        "centered": centered
    }
