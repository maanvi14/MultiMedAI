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
5. Neutral expression (basic)

Returns:
- quality_ok (bool)
- message (UI guidance)
- diagnostic metrics
"""

import cv2
import numpy as np

# -------------------------------------------------
# Constants & thresholds (clinically chosen)
# -------------------------------------------------

IRIS_DIAMETER_MM = 11.7
DIST_MIN_CM = 40
DIST_MAX_CM = 60

BLUR_THRESHOLD = 100.0
LIGHTING_DIFF_THRESHOLD = 40

SMILE_THRESHOLD = 0.10
BLINK_THRESHOLD = 0.20
JAW_THRESHOLD = 0.15
BROW_THRESHOLD = 0.15

CENTER_TOLERANCE = 0.15

# Cheek landmarks
LEFT_CHEEK = [123, 116]
RIGHT_CHEEK = [423, 345]

# Iris landmarks (use ONE eye only – more stable)
LEFT_IRIS = [474, 475, 476, 477]

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def estimate_distance_cm(landmarks, frame_w):
    """
    Estimate distance using iris diameter (single eye).
    """
    xs = [landmarks[i].x * frame_w for i in LEFT_IRIS]
    iris_pixel_diameter = max(xs) - min(xs)

    if iris_pixel_diameter <= 1:
        return None

    # Pinhole approximation (relative, not absolute optics)
    distance_cm = (IRIS_DIAMETER_MM * frame_w) / iris_pixel_diameter / 10
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

    left = cheek_mean(LEFT_CHEEK)
    right = cheek_mean(RIGHT_CHEEK)
    return abs(left - right)


def is_face_centered(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (
        abs(np.mean(xs) - 0.5) <= CENTER_TOLERANCE and
        abs(np.mean(ys) - 0.5) <= CENTER_TOLERANCE
    )


def is_expression_neutral(blendshapes):
    """
    Basic neutral expression check (MVP).
    Eye squint will be added later (remembered).
    """
    if not blendshapes:
        return True

    def score(name):
        for b in blendshapes:
            if b.category_name == name:
                return b.score
        return 0.0

    return (
        score("mouthSmileLeft") < SMILE_THRESHOLD and
        score("mouthSmileRight") < SMILE_THRESHOLD and
        score("eyeBlinkLeft") < BLINK_THRESHOLD and
        score("eyeBlinkRight") < BLINK_THRESHOLD and
        score("jawOpen") < JAW_THRESHOLD and
        score("browInnerUp") < BROW_THRESHOLD
    )

# -------------------------------------------------
# Main gatekeeper
# -------------------------------------------------

def check_frame_quality(frame_bgr, landmarks, blendshapes):
    h, w, _ = frame_bgr.shape
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    distance_cm = estimate_distance_cm(landmarks, w)
    blur_score = compute_blur_score(gray)
    lighting_diff = compute_lighting_symmetry(gray, landmarks, w, h)
    centered = is_face_centered(landmarks)
    expression_ok = is_expression_neutral(blendshapes)

    # -------- Decision logic (priority order) --------

    if distance_cm is None:
        return _fail("Adjust distance", distance_cm, blur_score, lighting_diff, centered, expression_ok)

    if distance_cm < DIST_MIN_CM:
        return _fail("Move back", distance_cm, blur_score, lighting_diff, centered, expression_ok)

    if distance_cm > DIST_MAX_CM:
        return _fail("Move closer", distance_cm, blur_score, lighting_diff, centered, expression_ok)

    if not centered:
        return _fail("Center your face", distance_cm, blur_score, lighting_diff, centered, expression_ok)

    if blur_score < BLUR_THRESHOLD:
        return _fail("Hold still (blur)", distance_cm, blur_score, lighting_diff, centered, expression_ok)

    if lighting_diff > LIGHTING_DIFF_THRESHOLD:
        return _fail("Fix lighting", distance_cm, blur_score, lighting_diff, centered, expression_ok)

    if not expression_ok:
        return _fail("Relax face", distance_cm, blur_score, lighting_diff, centered, expression_ok)

    # -------- PASS --------
    return {
        "quality_ok": True,
        "message": "Quality OK",
        "distance_cm": distance_cm,
        "blur_score": blur_score,
        "lighting_status": "OK",
        "expression_status": "Neutral"
    }


def _fail(msg, dist, blur, light, centered, expr):
    return {
        "quality_ok": False,
        "message": msg,
        "distance_cm": dist,
        "blur_score": blur,
        "lighting_status": "Uneven" if light > LIGHTING_DIFF_THRESHOLD else "OK",
        "expression_status": "Not neutral" if not expr else "OK",
        "centered": centered
    }
