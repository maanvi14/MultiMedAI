"""
quality_gatekeeper.py

LAYER 3A: Quality Gatekeeper (MODE-AWARE)
---------------------------------------
Purpose:
- Ensure a single frame is Prakriti-ready

Supports:
- FRONTAL
- LEFT_PROFILE
- RIGHT_PROFILE
- TEETH_SMILE
- TEETH_OPEN
"""

import cv2
import numpy as np
from collections import deque

# -------------------------------------------------
# Constants (REAL-WORLD SAFE)
# -------------------------------------------------

IRIS_DIAMETER_MM = 11.7
DIST_MIN_CM = 40
DIST_MAX_CM = 60

BLUR_THRESHOLD = 20
LIGHTING_DIFF_THRESHOLD = 50
CENTER_TOLERANCE = 0.15

# Expression logic
EXPR_DELTA_TOLERANCE = 0.08
EXPR_VARIANCE_LIMIT = 0.005
EXPR_BUFFER_SIZE = 5

# Teeth thresholds
TEETH_CLOSED_MAX = 0.30   # TEETH_SMILE should be <= this
TEETH_OPEN_MIN   = 0.35   # TEETH_OPEN should be >= this

# Landmarks
LEFT_CHEEK = [123, 116]
RIGHT_CHEEK = [423, 345]
LEFT_IRIS = [474, 475, 476, 477]

# -------------------------------------------------
# Rolling expression buffer (INTENTIONAL GLOBAL)
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

    return round((IRIS_DIAMETER_MM * frame_w) / iris_px / 10, 1)


def compute_blur_score(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_cheek_brightness(gray, landmarks, indices, w, h):
    vals = []
    for idx in indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        if 0 <= x < w and 0 <= y < h:
            vals.append(gray[y, x])
    return np.mean(vals) if vals else 0


def is_face_centered(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (
        abs(np.mean(xs) - 0.5) <= CENTER_TOLERANCE and
        abs(np.mean(ys) - 0.5) <= CENTER_TOLERANCE
    )

# -------------------------------------------------
# Expression Logic (Dynamic + Smoothed)
# -------------------------------------------------

def is_expression_neutral_dynamic(blendshapes, baseline):
    if not blendshapes:
        return True

    scores = {b.category_name: b.score for b in blendshapes}
    _expression_buffer.append(scores)

    if len(_expression_buffer) < EXPR_BUFFER_SIZE:
        return True

    keys = [
        "mouthSmileLeft", "mouthSmileRight",
        "eyeBlinkLeft", "eyeBlinkRight",
        "jawOpen", "browInnerUp"
    ]

    for key in keys:
        values = [f.get(key, 0) for f in _expression_buffer]

        if np.var(values) > EXPR_VARIANCE_LIMIT:
            return False

        if baseline and key in baseline:
            if abs(values[-1] - baseline[key]) > EXPR_DELTA_TOLERANCE:
                return False

    return True


def mouth_open_ratio(face, w, h):
    """
    (upper-lower lip distance) / (mouth width)
    """
    upper = np.array([face[13].x * w, face[13].y * h])
    lower = np.array([face[14].x * w, face[14].y * h])
    left  = np.array([face[61].x * w, face[61].y * h])
    right = np.array([face[291].x * w, face[291].y * h])

    return np.linalg.norm(upper - lower) / (np.linalg.norm(left - right) + 1e-6)

# -------------------------------------------------
# MAIN QUALITY GATEKEEPER (MODE-AWARE)
# -------------------------------------------------

def check_frame_quality(frame_bgr, landmarks, blendshapes, baseline=None, capture_mode="FRONTAL"):
    h, w, _ = frame_bgr.shape
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    distance_cm = estimate_distance_cm(landmarks, w)
    blur_score = compute_blur_score(gray)
    expression_ok = is_expression_neutral_dynamic(blendshapes, baseline)

    # ---- Distance ----
    if distance_cm is None:
        return _fail("Adjust distance", distance_cm, blur_score)

    if distance_cm < DIST_MIN_CM:
        return _fail("Move back", distance_cm, blur_score)

    if distance_cm > DIST_MAX_CM:
        return _fail("Move closer", distance_cm, blur_score)

    # ---- Blur ----
    if blur_score < BLUR_THRESHOLD:
        return _fail("Hold still", distance_cm, blur_score)

    # ---- Mode-specific checks ----
    if capture_mode == "FRONTAL":
        if not is_face_centered(landmarks):
            return _fail("Center your face", distance_cm, blur_score)

        left = compute_cheek_brightness(gray, landmarks, LEFT_CHEEK, w, h)
        right = compute_cheek_brightness(gray, landmarks, RIGHT_CHEEK, w, h)
        if abs(left - right) > LIGHTING_DIFF_THRESHOLD:
            return _fail("Fix lighting", distance_cm, blur_score)

    elif capture_mode == "LEFT_PROFILE":
        left = compute_cheek_brightness(gray, landmarks, LEFT_CHEEK, w, h)
        if left == 0:
            return _fail("Improve lighting (left side)", distance_cm, blur_score)

    elif capture_mode == "RIGHT_PROFILE":
        right = compute_cheek_brightness(gray, landmarks, RIGHT_CHEEK, w, h)
        if right == 0:
            return _fail("Improve lighting (right side)", distance_cm, blur_score)

    # ---- TEETH Modes ----
    # Teeth does NOT need lighting checks like cheeks, but mouth ratio gating is needed
    if capture_mode in ["TEETH_SMILE", "TEETH_OPEN"]:
        ratio = mouth_open_ratio(landmarks, w, h)

        if capture_mode == "TEETH_SMILE":
            teeth_ok = ratio <= TEETH_CLOSED_MAX
            if not teeth_ok:
                return {
                    "quality_ok": False,
                    "message": "Smile naturally (close mouth slightly)",
                    "distance_cm": distance_cm,
                    "blur_score": blur_score,
                    "mouth_open_ratio": float(ratio),
                    "teeth_ok": False
                }

        elif capture_mode == "TEETH_OPEN":
            teeth_ok = ratio >= TEETH_OPEN_MIN
            if not teeth_ok:
                return {
                    "quality_ok": False,
                    "message": "Open mouth gently (show teeth)",
                    "distance_cm": distance_cm,
                    "blur_score": blur_score,
                    "mouth_open_ratio": float(ratio),
                    "teeth_ok": False
                }

    # ---- Expression ----
    if not expression_ok:
        return _fail("Keep neutral expression", distance_cm, blur_score)

    # ---- PASS ----
    out = {
        "quality_ok": True,
        "message": "Quality OK",
        "distance_cm": distance_cm,
        "blur_score": blur_score,
        "expression_status": "Stable"
    }

    # include ratio for teeth modes even when pass
    if capture_mode in ["TEETH_SMILE", "TEETH_OPEN"]:
        ratio = mouth_open_ratio(landmarks, w, h)
        out["mouth_open_ratio"] = float(ratio)
        out["teeth_ok"] = True

    return out


def _fail(msg, dist, blur):
    return {
        "quality_ok": False,
        "message": msg,
        "distance_cm": dist,
        "blur_score": blur
    }
