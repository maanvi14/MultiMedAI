"""
Eye ROI extraction (MODEL-COMPATIBLE)

Outputs:
- left_eye.jpg
- right_eye.jpg
- both_eyes_eyebrows.jpg

Uses:
- MediaPipe Face Mesh landmarks
- RGB rectangular crops (NO polygon mask)
"""

import os
import json
import cv2
import numpy as np

# -------------------------------------------------
# MediaPipe landmark indices
# -------------------------------------------------

LEFT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

RIGHT_EYE = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]

LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]

# -------------------------------------------------
# Loaders
# -------------------------------------------------

def load_mesh(session_dir):
    path = os.path.join(session_dir, "meshes", "FRONTAL.json")
    with open(path, "r") as f:
        data = json.load(f)
    return np.array(data["mesh_3d"], dtype=np.float32)

def load_image(session_dir):
    img_path = os.path.join(session_dir, "images", "FRONTAL_RAW.jpg")
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError("FRONTAL_RAW.jpg not found")
    return img

# -------------------------------------------------
# Landmark → Pixel conversion
# -------------------------------------------------

def mesh_to_pixel(mesh, img_shape):
    h, w = img_shape[:2]
    px = np.zeros((mesh.shape[0], 2), dtype=np.int32)
    px[:, 0] = (mesh[:, 0] * w).astype(int)
    px[:, 1] = (mesh[:, 1] * h).astype(int)
    return px

# -------------------------------------------------
# Bounding box helper
# -------------------------------------------------

def bbox(points, indices, pad=10):
    xs = points[indices, 0]
    ys = points[indices, 1]

    x1 = max(xs.min() - pad, 0)
    y1 = max(ys.min() - pad, 0)
    x2 = xs.max() + pad
    y2 = ys.max() + pad

    return int(x1), int(y1), int(x2), int(y2)

# -------------------------------------------------
# MAIN
# -------------------------------------------------

def extract_eye_rois(session_dir):
    print("👁 Extracting MODEL-COMPATIBLE eye ROIs...")

    img = load_image(session_dir)
    mesh = load_mesh(session_dir)
    pts = mesh_to_pixel(mesh, img.shape)

    out_dir = os.path.join(session_dir, "analysis", "eyes")
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # LEFT EYE
    # -------------------------
    x1, y1, x2, y2 = bbox(pts, LEFT_EYE, pad=12)
    left_eye = img[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(out_dir, "left_eye.jpg"), left_eye)

    # -------------------------
    # RIGHT EYE
    # -------------------------
    x1, y1, x2, y2 = bbox(pts, RIGHT_EYE, pad=12)
    right_eye = img[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(out_dir, "right_eye.jpg"), right_eye)

    # -------------------------
    # BOTH EYES + EYEBROWS
    # -------------------------
    combined = LEFT_EYE + RIGHT_EYE + LEFT_EYEBROW + RIGHT_EYEBROW
    x1, y1, x2, y2 = bbox(pts, combined, pad=18)
    both = img[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(out_dir, "both_eyes_eyebrows.jpg"), both)

    print("✅ Eye ROIs saved (RGB, model-ready)")

# -------------------------------------------------
# CLI
# -------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m canonical.roi.eyes <SESSION_DIR>")
        sys.exit(1)

    extract_eye_rois(sys.argv[1])
