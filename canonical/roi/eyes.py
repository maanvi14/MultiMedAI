# canonical/roi/eyes.py
"""
Canonical ROI extraction — EYES feature (FRONTAL)

Uses:
- canonical/FRONTAL.json → canonical_2d
- meshes/FRONTAL.json    → original image space
"""

import json
import os
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

def load_canonical(session_dir):
    path = os.path.join(session_dir, "canonical", "FRONTAL.json")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing canonical file: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if "canonical_2d" not in data:
        raise KeyError("canonical_2d missing in FRONTAL.json")

    canonical = np.array(data["canonical_2d"], dtype=np.float32)

    assert canonical.shape[1] == 2, "Canonical landmarks must be 2D"
    return canonical


def load_mesh_landmarks(session_dir):
    path = os.path.join(session_dir, "meshes", "FRONTAL.json")

    with open(path, "r") as f:
        data = json.load(f)

    return np.array(data["mesh_3d"], dtype=np.float32)


def load_image(session_dir):
    path = os.path.join(session_dir, "images", "FRONTAL_RAW.jpg")
    img = cv2.imread(path)

    if img is None:
        raise RuntimeError("FRONTAL_RAW.jpg not found")

    return img


# -------------------------------------------------
# Canonical → Image mapping
# -------------------------------------------------

def canonical_to_pixel(canonical_pts, mesh_3d, img_shape):
    """
    Maps canonical landmarks back into image space
    using original face bounding box.
    """

    h, w = img_shape[:2]

    xs = mesh_3d[:, 0] * w
    ys = mesh_3d[:, 1] * h

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    face_w = max_x - min_x
    face_h = max_y - min_y

    # Normalize canonical space
    cmin = canonical_pts.min(axis=0)
    cmax = canonical_pts.max(axis=0)
    norm = (canonical_pts - cmin) / (cmax - cmin + 1e-6)

    px = min_x + norm[:, 0] * face_w
    py = min_y + norm[:, 1] * face_h

    return np.stack([px, py], axis=1).astype(np.int32)


# -------------------------------------------------
# ROI helpers
# -------------------------------------------------

def bbox_from_indices(points, indices, pad=12, down_bias=0.35):
    xs = points[indices, 0]
    ys = points[indices, 1]

    h = ys.max() - ys.min()
    extra_down = int(h * down_bias)

    x1 = max(int(xs.min() - pad), 0)
    y1 = max(int(ys.min() - pad), 0)
    x2 = int(xs.max() + pad)
    y2 = int(ys.max() + pad + extra_down)

    return x1, y1, x2, y2

# -------------------------------------------------
# MAIN
# -------------------------------------------------

def extract_eye_rois(session_dir):

    canonical = load_canonical(session_dir)
    mesh_3d = load_mesh_landmarks(session_dir)
    image = load_image(session_dir)

    pixel_pts = canonical_to_pixel(canonical, mesh_3d, image.shape)

    out_dir = os.path.join(session_dir, "analysis", "eyes")
    os.makedirs(out_dir, exist_ok=True)

    # LEFT EYE
    x1, y1, x2, y2 = bbox_from_indices(pixel_pts, LEFT_EYE)
    cv2.imwrite(os.path.join(out_dir, "left_eye.jpg"), image[y1:y2, x1:x2])

    # RIGHT EYE
    x1, y1, x2, y2 = bbox_from_indices(pixel_pts, RIGHT_EYE)
    cv2.imwrite(os.path.join(out_dir, "right_eye.jpg"), image[y1:y2, x1:x2])

    # BOTH EYES + EYEBROWS
    combined = LEFT_EYE + RIGHT_EYE + LEFT_EYEBROW + RIGHT_EYEBROW
    x1, y1, x2, y2 = bbox_from_indices(pixel_pts, combined, pad=20)
    cv2.imwrite(
        os.path.join(out_dir, "both_eyes_eyebrows.jpg"),
        image[y1:y2, x1:x2]
    )

    print("✅ Eyes ROIs extracted correctly")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m canonical.roi.eyes <SESSION_DIR>")
        sys.exit(1)

    extract_eye_rois(sys.argv[1])

