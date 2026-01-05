
# canonical/roi/cheeks.py
"""
Canonical ROI extraction — CHEEKS feature (FRONTAL)

Uses:
- canonical/FRONTAL.json → canonical_2d
- meshes/FRONTAL.json    → image-space bounds

Outputs:
- left_cheek.jpg
- right_cheek.jpg
"""

import json
import os
import cv2
import numpy as np

# -------------------------------------------------
# Cheek landmark indices (MediaPipe)
# -------------------------------------------------

LEFT_CHEEK = [123, 116, 117, 118]
RIGHT_CHEEK = [423, 345, 346, 347]

# -------------------------------------------------
# Loaders
# -------------------------------------------------

def load_canonical(session_dir):
    path = os.path.join(session_dir, "canonical", "FRONTAL.json")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing canonical file: {path}")

    with open(path, "r") as f:
        data = json.load(f)

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
    h, w = img_shape[:2]

    xs = mesh_3d[:, 0] * w
    ys = mesh_3d[:, 1] * h

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    face_w = max_x - min_x
    face_h = max_y - min_y

    cmin = canonical_pts.min(axis=0)
    cmax = canonical_pts.max(axis=0)

    norm = (canonical_pts - cmin) / (cmax - cmin + 1e-6)

    px = min_x + norm[:, 0] * face_w
    py = min_y + norm[:, 1] * face_h

    return np.stack([px, py], axis=1).astype(np.int32)


# -------------------------------------------------
# ROI helper
# -------------------------------------------------

def bbox_from_indices(points, indices, pad=25):
    xs = points[indices, 0]
    ys = points[indices, 1]

    x1 = max(int(xs.min() - pad), 0)
    y1 = max(int(ys.min() - pad), 0)
    x2 = int(xs.max() + pad)
    y2 = int(ys.max() + pad)

    return x1, y1, x2, y2


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def extract_cheek_rois(session_dir):

    canonical = load_canonical(session_dir)
    mesh_3d = load_mesh_landmarks(session_dir)
    image = load_image(session_dir)

    pixel_pts = canonical_to_pixel(canonical, mesh_3d, image.shape)

    out_dir = os.path.join(session_dir, "analysis", "cheeks")
    os.makedirs(out_dir, exist_ok=True)

    # LEFT CHEEK
    x1, y1, x2, y2 = bbox_from_indices(pixel_pts, LEFT_CHEEK)
    cv2.imwrite(
        os.path.join(out_dir, "left_cheek.jpg"),
        image[y1:y2, x1:x2]
    )

    # RIGHT CHEEK
    x1, y1, x2, y2 = bbox_from_indices(pixel_pts, RIGHT_CHEEK)
    cv2.imwrite(
        os.path.join(out_dir, "right_cheek.jpg"),
        image[y1:y2, x1:x2]
    )

    print("✅ Cheek ROIs extracted successfully")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m canonical.roi.cheeks <SESSION_DIR>")
        sys.exit(1)

    extract_cheek_rois(sys.argv[1])
