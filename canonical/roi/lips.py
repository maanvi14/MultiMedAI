"""
Semantic LIPS ROI extraction — FINAL VERSION (FRONTAL)

✔ Upper + lower lips
✔ Slight skin margin
✔ No chin / no nose bleed
✔ Stable across distance
"""

import json
import os
import cv2
import numpy as np

# -------------------------------------------------
# Lip landmarks (MediaPipe)
# -------------------------------------------------
UPPER_LIP = [0, 13, 14]
LOWER_LIP = [17, 18]
MOUTH_CORNERS = [61, 291]

LIP_INDICES = UPPER_LIP + LOWER_LIP + MOUTH_CORNERS

# -------------------------------------------------
# Loaders
# -------------------------------------------------

def load_mesh_landmarks(session_dir):
    with open(os.path.join(session_dir, "meshes", "FRONTAL.json")) as f:
        return np.array(json.load(f)["mesh_3d"], dtype=np.float32)

def load_image(session_dir):
    img = cv2.imread(os.path.join(session_dir, "images", "FRONTAL_RAW.jpg"))
    if img is None:
        raise RuntimeError("FRONTAL_RAW.jpg not found")
    return img

# -------------------------------------------------
# Mesh → Pixel
# -------------------------------------------------

def mesh_to_pixel(mesh, img_shape):
    h, w = img_shape[:2]
    pts = np.zeros((mesh.shape[0], 2), dtype=np.int32)
    pts[:, 0] = (mesh[:, 0] * w).astype(np.int32)
    pts[:, 1] = (mesh[:, 1] * h).astype(np.int32)
    return pts

# -------------------------------------------------
# LIPS ROI (semantic)
# -------------------------------------------------

def compute_lips_bbox(pts, img_shape):
    h, w = img_shape[:2]

    xs = pts[LIP_INDICES, 0]
    ys = pts[LIP_INDICES, 1]

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    lip_h = y_max - y_min
    lip_w = x_max - x_min

    # Horizontal: reduce side skin
    x1 = x_min - int(0.20 * lip_w)
    x2 = x_max + int(0.20 * lip_w)

    # Vertical: reduce upper skin & chin bleed
    y1 = y_min - int(0.25 * lip_h)
    y2 = y_max + int(0.35 * lip_h)


    return (
        max(0, x1),
        max(0, y1),
        min(w, x2),
        min(h, y2)
    )

# -------------------------------------------------
# MAIN
# -------------------------------------------------

def extract_lips_roi(session_dir):

    img = load_image(session_dir)
    mesh = load_mesh_landmarks(session_dir)
    pts = mesh_to_pixel(mesh, img.shape)

    x1, y1, x2, y2 = compute_lips_bbox(pts, img.shape)

    out_dir = os.path.join(session_dir, "analysis", "lips")
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(
        os.path.join(out_dir, "lips.jpg"),
        img[y1:y2, x1:x2]
    )

    print("✅ Lips ROI saved")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m canonical.roi.lips <SESSION_DIR>")
        sys.exit(1)

    extract_lips_roi(sys.argv[1])
