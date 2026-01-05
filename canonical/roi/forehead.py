# canonical/roi/forehead.py
"""
Canonical ROI extraction — FOREHEAD

✔ Eyebrow-anchored (clinically correct)
✔ Hair-safe upper bound
✔ Wide, stable forehead slab
✔ Uses canonical → image reprojection
"""

import json
import os
import cv2
import numpy as np

# -------------------------------------------------
# MediaPipe landmarks
# -------------------------------------------------

LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]

FOREHEAD_TOP = [10]  # upper forehead anchor (below hairline)

# -------------------------------------------------
# Loaders
# -------------------------------------------------

def load_canonical(session_dir):
    path = os.path.join(session_dir, "canonical", "FRONTAL.json")
    if not os.path.exists(path):
        raise FileNotFoundError("Missing canonical FRONTAL.json")

    with open(path) as f:
        data = json.load(f)

    return np.array(data["canonical_2d"], dtype=np.float32)


def load_mesh(session_dir):
    with open(os.path.join(session_dir, "meshes", "FRONTAL.json")) as f:
        return np.array(json.load(f)["mesh_3d"], dtype=np.float32)


def load_image(session_dir):
    img = cv2.imread(os.path.join(session_dir, "images", "FRONTAL_RAW.jpg"))
    if img is None:
        raise RuntimeError("FRONTAL_RAW.jpg not found")
    return img


# -------------------------------------------------
# Canonical → image reprojection
# -------------------------------------------------

def canonical_to_pixel(canonical, mesh, img_shape):
    h, w = img_shape[:2]

    xs = mesh[:, 0] * w
    ys = mesh[:, 1] * h

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    face_w = max_x - min_x
    face_h = max_y - min_y

    cmin = canonical.min(axis=0)
    cmax = canonical.max(axis=0)
    norm = (canonical - cmin) / (cmax - cmin + 1e-6)

    px = min_x + norm[:, 0] * face_w
    py = min_y + norm[:, 1] * face_h

    return np.stack([px, py], axis=1).astype(np.int32)


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def extract_forehead_roi(session_dir):

    canonical = load_canonical(session_dir)
    mesh = load_mesh(session_dir)
    img = load_image(session_dir)

    pts = canonical_to_pixel(canonical, mesh, img.shape)

    eyebrow_pts = pts[LEFT_EYEBROW + RIGHT_EYEBROW]
    forehead_top = pts[FOREHEAD_TOP][0]

    # ---- Vertical bounds (CRITICAL)
    y_bottom = eyebrow_pts[:, 1].min() - 4        # just above brows
    y_top = forehead_top[1] + 14                  # safe from hair

    # ---- Horizontal bounds (wide slab)
    x_left = eyebrow_pts[:, 0].min() - 25
    x_right = eyebrow_pts[:, 0].max() + 25

    h, w = img.shape[:2]
    x1 = max(int(x_left), 0)
    x2 = min(int(x_right), w)
    y1 = max(int(y_top), 0)
    y2 = min(int(y_bottom), h)

    roi = img[y1:y2, x1:x2]

    out_dir = os.path.join(session_dir, "analysis", "forehead")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "forehead.jpg")
    cv2.imwrite(out_path, roi)

    print(f"✅ Forehead ROI saved: {out_path}")


if __name__ == "__main__":
    import sys
    extract_forehead_roi(sys.argv[1])

