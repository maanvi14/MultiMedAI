# canonical/roi/chin.py
"""
CHIN ROI extraction — SEMANTIC + CANONICAL (FRONTAL)

Design:
- Anchor chin BELOW lower lip using anatomy
- Use canonical 2D landmarks for invariance
- Map back to image space using original face bounds

Output:
analysis/chin/chin.jpg
"""

import json
import os
import cv2
import numpy as np

# -------------------------------------------------
# Landmark indices (MediaPipe)
# -------------------------------------------------

LOWER_LIP = 17
MOUTH_CORNERS = [61, 291]

# -------------------------------------------------
# Loaders
# -------------------------------------------------

def load_canonical(session_dir):
    path = os.path.join(session_dir, "canonical", "FRONTAL.json")
    with open(path) as f:
        data = json.load(f)

    canonical = np.array(data["canonical_2d"], dtype=np.float32)
    assert canonical.shape[1] == 2, "Canonical landmarks must be 2D"
    return canonical


def load_mesh_landmarks(session_dir):
    path = os.path.join(session_dir, "meshes", "FRONTAL.json")
    with open(path) as f:
        data = json.load(f)
    return np.array(data["mesh_3d"], dtype=np.float32)


def load_image(session_dir):
    img = cv2.imread(os.path.join(session_dir, "images", "FRONTAL_RAW.jpg"))
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
# Chin ROI (semantic, canonical)
# -------------------------------------------------

def extract_chin_roi(session_dir):

    canonical = load_canonical(session_dir)
    mesh_3d = load_mesh_landmarks(session_dir)
    img = load_image(session_dir)

    pts = canonical_to_pixel(canonical, mesh_3d, img.shape)
    h, w = img.shape[:2]

    # --- TOP: below lower lip (semantic anchor) ---
    lip_y = int(pts[LOWER_LIP, 1])
    top_offset = max(int(0.03 * h), 8)
    y1 = np.clip(lip_y + top_offset, 0, h - 2)

    # --- BOTTOM: semantic depth ---
    y2 = int(y1 + 0.20 * h)
    y2 = np.clip(y2, y1 + 1, h)

    # --- WIDTH: mouth adaptive ---
    mouth_x = pts[MOUTH_CORNERS, 0]
    x_center = int(mouth_x.mean())
    mouth_width = abs(int(mouth_x[1]) - int(mouth_x[0]))

    half_width = int(max(mouth_width * 0.6, 0.22 * w))

    x1 = np.clip(x_center - half_width, 0, w)
    x2 = np.clip(x_center + half_width, 0, w)

    out_dir = os.path.join(session_dir, "analysis", "chin")
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(
        os.path.join(out_dir, "chin.jpg"),
        img[y1:y2, x1:x2]
    )

    print("✅ Chin ROI extracted (canonical + semantic)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m canonical.roi.chin <SESSION_DIR>")
        sys.exit(1)

    extract_chin_roi(sys.argv[1])
