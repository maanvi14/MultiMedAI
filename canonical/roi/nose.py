"""
Canonical ROI extraction — NOSE feature (FINAL, STABLE)

✔ Proven correct cutting
✔ Uses RAW mesh → pixel
✔ No canonical reprojection
✔ Works for FRONTAL / LEFT_PROFILE / RIGHT_PROFILE
"""

import json
import os
import cv2
import numpy as np

# -------------------------------------------------
# Nose landmark indices (MediaPipe – stable)
# -------------------------------------------------

NOSE_INDICES = [1, 2, 98, 327, 168]

# -------------------------------------------------
# Loaders
# -------------------------------------------------

def load_mesh_landmarks(session_dir, mode):
    path = os.path.join(session_dir, "meshes", f"{mode}.json")
    with open(path, "r") as f:
        data = json.load(f)
    return np.array(data["mesh_3d"], dtype=np.float32)


def load_image(session_dir, mode):
    path = os.path.join(session_dir, "images", f"{mode}_RAW.jpg")
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"{mode}_RAW.jpg not found")
    return img


# -------------------------------------------------
# RAW mesh → pixel (THIS IS THE KEY)
# -------------------------------------------------

def mesh_to_pixel(mesh_3d, img_shape):
    h, w = img_shape[:2]
    pts = np.zeros((mesh_3d.shape[0], 2), dtype=np.int32)
    pts[:, 0] = (mesh_3d[:, 0] * w).astype(np.int32)
    pts[:, 1] = (mesh_3d[:, 1] * h).astype(np.int32)
    return pts


# -------------------------------------------------
# SIMPLE bbox (WORKING & CORRECT)
# -------------------------------------------------

def bbox_from_indices(points, indices, pad=18):
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

def extract_nose_rois(session_dir):

    out_dir = os.path.join(session_dir, "analysis", "nose")
    os.makedirs(out_dir, exist_ok=True)

    for mode in ["FRONTAL", "LEFT_PROFILE", "RIGHT_PROFILE"]:

        mesh_3d = load_mesh_landmarks(session_dir, mode)
        image = load_image(session_dir, mode)

        pixel_pts = mesh_to_pixel(mesh_3d, image.shape)

        x1, y1, x2, y2 = bbox_from_indices(pixel_pts, NOSE_INDICES)

        roi = image[y1:y2, x1:x2]
        out_path = os.path.join(out_dir, f"{mode.lower()}.jpg")
        cv2.imwrite(out_path, roi)

        print(f"✅ Nose ROI saved: {out_path}")


if __name__ == "__main__":
    import sys
    extract_nose_rois(sys.argv[1])