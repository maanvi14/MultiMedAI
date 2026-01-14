"""
Semantic TEETH ROI extraction — FINAL CLEAN

✔ Works for TEETH_SMILE and TEETH_OPEN
✔ OUTER_LIPS bbox (stable)
✔ DEBUG = full face with ONE bbox only
✔ ROI = clean crop (NO green line)
"""

import json
import os
import cv2
import numpy as np

OUTER_LIPS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146
]


def load_mesh_landmarks(session_dir, mode):
    mesh_path = os.path.join(session_dir, "meshes", f"{mode}.json")
    if not os.path.exists(mesh_path):
        raise RuntimeError(f"Mesh not found: {mesh_path}")

    with open(mesh_path, "r") as f:
        data = json.load(f)

    return np.array(data["mesh_3d"], dtype=np.float32)


def load_image(session_dir, mode):
    img_path = os.path.join(session_dir, "images", f"{mode}_RAW.jpg")
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"{mode}_RAW.jpg not found at: {img_path}")
    return img


def mesh_to_pixel(mesh, img_shape):
    h, w = img_shape[:2]
    pts = np.zeros((mesh.shape[0], 2), dtype=np.int32)
    pts[:, 0] = (mesh[:, 0] * w).astype(np.int32)
    pts[:, 1] = (mesh[:, 1] * h).astype(np.int32)
    return pts


def compute_teeth_bbox(pts, img_shape):
    h, w = img_shape[:2]

    xs = pts[OUTER_LIPS, 0]
    ys = pts[OUTER_LIPS, 1]

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    mouth_h = y_max - y_min
    mouth_w = x_max - x_min

    # ✅ Balanced padding
    x1 = x_min - int(0.15 * mouth_w)
    x2 = x_max + int(0.15 * mouth_w)
    y1 = y_min - int(0.20 * mouth_h)
    y2 = y_max + int(0.35 * mouth_h)

    # ✅ Minimum ROI size (prevents tiny ROI)
    min_w = 120
    min_h = 90

    if (x2 - x1) < min_w:
        cx = (x1 + x2) // 2
        x1 = cx - min_w // 2
        x2 = cx + min_w // 2

    if (y2 - y1) < min_h:
        cy = (y1 + y2) // 2
        y1 = cy - min_h // 2
        y2 = cy + min_h // 2

    # ✅ Clamp
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # ✅ Safe
    if x2 <= x1:
        x2 = min(w, x1 + 2)
    if y2 <= y1:
        y2 = min(h, y1 + 2)

    return x1, y1, x2, y2


def extract_teeth_roi(session_dir):
    modes = ["TEETH_SMILE", "TEETH_OPEN"]

    out_dir = os.path.join(session_dir, "analysis", "teeth_roi")
    os.makedirs(out_dir, exist_ok=True)

    for mode in modes:
        # ✅ Always keep a CLEAN original image for ROI crop
        img_original = load_image(session_dir, mode)

        mesh = load_mesh_landmarks(session_dir, mode)
        pts = mesh_to_pixel(mesh, img_original.shape)

        x1, y1, x2, y2 = compute_teeth_bbox(pts, img_original.shape)

        # ✅ ROI from clean image ONLY (no rectangles ever drawn here)
        roi = img_original[y1:y2, x1:x2]
        roi_path = os.path.join(out_dir, f"{mode}_ROI.jpg")
        cv2.imwrite(roi_path, roi)

        # ✅ DEBUG from a fresh copy + SINGLE rectangle only once
        debug = img_original.copy()
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        debug_path = os.path.join(out_dir, f"{mode}_DEBUG.jpg")
        cv2.imwrite(debug_path, debug)

        print(f"✅ {mode} ROI saved: {roi_path}")
        print(f"✅ {mode} DEBUG saved: {debug_path}")

    print("✅ Teeth ROI extraction completed.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m canonical.roi.teeth <SESSION_DIR>")
        sys.exit(1)

    extract_teeth_roi(sys.argv[1])
