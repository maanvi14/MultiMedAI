"""
Semantic TEETH ROI extraction — FINAL VERSION

✔ Works for TEETH_SMILE and TEETH_OPEN
✔ Uses OUTER_LIPS semantic bbox (stable)
✔ Adds padding for complete teeth visibility
✔ Saves both ROI + debug bbox image
"""

import json
import os
import cv2
import numpy as np

# -------------------------------------------------
# Mouth landmark indices (MediaPipe)
# -------------------------------------------------
OUTER_LIPS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146
]

# -------------------------------------------------
# Loaders
# -------------------------------------------------
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
# TEETH ROI bbox
# -------------------------------------------------
def compute_teeth_bbox(pts, img_shape):
    h, w = img_shape[:2]

    xs = pts[OUTER_LIPS, 0]
    ys = pts[OUTER_LIPS, 1]

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    mouth_h = y_max - y_min
    mouth_w = x_max - x_min

    # ✅ Keep ROI big enough to capture full teeth region
    x1 = x_min - int(0.25 * mouth_w)
    x2 = x_max + int(0.25 * mouth_w)

    y1 = y_min - int(0.35 * mouth_h)   # include upper lip/teeth
    y2 = y_max + int(0.55 * mouth_h)   # include lower teeth + small chin margin

    return (
        max(0, x1),
        max(0, y1),
        min(w, x2),
        min(h, y2)
    )

# -------------------------------------------------
# MAIN (extract for both modes)
# -------------------------------------------------
def extract_teeth_roi(session_dir):
    modes = ["TEETH_SMILE", "TEETH_OPEN"]

    out_dir = os.path.join(session_dir, "analysis", "teeth_roi")
    os.makedirs(out_dir, exist_ok=True)

    for mode in modes:
        img = load_image(session_dir, mode)
        mesh = load_mesh_landmarks(session_dir, mode)
        pts = mesh_to_pixel(mesh, img.shape)

        x1, y1, x2, y2 = compute_teeth_bbox(pts, img.shape)

        roi = img[y1:y2, x1:x2]

        # Save ROI
        roi_path = os.path.join(out_dir, f"{mode}_ROI.jpg")
        cv2.imwrite(roi_path, roi)

        # Save Debug Image (bbox visualization)
        debug = img.copy()
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        debug_path = os.path.join(out_dir, f"{mode}_DEBUG.jpg")
        cv2.imwrite(debug_path, debug)

        print(f"✅ {mode} ROI saved: {roi_path}")

    print("✅ Teeth ROI extraction completed.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m canonical.roi.teeth <SESSION_DIR>")
        sys.exit(1)

    extract_teeth_roi(sys.argv[1])
