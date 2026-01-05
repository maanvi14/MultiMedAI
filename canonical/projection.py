# canonical_projection.py
"""
Canonical facial projection (SESSION-AWARE)
-------------------------------------------
Produces TWO canonical representations:

1. canonical_3d_metric
   - Pose-normalized
   - NO scaling
   - Used for anthropometry / face structure

2. canonical_2d
   - Pose-normalized
   - Centered + scale-normalized
   - Used for ROIs, symmetry, visualization

Input:
- session_dir/meshes/{FRONTAL,LEFT_PROFILE,RIGHT_PROFILE}.json

Output:
- session_dir/canonical/{FRONTAL,LEFT_PROFILE,RIGHT_PROFILE}.json
"""

import json
import numpy as np
import os

# -------------------------------------------------
# Load golden mesh
# -------------------------------------------------
def load_golden_mesh(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    mesh_3d = np.array(data["mesh_3d"], dtype=np.float32)      # [N, 3]
    transform = np.array(data["transform"], dtype=np.float32) # [4, 4]

    return mesh_3d, transform


# -------------------------------------------------
# Canonical projection
# -------------------------------------------------
def project_to_canonical(mesh_3d, transform):
    """
    Steps:
    1. Convert landmarks to homogeneous coordinates
    2. Apply inverse face transform (pose normalization)
    3. Preserve metric canonical 3D
    4. Generate centered + scaled canonical 2D for ROIs
    """

    # 1. Homogeneous coordinates
    ones = np.ones((mesh_3d.shape[0], 1), dtype=np.float32)
    mesh_h = np.hstack([mesh_3d, ones])  # [N, 4]

    # 2. Inverse transform (pose normalization)
    T_inv = np.linalg.inv(transform)
    canonical_3d = (T_inv @ mesh_h.T).T[:, :3]  # [N, 3]

    # ---- METRIC CANONICAL 3D (DO NOT TOUCH SCALE) ----
    canonical_3d_metric = canonical_3d.copy()

    # ---- CANONICAL 2D (FOR ROIs / VISUALIZATION) ----
    canonical_2d = canonical_3d[:, :2]
    center = canonical_2d.mean(axis=0)
    canonical_2d -= center

    scale = np.linalg.norm(canonical_2d, axis=1).max()
    canonical_2d /= (scale + 1e-6)

    return canonical_3d_metric, canonical_2d


# -------------------------------------------------
# Save canonical output
# -------------------------------------------------
def save_canonical(canonical_3d_metric, canonical_2d, out_path):
    payload = {
        "canonical_3d_metric": canonical_3d_metric.tolist(),
        "canonical_2d": canonical_2d.tolist()
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"✅ Canonical saved: {out_path}")


# -------------------------------------------------
# AUTO-RUN FOR ONE SESSION
# -------------------------------------------------
def run_canonical_for_session(session_dir):
    mesh_dir = os.path.join(session_dir, "meshes")
    out_dir = os.path.join(session_dir, "canonical")
    os.makedirs(out_dir, exist_ok=True)

    MODES = ["FRONTAL", "LEFT_PROFILE", "RIGHT_PROFILE"]

    for mode in MODES:
        mesh_path = os.path.join(mesh_dir, f"{mode}.json")
        if not os.path.exists(mesh_path):
            print(f"⚠️ Missing mesh: {mesh_path}")
            continue

        mesh_3d, transform = load_golden_mesh(mesh_path)
        canonical_3d_metric, canonical_2d = project_to_canonical(
            mesh_3d, transform
        )

        out_path = os.path.join(out_dir, f"{mode}.json")
        save_canonical(canonical_3d_metric, canonical_2d, out_path)


# -------------------------------------------------
# CLI ENTRY POINT
# -------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m canonical.projection <SESSION_DIR>")
        sys.exit(1)

    session_dir = sys.argv[1]
    print(f"🧭 Running canonical projection for session: {session_dir}")
    run_canonical_for_session(session_dir)

