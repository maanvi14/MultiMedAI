"""
pose_gatekeeper.py

LAYER 2: Pose Gatekeeper (MediaPipe-native)
------------------------------------------
Uses MediaPipe facial transformation matrix
to estimate head orientation reliably.
"""

import numpy as np

def estimate_head_pose_from_matrix(transform_matrix):
    """
    Extract yaw, pitch, roll from MediaPipe face transformation matrix.
    """
    R = np.array(transform_matrix).reshape(4, 4)[:3, :3]

    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw   = np.arctan2(-R[2, 0], sy)
        roll  = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw   = np.arctan2(-R[2, 0], sy)
        roll  = 0

    return {
        "yaw":   np.degrees(yaw),
        "pitch": np.degrees(pitch),
        "roll":  np.degrees(roll)
    }


def is_pose_valid(pose, yaw_thresh=12, pitch_thresh=12, roll_thresh=8):
    """
    Strict but correct thresholds (now meaningful).
    """
    return (
        abs(pose["yaw"])   <= yaw_thresh and
        abs(pose["pitch"]) <= pitch_thresh and
        abs(pose["roll"])  <= roll_thresh
    )
