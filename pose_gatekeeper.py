"""
pose_gatekeeper.py

LAYER 2: Pose Gatekeeper (MODE-AWARE)
------------------------------------
Validates head pose based on capture mode:
- FRONTAL
- LEFT_PROFILE
- RIGHT_PROFILE
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


def is_pose_valid(pose, mode="FRONTAL"):
    """
    Mode-aware pose validation.
    """

    yaw   = pose["yaw"]
    pitch = pose["pitch"]
    roll  = pose["roll"]

    # -------------------------------
    # FRONTAL CAPTURE
    # -------------------------------
    if mode == "FRONTAL":
        return (
            abs(yaw)   <= 12 and
            abs(pitch) <= 12 and
            abs(roll)  <= 8
        )

    # -------------------------------
    # LEFT PROFILE CAPTURE
    # -------------------------------
    elif mode == "LEFT_PROFILE":
        return (
            -60 <= yaw <= -30 and   # face turned LEFT
            abs(pitch) <= 15 and
            abs(roll)  <= 10
        )

    # -------------------------------
    # RIGHT PROFILE CAPTURE
    # -------------------------------
    elif mode == "RIGHT_PROFILE":
        return (
            30 <= yaw <= 60 and     # face turned RIGHT
            abs(pitch) <= 15 and
            abs(roll)  <= 10
        )

    # -------------------------------
    # UNKNOWN MODE (fail safe)
    # -------------------------------
    return False
