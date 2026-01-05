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

    Returns a dict:{"valid": bool, "reason": str, "metrics": {...}}
    """
    yaw = float(pose.get("yaw", 0))
    pitch = float(pose.get("pitch", 0))
    roll = float(pose.get("roll", 0))

    # Configurable thresholds (tuned for robustness in lower-quality cameras)
    FRONTAL_YAW_MAX = 12.0
    FRONTAL_PITCH_MAX = 12.0
    FRONTAL_ROLL_MAX = 10.0

    PROFILE_YAW_MIN = 28.0
    PROFILE_YAW_MAX = 70.0
    PROFILE_PITCH_MAX = 18.0
    PROFILE_ROLL_MAX = 12.0

    metrics = {"yaw": yaw, "pitch": pitch, "roll": roll}

    # FRONTAL
    if mode == "FRONTAL":
        if abs(yaw) > FRONTAL_YAW_MAX:
            return {"valid": False, "reason": "Please face forward (center your face)", "metrics": metrics}
        if abs(pitch) > FRONTAL_PITCH_MAX:
            return {"valid": False, "reason": "Adjust head pitch", "metrics": metrics}
        if abs(roll) > FRONTAL_ROLL_MAX:
            return {"valid": False, "reason": "Reduce head tilt", "metrics": metrics}
        return {"valid": True, "reason": "Pose OK", "metrics": metrics}

    # LEFT PROFILE: yaw should be sufficiently negative
    if mode == "LEFT_PROFILE":
        if not ( -PROFILE_YAW_MAX <= yaw <= -PROFILE_YAW_MIN ):
            return {"valid": False, "reason": "Turn face LEFT more", "metrics": metrics}
        if abs(pitch) > PROFILE_PITCH_MAX:
            return {"valid": False, "reason": "Adjust head pitch", "metrics": metrics}
        if abs(roll) > PROFILE_ROLL_MAX:
            return {"valid": False, "reason": "Reduce head tilt", "metrics": metrics}
        return {"valid": True, "reason": "Pose OK", "metrics": metrics}

    # RIGHT PROFILE: yaw should be sufficiently positive
    if mode == "RIGHT_PROFILE":
        if not ( PROFILE_YAW_MIN <= yaw <= PROFILE_YAW_MAX ):
            return {"valid": False, "reason": "Turn face RIGHT more", "metrics": metrics}
        if abs(pitch) > PROFILE_PITCH_MAX:
            return {"valid": False, "reason": "Adjust head pitch", "metrics": metrics}
        if abs(roll) > PROFILE_ROLL_MAX:
            return {"valid": False, "reason": "Reduce head tilt", "metrics": metrics}
        return {"valid": True, "reason": "Pose OK", "metrics": metrics}

    return {"valid": False, "reason": "Unknown capture mode", "metrics": metrics}

