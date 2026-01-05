"""
stability_gatekeeper.py

LAYER 3B: Stability Gatekeeper (Temporal Trust)
----------------------------------------------
Purpose:
- Ensure pose + quality are stable over time
- Decide when to freeze ONE trusted 3D face mesh

Inputs (per frame):
- pose_ok (bool)
- quality_ok (bool)
- landmarks_3d (np.array shape: [468, 3])

Output:
- ready_to_capture (bool)
"""

import numpy as np
from collections import deque

class StabilityGatekeeper:
    def __init__(
        self,
        required_frames=60,        # ~2 seconds @ 30 FPS
        jitter_threshold=0.002     # normalized landmark std-dev
    ):
        self.required_frames = required_frames
        self.jitter_threshold = jitter_threshold

        self.landmark_buffer = deque(maxlen=required_frames)
        self.valid_frame_count = 0
        self.ready = False

    # -------------------------------------------------
    # Reset logic (hard reset)
    # -------------------------------------------------
    def reset(self):
        self.landmark_buffer.clear()
        self.valid_frame_count = 0
        self.ready = False

    # -------------------------------------------------
    # Main update per frame
    # -------------------------------------------------
    def update(self, pose_ok, quality_ok, landmarks_3d):
        """
        Call this once per frame.

        Returns:
            ready_to_capture (bool)
        """

        # Gate 1: Pose & Quality must pass
        if not pose_ok or not quality_ok:
            self.reset()
            return False

        # Gate 2: Accumulate landmarks
        self.landmark_buffer.append(landmarks_3d)
        self.valid_frame_count += 1

        # Gate 3: Need enough frames
        if self.valid_frame_count < self.required_frames:
            return False

        # Gate 4: Jitter check (mathematical stability)
        stacked = np.stack(self.landmark_buffer, axis=0)  # [N, 468, 3]
        jitter = np.std(stacked, axis=0).mean()

        if jitter > self.jitter_threshold:
            self.reset()
            return False

        # All gates passed
        self.ready = True
        return True
