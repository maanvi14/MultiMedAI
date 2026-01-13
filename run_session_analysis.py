"""
run_session_analysis.py
=======================

MASTER ANALYSIS ORCHESTRATOR
---------------------------
Runs the complete MultiMedAI session pipeline in a fixed, safe order:

1. Canonical projection (pose & camera normalization)
2. ROI extraction (eyes, nose, cheeks, lips, forehead, teeth)
3. Feature building (geometry-based, explainable)
4. Teeth ML Analysis (size + regularity)  ✅ NEW
5. Prakriti mapping (rule-based, doctor-assistive)
"""

import os
import sys

# -------------------------------------------------
# Imports
# -------------------------------------------------

from canonical.projection import run_canonical_for_session

from canonical.face_structure import extract_face_structure
from canonical.roi.eyes import extract_eye_rois
from canonical.roi.nose import extract_nose_rois
from canonical.roi.cheeks import extract_cheek_rois
from canonical.roi.lips import extract_lips_roi
from canonical.roi.forehead import extract_forehead_roi

# ✅ NEW: TEETH ROI + ML inference
from canonical.roi.teeth import extract_teeth_roi
from feature_extractors.teeth_features import extract_teeth_features

from build_features import build_features
from prakriti_mapping.run_prakriti import run_prakriti_analysis


# -------------------------------------------------
# Validation
# -------------------------------------------------

def validate_session(session_dir: str):
    """
    Ensures required inputs exist before analysis starts.
    Prevents silent or partial failures.
    """
    required = [
        "images/FRONTAL_RAW.jpg",
        "meshes/FRONTAL.json",
    ]

    for r in required:
        path = os.path.join(session_dir, r)
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Missing required file: {r}")

    print("✅ Session structure validated")


# -------------------------------------------------
# Orchestrator
# -------------------------------------------------

def run_analysis(
    session_dir: str,
    run_face_structure: bool = True,
    run_eyes: bool = True,
    run_nose: bool = True,
    run_cheeks: bool = True,
    run_lips: bool = True,
    run_forehead: bool = True,
    run_chin: bool = False,
    run_teeth: bool = True,       # ✅ NEW
    run_teeth_ml: bool = True,    # ✅ NEW
):
    print("\n==============================")
    print("🔬 Starting Session Analysis")
    print("==============================")
    print(f"📁 Session: {session_dir}\n")

    # Validate session structure
    validate_session(session_dir)

    # -------------------------------------------------
    # 1️⃣ Canonical Projection (MANDATORY)
    # -------------------------------------------------
    print("▶ Canonical projection...")
    run_canonical_for_session(session_dir)

    # -------------------------------------------------
    # 2️⃣ Structure & ROI Extraction
    # -------------------------------------------------
    if run_face_structure:
        print("▶ Face structure...")
        extract_face_structure(session_dir)

    if run_eyes:
        print("▶ Eyes...")
        extract_eye_rois(session_dir)

    if run_nose:
        print("▶ Nose...")
        extract_nose_rois(session_dir)

    if run_cheeks:
        print("▶ Cheeks...")
        extract_cheek_rois(session_dir)

    if run_lips:
        print("▶ Lips...")
        extract_lips_roi(session_dir)

    if run_forehead:
        print("▶ Forehead...")
        extract_forehead_roi(session_dir)

    if run_chin:
        print("▶ Chin...")
        try:
            from canonical.roi.chin import extract_chin_roi
            extract_chin_roi(session_dir)
        except Exception as e:
            print(f"⚠ Chin skipped: {e}")

    # ✅ NEW: TEETH ROI
    if run_teeth:
        print("▶ Teeth ROI...")
        extract_teeth_roi(session_dir)

    # -------------------------------------------------
    # 3️⃣ Feature Building
    # -------------------------------------------------
    print("▶ Building features...")
    build_features(session_dir)

    # ✅ NEW: TEETH ML ANALYSIS (post ROI + post build_features is also fine)
    if run_teeth_ml:
        print("▶ Teeth ML analysis...")
        extract_teeth_features(session_dir)

    # -------------------------------------------------
    # 4️⃣ Prakriti Mapping
    # -------------------------------------------------
    print("▶ Prakriti...")
    run_prakriti_analysis(session_dir)

    print("\n==============================")
    print("✅ Analysis Complete")
    print("==============================\n")


# -------------------------------------------------
# CLI ENTRY POINT
# -------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run_session_analysis.py <SESSION_DIR>")
        sys.exit(1)

    SESSION_DIR = sys.argv[1]

    if not os.path.isdir(SESSION_DIR):
        raise RuntimeError(f"Invalid session directory: {SESSION_DIR}")

    run_analysis(
        session_dir=SESSION_DIR,
        run_face_structure=True,
        run_eyes=True,
        run_nose=True,
        run_cheeks=True,
        run_lips=True,
        run_forehead=True,
        run_chin=False,
        run_teeth=True,
        run_teeth_ml=True
    )

    print("[DONE]")
