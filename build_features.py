"""
build_features.py
=================
MASTER FEATURE BUILDER
"""

import os
import json
import sys
from datetime import datetime, UTC

# -------------------------------------------------
# Import feature extractors
# -------------------------------------------------

from feature_extractors.face_structure_features import (
    compute_face_structure_features
)
from feature_extractors.eye_features import(
    extract_eye_features
    
)
from feature_extractors.teeth_features import(
    extract_teeth_features
)


# -------------------------------------------------
# Validation
# -------------------------------------------------

def validate_session(session_dir: str):
    required = [
        "meshes/FRONTAL.json"
    ]

    for r in required:
        path = os.path.join(session_dir, r)
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Missing required file: {r}")

    print("✅ Session validated for feature extraction")


# -------------------------------------------------
# Feature Builder
# -------------------------------------------------

def build_features(session_dir: str):

    print("\n==============================")
    print("📊 Building Features")
    print("==============================")
    print(f"📁 Session: {session_dir}\n")

    validate_session(session_dir)

    features = {
        "session_id": os.path.basename(session_dir),
        "generated_at": datetime.now(UTC).isoformat(),
        "features": {}
    }

    # ---------------- FACE STRUCTURE ----------------
    print("▶ Face structure features...")

    face_structure = compute_face_structure_features(session_dir)
    features["features"]["face_structure"] = face_structure

    eye_features = extract_eye_features(session_dir)
    features["features"]["eyes"] = eye_features

    print("▶ Teeth features...")
    teeth_features = extract_teeth_features(session_dir)
    features["features"]["teeth"] = teeth_features


    # ---------------- FUTURE MODULES ----------------
    # Eyes, cheeks, nose intentionally disabled for now

    # -------------------------------------------------
    # Save features.json
    # -------------------------------------------------
    out_path = os.path.join(session_dir, "analysis", "features.json")
    with open(out_path, "w") as f:
        json.dump(features, f, indent=2)

    print("\n==============================")
    print("✅ Features built successfully")
    print(f"📄 Saved to: {out_path}")
    print("==============================\n")


# -------------------------------------------------
# CLI Entry
# -------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python build_features.py <SESSION_DIR>")
        sys.exit(1)

    SESSION_DIR = sys.argv[1]
    build_features(SESSION_DIR)
