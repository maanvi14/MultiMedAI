"""
Final Prakriti Runner
====================
Combines face structure (PRIMARY) + eye EAR (SUPPORTIVE)
Produces explainable, doctor-assistive Prakriti output
"""

import os
import json

from prakriti_mapping.face_structure_mapping import map_face_structure_to_prakriti
from prakriti_mapping.eye_mapping import map_eye_to_prakriti


# ----------------------------
# Weights (explicit & tunable)
# ----------------------------
EYE_SUPPORT_WEIGHT = 0.3   # supportive only, never dominant


def run_prakriti_analysis(session_dir: str):
    features_path = os.path.join(session_dir, "analysis", "features.json")

    if not os.path.exists(features_path):
        raise FileNotFoundError("features.json not found. Run build_features first.")

    with open(features_path, "r") as f:
        data = json.load(f)

    # -------------------------------------------------
    # Load features safely
    # -------------------------------------------------
    features = data.get("features", {})

    face_features = features.get("face_structure")
    eye_features  = features.get("eyes")   # plural by design

    if not face_features:
        raise ValueError(
            "Face structure features missing. "
            "Prakriti analysis requires face structure."
        )

    # -------------------------------------------------
    # 1️⃣ Primary mapping: Face Structure
    # -------------------------------------------------
    face_result = map_face_structure_to_prakriti(face_features)

    final_scores = face_result["prakriti_scores"].copy()
    explanation  = face_result["explanation"].copy()

    # -------------------------------------------------
    # 2️⃣ Supportive mapping: Eyes (EAR-based)
    # -------------------------------------------------
    if eye_features:
        eye_result = map_eye_to_prakriti(eye_features)

        if eye_result and "prakriti_scores" in eye_result:
            for dosha, val in eye_result["prakriti_scores"].items():
                final_scores[dosha] += val * EYE_SUPPORT_WEIGHT

        if eye_result and "explanation" in eye_result:
            # single source of truth for eye explanation
            explanation["eyes"] = eye_result["explanation"]

    # -------------------------------------------------
    # 3️⃣ Normalize final scores
    # -------------------------------------------------
    total = sum(final_scores.values()) + 1e-6
    for k in final_scores:
        final_scores[k] = round(final_scores[k] / total, 3)

    # -------------------------------------------------
    # 4️⃣ Confidence estimation (agreement strength)
    # -------------------------------------------------
    sorted_vals = sorted(final_scores.values(), reverse=True)
    confidence = sorted_vals[0] - sorted_vals[1]

    # Clamp confidence for interpretability
    confidence = round(max(0.0, min(confidence, 1.0)), 3)

    # -------------------------------------------------
    # Final output
    # -------------------------------------------------
    result = {
        "prakriti_scores": final_scores,
        "dominant_prakriti": max(final_scores, key=final_scores.get).upper(),
        "confidence": confidence,
        "explanation": explanation,
        "note": (
            "Face structure is the primary constitutional signal. "
            "Eye EAR contributes supportive soft-tissue evidence only. "
            "Results are assistive and must be interpreted by a clinician."
        ),
    }

    # -------------------------------------------------
    # Save output
    # -------------------------------------------------
    out_dir = os.path.join(session_dir, "analysis", "prakriti")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "final_prakriti.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("🧘 Prakriti Analysis Complete")
    print(f"Dominant Prakriti → {result['dominant_prakriti']}")
    print(f"Confidence → {confidence}")

