"""
Final Prakriti Runner (EQUAL WEIGHT)
===================================
Combines:
✅ Face Structure
✅ Eyes (EAR based)
✅ Teeth

All features contribute with SAME WEIGHT (equal ensemble).

Produces explainable, doctor-assistive Prakriti output.
"""

import os
import json
import sys

from prakriti_mapping.face_structure_mapping import map_face_structure_to_prakriti
from prakriti_mapping.eye_mapping import map_eye_to_prakriti
from prakriti_mapping.teeth_mapping import build_teeth_section


def _init_scores():
    return {"VATA": 0.0, "PITTA": 0.0, "KAPHA": 0.0}


def _safe_normalize(scores: dict):
    total = sum(scores.values()) + 1e-6
    for k in scores:
        scores[k] = round(scores[k] / total, 3)
    return scores


def run_prakriti_analysis(session_dir: str):
    features_path = os.path.join(session_dir, "analysis", "features.json")

    if not os.path.exists(features_path):
        raise FileNotFoundError("❌ features.json not found. Run build_features first.")

    with open(features_path, "r") as f:
        data = json.load(f)

    # -------------------------------------------------
    # Load Features
    # -------------------------------------------------
    features = data.get("features", {})
    face_features = features.get("face_structure")
    eye_features = features.get("eyes")  # plural by design

    if not face_features:
        raise ValueError("❌ face_structure missing in features.json")

    # -------------------------------------------------
    # Collect module outputs
    # -------------------------------------------------
    modules = []

    # 1) Face
    face_result = map_face_structure_to_prakriti(face_features)
    if face_result and "prakriti_scores" in face_result:
        modules.append(("face_structure", face_result))

    # 2) Eyes (optional)
    if eye_features:
        eye_result = map_eye_to_prakriti(eye_features)
        if eye_result and "prakriti_scores" in eye_result:
            modules.append(("eyes", eye_result))

    # 3) Teeth (optional)
    teeth_result = build_teeth_section(session_dir)
    if teeth_result and isinstance(teeth_result, dict) and "prakriti_scores" in teeth_result:
        modules.append(("teeth", teeth_result))

    if len(modules) == 0:
        raise RuntimeError("❌ No valid mapping modules produced prakriti_scores.")

    # -------------------------------------------------
    # Equal-weight fusion
    # -------------------------------------------------
    final_scores = _init_scores()
    explanation = {}

    module_count = len(modules)

    for name, mod in modules:
        # Merge scores equally
        mod_scores = mod.get("prakriti_scores", {})
        for dosha in final_scores.keys():
            final_scores[dosha] += float(mod_scores.get(dosha, 0.0)) / module_count

        # Store explanation per module
        if "explanation" in mod:
            explanation[name] = mod["explanation"]

    # Normalize final scores
    final_scores = _safe_normalize(final_scores)

    # -------------------------------------------------
    # Confidence = top1 - top2
    # -------------------------------------------------
    sorted_vals = sorted(final_scores.values(), reverse=True)
    confidence = round(max(0.0, min(sorted_vals[0] - sorted_vals[1], 1.0)), 3)

    dominant = max(final_scores, key=final_scores.get)

    # -------------------------------------------------
    # Final Output
    # -------------------------------------------------
    result = {
        "prakriti_scores": final_scores,
        "dominant_prakriti": dominant,
        "confidence": confidence,
        "modules_used": [name for name, _ in modules],
        "explanation": explanation,
    }

    # -------------------------------------------------
    # Save output
    # -------------------------------------------------
    out_dir = os.path.join(session_dir, "analysis", "prakriti")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "final_prakriti.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("🧘 Prakriti Analysis Complete ✅")
    print(f"Dominant Prakriti → {dominant}")
    print(f"Confidence → {confidence}")
    print(f"Modules Used → {result['modules_used']}")
    print(f"Saved → {out_path}")

    return result


# -------------------------------------------------
# CLI ENTRY
# -------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m prakriti_mapping.run_prakriti <SESSION_DIR>")
        sys.exit(1)

    run_prakriti_analysis(sys.argv[1])
