import os
import json
from prakriti_mapping.face_structure_mapping import map_face_structure_to_prakriti

def run_prakriti_analysis(session_dir):
    features_path = os.path.join(session_dir, "analysis", "features.json")

    if not os.path.exists(features_path):
        raise FileNotFoundError("features.json not found. Run build_features first.")

    with open(features_path, "r") as f:
        data = json.load(f)

    face_features = data["features"]["face_structure"]

    # ✅ UPDATED: mapping returns full explainable result
    result = map_face_structure_to_prakriti(face_features)

    out_dir = os.path.join(session_dir, "analysis", "prakriti")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "final_prakriti.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"🧘 Dominant Prakriti: {result['dominant_prakriti']}")