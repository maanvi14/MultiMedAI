"""
Face Structure → Prakriti Mapping (FINAL, CALIBRATED)
====================================================
✔ Calibrated to IOD-normalized canonical 3D features
✔ Structural traits dominate (Ayurvedic priority)
✔ Kapha breadth & stability preserved
✔ Explainable, rule-based, clinician-safe
"""

# -------------------------------------------------
# Thresholds (derived from canonical 3D anatomy)
# -------------------------------------------------
# These ranges assume:
# - facial_index_ratio ≈ 1.15–1.70
# - jaw_roundness     ≈ 0.25–0.65
# - chin_shape_ratio  ≈ 0.20–0.60
# - symmetry_score    ≈ 0.60–0.95


def map_face_structure_to_prakriti(face_features: dict):

    scores = {"vata": 0.0, "pitta": 0.0, "kapha": 0.0}
    explanation = {}
    
    fl = face_features.get("face_length_norm")
    fw = face_features.get("face_width_norm")

    # ==================================================
    # Facial dominance (Ayurvedic correct)
    # ==================================================
    if fl is not None and fw is not None:

        if fl >= 1.65 and fw <= 1.30:
            scores["vata"] += 0.45
            explanation["face"] = "Long, narrow face → Vata tendency"

        elif fl <= 1.50 and fw >= 1.45:
            scores["kapha"] += 0.45
            explanation["face"] = "Broad, wide face → Kapha tendency"

        else:
            scores["pitta"] += 0.30
            explanation["face"] = "Balanced face proportions → Pitta tendency"


    # ==================================================
    # 2️⃣ Jaw Roundness (CORRECTED MAPPING)
    # ==================================================
    jr = face_features.get("jaw_roundness")

    if jr is not None:
        # LOW ratio = Wide Width + Low Depth = Kapha (Broad/Flat)
        if jr <= 0.35: 
            scores["kapha"] += 0.30
            explanation["jaw_roundness"] = "Broad, non-projecting jaw → Kapha tendency"
        
        # HIGH ratio = Narrow Width + High Depth = Vata (Angular/Protruding)
        elif jr >= 0.50: 
            scores["vata"] += 0.25
            explanation["jaw_roundness"] = "Narrow, angular jaw → Vata tendency"
            
        # Middle range = Pitta
        else:
            scores["pitta"] += 0.20
            explanation["jaw_roundness"] = "Defined, balanced jaw → Pitta tendency"

    # ==================================================
    # 3️⃣ Chin Shape Ratio (SHARPNESS vs ROUNDEDNESS)
    # ==================================================
    cr = face_features.get("chin_shape_ratio")

    if cr is not None:
        if cr >= 0.45:
            scores["vata"] += 0.25
            explanation["chin_shape"] = "Pointed, projecting chin → Vata tendency"
        elif cr >= 0.30:
            scores["pitta"] += 0.15
            explanation["chin_shape"] = "Moderately defined chin → Pitta tendency"
        else:
            scores["kapha"] += 0.25
            explanation["chin_shape"] = "Rounded, blunt chin → Kapha tendency"

    # ==================================================
    # 4️⃣ Facial Symmetry (STABILITY INDICATOR)
    # ==================================================
    sym = face_features.get("symmetry_score")

    if sym is not None:
        if sym >= 0.80:
            scores["kapha"] += 0.20
            explanation["symmetry"] = "High facial symmetry → Kapha stability"
        elif sym >= 0.65:
            scores["pitta"] += 0.15
            explanation["symmetry"] = "Moderate facial symmetry → Pitta balance"
        else:
            scores["vata"] += 0.20
            explanation["symmetry"] = "Irregular facial symmetry → Vata variability"

    # ==================================================
    # Output (STRUCTURAL, PRIMARY)
    # ==================================================
    return {
        "prakriti_scores": scores,
        "explanation": explanation
    }
