def map_face_structure_to_prakriti(f):
    scores = {"vata": 0.0, "pitta": 0.0, "kapha": 0.0}
    explanation = {}

    # 1️⃣ Facial Index
    fi = f["facial_index_ratio"]
    if fi < 1.05:
        scores["kapha"] += 0.3
        explanation["facial_index"] = "Low facial index → broad face → Kapha"
    elif fi < 1.20:
        scores["pitta"] += 0.25
        explanation["facial_index"] = "Medium facial index → balanced face → Pitta"
    else:
        scores["vata"] += 0.35
        explanation["facial_index"] = "High facial index → elongated face → Vata"

    # 2️⃣ Jaw Roundness (PRIMARY)
    jr = f["jaw_roundness"]
    if jr > 0.65:
        scores["kapha"] += 0.4
        explanation["jaw_roundness"] = "Rounded jaw → Kapha dominance"
    elif jr > 0.5:
        scores["pitta"] += 0.25
        explanation["jaw_roundness"] = "Moderately defined jaw → Pitta"
    else:
        scores["vata"] += 0.3
        explanation["jaw_roundness"] = "Sharp jaw → Vata"

    # 3️⃣ Chin Taper
    ct = f["chin_taper_ratio"]
    if ct < 0.25:
        scores["kapha"] += 0.35
        explanation["chin_taper"] = "Blunt chin → Kapha"
    elif ct < 0.4:
        scores["pitta"] += 0.25
        explanation["chin_taper"] = "Moderate chin taper → Pitta"
    else:
        scores["vata"] += 0.35
        explanation["chin_taper"] = "Pointed chin → Vata"

    # 4️⃣ Symmetry (Modifier)
    sym = f["symmetry_score"]
    if sym > 0.65:
        scores["kapha"] += 0.2
        explanation["symmetry"] = "High facial symmetry → stable Kapha traits"
    elif sym > 0.45:
        scores["pitta"] += 0.15
        explanation["symmetry"] = "Moderate symmetry → Pitta"
    else:
        scores["vata"] += 0.2
        explanation["symmetry"] = "Low symmetry → Vata instability"

    # Normalize
    total = sum(scores.values()) + 1e-6
    for k in scores:
        scores[k] = round(scores[k] / total, 3)

    dominant = max(scores, key=scores.get).upper()

    return {
        "prakriti_scores": scores,
        "dominant_prakriti": dominant,
        "explanation": explanation
    }