def map_eye_to_prakriti(eye_features):
    scores = {"vata": 0.0, "pitta": 0.0, "kapha": 0.0}
    explanation = {}

    ear = eye_features.get("avg_EAR", None)
    if ear is None:
        return {"prakriti_scores": scores, "explanation": {"eyes": "EAR not available"}}

    ear = float(ear)

    if ear < 0.55:
        scores["vata"] += 0.10
        explanation["eyes"] = f"Low eye openness (EAR={ear:.3f}) → Vata (supportive)"

    elif ear < 0.60:
        scores["pitta"] += 0.07
        explanation["eyes"] = f"Moderate eye openness (EAR={ear:.3f}) → Pitta (supportive)"

    else:
        scores["kapha"] += 0.10
        explanation["eyes"] = f"Wide/soft eye openness (EAR={ear:.3f}) → Kapha (supportive)"

    return {"prakriti_scores": scores, "explanation": explanation}
