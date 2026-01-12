def map_eye_to_prakriti(eye_features):
    """
    Eye → Prakriti Mapping (Supportive, Calibrated)
    -----------------------------------------------
    ✔ Uses IOD-normalized EAR
    ✔ Soft-tissue supportive signal only
    ✔ Cannot flip dominance alone
    ✔ Explainable & clinician-safe
    """

    scores = {"vata": 0.0, "pitta": 0.0, "kapha": 0.0}
    explanation = {}

    # ----------------------------
    # Defensive read
    # ----------------------------
    ear = eye_features.get("avg_EAR", None)
    if ear is None:
        return {
            "prakriti_scores": scores,
            "explanation": {"eyes": "EAR not available"}
        }

    ear = float(ear)

    # ----------------------------
    # Calibrated thresholds
    # (post IOD-normalization)
    # ----------------------------
    if ear < 0.24:
        scores["vata"] += 0.10
        explanation["eyes"] = (
            f"Low eye openness (EAR={ear:.3f}) → Vata tendency (supportive)"
        )

    elif ear < 0.29:
        scores["pitta"] += 0.07
        explanation["eyes"] = (
            f"Moderate eye openness (EAR={ear:.3f}) → Pitta tendency (supportive)"
        )

    else:
        scores["kapha"] += 0.10
        explanation["eyes"] = (
            f"Wide, soft eye openness (EAR={ear:.3f}) → Kapha tendency (supportive)"
        )

    return {
        "prakriti_scores": scores,
        "explanation": explanation
    }
