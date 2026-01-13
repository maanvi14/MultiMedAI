"""
teeth_mapping.py
================

Doctor-assistive interpretation for Teeth features.

Input:
- analysis/teeth_analysis.json  (produced by feature_extractors/teeth_features.py)

Output:
- A safe, explainable "supportive_observations" list
- Teeth summary fields that can be merged into final Prakriti report

⚠️ NOTE (AYUSH-safe):
This module does NOT diagnose Prakriti.
It only provides supportive observations based on captured teeth ROI + ML labels.
"""

import os
import json


# -------------------------------------------------
# Loader
# -------------------------------------------------

def load_teeth_analysis(session_dir: str):
    """
    Loads teeth analysis JSON saved by teeth_features.py
    Returns dict like:
    {
      "smile": {...},
      "open": {...}
    }
    """
    path = os.path.join(session_dir, "analysis", "teeth_analysis.json")
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        data = json.load(f)

    return data.get("teeth", None)


# -------------------------------------------------
# Safe Interpretation
# -------------------------------------------------

def interpret_teeth_for_prakriti(teeth: dict):
    """
    Returns supportive observations (safe + explainable).
    """
    if not teeth:
        return []

    notes = []

    # prefer smile mode for stable tooth appearance
    smile = teeth.get("smile", {})
    open_ = teeth.get("open", {})

    smile_size = (smile.get("size") or {}).get("label", None)
    smile_reg = (smile.get("regularity") or {}).get("label", None)

    open_size = (open_.get("size") or {}).get("label", None)
    open_reg = (open_.get("regularity") or {}).get("label", None)

    # ----------------------------
    # Size interpretation (soft)
    # ----------------------------
    if smile_size:
        if smile_size == "Small":
            notes.append("Teeth size appears smaller (supportive trait; may align with Vata-type variability).")
        elif smile_size == "Large":
            notes.append("Teeth size appears larger (supportive trait; may align with Kapha-type robustness).")
        else:
            notes.append("Teeth size appears medium (neutral supportive trait).")

    # ----------------------------
    # Regularity interpretation (soft)
    # ----------------------------
    if smile_reg:
        if smile_reg == "Irregular":
            notes.append("Teeth regularity appears irregular (supportive trait; may align with Vata-type roughness/variation).")
        else:
            notes.append("Teeth regularity appears regular (neutral supportive trait).")

    # ----------------------------
    # Cross-check with OPEN capture
    # ----------------------------
    if open_reg and smile_reg and open_reg != smile_reg:
        notes.append("Smile vs Open regularity differs → ensure ROI quality / capture stability.")

    if open_size and smile_size and open_size != smile_size:
        notes.append("Smile vs Open size differs → ensure consistent lighting and mouth ROI crop.")

    return notes


# -------------------------------------------------
# Public API for prakriti mapping
# -------------------------------------------------

def build_teeth_section(session_dir: str):
    """
    Returns a final structured dict to merge into report JSON.
    """
    teeth = load_teeth_analysis(session_dir)

    if not teeth:
        return {
            "status": "not_available",
            "message": "teeth_analysis.json not found",
            "supportive_observations": []
        }

    return {
        "status": "available",
        "teeth": teeth,
        "supportive_observations": interpret_teeth_for_prakriti(teeth)
    }
