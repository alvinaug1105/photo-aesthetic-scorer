"""
src/explainer.py — Generate tips from CV feature scores v2.0
Handles non-numeric fields (e.g. comp_style) safely.
"""

THRESHOLDS = {
    "brightness":  (6.0, 8.0),
    "sharpness":   (6.0, 8.5),
    "composition": (5.5, 8.0),
    "color":       (6.0, 8.0),
    "noise":       (6.5, 9.0),
    "contrast":    (5.5, 8.0),
}

TIPS = {
    "brightness": {
        "low":  "⚠️ Underexposed — try +0.5 to +1.0 EV or lift Shadows in Lightroom",
        "high": "⚠️ Overexposed — reduce EV or pull down Highlights",
        "ok":   "✅ Brightness is well-balanced",
    },
    "sharpness": {
        "low":  "⚠️ Soft/blurry — check focus point, use faster shutter or tripod",
        "high": "✅ Excellent sharpness — subject is crisp and well focused",
        "ok":   "✅ Sharpness is acceptable",
    },
    "composition": {
        "low":  "⚠️ Composition could be stronger — try Rule of Thirds, symmetry, or deliberate negative space",
        "high": "✅ Strong composition detected",
        "ok":   "✅ Composition is decent",
    },
    "color": {
        "low":  "⚠️ Color looks flat — try Vibrance +15 or fix white balance",
        "high": "✅ Great color harmony and saturation",
        "ok":   "✅ Color balance is fine",
    },
    "noise": {
        "low":  "⚠️ High noise — shoot ISO ≤800 or apply Noise Reduction +30",
        "high": "✅ Very clean image — minimal noise",
        "ok":   "✅ Noise level is acceptable",
    },
    "contrast": {
        "low":  "⚠️ Low contrast — add an S-curve or Contrast +10 in post",
        "high": "✅ Good contrast range",
        "ok":   "✅ Contrast is balanced",
    },
}


def generate_tips(features: dict) -> list[str]:
    tips = []
    for feat, score in features.items():
        # ✅ Skip non-numeric values (e.g. comp_style string label)
        if not isinstance(score, (int, float)):
            continue
        # Skip unknown features not in TIPS
        if feat not in TIPS:
            continue
        lo, hi = THRESHOLDS.get(feat, (5.0, 8.0))
        if score < lo:
            tips.append(TIPS[feat]["low"])
        elif score >= hi:
            tips.append(TIPS[feat]["high"])
        else:
            tips.append(TIPS[feat]["ok"])
    return tips


def overall_verdict(nima_score: float, features: dict) -> tuple[str, str]:
    # ✅ Only average numeric values
    numeric_vals = [v for v in features.values() if isinstance(v, (int, float))]
    avg_cv   = sum(numeric_vals) / len(numeric_vals) if numeric_vals else 5.0
    combined = nima_score * 0.6 + avg_cv * 0.4
    if   combined >= 7.5: return "🌟 Portfolio-worthy shot!", "#00D2BE"
    elif combined >= 6.5: return "👍 Good photo with minor improvements", "#FFD700"
    elif combined >= 5.5: return "😐 Average — see suggestions below", "#FF8C00"
    else:                 return "📉 Needs improvement", "#E8002D"