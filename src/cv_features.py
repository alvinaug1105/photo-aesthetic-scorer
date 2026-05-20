"""
src/cv_features.py — Computer Vision Feature Extraction v2.0
Multi-intent composition scoring: Rule of Thirds + Symmetry + Negative Space
"""

import logging
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MAX_DIM = 800  # resize before CV analysis (speed)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_cv(img_path: str, max_dim: int = MAX_DIM):
    """Load image as RGB numpy array, resize for speed."""
    img = Image.open(img_path).convert("RGB")
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    arr = np.array(img)
    return arr


def _to_gray(arr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def _normalise(val: float, lo: float, hi: float) -> float:
    """Clamp + normalise to 0–10."""
    return round(float(np.clip((val - lo) / (hi - lo + 1e-9), 0.0, 1.0) * 10), 2)


# ── Individual feature functions ──────────────────────────────────────────────

def compute_sharpness(gray: np.ndarray) -> float:
    """Laplacian variance — higher = sharper."""
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return _normalise(lap_var, 0, 1500)


def compute_brightness(arr: np.ndarray) -> float:
    """Mean V-channel in HSV."""
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    mean_v = float(hsv[:, :, 2].mean())
    return _normalise(mean_v, 20, 235)


def compute_contrast(gray: np.ndarray) -> float:
    """Standard deviation of grayscale — higher = more contrast."""
    return _normalise(float(gray.std()), 5, 80)


def compute_color(arr: np.ndarray) -> float:
    """Mean saturation in HSV."""
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    mean_s = float(hsv[:, :, 1].mean())
    return _normalise(mean_s, 10, 200)


def compute_noise(gray: np.ndarray) -> float:
    """
    Noise score — lower high-freq variance = less noise = higher score.
    Uses difference between original and Gaussian-blurred image.
    """
    blurred   = cv2.GaussianBlur(gray, (5, 5), 0)
    diff      = cv2.absdiff(gray, blurred).astype(np.float32)
    noise_lvl = float(diff.mean())
    # Invert: low noise → high score
    return _normalise(30 - noise_lvl, 0, 30)


def compute_composition(arr: np.ndarray) -> float:
    """
    Multi-intent composition scoring v2.0
    Detects THREE valid composition styles and returns the best match:

    1. Rule of Thirds  — subject near 1/3 grid lines
    2. Symmetry        — left-right or top-bottom mirror
    3. Negative Space  — large clean area = intentional minimalism

    Final score = weighted best intent (not forcing ONE style).
    """
    gray = _to_gray(arr)
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150)

    # ── 1. Rule of Thirds ────────────────────────────────
    tx = [w // 3, 2 * w // 3]
    ty = [h // 3, 2 * h // 3]
    margin = max(15, w // 40)

    def _band_mean(axis: str, pos: int, m: int) -> float:
        if axis == "x":
            region = edges[:, max(0, pos - m):min(w, pos + m)]
        else:
            region = edges[max(0, pos - m):min(h, pos + m), :]
        return float(region.mean()) if region.size > 0 else 0.0

    rot_density = np.mean([
        _band_mean("x", tx[0], margin),
        _band_mean("x", tx[1], margin),
        _band_mean("y", ty[0], margin),
        _band_mean("y", ty[1], margin),
    ])
    rot_score = float(np.clip(rot_density / 35.0, 0.0, 1.0))

    # ── 2. Symmetry (left-right & top-bottom) ────────────
    # Left-right
    left  = gray[:, :w // 2].astype(float)
    right = cv2.flip(gray[:, w // 2:w // 2 * 2], 1).astype(float)
    lr_diff = np.abs(left - right).mean() / 255.0
    lr_sym  = 1.0 - lr_diff

    # Top-bottom
    top    = gray[:h // 2, :].astype(float)
    bottom = cv2.flip(gray[h // 2:h // 2 * 2, :], 0).astype(float)
    tb_diff = np.abs(top - bottom).mean() / 255.0
    tb_sym  = 1.0 - tb_diff

    sym_score = float(max(lr_sym, tb_sym))

    # ── 3. Negative Space ────────────────────────────────
    # Large smooth area with low edge density = intentional empty space
    _, edge_thresh = cv2.threshold(edges, 25, 255, cv2.THRESH_BINARY)
    edge_ratio     = float((edge_thresh > 0).mean())   # 0 = all empty
    neg_space_score = float(np.clip(1.0 - edge_ratio * 3.5, 0.0, 1.0))

    # ── 4. Centre Composition ────────────────────────────
    # For minimalist / portrait shots: subject centred can be strong
    cx, cy    = w // 2, h // 2
    cx_margin = w // 6
    cy_margin = h // 6
    centre_region = edges[
        max(0, cy - cy_margin): min(h, cy + cy_margin),
        max(0, cx - cx_margin): min(w, cx + cx_margin),
    ]
    centre_density = float(centre_region.mean()) if centre_region.size > 0 else 0.0
    centre_score   = float(np.clip(centre_density / 40.0, 0.0, 1.0))

    # ── Final: best-intent scoring ────────────────────────
    # Photographer uses ONE dominant style; reward the best match
    best = max(
        rot_score    * 1.00,   # RoT — classic & rewarded most
        sym_score    * 0.92,   # Symmetry — strong but slightly less common
        neg_space_score * 0.85, # Negative space — intentional minimalism
        centre_score * 0.80,   # Centre — valid for portraits
    )

    # Blend with a mild overall-edge floor to avoid 0 on busy shots
    edge_floor   = float(np.clip(edge_ratio * 2.5, 0.0, 0.5))
    final_score  = best * 0.85 + edge_floor * 0.15

    return round(float(np.clip(final_score * 10, 0.0, 10.0)), 2)


# ── Composition style detector (for UI label) ─────────────────────────────────

def detect_composition_style(arr: np.ndarray) -> str:
    """
    Returns a human-readable composition style label.
    Used in app.py to show 'Style: Symmetry' etc.
    """
    gray  = _to_gray(arr)
    h, w  = gray.shape
    edges = cv2.Canny(gray, 50, 150)

    # RoT density
    tx     = [w // 3, 2 * w // 3]
    ty     = [h // 3, 2 * h // 3]
    m      = max(15, w // 40)
    rot_d  = np.mean([
        edges[:, max(0, tx[0]-m):tx[0]+m].mean(),
        edges[:, max(0, tx[1]-m):tx[1]+m].mean(),
        edges[max(0, ty[0]-m):ty[0]+m, :].mean(),
        edges[max(0, ty[1]-m):ty[1]+m, :].mean(),
    ])

    # Symmetry
    left   = gray[:, :w//2].astype(float)
    right  = cv2.flip(gray[:, w//2:w//2*2], 1).astype(float)
    lr_sym = 1.0 - np.abs(left - right).mean() / 255.0

    # Negative space
    _, et     = cv2.threshold(edges, 25, 255, cv2.THRESH_BINARY)
    neg_score = 1.0 - float((et > 0).mean()) * 3.5

    # Centre
    cx, cy = w // 2, h // 2
    cw, ch = w // 6, h // 6
    centre_d = edges[max(0,cy-ch):cy+ch, max(0,cx-cw):cx+cw].mean()

    scores = {
        "📐 Rule of Thirds": float(np.clip(rot_d / 35.0, 0, 1)),
        "🪞 Symmetry":       float(lr_sym),
        "⬜ Negative Space": float(np.clip(neg_score, 0, 1)),
        "🎯 Centre Focus":   float(np.clip(centre_d / 40.0, 0, 1)),
    }

    best_style = max(scores.keys(), key=lambda k: scores[k])
    return best_style


# ── Main public API ───────────────────────────────────────────────────────────

def extract_features(img_path: str) -> dict:
    """
    Extract all 6 CV dimensions + composition style label.

    Returns:
        {
            "sharpness":    float (0–10),
            "brightness":   float (0–10),
            "contrast":     float (0–10),
            "color":        float (0–10),
            "noise":        float (0–10),
            "composition":  float (0–10),
            "comp_style":   str   (e.g. "📐 Rule of Thirds"),
        }
    """
    try:
        arr  = _load_cv(img_path)
        gray = _to_gray(arr)

        feats = {
            "sharpness":   compute_sharpness(gray),
            "brightness":  compute_brightness(arr),
            "contrast":    compute_contrast(gray),
            "color":       compute_color(arr),
            "noise":       compute_noise(gray),
            "composition": compute_composition(arr),
            "comp_style":  detect_composition_style(arr),
        }
        logger.info("CV features: %s", feats)
        return feats

    except Exception as e:
        logger.warning("cv_features failed: %s", e)
        return {
            "sharpness":   5.0,
            "brightness":  5.0,
            "contrast":    5.0,
            "color":       5.0,
            "noise":       5.0,
            "composition": 5.0,
            "comp_style":  "Unknown",
        }