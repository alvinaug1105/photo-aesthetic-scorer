"""
src/nima_scorer.py — NIMA MobileNet inference v2.1
"""

import logging
import numpy as np
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "nima_mobilenet.h5"
IMG_SIZE   = (224, 224)
MAX_DIM    = 1200


def _build_nima_model():
    from keras.applications.mobilenet import MobileNet
    from keras.layers import Dropout, Dense
    from keras.models import Model
    base = MobileNet(input_shape=(*IMG_SIZE, 3), include_top=False,
                     pooling="avg", weights="imagenet")
    x = Dropout(0.75)(base.output)
    x = Dense(10, activation="softmax")(x)
    return Model(base.input, x)


def load_model():
    if not MODEL_PATH.exists():
        logger.info("⬇️  Downloading NIMA weights...")
        _download_weights()
    model = _build_nima_model()
    model.load_weights(str(MODEL_PATH))
    logger.info("✅ NIMA model loaded")
    return model


def _download_weights():
    import urllib.request
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://github.com/idealo/image-quality-assessment"
        "/raw/master/models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5"
    )
    urllib.request.urlretrieve(url, str(MODEL_PATH))
    logger.info("✅ Weights saved → %s", MODEL_PATH)


def compress_image(img: Image.Image, max_dim: int = MAX_DIM) -> Image.Image:
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        logger.info("🗜️  Compressed to %s", img.size)
    return img


def preprocess_image(img_path: str):
    from keras.applications.mobilenet import preprocess_input
    img       = Image.open(img_path).convert("RGB")
    img       = compress_image(img)
    preview   = img.copy()
    img_small = img.resize(IMG_SIZE)
    arr       = np.array(img_small, dtype=np.float32)
    arr       = preprocess_input(arr)
    return arr[np.newaxis, :], preview   # ✅ numpy 2.x compatible


def mean_score(dist: np.ndarray) -> float:
    return float(np.sum(np.arange(1, 11, dtype=float) * dist))


def std_score(dist: np.ndarray) -> float:
    scores = np.arange(1, 11, dtype=float)
    mean   = mean_score(dist)
    return float(np.sqrt(np.sum(dist * (scores - mean) ** 2)))


def score_image(model, img_path: str) -> dict:
    x, preview = preprocess_image(img_path)
    dist       = model.predict(x, verbose=0)[0]
    s          = mean_score(dist)
    sd         = std_score(dist)
    if   s >= 7.5: rating, color = "Excellent 🌟", "#00D2BE"
    elif s >= 6.5: rating, color = "Good 👍",      "#FFD700"
    elif s >= 5.5: rating, color = "Average 😐",   "#FF8C00"
    else:          rating, color = "Needs Work 📉", "#E8002D"
    return {
        "score":        round(s, 2),
        "std":          round(sd, 2),
        "distribution": dist.tolist(),
        "rating":       rating,
        "color":        color,
        "preview":      preview,
    }


def score_batch(model, img_paths: list[str]) -> list[dict]:
    results = []
    for path in img_paths:
        try:
            r = score_image(model, path)
            r["path"] = path
            results.append(r)
        except Exception as e:
            logger.warning("Failed: %s — %s", path, e)
    return sorted(results, key=lambda x: x["score"], reverse=True)