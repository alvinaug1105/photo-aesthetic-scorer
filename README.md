# 📸 Photo Aesthetic Scorer

AI-powered photo quality scorer built with NIMA (Neural Image Assessment) + OpenCV.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange)

## ✨ Features
- 🎯 **NIMA MobileNet** aesthetic scoring (trained on 255k AVA photos)
- 📊 **6 CV dimensions**: Sharpness, Composition, Brightness, Contrast, Color, Noise
- 📐 **Multi-intent composition detection**: Rule of Thirds / Symmetry / Negative Space / Centre Focus
- 📈 **Batch analyzer** with score trends and rolling average
- 🏆 **Gallery stats** dashboard with dimension breakdown
- 🌟 **Percentile rank** vs AVA dataset
- ✨ **Glow animations** + Pro Benchmark radar chart

## 🚀 Setup

```bash
git clone https://github.com/YOUR_USERNAME/photo-aesthetic-scorer
cd photo-aesthetic-scorer
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Model Weights

Weights are downloaded automatically on first run. To download manually:

```bash
mkdir -p models
wget -O models/nima_mobilenet.h5 \
  https://github.com/idealo/image-quality-assessment/raw/master/models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5
```

## 📁 Project Structure

```
photo-aesthetic-scorer/
├── app.py                  # Main Streamlit app
├── requirements.txt
├── README.md
├── .gitignore
└── src/
    ├── nima_scorer.py      # NIMA MobileNet inference
    ├── cv_features.py      # OpenCV feature extraction
    └── explainer.py        # Tips & verdict generation
```

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Streamlit | Web UI |
| TensorFlow / Keras | NIMA model inference |
| OpenCV | CV feature extraction |
| Plotly | Interactive charts |
| SciPy | Percentile calculation |
| Pillow | Image processing |

## 📊 Scoring Guide

| NIMA Score | Meaning |
|-----------|---------|
| < 4.5 | Below average |
| 4.5 – 5.5 | Average (most photos) |
| 5.5 – 6.5 | Good, above average |
| 6.5 – 7.5 | Excellent, professional level |
| > 7.5 | Top tier, competition level |

## 📄 License

MIT License — feel free to use and modify.
