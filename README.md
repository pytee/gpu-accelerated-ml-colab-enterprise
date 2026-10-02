# Hands-On GPU-Accelerated Machine Learning in Colab Enterprise

An interactive, self-contained codelab walking through GPU-accelerated machine learning on
an NVIDIA L4 GPU in Google Cloud Colab Enterprise — using `cuDF`, `cuML`, and XGBoost to
accelerate a `pandas` / `scikit-learn` NYC Yellow Taxi tip-prediction workflow with no
rewrite of the underlying code.

## Contents

- 15 steps, ~2–3 hours, intermediate level
- Configuring a GPU-backed Colab Enterprise runtime template
- Accelerating `pandas` with `cudf.pandas` and `scikit-learn` with `cuml.accel`
- Cross-validated training, ensembling, and end-to-end pipeline evaluation
- CPU vs. GPU benchmarking and profiling to catch CPU fallbacks

## Viewing

Open `index.html` in a browser, or serve it locally:

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000`.
