# 🧰 Technology Stack & Tooling Breakdown

This document details all software libraries, machine learning frameworks, computer vision architectures, deployment tools, and system requirements utilized in the **Environment — Plastic Waste & Ocean Pollution Detection** project.

---

## 1. Core Programming Languages & Runtimes
- **Python 3.10+**: Core programming language for data preprocessing, ML/DL pipelines, and web app dashboard.
- **Shell / Bash & Windows Batch**: Command-line launchers (`run_dashboard.sh`, `run_dashboard.bat`).
- **Markdown & GFM**: Structured phase documentation and technical reporting.

---

## 2. Machine Learning & Data Science Libraries
- **Scikit-Learn (`scikit-learn >= 1.2.0`)**:
  - Feature normalization via `StandardScaler`.
  - Missing value imputation via `SimpleImputer(strategy="median")`.
  - Model metrics calculation: MAE, RMSE, $R^2$, Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
  - Categorical target encoding via `LabelEncoder`.
- **XGBoost (`xgboost >= 1.7.0`)**:
  - `XGBRegressor`: Continuous `pollution_index` (0-100) prediction ($R^2 = 0.953$).
  - `XGBClassifier`: Categorical `pollution_level` risk classification.
- **Random Forest (`scikit-learn`)**:
  - `RandomForestRegressor`: Multi-tree ensemble regression.
  - `RandomForestClassifier`: Categorical risk classification (Accuracy = 91.73%).
- **Pandas (`pandas >= 2.0.0`)**: Tabular dataset processing, CSV reading/writing, and data frame manipulation.
- **NumPy (`numpy >= 1.24.0`)**: Array numerical computations and synthetic data math.
- **Joblib (`joblib >= 1.2.0`)**: Model serialization for saving/loading `.joblib` model binary files.

---

## 3. Deep Learning & Computer Vision (DL / CV)
- **Ultralytics YOLOv8 (`ultralytics >= 8.0.0`)**: Real-time object detection model fine-tuned for marine waste detection (`plastic_bottle`, `plastic_bag`, `fishing_net`, `other_waste`).
- **PyTorch (`torch >= 2.0.0`, `torchvision >= 0.15.0`)**: Deep learning tensor backend framework.
- **OpenCV (`opencv-python >= 4.7.0`)**: Image BGR/RGB conversion, contour analysis, image resizing, and bounding box drawing.
- **Pillow / PIL (`Pillow >= 9.5.0`)**: Image creation, shape drawing, and texture manipulation.

---

## 4. Web Application & Data Visualization
- **Streamlit (`streamlit >= 1.22.0`)**: Interactive web framework for rendering the visual dashboard UI with dark glassmorphism styling.
- **Plotly (`plotly >= 5.14.0`)**: Interactive gauge meters, donut charts, scatter plots, and Carto mapbox maps.
- **Seaborn (`seaborn >= 0.12.0`) & Matplotlib (`matplotlib >= 3.7.0`)**: Static publication-quality statistical charts, correlation heatmaps, KDE distributions, and residual plots.

---

## 5. Containerization & Deployment
- **Docker**: `Dockerfile` defining Debian slim base image with OpenCV C++ dependencies (`libgl1-mesa-glx`, `libglib2.0-0`), Python environment, healthcheck, and Streamlit entrypoint.
- **Git & GitHub**: Version control tracking for 10 sequential phase commits.

---

## 6. Summary Table of Dependencies

| Library / Tool | Primary Purpose | Version |
| :--- | :--- | :--- |
| `streamlit` | Visual Interactive Web Dashboard | $\ge 1.22.0$ |
| `ultralytics` | YOLOv8 Deep Learning Object Detection | $\ge 8.0.0$ |
| `torch` / `torchvision` | PyTorch Deep Learning Backend | $\ge 2.0.0$ |
| `xgboost` | Gradient Boosting Pollution Prediction | $\ge 1.7.0$ |
| `scikit-learn` | Random Forest & Preprocessing Pipeline | $\ge 1.2.0$ |
| `opencv-python` | Image Processing & Bounding Box Drawing | $\ge 4.7.0$ |
| `pandas` / `numpy` | Data Manipulation & Matrix Computation | $\ge 2.0.0$ |
| `plotly` / `seaborn` | Interactive & Statistical Visualizations | $\ge 5.14.0$ |
| `docker` | Containerized App Deployment | Docker Engine 20+ |
