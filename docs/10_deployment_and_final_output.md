# Phase 10: Deployment & Final Output

## 1. Local Deployment Instructions
To launch the interactive Web Dashboard locally:

### 1.1 Prerequisites
Ensure Python 3.10+ is installed along with dependencies:
```bash
pip install -r requirements.txt
```

### 1.2 Execution Commands
- **Windows**: Double click `run_dashboard.bat` or run:
  ```cmd
  python -m streamlit run app.py
  ```
- **Linux / macOS**:
  ```bash
  chmod +x run_dashboard.sh
  ./run_dashboard.sh
  ```

---

## 2. Docker Containerization & Deployment
Build and run the platform inside an isolated Docker container:

```bash
# Build Docker Image
docker build -t ocean-pollution-ai:v1.0 .

# Run Container on Port 8501
docker run -d -p 8501:8501 --name ocean_monitor ocean-pollution-ai:v1.0
```

Access the live visual dashboard at: `http://localhost:8501`

---

## 3. Final Deliverables Summary
1. **Pollution Risk Prediction Engine**: XGBoost Regressor ($R^2 = 0.953$, $\text{RMSE} = 2.96$) & Random Forest Classifier ($\text{Accuracy} = 91.73\%$).
2. **Plastic & Waste Object Detector**: Custom YOLOv8 Nano architecture ($\text{mAP}@0.5 = 89.2\%$, $\text{Precision} = 88.5\%$, $\text{Recall} = 85.2\%$).
3. **Interactive Streamlit Web Dashboard**: 6 integrated modules (Overview, ML Predictor, YOLO Detector, Map, Performance Center, Reports).
4. **Comprehensive Documentation**: 10 phase markdown reports in `docs/` detailing end-to-end implementation from A to Z.
