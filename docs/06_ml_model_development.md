# Phase 6: ML Model Development

## 1. Overview
Phase 6 developed both regression and classification algorithms to forecast continuous pollution risk scores ($0.0 - 100.0$) and classify categorical risk levels (`Low`, `Medium`, `High`, `Critical`).

Two ensemble model families were trained and tuned:
1. **Random Forest**: `RandomForestRegressor` and `RandomForestClassifier` (200 trees, `max_depth=12`).
2. **XGBoost**: `XGBRegressor` and `XGBClassifier` (250 estimators, `learning_rate=0.05`, `max_depth=6`).

---

## 2. Validation Metrics Breakdown

| Model Architecture | Target Task | Primary Metric | Score | Secondary Metric | Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost Regressor** | Continuous `pollution_index` | $R^2$ Score | **0.9534** | RMSE | **2.9607** |
| **Random Forest Regressor** | Continuous `pollution_index` | $R^2$ Score | **0.9479** | RMSE | **3.1294** |
| **Random Forest Classifier**| Categorical `pollution_level` | Accuracy | **91.73%** | F1-Score | **0.9149** |
| **XGBoost Classifier** | Categorical `pollution_level` | Accuracy | **91.47%** | F1-Score | **0.9151** |

---

## 3. Saved Model Artifacts
Trained model binary weights are persisted under `models/`:
- `random_forest_regressor.joblib`
- `random_forest_classifier.joblib`
- `xgboost_regressor.joblib`
- `xgboost_classifier.joblib`
- `ml_validation_metrics.json`
