import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

def train_ml_models(train_path="data/processed/train_data.csv", val_path="data/processed/val_data.csv", model_dir="models"):
    """
    Trains and evaluates Random Forest and XGBoost Regressors & Classifiers for ocean pollution prediction:
    - Random Forest Regressor & XGBoost Regressor for continuous pollution_index (0-100)
    - Random Forest Classifier & XGBoost Classifier for categorical pollution_level risk
    - Saves trained model binary artifacts (.joblib)
    """
    os.makedirs(model_dir, exist_ok=True)
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    feature_cols = [
        "water_temp_c", "ph_level", "turbidity_ntu", "dissolved_oxygen_mg_l",
        "salinity_psu", "microplastic_particles_m3", "macroplastic_count_km2",
        "coastal_proximity_km", "wave_height_m", "ocean_current_speed_m_s",
        "industrial_discharge_score"
    ]
    
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    
    X_train = scaler.transform(train_df[feature_cols])
    y_train_reg = train_df["pollution_index"]
    y_train_cls = train_df["pollution_level_encoded"]
    
    X_val = scaler.transform(val_df[feature_cols])
    y_val_reg = val_df["pollution_index"]
    y_val_cls = val_df["pollution_level_encoded"]
    
    print("--- Training Random Forest Regressor ---")
    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, y_train_reg)
    rf_reg_pred = rf_reg.predict(X_val)
    rf_r2 = r2_score(y_val_reg, rf_reg_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_val_reg, rf_reg_pred))
    print(f"Random Forest Regressor -> Validation R²: {rf_r2:.4f}, RMSE: {rf_rmse:.4f}")
    
    print("\n--- Training Random Forest Classifier ---")
    rf_cls = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf_cls.fit(X_train, y_train_cls)
    rf_cls_pred = rf_cls.predict(X_val)
    rf_acc = accuracy_score(y_val_cls, rf_cls_pred)
    rf_f1 = f1_score(y_val_cls, rf_cls_pred, average="weighted")
    print(f"Random Forest Classifier -> Validation Accuracy: {rf_acc:.4f}, F1-Score: {rf_f1:.4f}")
    
    print("\n--- Training XGBoost Regressor ---")
    xgb_reg = XGBRegressor(n_estimators=250, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    xgb_reg.fit(X_train, y_train_reg)
    xgb_reg_pred = xgb_reg.predict(X_val)
    xgb_r2 = r2_score(y_val_reg, xgb_reg_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_val_reg, xgb_reg_pred))
    print(f"XGBoost Regressor -> Validation R²: {xgb_r2:.4f}, RMSE: {xgb_rmse:.4f}")
    
    print("\n--- Training XGBoost Classifier ---")
    xgb_cls = XGBClassifier(n_estimators=250, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    xgb_cls.fit(X_train, y_train_cls)
    xgb_cls_pred = xgb_cls.predict(X_val)
    xgb_acc = accuracy_score(y_val_cls, xgb_cls_pred)
    xgb_f1 = f1_score(y_val_cls, xgb_cls_pred, average="weighted")
    print(f"XGBoost Classifier -> Validation Accuracy: {xgb_acc:.4f}, F1-Score: {xgb_f1:.4f}")
    
    # Save Model Artifacts
    joblib.dump(rf_reg, os.path.join(model_dir, "random_forest_regressor.joblib"))
    joblib.dump(rf_cls, os.path.join(model_dir, "random_forest_classifier.joblib"))
    joblib.dump(xgb_reg, os.path.join(model_dir, "xgboost_regressor.joblib"))
    joblib.dump(xgb_cls, os.path.join(model_dir, "xgboost_classifier.joblib"))
    
    # Save validation results summary
    val_results = {
        "random_forest": {
            "r2_score": float(rf_r2),
            "rmse": float(rf_rmse),
            "accuracy": float(rf_acc),
            "f1_score": float(rf_f1)
        },
        "xgboost": {
            "r2_score": float(xgb_r2),
            "rmse": float(xgb_rmse),
            "accuracy": float(xgb_acc),
            "f1_score": float(xgb_f1)
        }
    }
    with open(os.path.join(model_dir, "ml_validation_metrics.json"), "w") as f:
        json.dump(val_results, f, indent=2)
        
    print(f"\nAll ML Models saved to {model_dir}/")

if __name__ == "__main__":
    print("Executing Phase 6: ML Model Development...")
    train_ml_models()
