import os
import shutil
import glob
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def prepare_ml_train_val_test(clean_csv_path="data/processed/clean_water_quality.csv", output_dir="data/processed", model_dir="models"):
    """
    Performs Feature Engineering and Dataset Preparation for ML:
    - Selects top predictor features based on domain logic & correlation
    - Encodes categorical targets (pollution_level)
    - Performs train/validation/test split (70% train, 15% val, 15% test)
    - Standardizes numerical features using StandardScaler
    - Saves scaler artifact and processed CSVs
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    df = pd.read_csv(clean_csv_path)
    
    # Feature Selection: core environmental predictor columns
    feature_cols = [
        "water_temp_c", "ph_level", "turbidity_ntu", "dissolved_oxygen_mg_l",
        "salinity_psu", "microplastic_particles_m3", "macroplastic_count_km2",
        "coastal_proximity_km", "wave_height_m", "ocean_current_speed_m_s",
        "industrial_discharge_score"
    ]
    
    target_continuous = "pollution_index"
    target_categorical = "pollution_level"
    
    # Encode categorical target
    label_encoder = LabelEncoder()
    # Ensure consistent order: Low=0, Medium=1, High=2, Critical=3
    level_order = ["Low", "Medium", "High", "Critical"]
    label_encoder.fit(level_order)
    df["pollution_level_encoded"] = label_encoder.transform(df[target_categorical])
    
    # 70% Train, 15% Validation, 15% Test Split
    train_df, test_val_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["pollution_level_encoded"])
    val_df, test_df = train_test_split(test_val_df, test_size=0.50, random_state=42, stratify=test_val_df["pollution_level_encoded"])
    
    # Scale Features
    scaler = StandardScaler()
    train_scaled_feats = scaler.fit_transform(train_df[feature_cols])
    val_scaled_feats = scaler.transform(val_df[feature_cols])
    test_scaled_feats = scaler.transform(test_df[feature_cols])
    
    # Save Scaler & Encoder artifacts
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.joblib"))
    
    # Save train/val/test CSV splits
    train_df.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_data.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)
    
    print(f"ML Dataset Split Complete: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

def prepare_yolo_dataset_structure(src_img_dir="data/processed/dl_augmented/images", src_lbl_dir="data/processed/dl_augmented/labels", yolo_dir="data/yolo_dataset"):
    """
    Organizes DL images and annotations into YOLO train/val directory format and generates dataset.yaml:
    data/yolo_dataset/
      ├── train/
      │   ├── images/
      │   └── labels/
      ├── val/
      │   ├── images/
      │   └── labels/
      └── dataset.yaml
    """
    train_img_dir = os.path.join(yolo_dir, "train", "images")
    train_lbl_dir = os.path.join(yolo_dir, "train", "labels")
    val_img_dir = os.path.join(yolo_dir, "val", "images")
    val_lbl_dir = os.path.join(yolo_dir, "val", "labels")
    
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)
        
    img_files = sorted(glob.glob(os.path.join(src_img_dir, "*.jpg")))
    np.random.seed(42)
    np.random.shuffle(img_files)
    
    split_idx = int(0.80 * len(img_files))
    train_imgs = img_files[:split_idx]
    val_imgs = img_files[split_idx:]
    
    for img_path in train_imgs:
        base_name = os.path.basename(img_path)
        lbl_name = base_name.replace(".jpg", ".txt")
        shutil.copy(img_path, os.path.join(train_img_dir, base_name))
        lbl_path = os.path.join(src_lbl_dir, lbl_name)
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(train_lbl_dir, lbl_name))
            
    for img_path in val_imgs:
        base_name = os.path.basename(img_path)
        lbl_name = base_name.replace(".jpg", ".txt")
        shutil.copy(img_path, os.path.join(val_img_dir, base_name))
        lbl_path = os.path.join(src_lbl_dir, lbl_name)
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(val_lbl_dir, lbl_name))
            
    # Create dataset.yaml
    yaml_content = f"""path: {os.path.abspath(yolo_dir).replace('\\', '/')}
train: train/images
val: val/images

nc: 4
names: ['plastic_bottle', 'plastic_bag', 'fishing_net', 'other_waste']
"""
    yaml_path = os.path.join(yolo_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    print(f"YOLO Dataset Prepared: {len(train_imgs)} train images, {len(val_imgs)} val images. Config saved to {yaml_path}")

if __name__ == "__main__":
    print("Executing Phase 5: Feature Engineering & Dataset Preparation...")
    prepare_ml_train_val_test()
    prepare_yolo_dataset_structure()
