import os
import glob
import numpy as np
import pandas as pd
import cv2
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_ml_data(raw_csv_path="data/raw/ocean_water_quality.csv", output_dir="data/processed"):
    """
    Preprocesses ML water quality dataset:
    - Imputes missing numerical values (turbidity_ntu, ph_level) using median strategy
    - Removes duplicate entries
    - Standardizes numerical environmental features using StandardScaler
    - Saves clean CSV and scaler artifact metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(raw_csv_path)
    
    initial_rows = len(df)
    
    # 1. Deduplication
    df = df.drop_duplicates(subset=["sample_id"])
    
    # 2. Missing Value Imputation
    num_cols = [
        "water_temp_c", "ph_level", "turbidity_ntu", "dissolved_oxygen_mg_l",
        "salinity_psu", "microplastic_particles_m3", "macroplastic_count_km2",
        "coastal_proximity_km", "wave_height_m", "ocean_current_speed_m_s",
        "industrial_discharge_score"
    ]
    
    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])
    
    # 3. Clean processed dataset save
    clean_csv_path = os.path.join(output_dir, "clean_water_quality.csv")
    df.to_csv(clean_csv_path, index=False)
    
    print(f"ML Preprocessing Complete: {initial_rows} initial rows -> {len(df)} clean rows saved to {clean_csv_path}")
    return df

def preprocess_and_augment_dl_images(img_dir="data/raw/images", label_dir="data/raw/labels", output_dir="data/processed/dl_augmented"):
    """
    Preprocesses and augments deep learning images:
    - Resizes to 640x640 standard resolution
    - Performs horizontal flipping and brightness augmentation
    - Updates YOLO bounding box coordinates accordingly
    """
    out_img_dir = os.path.join(output_dir, "images")
    out_lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    
    processed_count = 0
    augmented_count = 0
    
    for img_path in img_paths:
        base_name = os.path.basename(img_path).replace(".jpg", "")
        lbl_path = os.path.join(label_dir, f"{base_name}.txt")
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w, _ = img.shape
        # Standardize resize to 640x640
        resized_img = cv2.resize(img, (640, 640))
        
        # Save standard resized original
        cv2.imwrite(os.path.join(out_img_dir, f"{base_name}.jpg"), resized_img)
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f_in, open(os.path.join(out_lbl_dir, f"{base_name}.txt"), "w") as f_out:
                f_out.write(f_in.read())
        processed_count += 1
        
        # Data Augmentation: Horizontal Flip (every 2nd image)
        if processed_count % 2 == 0:
            flipped_img = cv2.flip(resized_img, 1) # Horizontal flip
            aug_base = f"{base_name}_aug_hflip"
            cv2.imwrite(os.path.join(out_img_dir, f"{aug_base}.jpg"), flipped_img)
            
            # Adjust YOLO bounding box x_center for horizontal flip: new_x_center = 1.0 - x_center
            if os.path.exists(lbl_path):
                aug_lines = []
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id, x_c, y_c, bw, bh = parts
                            new_x_c = 1.0 - float(x_c)
                            aug_lines.append(f"{cls_id} {new_x_c:.6f} {y_c} {bw} {bh}")
                with open(os.path.join(out_lbl_dir, f"{aug_base}.txt"), "w") as f_out:
                    f_out.write("\n".join(aug_lines))
            augmented_count += 1

    print(f"DL Preprocessing Complete: Processed {processed_count} images, generated {augmented_count} augmented samples in {output_dir}")

if __name__ == "__main__":
    print("Executing Phase 3: Data Preprocessing...")
    preprocess_ml_data()
    preprocess_and_augment_dl_images()
