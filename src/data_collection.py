import os
import json
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFilter

def generate_ml_water_quality_dataset(n_samples=2500, random_state=42):
    """
    Generates a realistic coastal and ocean environmental water quality dataset.
    Features include temperature, pH, turbidity, dissolved oxygen, salinity,
    microplastic particle density, macroplastic count, wave height, coastal proximity,
    industrial discharge score, and location coordinates.
    """
    np.random.seed(random_state)
    
    # Geographic distribution around realistic ocean/coastal zones (e.g., Pacific, Atlantic, Indian Ocean coastal hotspots)
    base_lats = np.random.choice([13.08, 19.07, 34.05, 1.35, -6.20, 25.76, 35.68], size=n_samples)
    base_lons = np.random.choice([80.27, 72.87, -118.24, 103.81, 106.84, -80.19, 139.69], size=n_samples)
    
    latitudes = base_lats + np.random.uniform(-1.5, 1.5, n_samples)
    longitudes = base_lons + np.random.uniform(-1.5, 1.5, n_samples)
    
    # Environmental physical parameters
    water_temp = np.random.normal(24.5, 4.2, n_samples).clip(10.0, 36.0)  # °C
    ph_level = np.random.normal(7.8, 0.45, n_samples).clip(6.0, 9.0)        # pH scale
    turbidity = np.random.exponential(12.0, n_samples).clip(0.5, 95.0)     # NTU
    dissolved_oxygen = np.random.normal(6.5, 1.8, n_samples).clip(1.5, 12.0) # mg/L
    salinity = np.random.normal(34.5, 2.5, n_samples).clip(20.0, 42.0)       # PSU
    
    # Pollution metrics
    coastal_proximity_km = np.random.exponential(15.0, n_samples).clip(0.2, 100.0)
    industrial_discharge = np.random.uniform(0.0, 10.0, n_samples)
    wave_height = np.random.gamma(2.0, 0.8, n_samples).clip(0.1, 6.0)         # meters
    ocean_current_speed = np.random.uniform(0.05, 2.2, n_samples)            # m/s
    
    # Microplastic & Macroplastic levels correlated with proximity & industrial score
    microplastic_particles = (
        (100.0 / (coastal_proximity_km + 1.0)) * 45.0 + 
        industrial_discharge * 80.0 + 
        turbidity * 15.0 + 
        np.random.normal(50, 20, n_samples)
    ).clip(5.0, 3500.0)  # particles/m³
    
    macroplastic_count = (
        (50.0 / (coastal_proximity_km + 0.5)) * 12.0 + 
        industrial_discharge * 18.0 + 
        np.random.normal(10, 5, n_samples)
    ).clip(0.0, 500.0)  # items/km²
    
    # Calculate Pollution Index (0 to 100) based on environmental formula
    # Higher turbidity, higher microplastics, higher industrial discharge, lower dissolved oxygen = higher pollution
    norm_turbidity = turbidity / 95.0
    norm_micro = microplastic_particles / 3500.0
    norm_macro = macroplastic_count / 500.0
    norm_do_inv = (12.0 - dissolved_oxygen) / 10.5
    norm_ind = industrial_discharge / 10.0
    norm_ph_dev = np.abs(ph_level - 8.1) / 2.1
    
    pollution_index_raw = (
        norm_turbidity * 22.0 +
        norm_micro * 30.0 +
        norm_macro * 20.0 +
        norm_do_inv * 15.0 +
        norm_ind * 10.0 +
        norm_ph_dev * 3.0 +
        np.random.normal(0, 2.5, n_samples)
    )
    
    pollution_index = np.clip(pollution_index_raw, 0.0, 100.0)
    
    # Assign Pollution Level Category
    pollution_level = []
    for idx in pollution_index:
        if idx < 25.0:
            pollution_level.append("Low")
        elif idx < 50.0:
            pollution_level.append("Medium")
        elif idx < 75.0:
            pollution_level.append("High")
        else:
            pollution_level.append("Critical")
            
    df = pd.DataFrame({
        "sample_id": [f"ENV_SAMP_{i+1:04d}" for i in range(n_samples)],
        "latitude": np.round(latitudes, 5),
        "longitude": np.round(longitudes, 5),
        "water_temp_c": np.round(water_temp, 2),
        "ph_level": np.round(ph_level, 2),
        "turbidity_ntu": np.round(turbidity, 2),
        "dissolved_oxygen_mg_l": np.round(dissolved_oxygen, 2),
        "salinity_psu": np.round(salinity, 2),
        "microplastic_particles_m3": np.round(microplastic_particles, 2),
        "macroplastic_count_km2": np.round(macroplastic_count, 2),
        "coastal_proximity_km": np.round(coastal_proximity_km, 2),
        "wave_height_m": np.round(wave_height, 2),
        "ocean_current_speed_m_s": np.round(ocean_current_speed, 2),
        "industrial_discharge_score": np.round(industrial_discharge, 2),
        "pollution_index": np.round(pollution_index, 2),
        "pollution_level": pollution_level
    })
    
    # Introduce ~1% synthetic missing values to simulate realistic sensor telemetry noise
    missing_indices = np.random.choice(df.index, size=int(0.01 * n_samples), replace=False)
    df.loc[missing_indices, "turbidity_ntu"] = np.nan
    
    missing_ph = np.random.choice(df.index, size=int(0.008 * n_samples), replace=False)
    df.loc[missing_ph, "ph_level"] = np.nan

    return df

def draw_water_background(width=640, height=640):
    """Generates a realistic ocean/coastal water background image with waves and textures."""
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create realistic water gradient (deep teal/blue to shallow aqua)
    for y in range(height):
        r = int(10 + (y / height) * 20)
        g = int(80 + (y / height) * 60)
        b = int(140 + (y / height) * 70)
        bg[y, :] = [b, g, r] # BGR
        
    # Add subtle wave ripple textures
    noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
    bg_noisy = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    bg_blur = cv2.GaussianBlur(bg_noisy, (15, 15), 0)
    return bg_blur

def generate_dl_plastic_images(n_images=300, output_img_dir="data/raw/images", output_lbl_dir="data/raw/labels"):
    """
    Generates synthetic plastic and marine waste image dataset with bounding boxes in YOLO format.
    Classes:
    0: plastic_bottle
    1: plastic_bag
    2: fishing_net
    3: other_waste
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)
    
    classes = ["plastic_bottle", "plastic_bag", "fishing_net", "other_waste"]
    img_size = 640
    
    for idx in range(n_images):
        img = draw_water_background(img_size, img_size)
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        n_objects = np.random.randint(1, 5)
        labels = []
        
        for _ in range(n_objects):
            cls_id = np.random.randint(0, len(classes))
            
            # Object bounding box size
            w = np.random.randint(50, 150)
            h = np.random.randint(50, 150)
            x_min = np.random.randint(20, img_size - w - 20)
            y_min = np.random.randint(20, img_size - h - 20)
            x_max = x_min + w
            y_max = y_min + h
            
            # Draw synthetic plastic item with realistic shapes and semi-transparency
            if cls_id == 0:  # plastic bottle
                # Bottle shape (rectangle body + bottle neck)
                draw.rectangle([x_min, y_min + int(h*0.3), x_max, y_max], fill=(220, 240, 255), outline=(100, 180, 255), width=3)
                draw.rectangle([x_min + int(w*0.3), y_min, x_min + int(w*0.7), y_min + int(h*0.3)], fill=(0, 150, 255), outline=(0, 100, 200), width=2)
            elif cls_id == 1: # plastic bag
                # Irregular bag polygon
                points = [
                    (x_min + w*0.2, y_min),
                    (x_min + w*0.8, y_min),
                    (x_max, y_min + h*0.5),
                    (x_min + w*0.9, y_max),
                    (x_min + w*0.1, y_max),
                    (x_min, y_min + h*0.4)
                ]
                draw.polygon(points, fill=(240, 240, 240), outline=(180, 180, 180), width=2)
            elif cls_id == 2: # fishing net
                # Grid pattern for net
                draw.rectangle([x_min, y_min, x_max, y_max], outline=(50, 50, 50), width=2)
                for gx in range(x_min, x_max, 15):
                    draw.line([(gx, y_min), (gx, y_max)], fill=(70, 70, 70), width=2)
                for gy in range(y_min, y_max, 15):
                    draw.line([(x_min, gy), (x_max, gy)], fill=(70, 70, 70), width=2)
            else: # other waste (can, jug, container)
                draw.ellipse([x_min, y_min, x_max, y_max], fill=(230, 120, 50), outline=(180, 80, 20), width=3)
                
            # YOLO format: class_id x_center y_center width height (normalized 0-1)
            x_center = (x_min + w / 2.0) / img_size
            y_center = (y_min + h / 2.0) / img_size
            norm_w = w / img_size
            norm_h = h / img_size
            
            labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
            
        # Convert back to OpenCV & save image
        out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        img_name = f"ocean_waste_{idx+1:04d}.jpg"
        lbl_name = f"ocean_waste_{idx+1:04d}.txt"
        
        cv2.imwrite(os.path.join(output_img_dir, img_name), out_bgr)
        with open(os.path.join(output_lbl_dir, lbl_name), "w") as f:
            f.write("\n".join(labels))

if __name__ == "__main__":
    print("Generating ML Environmental Water Quality Dataset...")
    os.makedirs("data/raw", exist_ok=True)
    df_ml = generate_ml_water_quality_dataset(n_samples=2500, random_state=42)
    ml_path = "data/raw/ocean_water_quality.csv"
    df_ml.to_csv(ml_path, index=False)
    print(f"ML Dataset saved to {ml_path} ({len(df_ml)} rows, {df_ml.shape[1]} columns)")
    
    print("\nGenerating Deep Learning Plastic Waste Image Dataset & Bounding Box Annotations...")
    generate_dl_plastic_images(n_images=300, output_img_dir="data/raw/images", output_lbl_dir="data/raw/labels")
    print(f"DL Image Dataset generated: 300 images saved to data/raw/images/ and annotations to data/raw/labels/")
    
    # Save dataset metadata summary
    summary = {
        "ml_dataset": {
            "total_samples": len(df_ml),
            "features": list(df_ml.columns),
            "pollution_levels": df_ml["pollution_level"].value_counts().to_dict(),
            "file_path": ml_path
        },
        "dl_dataset": {
            "total_images": 300,
            "image_size": [640, 640],
            "classes": ["plastic_bottle", "plastic_bag", "fishing_net", "other_waste"],
            "image_dir": "data/raw/images",
            "label_dir": "data/raw/labels"
        }
    }
    with open("data/raw/dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Dataset Summary saved to data/raw/dataset_summary.json")
