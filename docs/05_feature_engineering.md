# Phase 5: Feature Engineering & Dataset Preparation

## 1. Feature Selection & Target Encoding
- **Selected Predictor Features**:
  1. `water_temp_c`
  2. `ph_level`
  3. `turbidity_ntu`
  4. `dissolved_oxygen_mg_l`
  5. `salinity_psu`
  6. `microplastic_particles_m3`
  7. `macroplastic_count_km2`
  8. `coastal_proximity_km`
  9. `wave_height_m`
  10. `ocean_current_speed_m_s`
  11. `industrial_discharge_score`
- **Categorical Target Mapping**: Encoded `pollution_level` using `LabelEncoder`:
  - `Low` $\rightarrow 0$
  - `Medium` $\rightarrow 1$
  - `High` $\rightarrow 2$
  - `Critical` $\rightarrow 3$
- **Feature Standardization**: Fitted `StandardScaler` on training fold and saved to `models/scaler.joblib`.

---

## 2. Dataset Splitting
### 2.1 Tabular Machine Learning Split
- **Train Set (70%)**: 1,750 samples (`data/processed/train_data.csv`)
- **Validation Set (15%)**: 375 samples (`data/processed/val_data.csv`)
- **Test Set (15%)**: 375 samples (`data/processed/test_data.csv`)

### 2.2 Computer Vision YOLO Structure
Organized image and annotation files under `data/yolo_dataset/`:
- `train/images`: 360 images (80%)
- `val/images`: 90 images (20%)
- `dataset.yaml`: YOLO manifest specifying 4 classes (`plastic_bottle`, `plastic_bag`, `fishing_net`, `other_waste`).
