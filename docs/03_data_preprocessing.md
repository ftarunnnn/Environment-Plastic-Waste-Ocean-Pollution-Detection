# Phase 3: Data Preprocessing

## 1. Machine Learning Data Preprocessing
- **Missing Value Handling**: Used `SimpleImputer` with median strategy on `turbidity_ntu` and `ph_level` to preserve non-Gaussian distributions without introducing skewness.
- **Deduplication**: Verified uniqueness of all sample IDs (`sample_id`).
- **Feature Normalization**: Configured `StandardScaler` pipeline for numerical features.
- **Output File**: Saved clean preprocessed tabular dataset to `data/processed/clean_water_quality.csv` (2,500 clean rows, 0 nulls).

---

## 2. Deep Learning Image Preprocessing & Augmentation
- **Resolution Standardization**: Resized all input images to a uniform 640×640×3 resolution.
- **Data Augmentation**:
  - Horizontal Flipping: Generated 150 augmented image frames.
  - Coordinate Recalculation: Transformed bounding box $x_{center}' = 1.0 - x_{center}$ for all flipped annotations.
- **Output Directory**: Saved 450 total processed and augmented images with matching YOLO coordinate `.txt` files in `data/processed/dl_augmented/`.
