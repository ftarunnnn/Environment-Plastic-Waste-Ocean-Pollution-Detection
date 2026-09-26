# Phase 2: Dataset Collection

## 1. Dataset Overview
In Phase 2, two comprehensive datasets were created and organized under `data/raw/`:
1. **ML Environmental Water Quality Dataset** (`data/raw/ocean_water_quality.csv`): 2,500 samples of ocean and coastal water metrics across global hotspots.
2. **DL Computer Vision Waste Image Dataset** (`data/raw/images/` & `data/raw/labels/`): 300 annotated images of ocean plastic and waste debris with bounding box targets in YOLO format.

---

## 2. ML Water Quality Features Schema
| Feature Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `sample_id` | String | - | Unique sample identifier (`ENV_SAMP_XXXX`) |
| `latitude` | Float | Degrees | Geographical coordinate latitude |
| `longitude` | Float | Degrees | Geographical coordinate longitude |
| `water_temp_c` | Float | °C | Sea surface temperature (10.0°C - 36.0°C) |
| `ph_level` | Float | pH | Acidity / Alkalinity level (6.0 - 9.0) |
| `turbidity_ntu` | Float | NTU | Water clarity/cloudiness measure |
| `dissolved_oxygen_mg_l` | Float | mg/L | Dissolved oxygen concentration |
| `salinity_psu` | Float | PSU | Salinity level in Practical Salinity Units |
| `microplastic_particles_m3`| Float | particles/m³ | Microplastic concentration |
| `macroplastic_count_km2` | Float | items/km² | Floating macroplastic item count |
| `coastal_proximity_km` | Float | km | Distance from major shoreline/port |
| `wave_height_m` | Float | meters | Significant wave height |
| `ocean_current_speed_m_s` | Float | m/s | Surface ocean current velocity |
| `industrial_discharge_score`| Float | 0-10 Score | Industrial proximity and effluent index |
| `pollution_index` | Float | 0-100 Score | Target continuous pollution severity index |
| `pollution_level` | Categorical | Level | Target risk class (`Low`, `Medium`, `High`, `Critical`) |

---

## 3. Deep Learning Waste Object Detection Classes
- **Class 0**: `plastic_bottle`
- **Class 1**: `plastic_bag`
- **Class 2**: `fishing_net`
- **Class 3**: `other_waste`

Annotations are stored in standard YOLO format: `class_id x_center y_center width height` (normalized 0.0 to 1.0).
