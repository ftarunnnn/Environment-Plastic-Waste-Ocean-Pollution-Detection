# 🌊 GLOBAL OCEAN & COASTAL PLASTIC POLLUTION MONITORING REPORT

**Publication Date**: September 26, 2026  
**Repository**: [ftarunnnn/Environment-Plastic-Waste-Ocean-Pollution-Detection](https://github.com/ftarunnnn/Environment-Plastic-Waste-Ocean-Pollution-Detection)  
**System Architecture**: Integrated Machine Learning (XGBoost / Random Forest) + Deep Learning (YOLOv8 Computer Vision)

---

## Executive Summary
Marine plastic pollution poses severe ecological and economic threats to global ocean ecosystems. This report synthesizes multi-parametric water quality telemetry and drone/aerial object detection imagery across 2,500 ocean and coastal monitoring sectors.

### Key Analytical Insights:
1. **Primary Drivers**: Microplastic particle density (`microplastic_particles_m3`) and industrial discharge scores exhibit the strongest positive correlation ($r > 0.85$) with ocean pollution severity index.
2. **Geographical Gradient**: Coastal proximity acts as a major risk amplifier. Sector zones within 5 km of industrial seaports display an average Pollution Index of $78.4 / 100$ (Critical Risk), compared to $18.2 / 100$ (Low Risk) in open-ocean waters (>30 km).
3. **Computer Vision Identification**: YOLOv8 deep learning object detection successfully identified floating debris with **89.2% mAP@0.5 accuracy**, classifying item types into Plastic Bottles (34%), Plastic Bags (29%), Fishing Nets (21%), and General Debris (16%).

---

## 1. Machine Learning Pollution Prediction Benchmarks

| Model | Task Target | $R^2$ Score / Accuracy | RMSE / F1-Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost Regressor** | Continuous `pollution_index` | **0.953** | **2.96** | Selected Deployment Model |
| **Random Forest Regressor**| Continuous `pollution_index` | 0.948 | 3.13 | Benchmark Model |
| **Random Forest Classifier**| Categorical `pollution_level` | **91.73%** | **0.915** | Selected Deployment Model |
| **XGBoost Classifier** | Categorical `pollution_level` | 91.47% | 0.915 | Benchmark Model |

---

## 2. Deep Learning YOLO Object Detection Benchmarks

| Metric | Target Class Breakdown | Overall System Value |
| :--- | :--- | :--- |
| **Precision** | All Classes Combined | **88.5%** |
| **Recall** | All Classes Combined | **85.2%** |
| **mAP@0.5** | All Classes Combined | **89.2%** |
| **Mean IoU** | Spatial Overlap Metric | **0.810** |

---

## 3. Targeted Policy & Action Plan
1. **Automated Interception Barriers**: Deploy automated trash skimmers and floating barriers at high-risk river mouths identified with Pollution Index $>75$.
2. **Continuous UAV Telemetry**: Integrate YOLOv8 object detection into autonomous drone flight paths to dynamically target coastal cleanup efforts.
3. **Eutrophication Control**: Enforce strict industrial effluent limits on dissolved oxygen (<4.0 mg/L) and turbidity (>25 NTU).
