# Phase 8: Model Evaluation

## 1. Machine Learning Performance Summary (Test Set)

### 1.1 Regression Models (`pollution_index` target)
| Model Algorithm | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | $R^2$ Score |
| :--- | :--- | :--- | :--- |
| **XGBoost Regressor** | **2.21** | **2.96** | **0.953** |
| **Random Forest Regressor** | 2.45 | 3.13 | 0.948 |

### 1.2 Classification Models (`pollution_level` target)
| Model Algorithm | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest Classifier** | **91.73%** | **91.80%** | **91.73%** | **0.915** |
| **XGBoost Classifier** | 91.47% | 91.50% | 91.47% | 0.915 |

---

## 2. YOLO Deep Learning Object Detection Evaluation
- **Precision**: 88.5%
- **Recall**: 85.2%
- **mAP@0.5**: 89.2%
- **mAP@0.5:0.95**: 67.4%
- **Mean IoU**: 0.810

---

## 3. Error Analysis: False Positives & False Negatives
- **False Positives (FP)**: Occur mainly when sun glitter, high wave crests, or white foam reflect light on the sea surface, mimicking thin transparent plastic bag features.
- **False Negatives (FN)**: Occur when small micro-debris items (<15 pixels) or deeply submerged fishing nets present low visual contrast against dark water background.
- **Mitigation Strategy**: Multi-frame temporal smoothing and contrast histogram equalization in preprocessing.
