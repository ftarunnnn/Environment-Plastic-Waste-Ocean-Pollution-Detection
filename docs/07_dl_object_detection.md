# Phase 7: DL Object Detection Development

## 1. Deep Learning Model Architecture
- **Model Framework**: Ultralytics YOLOv8 Nano Architecture (`yolov8n`) coupled with OpenCV contour-assisted fallback engine.
- **Input Dimensions**: 640×640×3 (RGB).
- **Target Classes**:
  1. `plastic_bottle` (ID 0)
  2. `plastic_bag` (ID 1)
  3. `fishing_net` (ID 2)
  4. `other_waste` (ID 3)
- **Output Capabilities**: Bounding box coordinates $[x_1, y_1, x_2, y_2]$, class predictions, confidence scores, and instance counter.

---

## 2. Detection Engine Architecture (`src/dl_detector.py`)
The `PlasticWasteDetector` class provides unified inference:
- Accepts raw images, file paths, PIL Image objects, or video frame buffers.
- Draws color-coded bounding box tags:
  - **Pink**: Plastic Bottle
  - **Gold**: Plastic Bag
  - **Green**: Fishing Net
  - **Blue**: General Debris / Other Waste
- Generates structured dictionary outputs containing bounding box metadata and counts.
- Evaluated on test imagery; saved output frame to `models/sample_detection_output.jpg`.
