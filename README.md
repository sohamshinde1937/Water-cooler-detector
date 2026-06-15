# Water Cooler AI 💧🤖

A YOLOv8-based computer vision project for real-time **water cooler detection** using a custom-trained object detection model.

This project extracts frames from video footage, uses them to train a YOLOv8 model on custom classes, and runs live detection through a webcam feed.

---

## ✨ Features

- 🎯 **Custom-trained YOLOv8 model** for water cooler detection (2 classes)
- 🎥 **Frame extraction tool** to build a dataset from video recordings
- 📦 **Pre-labeled dataset** with train/val splits (images + YOLO-format labels)
- 🖥️ **Live webcam detection** with bounding box visualization
- 📊 **Full training results** included (metrics, curves, confusion matrix, sample predictions)

---

## 📁 Project Structure

```
water_cooler_ai/
├── data.yaml                 # Dataset config (classes, paths)
├── train.py                  # Training script
├── extract_frames.py         # Extracts frames from a video for labeling
├── detect_camera.py          # Runs live detection via webcam
├── requirements.txt          # Python dependencies
├── yolov8n.pt                 # Base YOLOv8 nano pretrained weights
├── dataset/                    # ⬇️ Download separately (see Dataset section below)
│   ├── images/
│   │   ├── train/             # Training images (frames)
│   │   └── val/                # Validation images
│   └── labels/
│       ├── train/             # YOLO-format labels for training
│       └── val/                 # YOLO-format labels for validation
└── runs/detect/train/
    ├── weights/
    │   ├── best.pt             # Best trained model weights
    │   └── last.pt              # Last epoch weights
    ├── results.csv              # Per-epoch training metrics
    ├── results.png               # Training curves
    ├── confusion_matrix.png      # Confusion matrix
    └── ...                        # Additional plots & sample batches
```

---

## 📦 Dataset

The full dataset (images + labels for train/val) is too large to host directly on GitHub. Download it from Google Drive:

🔗 **[Download Dataset](PASTE_YOUR_GOOGLE_DRIVE_LINK_HERE)**

After downloading, extract it so the folder structure matches:

```
water_cooler_ai/
└── dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

---

## 🏷️ Classes

The model is trained to detect **2 classes**:

| Class ID | Name |
|---|---|
| 0 | water cooler |
| 1 | water cooler 23 |

---

## 📋 Requirements

- Python 3.8+
- A webcam (for live detection)

Install dependencies:

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `ultralytics` (YOLOv8)
- `opencv-python`
- `numpy`

---

## 🚀 Usage

### 1. Extract frames from a video (for building/expanding your dataset)

Edit `extract_frames.py` to point to your video file, then run:

```bash
python extract_frames.py
```

Extracted frames are saved to the `frames/` directory.

### 2. Train the model

```bash
python train.py
```

This trains a YOLOv8 nano model using `data.yaml` for 50 epochs at 640px image size. Training outputs (weights, metrics, plots) are saved to `runs/detect/train/`.

### 3. Run live detection

```bash
python detect_camera.py
```

This loads the trained weights (`runs/detect/train/weights/best.pt`) and runs real-time detection on your webcam feed. Press **`q`** to quit.

---

## 📊 Training Results

Training was run for 50 epochs on a CPU. Final metrics:

| Metric | Value |
|---|---|
| Precision | ~0.999 |
| Recall | 1.0 |
| mAP50 | 0.995 |
| mAP50-95 | 0.995 |

> See `runs/detect/train/results.png` and `confusion_matrix.png` for visualized training performance.

---

## ⚙️ Configuration

To retrain with different settings, edit `train.py`:

```python
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)
```

To adjust detection confidence threshold, edit `detect_camera.py`:

```python
results = model(frame, conf=0.5)
```

---

## ⚠️ Notes

- The `dataset/` folder is hosted on Google Drive (not included in this repo) — see the **Dataset** section above.
- `runs/` contains generated training artifacts — feel free to `.gitignore` this if you plan to retrain from scratch.
- `device: cpu` was used for training (see `runs/detect/train/args.yaml`); training on GPU will be significantly faster.

---

## 🤝 Contributing

Pull requests, issue reports, and feature suggestions are welcome!

---

## 📄 License

This project is open source. Add your preferred license (e.g. MIT) here.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
