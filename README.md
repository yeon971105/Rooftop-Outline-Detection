# Rooftop Outline Detection with U-Net

This project trains a deep learning model to detect rooftops in satellite images and extract their edge outlines. It uses the SpaceNet 2 dataset and a custom U-Net implementation to produce segmentation masks, which are converted into vector-style rooftop outlines.

## Key Features
- U-Net architecture for semantic segmentation
- Combined **BCEWithLogits + Dice Loss** for stable training
- Data augmentation: rotation, flip, color jitter
- Post-processing: OpenCV contour extraction for rooftop outlines
- Achieved **IoU > 90%** on the training set

---

## 📁 Project Structure (for GitHub)

```
rooftop-outline-detection/
├── data/
│   ├── images/           # PNG input images (from SpaceNet tiles)
│   └── masks/            # PNG binary masks
├── notebooks/
│   └── training.ipynb    # Kaggle-compatible training notebook
├── models/
│   └── unet.py           # U-Net architecture
├── utils/
│   ├── loss.py           # Dice + BCE loss
│   ├── metrics.py        # IoU computation
│   └── postprocess.py    # Contour extraction, SVG export
├── scripts/
│   └── predict.py        # Run prediction on new images
├── checkpoints/
│   └── unet_rooftop.pth  # Trained model weights
├── README.md             # Project overview
└── requirements.txt      # Dependencies
```

---

## Training Pipeline
1. Load and preprocess SpaceNet dataset (Paris AOI)
2. Apply data augmentation (rotation, flip, jitter)
3. Train U-Net using BCEWithLogitsLoss + DiceLoss
4. Track training performance with IoU
5. Save best model as `.pth`

---

## Output Examples
- **Input**: Satellite tile from SpaceNet 2
- **Predicted Mask**: Binary rooftop regions
- **Outlined Image**: Contours drawn on the image

---

## Setup
```bash
# Create environment
pip install -r requirements.txt

# Run training (Jupyter recommended)
python notebooks/training.ipynb

# Predict on new image
python scripts/predict.py --image path/to/image.png --weights checkpoints/unet_rooftop.pth
```

---

## Evaluation
- IoU on training set: **~0.90+**
- Visual inspection: clear rooftop outlines

---

##  Author
Jewon Yeon – yeon971105@icloud.com

---

Happy to help you polish this more for GitHub or portfolio if you like!
