# Rooftop Outline Detection with U-Net

This project trains a U-Net-based deep learning model to detect rooftops in top-down satellite images and extract their edge outlines. It uses a modified U-Net architecture, binary segmentation masks, and post-processing with OpenCV to generate clean, visual rooftop outlines from raw imagery.

## Features

- U-Net architecture trained on SpaceNet 2 (Paris AOI)
- BCEWithLogits + Dice Loss for accurate binary segmentation
- Data augmentation (flip, rotate, color jitter)
- OpenCV-based contour extraction from predicted masks
- Final model achieves **IoU > 0.90**
- Output options: binary masks, rooftop outlines

## Files

- `rooftop-outline-detection-with-u-net.ipynb` — Full training and inference pipeline
- `unet_rooftop.pth` — Trained model weights
- `requirements.txt` — Dependency list
- `README.md` — Project overview (this file)

## Training Pipeline

1. Load SpaceNet imagery + mask pairs
2. Resize and augment input images
3. Train U-Net using BCEWithLogits + Dice loss
4. Monitor IoU during training
5. Save model and visualize results

## Results

- Trained over 100 epochs
- Achieved **IoU ~ 0.91** on training set
- Visual output includes: input image, ground truth mask, predicted mask, and rooftop outline

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```python
# In a new script or notebook
from unet import UNet
model = UNet()
model.load_state_dict(torch.load("unet_rooftop.pth"))
model.eval()
```

For full training and prediction workflow, run the notebook:
```
rooftop-outline-detection-with-u-net.ipynb
```


## License

MIT

## ✍Author

Jewon Yeon – yeon971105@icloud.com
