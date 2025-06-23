# -*- coding: utf-8 -*-
"""roof_seg-Copy2.py

Clean Python module for roof segmentation.
"""

# Cell 2 — Imports
import os, random
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

# ──────────────────────────────────────────────────────────────────────────────
IMG_DIR     = 'images_combined'
MASK_DIR    = 'masks_combined'
BACKBONE    = 'efficientnet-b7'    # try efficientnet-b5/-b7 or 'swin_tiny'
IMG_SIZE    = 256
BATCH_SIZE  = 64                   # on an A100 you can often go to 128+
LR          = 6e-4                  # if you double batch, try doubling LR
EPOCHS      = 50
VAL_SPLIT   = 0.2
PATIENCE    = 5
CHECKPOINT  = 'best_model_focaldice.pth'
FULL_IMG    = 'full_satellite1.png' # for inference
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

# Cell 3 — Dataset + Transforms
class RoofDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size, tf=None):
        self.imgs = sorted(glob(f"{img_dir}/*"))
        self.msks = sorted(glob(f"{mask_dir}/*"))
        self.tf   = tf
        self.sz   = img_size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = Image.open(self.imgs[i]).convert('RGB')
        msk = Image.open(self.msks[i]).convert('L')

        # image transform
        if self.tf:
            img = self.tf(img)
        else:
            img = transforms.Resize((self.sz,self.sz))(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize((0.485,0.456,0.406),
                                       (0.229,0.224,0.225))(img)

        # mask: always resize + binarize
        msk = transforms.Resize((self.sz,self.sz))(msk)
        msk = transforms.ToTensor()(msk)
        msk = (msk > 0.5).float()

        return img, msk

def get_transforms(sz):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(sz, scale=(0.8,1.0), ratio=(0.75,1.333)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2),
        transforms.Resize((sz,sz)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),
                             (0.229,0.224,0.225)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((sz,sz)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),
                             (0.229,0.224,0.225)),
    ])
    return train_tf, val_tf

# Cell 4 — Train / Validate loops
def train_epoch(model, loader, loss_fn, opt, device):
    model.train()
    total_loss = 0.0
    for x,y in tqdm(loader, desc='Train', leave=False):
        x,y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    for x,y in tqdm(loader, desc='Val  ', leave=False):
        x,y = x.to(device), y.to(device)
        total_loss += loss_fn(model(x), y).item()
    return total_loss / len(loader)

# Cell 5 — Build model, optimizer, scheduler, DataLoaders
def setup_training():
    print("Running on", DEVICE)
    device = torch.device(DEVICE)

    # data
    tr_tf, vl_tf = get_transforms(IMG_SIZE)
    full_ds = RoofDataset(IMG_DIR, MASK_DIR, IMG_SIZE, tf=None)
    n_val   = int(len(full_ds) * VAL_SPLIT)
    n_trn   = len(full_ds) - n_val
    tr_ds, vl_ds = random_split(full_ds, [n_trn,n_val],
                                 generator=torch.Generator().manual_seed(42))

    # assign transforms
    tr_ds.dataset.tf = tr_tf
    vl_ds.dataset.tf = vl_tf

    train_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # model
    model = smp.Unet(
        encoder_name    = BACKBONE,
        encoder_weights = 'imagenet',
        in_channels     = 3,
        classes         = 1,
    ).to(device)

    # focal + dice
    focal = FocalLoss(mode='binary', alpha=0.25, gamma=2.0)
    dice  = DiceLoss(mode='binary')
    def loss_fn(p, t):
        return focal(p, t) + dice(p, t)

    opt   = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    
    return model, train_loader, val_loader, loss_fn, opt, sched, device

# Cell 6 — Training loop with checkpointing
def train_model():
    model, train_loader, val_loader, loss_fn, opt, sched, device = setup_training()
    
    best_val = float('inf')
    patience_ctr = 0

    for epoch in range(1, EPOCHS+1):
        tr_loss = train_epoch(model, train_loader, loss_fn, opt, device)
        vl_loss = validate(model,   val_loader,   loss_fn, device)
        sched.step()

        print(f"Epoch {epoch}/{EPOCHS} → train: {tr_loss:.4f} | val: {vl_loss:.4f}", end='')
        if vl_loss < best_val - 1e-4:
            best_val = vl_loss
            torch.save(model.state_dict(), CHECKPOINT)
            patience_ctr = 0
            print("  🏆 saved", CHECKPOINT)
        else:
            patience_ctr += 1
            print(f"  (no improve: {patience_ctr}/{PATIENCE})")
            if patience_ctr >= PATIENCE:
                print("  ⏹ Early stopping.")
                break

# Cell 7 — Final Evaluation (Dice & IoU)
def evaluate_model():
    device = torch.device(DEVICE)
    model, train_loader, val_loader, loss_fn, opt, sched, device = setup_training()
    
    # reload best
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    total_dice, total_iou = 0.0, 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Eval"):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            probs  = torch.sigmoid(logits)
            preds  = (probs > 0.5).float()

            p = preds.view(preds.size(0), -1)
            t = masks.view(masks.size(0), -1)

            inter = (p * t).sum(dim=1)
            union = p.sum(dim=1) + t.sum(dim=1)

            dice = (2*inter + 1e-6) / (union + 1e-6)
            iou  = (inter + 1e-6) / (union - inter + 1e-6)

            total_dice += dice.sum().item()
            total_iou  += iou.sum().item()

    n = len(val_loader.dataset)
    print(f"\n📊 Val Dice: {total_dice/n:.4f} | Val IoU: {total_iou/n:.4f}")

# Cell 8 — Inference & overlay
def run_inference():
    device = torch.device(DEVICE)
    model, train_loader, val_loader, loss_fn, opt, sched, device = setup_training()
    
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    # load & preprocess
    im = Image.open(FULL_IMG).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    x = tf(im).unsqueeze(0).to(device)

    # predict
    with torch.no_grad():
        p = torch.sigmoid(model(x))[0,0].cpu().numpy()
    mask    = (p > 0.5).astype(np.uint8)
    overlay = np.array(im.resize((IMG_SIZE,IMG_SIZE))).copy()
    overlay[mask==1] = (overlay[mask==1]*0.4 + np.array([255,0,0])*0.6).astype(np.uint8)

    # show
    fig,axs = plt.subplots(1,3,figsize=(15,5))
    axs[0].imshow(im);        axs[0].set_title("Input");   axs[0].axis('off')
    axs[1].imshow(mask, 'gray'); axs[1].set_title("Mask");    axs[1].axis('off')
    axs[2].imshow(overlay);    axs[2].set_title("Overlay"); axs[2].axis('off')
    plt.tight_layout()
    plt.show()

def get_model():
    """Create and return the model with the same architecture as training"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = smp.Unet(
        encoder_name    = BACKBONE,
        encoder_weights = None,  # Don't load ImageNet weights for inference
        in_channels     = 3,
        classes         = 1,
    ).to(device)
    return model

def get_transform(img_size):
    """Return the validation transform used during training"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

# Main execution
if __name__ == "__main__":
    # Uncomment the function you want to run:
    # train_model()
    # evaluate_model()
    # run_inference()
    pass