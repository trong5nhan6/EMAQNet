# shap_s5p_patch_explainer.py
"""
Patch-based SHAP approximation for Sentinel-5P branch:
 1. Compute per-patch attribution by masking patches on S5P input
 2. Visualize: original image, heatmap, overlay, and bar plot with cleaned ticks
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from model_single import get_model
from utils_single import load_data
from transforms_single import ChangeBandOrder, Normalize, DatasetStatistics, ToTensor
from dataset_single import NO2PredictionDataset

# ===== CONFIGURATION =====
SAMPLES_FILE    = "./data/multimodal/XAI_5_poll_data.csv"
DATADIR         = "D:/VS_code/AQNet/Datadir/eea"
CHECKPOINT_PATH = (
    "D:/VS_code/AQNet/Modules/SingleOutput/" +
    "results/20250408_1903_30_epochs_no2_SatAndTabData/" +
    "ImageNet_30_epochs.model"
)
USE_TABULAR = True
USE_S5P     = True
DEVICE      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Directory to save outputs
OUTPUT_DIR = "shap_explanations/S5P"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 1) LOAD MODEL =====
model = get_model(DEVICE, "resnet18", USE_TABULAR, USE_S5P, checkpoint=True)
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval().to(DEVICE)

# ===== 2) LOAD ONE SAMPLE =====
samples, stations = load_data(DATADIR, SAMPLES_FILE)
datastats = DatasetStatistics()
tf = transforms.Compose([ChangeBandOrder(), Normalize(datastats), ToTensor()])
dataset = NO2PredictionDataset(DATADIR, samples[3:5], transforms=tf, station_imgs=stations)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
sample = next(iter(dataloader))

# Extract inputs
s2  = sample['img'].float().to(DEVICE)        # (1,12,H_S2,W_S2)
s5p = sample['s5p'].float().unsqueeze(1).to(DEVICE)  # (1,1,H_S5P,W_S5P)
tab = torch.stack([
    sample['Altitude'], sample['rural'], sample['suburban'], sample['urban'], sample['traffic'],
    sample['industrial'], sample['background'], sample['rural-nearcity'], sample['rural-regional'],
    sample['tavg'], sample['prcp'], sample['wspd'], sample['pres'], sample['tsun']
], dim=1).float().to(DEVICE)  # (1,14)

# ===== 3) PATCH-BASED ATTRIBUTION ON S5P =====
PATCH_SIZE = 16
_, _, H, W = s5p.shape
out_h, out_w = H // PATCH_SIZE, W // PATCH_SIZE
attrib_map = np.zeros((out_h, out_w))

# Original prediction
with torch.no_grad():
    y_orig = model({'img': s2, 's5p': s5p, 'tabular': tab}).item()

# Iterate through patches and compute attribution (y_orig - y_masked)
for i in range(out_h):
    for j in range(out_w):
        s5p_masked = s5p.clone()
        y0, y1 = i * PATCH_SIZE, (i + 1) * PATCH_SIZE
        x0, x1 = j * PATCH_SIZE, (j + 1) * PATCH_SIZE
        s5p_masked[:, :, y0:y1, x0:x1] = 0
        with torch.no_grad():
            y_masked = model({'img': s2, 's5p': s5p_masked, 'tabular': tab}).item()
        attrib_map[i, j] = y_orig - y_masked

# Normalize attribution to [0,1]
attrib_norm = (attrib_map - attrib_map.min()) / (attrib_map.max() - attrib_map.min() + 1e-8)

# ===== 4) VISUALIZATION =====
# 4.1 Original S5P image (no colorbar)
s5p_vis = s5p[0,0].cpu().numpy()
plt.figure(figsize=(6,5))
plt.imshow(s5p_vis, cmap='viridis')
plt.title('Original S5P input (viridis)')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 's5p_original_viridis.png'), dpi=300)
plt.close()

# 4.2 Patch-based heatmap
plt.figure(figsize=(6,5))
plt.imshow(attrib_norm, cmap='viridis', interpolation='nearest')
plt.title('Patch-based SHAP heatmap on S5P')
plt.colorbar(label='Attribution')
plt.xlabel('Patch X')
plt.ylabel('Patch Y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 's5p_patch_shap_heatmap.png'), dpi=300)
plt.close()

# 4.3 Overlay of heatmap on original
plt.figure(figsize=(6,5))
plt.imshow(s5p_vis, cmap='gray', alpha=0.8)
plt.imshow(attrib_norm, cmap='jet', alpha=0.5, extent=(0, W, H, 0))
plt.title('Overlay of SHAP heatmap on S5P')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 's5p_patch_overlay.png'), dpi=300)
plt.close()

# ===== 5) BAR PLOT OF PATCH ATTRIBUTIONS =====
flat_attrib = attrib_map.flatten()
patch_idx = np.arange(flat_attrib.shape[0])
# reduce tick labels to avoid overlap: show every Nth index
step = max(1, patch_idx.size // 10)
tick_idx = patch_idx[::step]

plt.figure(figsize=(8,4))
plt.bar(patch_idx, flat_attrib)
plt.xlabel('Patch index')
plt.ylabel('Attribution (Δ output)')
plt.title('Patch-wise attribution for S5P')
plt.xticks(tick_idx)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 's5p_patch_attribution_bar.png'), dpi=300)
plt.close()