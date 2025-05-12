# # shap_s2_patch_explainer.py

# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# from torchvision import transforms

# from model_single import get_model
# from utils_single import load_data
# from transforms_single import ChangeBandOrder, Normalize, DatasetStatistics, ToTensor
# from dataset_single import NO2PredictionDataset

# # ========== CONFIGURATION ==========
# SAMPLES_FILE    = "./data/multimodal/XAI_5_poll_data.csv"
# DATADIR         = "D:/VS_code/AQNet/Datadir/eea"
# CHECKPOINT_PATH = (
#     "D:/VS_code/AQNet/Modules/SingleOutput/" +
#     "results/20250408_1903_30_epochs_no2_SatAndTabData/" +
#     "ImageNet_30_epochs.model"
# )
# USE_TABULAR = True
# USE_S5P     = True
# DEVICE      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Directory for outputs
# OUTPUT_DIR = "shap_explanations/S2_patch"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ========== 1) LOAD MODEL ==========
# model = get_model(DEVICE, "resnet18", USE_TABULAR, USE_S5P, checkpoint=True)
# model.load_state_dict(torch.load(CHECKPOINT_PATH))
# model.eval().to(DEVICE)

# # ========== 2) LOAD ONE SAMPLE ==========
# samples, stations = load_data(DATADIR, SAMPLES_FILE)
# datastats = DatasetStatistics()
# tf = transforms.Compose([ChangeBandOrder(), Normalize(datastats), ToTensor()])
# dataset = NO2PredictionDataset(DATADIR, samples[:1], transforms=tf, station_imgs=stations)
# dataloader = DataLoader(dataset, batch_size=1)

# sample = next(iter(dataloader))
# s2  = sample['img'].float().to(DEVICE)   # (1,12,H,W)
# s5p = sample['s5p'].float().unsqueeze(1).to(DEVICE) # (1,1,H,W)
# tab = torch.stack([
#     sample['Altitude'], sample['rural'], sample['suburban'], sample['urban'], sample['traffic'],
#     sample['industrial'], sample['background'], sample['rural-nearcity'], sample['rural-regional'],
#     sample['tavg'], sample['prcp'], sample['wspd'], sample['pres'], sample['tsun']
# ], dim=1).float().to(DEVICE) # (1,14)

# # ========== 3) PATCH-BASED SHAP APPROXIMATION ==========
# PATCH_SIZE = 16  # adjust
# _, C, H, W = s2.shape
# out_h, out_w = H // PATCH_SIZE, W // PATCH_SIZE
# shap_map = np.zeros((out_h, out_w))

# # Original prediction
# with torch.no_grad():
#     y_orig = model({'img': s2, 's5p': s5p, 'tabular': tab}).item()

# # Iterate patches
# for i in range(out_h):
#     for j in range(out_w):
#         s2_masked = s2.clone()
#         s2_masked[:, :,
#                   i*PATCH_SIZE:(i+1)*PATCH_SIZE,
#                   j*PATCH_SIZE:(j+1)*PATCH_SIZE] = 0
#         with torch.no_grad():
#             y_masked = model({'img': s2_masked, 's5p': s5p, 'tabular': tab}).item()
#         shap_map[i, j] = y_orig - y_masked

# # Normalize shap_map to [0,1]
# shap_map_norm = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min())

# # ========== 4) VISUALIZATION ==========
# # Plot heatmap
# plt.figure(figsize=(6,5))
# plt.imshow(shap_map_norm, cmap='viridis', interpolation='nearest')
# plt.title('Approximate SHAP heatmap on S2')
# plt.colorbar(label='Attribution')
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, 's2_patch_shap_heatmap.png'), dpi=300)
# plt.close()

# # Overlay on one channel (e.g., channel 0)
# s2_vis = s2[0,0].cpu().numpy()
# plt.figure(figsize=(6,5))
# plt.imshow(s2_vis, cmap='gray', alpha=0.8)
# plt.imshow(shap_map_norm, cmap='jet', alpha=0.5, extent=(0, W, H, 0))
# plt.title('Overlay of SHAP heatmap on S2 channel 0')
# plt.axis('off')
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, 's2_patch_overlay.png'), dpi=300)
# plt.close()





# # shap_s2_channel_explainer.py

# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# from torchvision import transforms

# from model_single import get_model
# from utils_single import load_data
# from transforms_single import ChangeBandOrder, Normalize, DatasetStatistics, ToTensor
# from dataset_single import NO2PredictionDataset

# # ========== CONFIGURATION ==========
# SAMPLES_FILE    = "./data/multimodal/XAI_5_poll_data.csv"
# DATADIR         = "D:/VS_code/AQNet/Datadir/eea"
# CHECKPOINT_PATH = (
#     "D:/VS_code/AQNet/Modules/SingleOutput/" +
#     "results/20250408_1903_30_epochs_no2_SatAndTabData/" +
#     "ImageNet_30_epochs.model"
# )
# USE_TABULAR = True
# USE_S5P     = True
# DEVICE      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Directory to save outputs
# OUTPUT_DIR = "shap_explanations/S2_channel"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ========== 1) LOAD MODEL ==========
# model = get_model(DEVICE, "resnet18", USE_TABULAR, USE_S5P, checkpoint=True)
# model.load_state_dict(torch.load(CHECKPOINT_PATH))
# model.eval().to(DEVICE)

# # ========== 2) LOAD ONE SAMPLE ==========
# samples, stations = load_data(DATADIR, SAMPLES_FILE)
# datastats = DatasetStatistics()
# transforms_pipeline = transforms.Compose([
#     ChangeBandOrder(), Normalize(datastats), ToTensor()
# ])
# dataset = NO2PredictionDataset(DATADIR, samples[:1], transforms=transforms_pipeline, station_imgs=stations)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# sample = next(iter(dataloader))
# # S2: (1,12,H,W)
# s2  = sample['img'].float().to(DEVICE)
# # S5P: (1,1,H,W)
# s5p = sample['s5p'].float().unsqueeze(1).to(DEVICE)
# # Tabular: (1,14)
# tab = torch.stack([
#     sample['Altitude'], sample['rural'], sample['suburban'], sample['urban'], sample['traffic'],
#     sample['industrial'], sample['background'], sample['rural-nearcity'], sample['rural-regional'],
#     sample['tavg'], sample['prcp'], sample['wspd'], sample['pres'], sample['tsun']
# ], dim=1).float().to(DEVICE)

# # ========== 3) PLOT 12 CHANNELS ==========
# # Extract channel data
# channels = s2[0].cpu().numpy()  # shape (12,H,W)
# num_ch, H, W = channels.shape
# # Plot in a 3x4 grid
# grid_rows, grid_cols = 3, 4
# fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(12, 9))
# for idx in range(num_ch):
#     r, c = divmod(idx, grid_cols)
#     ax = axs[r, c]
#     im = ax.imshow(channels[idx], cmap='gray')
#     ax.set_title(f"Channel {idx}")
#     ax.axis('off')
# # Hide unused subplots if any
# for idx in range(num_ch, grid_rows*grid_cols):
#     axs.flatten()[idx].axis('off')
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, 's2_channels_grid.png'), dpi=300)
# plt.close()

# # ========== 4) CHANNEL-WISE SHAP APPROXIMATION ==========
# # Original prediction
# y_orig = model({'img': s2, 's5p': s5p, 'tabular': tab}).item()
# # Compute attribution per channel
# attrib = []
# for ch in range(num_ch):
#     s2_masked = s2.clone()
#     s2_masked[:, ch, :, :] = 0
#     y_masked = model({'img': s2_masked, 's5p': s5p, 'tabular': tab}).item()
#     attrib.append(y_orig - y_masked)
# attrib = np.array(attrib)

# # ========== 5) PLOT BAR CHART ==========
# plt.figure(figsize=(8,4))
# channels_idx = np.arange(num_ch)
# plt.bar(channels_idx, attrib)
# plt.xlabel('Channel')
# plt.ylabel('Attribution (Δ output)')
# plt.title('Channel-wise SHAP approximation for S2')
# plt.xticks(channels_idx, [f"Ch{c}" for c in channels_idx], rotation=45)
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, 's2_channel_attribution_bar.png'), dpi=300)
# plt.close()




# shap_s2_channel_explainer.py

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

# ========== CONFIGURATION ==========
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
OUTPUT_DIR = "shap_explanations/S2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 1) LOAD MODEL ==========
model = get_model(DEVICE, "resnet18", USE_TABULAR, USE_S5P, checkpoint=True)
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval().to(DEVICE)

# ========== 2) LOAD ONE SAMPLE ==========
samples, stations = load_data(DATADIR, SAMPLES_FILE)
datastats = DatasetStatistics()
transforms_pipeline = transforms.Compose([
    ChangeBandOrder(), Normalize(datastats), ToTensor()
])
dataset = NO2PredictionDataset(DATADIR, samples[5:10], transforms=transforms_pipeline, station_imgs=stations)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

sample = next(iter(dataloader))
# S2: (1,12,H,W)
s2  = sample['img'].float().to(DEVICE)
# S5P: (1,1,H,W)
s5p = sample['s5p'].float().unsqueeze(1).to(DEVICE)
# Tabular: (1,14)
tab = torch.stack([
    sample['Altitude'], sample['rural'], sample['suburban'], sample['urban'], sample['traffic'],
    sample['industrial'], sample['background'], sample['rural-nearcity'], sample['rural-regional'],
    sample['tavg'], sample['prcp'], sample['wspd'], sample['pres'], sample['tsun']
], dim=1).float().to(DEVICE)

# ========== 3) PLOT 12 CHANNELS ==========
# Extract channel data
channels = s2[0].cpu().numpy()  # shape (12,H,W)
num_ch, H, W = channels.shape
# Plot in a 3x4 grid
grid_rows, grid_cols = 3, 4
fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(12, 9))
for idx in range(num_ch):
    r, c = divmod(idx, grid_cols)
    ax = axs[r, c]
    im = ax.imshow(channels[idx], cmap='gray')
    ax.set_title(f"Channel {idx}")
    ax.axis('off')
# Hide unused subplots if any
for idx in range(num_ch, grid_rows*grid_cols):
    axs.flatten()[idx].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 's2_channels_grid.png'), dpi=300)
plt.close()

# ========== 4) CHANNEL-WISE SHAP APPROXIMATION ==========
# Original prediction
y_orig = model({'img': s2, 's5p': s5p, 'tabular': tab}).item()
# Compute attribution per channel
attrib = []
for ch in range(num_ch):
    s2_masked = s2.clone()
    s2_masked[:, ch, :, :] = 0
    y_masked = model({'img': s2_masked, 's5p': s5p, 'tabular': tab}).item()
    attrib.append(y_orig - y_masked)
attrib = np.array(attrib)

# ========== 5) PLOT BAR CHART ==========
plt.figure(figsize=(8,4))
channels_idx = np.arange(num_ch)
plt.bar(channels_idx, attrib)
plt.xlabel('Channel')
plt.ylabel('Attribution (Δ output)')
plt.title('Channel-wise SHAP approximation for S2')
plt.xticks(channels_idx, [f"Ch{c}" for c in channels_idx], rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 's2_channel_attribution_bar.png'), dpi=300)
plt.close()

# ========== 6) PLOT CHANNEL-LEVEL HEATMAP ==========
# Arrange channel attributions into a 3x4 grid for visualization
grid_rows, grid_cols = 3, 4
attrib_grid = attrib.reshape((grid_rows, grid_cols))
plt.figure(figsize=(6,5))
plt.imshow(attrib_grid, cmap='viridis', interpolation='nearest')
plt.title('Channel-wise attribution heatmap for S2')
plt.colorbar(label='Attribution')
# annotate channel number in each cell
for i in range(grid_rows):
    for j in range(grid_cols):
        ch = i * grid_cols + j
        plt.text(j, i, f"Ch{ch}", ha='center', va='center', color='white', fontsize=8)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 's2_channel_attribution_heatmap.png'), dpi=300)
plt.close()
