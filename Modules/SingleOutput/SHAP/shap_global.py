# import os
# import torch
# import numpy as np
# import shap
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from model_single import get_model
# from utils_single import load_data
# from transforms_single import ChangeBandOrder, Normalize, DatasetStatistics, ToTensor
# from dataset_single import NO2PredictionDataset
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt

# # ========== CONFIG ==========
# SAMPLES_FILE = "./data/multimodal/5_poll_data_train.csv"
# # SAMPLES_FILE = "./data/multimodal/XAI_5_poll_data.csv"
# DATADIR = "D:/VS_code/AQNet/Datadir/eea"
# CHECKPOINT_PATH = "D:/VS_code/AQNet/Modules/SingleOutput/results/20250408_1903_30_epochs_no2_SatAndTabData/ImageNet_30_epochs.model"
# USE_TABULAR = True
# USE_S5P    = True
# DEVICE     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Directory to save plots
# OUTPUT_DIR = "./shap_explanations"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ========== LOAD MODEL ==========
# model = get_model(DEVICE, "resnet18", USE_TABULAR, USE_S5P, checkpoint=True)
# model.load_state_dict(torch.load(CHECKPOINT_PATH))
# model.eval().to(DEVICE)

# # ========== LOAD DATA (10 samples) ==========
# samples, stations = load_data(DATADIR, SAMPLES_FILE)
# datastats = DatasetStatistics()
# tf = transforms.Compose([ChangeBandOrder(), Normalize(datastats), ToTensor()])
# dataset   = NO2PredictionDataset(DATADIR, list(samples)[:300], transforms=tf, station_imgs=stations)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# # ========== 1) Build meta-features & labels ==========
# meta_list = []
# y_list    = []

# for sample in dataloader:
#     # lấy 3 input như trước
#     s2  = sample["img"].float().to(DEVICE)                     # (1,12,H_S2,W_S2)
#     s5p = sample["s5p"].float().unsqueeze(1).to(DEVICE)        # (1,1,H_S5P,W_S5P)
#     tab = torch.stack([
#         sample["Altitude"], sample["rural"], sample["suburban"], sample["urban"], sample["traffic"],
#         sample["industrial"], sample["background"], sample["rural-nearcity"], sample["rural-regional"],
#         sample["tavg"], sample["prcp"], sample["wspd"], sample["pres"], sample["tsun"]
#     ], dim=1).float().to(DEVICE)                               # (1,14)

#     with torch.no_grad():
#         # tính embedding của mỗi branch rồi lấy mean pooling qua dimension feature
#         feat_s2  = model.backbone_S2(s2).mean(dim=1)           # (1,)
#         feat_s5p = model.backbone_S5P(s5p).mean(dim=1)         # (1,)
#         feat_tab = model.backbone_tabular(tab).mean(dim=1)     # (1,)

#         # prediction gốc
#         y = model({"img": s2, "s5p": s5p, "tabular": tab}).item()

#     # ghép 3 feature thành 1 vector (3,)
#     meta = torch.cat([feat_s2, feat_s5p, feat_tab], dim=0).cpu().numpy()
#     meta_list.append(meta)
#     y_list.append(y)

# meta_X = np.vstack(meta_list)    # shape (n_samples, 3)
# y      = np.array(y_list)        # shape (n_samples,)

# # ========== 2) Fit surrogate model trên 3 meta-feature ==========
# surrogate = LinearRegression()
# surrogate.fit(meta_X, y)

# # ========== 3) SHAP KernelExplainer ==========
# # chọn background (dùng toàn bộ hoặc sample  min(100, n_samples))
# bg = meta_X if meta_X.shape[0] <= 100 else meta_X[np.random.choice(meta_X.shape[0], 100, replace=False)]

# explainer    = shap.KernelExplainer(surrogate.predict, bg)
# shap_values  = explainer.shap_values(meta_X, nsamples=200)

# # ========== 4) Visualize ==========
# feature_names = ["S2_mean","S5P_mean","Tabular_mean"]
# shap.summary_plot(shap_values, meta_X, feature_names=feature_names)
# plt.savefig(os.path.join(OUTPUT_DIR, "summary_plot.png"))
# plt.close()

# shap.summary_plot(shap_values, meta_X, feature_names=feature_names, plot_type="bar")


# shap_meta_feature_explainer_bar.py

import os
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.linear_model import LinearRegression

from model_single import get_model
from utils_single import load_data
from transforms_single import ChangeBandOrder, Normalize, DatasetStatistics, ToTensor
from dataset_single import NO2PredictionDataset

# ========== CONFIGURATION ==========
SAMPLES_FILE    = "./data/multimodal/5_poll_data_train.csv"
DATADIR         = "D:/VS_code/AQNet/Datadir/eea"
CHECKPOINT_PATH = (
    "D:/VS_code/AQNet/Modules/SingleOutput/" +
    "results/20250408_1903_30_epochs_no2_SatAndTabData/" +
    "ImageNet_30_epochs.model"
)
USE_TABULAR = True
USE_S5P     = True
DEVICE      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Directory to save plots
OUTPUT_DIR = "./shap_explanations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 1) LOAD MODEL ==========
model = get_model(DEVICE, "resnet18", USE_TABULAR, USE_S5P, checkpoint=True)
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval().to(DEVICE)

# ========== 2) LOAD DATA ==========
samples, stations = load_data(DATADIR, SAMPLES_FILE)
datastats = DatasetStatistics()
tf = transforms.Compose([ChangeBandOrder(), Normalize(datastats), ToTensor()])
dataset   = NO2PredictionDataset(DATADIR, list(samples)[:300], transforms=tf, station_imgs=stations)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# ========== 3) EXTRACT META-FEATURES AND LABELS ==========
meta_list = []
y_list    = []

for sample in dataloader:
    s2  = sample["img"  ].float().to(DEVICE)                      # (1,12,H_S2,W_S2)
    s5p = sample["s5p" ].float().unsqueeze(1).to(DEVICE)         # (1,1,H_S5P,W_S5P)
    tab = torch.stack([
        sample["Altitude"], sample["rural"], sample["suburban"], sample["urban"], sample["traffic"],
        sample["industrial"], sample["background"], sample["rural-nearcity"], sample["rural-regional"],
        sample["tavg"], sample["prcp"], sample["wspd"], sample["pres"], sample["tsun"]
    ], dim=1).float().to(DEVICE)                                  # (1,14)

    with torch.no_grad():
        emb_s2  = model.backbone_S2(s2)       # (1, feature_dim)
        emb_s5p = model.backbone_S5P(s5p)     # (1, feature_dim)
        emb_tab = model.backbone_tabular(tab) # (1, feature_dim)

        feat_s2  = emb_s2.mean(dim=1)   # (1,)
        feat_s5p = emb_s5p.mean(dim=1)  # (1,)
        feat_tab = emb_tab.mean(dim=1)  # (1,)

        y = model({"img": s2, "s5p": s5p, "tabular": tab}).item()

    meta = torch.cat([feat_s2, feat_s5p, feat_tab], dim=0).cpu().numpy()
    meta_list.append(meta)
    y_list.append(y)

meta_X = np.vstack(meta_list)  # (n_samples, 3)
y      = np.array(y_list)      # (n_samples,)

# ========== 4) TRAIN SURROGATE MODEL ==========
surrogate = LinearRegression()
surrogate.fit(meta_X, y)

# ========== 5) SHAP EXPLANATION ==========
bg = meta_X if meta_X.shape[0] <= 100 else meta_X[np.random.choice(meta_X.shape[0], 100, replace=False)]
explainer   = shap.KernelExplainer(surrogate.predict, bg)
shap_values = explainer.shap_values(meta_X, nsamples=200)
feature_names = ["S2_mean", "S5P_mean", "Tabular_mean"]

# ========== 6) PLOT AND SAVE SUMMARY PLOT ==========
plt.figure()
shap.summary_plot(
    shap_values,
    meta_X,
    feature_names=feature_names,
    show=False
)
plt.savefig(os.path.join(OUTPUT_DIR, "global_summary_plot.pdf"))
plt.close()

# ========== 7) PLOT AND SAVE BAR PLOT ==========
plt.figure()
shap.summary_plot(
    shap_values,
    meta_X,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
plt.savefig(os.path.join(OUTPUT_DIR, "global_bar_plot.pdf"))
plt.close()

# # ========== 5) TÍNH SHAP VỚI KERNELEXPLAINER ==========
# # Chọn background (max 100 mẫu)
# bg = meta_X if meta_X.shape[0] <= 100 else meta_X[np.random.choice(meta_X.shape[0], 100, replace=False)]

# explainer   = shap.KernelExplainer(surrogate.predict, bg)
# shap_values = explainer.shap_values(meta_X, nsamples=200)

# feature_names = ["S2_mean", "S5P_mean", "Tabular_mean"]

# # ========== 6) VẼ SUMMARY PLOT ==========
# shap.summary_plot(
#     shap_values, 
#     meta_X, 
#     feature_names=feature_names, 
#     show=True
# )

# # ========== 7) VẼ BAR PLOT CHO MEAN |SHAP| ==========
# mean_abs_shap = np.abs(shap_values).mean(axis=0)  # (3,)

# plt.figure(figsize=(6,4))
# plt.bar(feature_names, mean_abs_shap)
# plt.xlabel('Feature')
# plt.ylabel('Mean |SHAP value|')
# plt.title('Feature importance based on SHAP')
# plt.xticks(rotation=30, ha='right')
# plt.tight_layout()
# plt.show()
