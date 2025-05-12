# shap_tabular_explainer.py

import os
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

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
OUTPUT_DIR = "./shap_explanations/tabular"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 1) LOAD MODEL ==========
model = get_model(DEVICE, "resnet18", USE_TABULAR, USE_S5P, checkpoint=True)
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval().to(DEVICE)

# ========== 2) LOAD DATA (FIRST 10 SAMPLES) ==========
samples, stations = load_data(DATADIR, SAMPLES_FILE)
datastats = DatasetStatistics()
transforms_pipeline = transforms.Compose([
    ChangeBandOrder(),
    Normalize(datastats),
    ToTensor()
])
# Limit to first 10 samples for quick test
subset_samples = list(samples)[:300]
dataset = NO2PredictionDataset(DATADIR, subset_samples, transforms=transforms_pipeline, station_imgs=stations)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# ========== 3) COMPUTE MEAN-IMAGE BASELINE & BUILD TABULAR INPUTS ==========
sum_s2, sum_s5p = None, None
X_tab_list = []
count = 0

for sample in dataloader:
    # Image inputs: DataLoader batch dim included
    # s2 shape: (1, 12, H, W)
    s2  = sample["img"].float().to(DEVICE)
    # s5p shape: (1, 1, H, W)
    s5p = sample["s5p"].float().to(DEVICE)

    # Tabular inputs: shape (1,14)
    tab = torch.stack([
        sample["Altitude"], sample["rural"], sample["suburban"], sample["urban"], sample["traffic"],
        sample["industrial"], sample["background"], sample["rural-nearcity"], sample["rural-regional"],
        sample["tavg"], sample["prcp"], sample["wspd"], sample["pres"], sample["tsun"]
    ], dim=1).float().to(DEVICE)

    # Accumulate for mean baseline
    if sum_s2 is None:
        sum_s2  = torch.zeros_like(s2)
        sum_s5p = torch.zeros_like(s5p)
    sum_s2  += s2
    sum_s5p += s5p

    # Collect tabular data
    X_tab_list.append(tab.cpu().numpy())
    count += 1

# Compute mean baseline images
mean_s2  = sum_s2  / count  # shape: (1,12,H,W)
mean_s5p = sum_s5p / count   # shape: (1,1,H,W)

# Build tabular matrix: (10,14)
X_tab = np.vstack(X_tab_list)

# ========== 4) DEFINE PREDICT FUNCTION USING MEAN-IMAGE BASELINE ==========
def predict_tabular(x_numpy):
    """
    x_numpy: numpy array shape (n_samples,14)
    Returns: numpy array shape (n_samples,) of model predictions
    """
    x_tensor = torch.from_numpy(x_numpy).float().to(DEVICE)
    batch    = x_tensor.size(0)
    # repeat mean-image baseline to match batch
    s2_batch  = mean_s2.repeat(batch, 1, 1, 1)
    s5p_batch = mean_s5p.repeat(batch, 1, 1, 1)
    with torch.no_grad():
        preds = model({"img": s2_batch, "s5p": s5p_batch, "tabular": x_tensor})
    return preds.cpu().numpy().ravel()

# ========== 5) SHAP EXPLANATION FOR TABULAR MODEL ==========
# Choose small background for speed
bg_tab = X_tab[np.random.choice(X_tab.shape[0], min(5, X_tab.shape[0]), replace=False)]
explainer_tab = shap.KernelExplainer(predict_tabular, bg_tab)
shap_vals_tab = explainer_tab.shap_values(X_tab, nsamples=50)

feature_names_tab = [
    "Altitude", "rural", "suburban", "urban", "traffic",
    "industrial", "background", "rural-nearcity", "rural-regional",
    "tavg", "prcp", "wspd", "pres", "tsun"
]

# ========== 6) PLOT AND SAVE SUMMARY PLOT ==========
plt.figure()
shap.summary_plot(
    shap_vals_tab,
    X_tab,
    feature_names=feature_names_tab,
    show=False
)
plt.savefig(os.path.join(OUTPUT_DIR, "tabular_summary.pdf"))
plt.savefig(os.path.join(OUTPUT_DIR, "tabular_summary.png"))
plt.close()

# ========== 7) PLOT AND SAVE BAR PLOT ==========
plt.figure()
shap.summary_plot(
    shap_vals_tab,
    X_tab,
    feature_names=feature_names_tab,
    plot_type="bar",
    show=False
)
plt.savefig(os.path.join(OUTPUT_DIR, "tabular_bar.pdf"))
plt.savefig(os.path.join(OUTPUT_DIR, "tabular_bar.png"))
plt.close()





# shap_tabular_explainer.py

# import os
# import torch
# import numpy as np
# import shap
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

# # Directory to save plots
# OUTPUT_DIR = "shap_explanations/tabular"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ========== 1) LOAD MODEL ==========
# model = get_model(DEVICE, "resnet18", USE_TABULAR, USE_S5P, checkpoint=True)
# model.load_state_dict(torch.load(CHECKPOINT_PATH))
# model.eval().to(DEVICE)

# # ========== 2) LOAD DATA (FIRST 10 SAMPLES FOR QUICK TEST) ==========
# samples, stations = load_data(DATADIR, SAMPLES_FILE)
# datastats = DatasetStatistics()
# tf = transforms.Compose([ChangeBandOrder(), Normalize(datastats), ToTensor()])
# # Limit to first 10 samples to reduce memory
# subset_samples = list(samples)[:10]
# dataset   = NO2PredictionDataset(DATADIR, subset_samples, transforms=tf, station_imgs=stations)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

# # ========== 3) BUILD TABULAR INPUTS AND DEFINE PREDICT FUNCTION ==========
# X_tab_list = []
# y_list     = []

# for sample in dataloader:
#     tab = torch.stack([
#         sample["Altitude"], sample["rural"], sample["suburban"], sample["urban"], sample["traffic"],
#         sample["industrial"], sample["background"], sample["rural-nearcity"], sample["rural-regional"],
#         sample["tavg"], sample["prcp"], sample["wspd"], sample["pres"], sample["tsun"]
#     ], dim=1).float().to(DEVICE)

#     zeros_s2  = torch.zeros(tab.size(0), 12, 224, 224, device=DEVICE)
#     zeros_s5p = torch.zeros(tab.size(0), 1, 128, 128, device=DEVICE)

#     # zeros_s2  = first_sample["img"] .float().to(DEVICE)        # (1,12,H_S2,W_S2)
#     # zeros_s5p = first_sample["s5p"].float().unsqueeze(1).to(DEVICE)  # (1,1,H_S5P,W_S5P)

#     with torch.no_grad():
#         preds = model({"img": zeros_s2, "s5p": zeros_s5p, "tabular": tab})

#     X_tab_list.append(tab.cpu().numpy())
#     y_list.append(preds.cpu().numpy())

# # Stack only first 10 samples
# X_tab = np.vstack(X_tab_list)[:10]
# y_tab = np.concatenate(y_list)[:10]

# # Predictor for SHAP
# def predict_tabular(x_numpy):
#     x_tensor = torch.from_numpy(x_numpy).float().to(DEVICE)
#     zeros_s2  = torch.zeros(x_tensor.size(0), 12, 224, 224, device=DEVICE)
#     zeros_s5p = torch.zeros(x_tensor.size(0), 1, 128, 128, device=DEVICE)
#     with torch.no_grad():
#         preds = model({"img": zeros_s2, "s5p": zeros_s5p, "tabular": x_tensor})
#     return preds.cpu().numpy().ravel()

# # ========== 4) SHAP EXPLAINER FOR TABULAR MODEL ==========
# bg_tab = X_tab[np.random.choice(X_tab.shape[0], min(5, X_tab.shape[0]), replace=False)]
# explainer_tab = shap.KernelExplainer(predict_tabular, bg_tab)
# shap_vals_tab = explainer_tab.shap_values(X_tab, nsamples=50)
# feature_names_tab = [
#     "Altitude", "rural", "suburban", "urban", "traffic",
#     "industrial", "background", "rural-nearcity", "rural-regional",
#     "tavg", "prcp", "wspd", "pres", "tsun"
# ]

# # ========== 5) PLOT AND SAVE SUMMARY ==========
# plt.figure()
# shap.summary_plot(
#     shap_vals_tab,
#     X_tab,
#     feature_names=feature_names_tab,
#     show=True
# )
# # plt.savefig(os.path.join(OUTPUT_DIR, "tabular_summary.png"), dpi=300)
# # plt.close()

# # ========== 6) PLOT AND SAVE BAR PLOT ==========
# plt.figure()
# shap.summary_plot(
#     shap_vals_tab,
#     X_tab,
#     feature_names=feature_names_tab,
#     plot_type="bar",
#     show=True
# )
# # plt.savefig(os.path.join(OUTPUT_DIR, "tabular_bar.png"), dpi=300)
# # plt.close()
