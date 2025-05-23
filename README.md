
# Multimodal Air Quality Forecasting with Explainable AI: EMAQNet for Transparent Environmental Policy Support

**EMAQNet** (Environmental Multi-output Air Quality Network) is a deep learning framework designed to forecast air pollutant concentrations at multiple monitoring stations over time. It supports training configurations for single-output (predicting one pollutant), three-output, and five-output scenarios.

## 📁 Project Structure

EMAQNet/  
├── Checkpoints/ # Saved model checkpoints  
├── Data/  
│ ├── data/  
│ ├── station_info/ # Metadata about monitoring stations  
│ └── timeseries_data/ # Time-series input data  
│ ├── EMAQNet.png # Model architecture diagram  
│ └── README_online_data.txt # Additional dataset information  
├── Datadir/ # Optional directory for processed data  
├── Modules/  
│ ├── SingleOutput/ # Code for single-pollutant prediction  
│	│ ├── SHAP/ # SHAP explainability code
│ ├── ThreeOutput/ # Code for 3-pollutant prediction  
│ └── FiveOutput/ # Code for 5-pollutant prediction  
├── Notebooks/  
│ └── weather/ # Weather-related preprocessing notebooks  
├── results/ # Evaluation outputs  
├── requirements.txt # Python dependencies  
└── README.md



## 🛠 Installation Guide

It is recommended to use Anaconda:

```bash
conda create --name aqnet --file requirements.txt 
conda activate aqnet
```

## 🛰️ Satellite Imagery Download

Before running experiments, you must download satellite images:

-   Download **Sentinel-2** and **Sentinel-5P** imagery.
    
-   Source: **[Scheibenreif et al.](https://zenodo.org/records/5764262#.YfJiPS1XYUs)**
    
-   Place the downloaded files inside the `/Datadir` directory.
    

This imagery is essential for multimodal data fusion and spatial modeling.

## 🚀 How to Run the Project

Each model variant has its own training script:

### Single-output model
```bash
python Modules/SingleOutput/training_single.py
```

### Optional SHAP analysis:


```bash
Modules/SingleOutput/SHAP/run_shap_analysis.py` 
```
### Three-output model
```bash
python Modules/ThreeOutput/training_3poll.py
```
### Five-output model
```bash
python Modules/FiveOutput/training_5poll.py
```
