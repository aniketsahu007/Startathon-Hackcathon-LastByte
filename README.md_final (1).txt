# 📌 Offroad Semantic Scene Segmentation — Hackathon Submission  
**Team Name:** Last Byte  


## 👥 Team Members
- **Aniket Sahu** – Model Training  
- **Gulshan Nijammudin Shaikh Mansuri** – Model Training  
- **Mohd. Wasil Azmi** – Designer  
- **Disha Dashore** – Documentation  

---

## 📌 Project Overview

This repository contains our end-to-end pipeline for **multi-class semantic segmentation** on synthetic desert scenes.  
The objective is to assign **every pixel** in an image to one of **six semantic classes**.

Within a ~10-hour hackathon timeframe, we focused on building a **simple, stable, and reproducible baseline** including:

- Dataset preprocessing  
- Model training using a U-Net architecture  
- Evaluation using Mean Intersection over Union (mIoU)  
- Visualization of predictions  

The emphasis was on creating a complete workflow rather than extensive hyperparameter tuning.

---

## 📂 Dataset

The Segmentation Track dataset contains:

- `train/` – RGB training images and masks  
- `val/` – validation images and masks  
- `testImages/` – unseen RGB images for final evaluation  

### 🔄 Label Mapping

All masks were remapped to a continuous range **[0–5]** to match the six model output channels:

| Original Label | Remapped Label |
|---------------|----------------|
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| 27 | 4 |
| 39 | 5 |
 **Important:** Test images were never used during training.

---

ENVIRONMENT SETUP
--------------------
To run this project, you must set up the 'EDU' environment as per the guidelines.

Option A: Windows (Recommended)
1. Open Anaconda Prompt.
2. Run the provided setup script:
   > setup_env.bat [cite: 139, 140]

Option B: Linux/Mac
1. Open a terminal.
2. Run the following commands:
   > conda create -n EDU python=3.10 -y
   > conda activate EDU
   > pip install torch torchvision opencv-python streamlit scikit-learn matplotlib seaborn

4. HOW TO RUN
-------------
Ensure the 'EDU' environment is activated:
> conda activate EDU [cite: 152]

To Launch the Web Application:
> streamlit run app.py
- Use the sidebar to navigate to 'Analyze Terrain'.
- Upload an image to see the AI Terrain Map and Overlay.

To Run Performance Evaluation:
> python evaluate.py

5. DATASET MAPPING
------------------
The model maps complex dataset IDs to 6 simplified classes for navigation:
ID 0: Background
ID 1: Walkable Path
ID 2: Rocky Zone
ID 3: Vegetation
ID 4: Rough Terrain
ID 5: Obstacle Area


6. DISQUALIFICATION COMPLIANCE
------------------------------
The training, validation, and testing sets were strictly separated. No images 
from the 'testimages' directory were used during the training process[cite: 224, 225]
