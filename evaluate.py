import torch
import cv2
import os
import numpy as np
from model import get_model
from dataset import SegmentationDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = get_model(num_classes=6).to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
model.eval()

# Load validation data
val_dataset = SegmentationDataset(
    image_dir="data/val/images",
    mask_dir="data/val/masks"
)

val_loader = DataLoader(val_dataset, batch_size=1)

all_preds = []
all_targets = []

with torch.no_grad():
    for images, masks in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy().flatten())
        all_targets.extend(masks.numpy().flatten())

# Confusion Matrix
cm = confusion_matrix(all_targets, all_preds)

print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
