import torch
import cv2
import os
import numpy as np
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Load Model
# --------------------
model = get_model(num_classes=6).to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
model.eval()

# --------------------
# Paths
# --------------------
input_dir = "data/test/images"  # Change if needed
output_dir = "predictions"
os.makedirs(output_dir, exist_ok=True)

# Reverse class mapping
mapping_back = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 27,
    5: 39
}

# --------------------
# Inference Loop
# --------------------
for img_name in os.listdir(input_dir):

    img_path = os.path.join(input_dir, img_name)

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))

    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Convert back to original labels
    final_mask = np.zeros_like(pred)
    for k, v in mapping_back.items():
        final_mask[pred == k] = v

    save_path = os.path.join(output_dir, img_name)
    cv2.imwrite(save_path, final_mask)

print("Test inference complete. Masks saved in 'predictions' folder.")
