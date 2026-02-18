import torch
from torch.utils.data import DataLoader
from dataset import SegmentationDataset
from model import get_model
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- DATA ----------------
train_dataset = SegmentationDataset(
    image_dir="data/train/images",
    mask_dir="data/train/masks"
)

val_dataset = SegmentationDataset(
    image_dir="data/val/images",
    mask_dir="data/val/masks"
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

# ---------------- MODEL ----------------
model = get_model(num_classes=6).to(device)

criterion = torch.nn.CrossEntropyLoss()

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)

    intersection = (pred == target).float().sum()
    union = pred.numel()

    return 1 - (2. * intersection + smooth) / (union + smooth)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

epochs = 60
best_iou = 0.0

os.makedirs("checkpoints", exist_ok=True)
checkpoint_path = "checkpoints/best_model.pth"

# -------- RESUME --------
if os.path.exists(checkpoint_path):
    print("Loading previous best model...")
    model.load_state_dict(torch.load(checkpoint_path))
    print("Resuming training...\n")

# -------- IoU --------
def compute_iou(pred, mask, num_classes=6):
    pred = torch.argmax(pred, dim=1)
    ious = []

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (mask == cls)

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            continue

        ious.append(intersection / union)

    if len(ious) == 0:
        return 0.0

    return sum(ious) / len(ious)

# -------- TRAIN --------
for epoch in range(epochs):

    model.train()
    total_loss = 0

    for batch_idx, (images, masks) in enumerate(train_loader):

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        ce = criterion(outputs, masks)
        dice = dice_loss(outputs, masks)
        loss = ce + 0.5 * dice

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 20 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)

    # -------- VALIDATION --------
    model.eval()
    total_iou = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            total_iou += compute_iou(outputs, masks)

    avg_iou = total_iou / len(val_loader)

    scheduler.step()

    print(f"\nEpoch {epoch+1} Complete")
    print(f"Train Loss: {avg_loss:.4f}")
    print(f"Validation IoU: {avg_iou:.4f}\n")

    if avg_iou > best_iou:
        best_iou = avg_iou
        torch.save(model.state_dict(), checkpoint_path)
        print("Best model saved based on IoU!\n")

print("Training Complete")
