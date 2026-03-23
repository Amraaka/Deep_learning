import os
import time
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# Best params from sanity check:
# lr=3e-4, optimizer=AdamW, weight_decay=0.0, loss=MSE, dropout=0.0


# ------------------ static config ------------------
CFG = {
    "seed": 42,
    "epochs": 40,
    "batch_size": 128,
    "save_every": 5,
    "grad_clip": 1.0,
    "use_amp": True,
    "out_dir": "./checkpoints_improved",
}

# Fixed training params selected from sanity check
LR = 3e-4
WEIGHT_DECAY = 0.0
DROPOUT = 0.0

os.makedirs(CFG["out_dir"], exist_ok=True)


# ------------------ reproducibility ------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CelebALandmarks(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "img_align_celeba")
        self.transform = transform

        partition_file = os.path.join(root_dir, "list_eval_partition.txt")
        landmark_file = os.path.join(root_dir,"landmarks", "list_landmarks_align_celeba.txt")

        self.partition = {}
        with open(partition_file, "r") as f:
            for line in f:
                name, p = line.strip().split()
                self.partition[name] = int(p)

        split_map = {"train": 0, "val": 1, "test": 2}
        self.target_split = split_map[split]

        self.samples = []
        with open(landmark_file, "r") as f:
            lines = f.readlines()

        lines = lines[2:]

        for line in lines:
            parts = line.strip().split()
            img_name = parts[0]
            coords = list(map(float, parts[1:]))

            if self.partition[img_name] == self.target_split:
                self.samples.append((img_name, coords))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, coords = self.samples[idx]

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        w, h = image.size

        landmarks = torch.tensor(coords, dtype=torch.float32).view(-1, 2)

        landmarks[:, 0] /= w
        landmarks[:, 1] /= h

        if self.transform:
            image = self.transform(image)

        return image, landmarks.view(-1)  # shape: (10,)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))


class InceptionBlock(nn.Module):
    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool
    ):

        super(InceptionBlock, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1, padding=0),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            ConvBlock(in_channels, out_pool, kernel_size=1),
        )

    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        return torch.cat([branch(x) for branch in branches], 1)
    


class InceptionModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = ConvBlock(3, 32, kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1)

        self.incept3a = InceptionBlock(64, 32, 32, 64, 16, 32, 32)
        self.incept3b = InceptionBlock(160, 64, 64, 96, 32, 64, 64)

        self.incept4a = InceptionBlock(288, 96, 64, 128, 32, 64, 64)
        self.incept4b = InceptionBlock(352, 96, 64, 128, 32, 64, 64)

        self.incept5a = InceptionBlock(352, 128, 96, 160, 32, 96, 96)

        self.dropout = nn.Dropout(p=DROPOUT)
        self.fc = nn.Linear(480, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.incept3a(x)
        x = self.incept3b(x)
        x = self.maxpool(x)

        x = self.incept4a(x)
        x = self.incept4b(x)
        x = self.maxpool(x)

        x = self.incept5a(x)

        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


data_root = "/home/toru2/Amara/Deep_learning/dl_lab345.ipynb/dataset"

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = CelebALandmarks(root_dir=data_root, split="train", transform=transform)
val_dataset = CelebALandmarks(root_dir=data_root, split="val", transform=transform)
test_dataset = CelebALandmarks(root_dir=data_root, split="test", transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=CFG["batch_size"],
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG["batch_size"],
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

set_seed(CFG["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------ model ------------------
model = InceptionModel(num_classes=10).to(device)

# ------------------ loss ------------------
criterion = nn.MSELoss()

# ------------------ optimizer ------------------
optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

scaler = torch.amp.GradScaler(device="cuda", enabled=(CFG["use_amp"] and device.type == "cuda"))

# ------------------ training utilities ------------------
def run_one_epoch(loader, train_mode=True):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True).float()

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=(CFG["use_amp"] and device.type == "cuda")):
                preds = model(images)
                loss = criterion(preds, targets)

            scaler.scale(loss).backward()

            if CFG["grad_clip"] is not None and CFG["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])

            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                preds = model(images)
                loss = criterion(preds, targets)

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / total_samples

# ------------------ logging + checkpoints ------------------
history = []
best_val_loss = float("inf")
best_epoch = -1

config_path = os.path.join(CFG["out_dir"], "train_config.json")
with open(config_path, "w") as f:
    json.dump({**CFG, "lr": LR, "optimizer": "AdamW", "weight_decay": WEIGHT_DECAY, "loss": "MSE", "dropout": DROPOUT}, f, indent=2)

start_all = time.time()

for epoch in range(1, CFG["epochs"] + 1):
    t0 = time.time()

    train_loss = run_one_epoch(train_loader, train_mode=True)
    val_loss = run_one_epoch(val_loader, train_mode=False)

    epoch_sec = time.time() - t0

    row = {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "lr": optimizer.param_groups[0]["lr"],
        "epoch_time_sec": round(epoch_sec, 2),
    }
    history.append(row)

    print(
        f"Epoch {epoch:03d}/{CFG['epochs']} | "
        f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
        f"time={epoch_sec:.1f}s"
    )

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_path = os.path.join(CFG["out_dir"], "best_model.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": CFG,
        }, best_path)
        print(f"  Saved BEST checkpoint -> {best_path}")

    # Save every N epochs
    if epoch % CFG["save_every"] == 0:
        nth_path = os.path.join(CFG["out_dir"], f"epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": CFG,
        }, nth_path)
        print(f"  Saved periodic checkpoint -> {nth_path}")

# Save final model
final_path = os.path.join(CFG["out_dir"], "final_model.pt")
torch.save({
    "epoch": CFG["epochs"],
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": CFG,
}, final_path)

# Save training log
hist_df = pd.DataFrame(history)
log_csv = os.path.join(CFG["out_dir"], "training_log.csv")
hist_df.to_csv(log_csv, index=False)

total_time = time.time() - start_all
print("\nTraining complete.")
print(f"Best epoch: {best_epoch} | Best val loss: {best_val_loss:.6f}")
print(f"Final model: {final_path}")
print(f"Training log CSV: {log_csv}")
print(f"Total time: {total_time/60:.1f} min")

# Optional: evaluate test set with best checkpoint
best_ckpt = torch.load(os.path.join(CFG["out_dir"], "best_model.pt"), map_location=device)
model.load_state_dict(best_ckpt["model_state_dict"])
test_loss = run_one_epoch(test_loader, train_mode=False)
print(f"Test loss (best checkpoint): {test_loss:.6f}")