import copy
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image


# --- Dataset ---

class CelebALandmarksDataset(Dataset):
    def __init__(self, landmarks_path, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.samples = []
        with open(landmarks_path) as f:
            lines = f.readlines()
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 11:
                fname = parts[0]
                coords = [float(x) for x in parts[1:11]]
                self.samples.append((fname, coords))

        self.samples = [(f, c) for f, c in self.samples if (self.img_dir / f).exists()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, coords = self.samples[idx]
        img = Image.open(self.img_dir / fname).convert("RGB")
        landmarks = torch.tensor(coords, dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        return img, landmarks


# --- Model ---

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_chanels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_chanels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_chanels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        super(InceptionBlock, self).__init__()

        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=3, padding=1),
            ConvBlock(out_5x5, out_5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_pool, kernel_size=1)
        )

        self.out_channels = out_1x1 + out_3x3 + out_5x5 + out_pool

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class InceptionResBlock(nn.Module):
    def __init__(self, in_channels):
        super(InceptionResBlock, self).__init__()
        self.block = InceptionBlock(in_channels, 64, 96, 128, 16, 32, 32)
        self.proj = nn.Conv2d(self.block.out_channels, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        out = self.block(x)
        out = self.proj(out)
        out = self.bn(out)
        return F.relu(out + x)


class InceptionLandmarkModel(nn.Module):
    def __init__(self, num_landmarks=5):
        super(InceptionLandmarkModel, self).__init__()

        self.conv_blocks = nn.Sequential(
            ConvBlock(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1),
            ConvBlock(64, 128, kernel_size=3, padding=1),
            ConvBlock(128, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.inception_block_group1 = nn.Sequential(
            InceptionBlock(192, 64, 96, 128, 16, 32, 32),
            InceptionBlock(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.inception_res_blocks = nn.Sequential(
            InceptionResBlock(480),
            InceptionResBlock(480),
            InceptionResBlock(480)
        )

        self.inception_block_group2 = nn.Sequential(
            InceptionBlock(480, 256, 160, 320, 32, 128, 128),
            InceptionBlock(832, 256, 192, 320, 48, 128, 128)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(832, num_landmarks * 2)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.inception_block_group1(x)
        x = self.inception_res_blocks(x)
        x = self.inception_block_group2(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x)


# --- Logging ---

def setup_logging(log_dir):
    """Configure logging to file only."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logging.info(f"Logging to {log_file}")
    return log_file


# --- Training ---

def train_epoch(model, loader, criterion, optimizer, device, img_w=178, img_h=218):
    model.train()
    total_loss = 0.0
    for imgs, landmarks in loader:
        imgs = imgs.float().to(device) / 255.0
        landmarks = landmarks.to(device)
        landmarks[:, 0::2] /= img_w
        landmarks[:, 1::2] /= img_h

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, landmarks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device, img_w=178, img_h=218):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, landmarks in loader:
            imgs = imgs.float().to(device) / 255.0
            landmarks = landmarks.to(device)
            landmarks[:, 0::2] /= img_w
            landmarks[:, 1::2] /= img_h

            preds = model(imgs)
            loss = criterion(preds, landmarks)
            total_loss += loss.item()
    return total_loss / len(loader)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss):
    """Save full training state for resume and analysis."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, path)


def main():
    # Training configuration (same values as previous CLI defaults).
    landmarks_path = Path(
        "/home/toru2/Amara/Deep_learning/dl_lab345.ipynb/dataset/landmarks/list_landmarks_align_celeba.txt"
    )
    img_dir = Path("/home/toru2/Amara/Deep_learning/dl_lab345.ipynb/dataset/img_align_celeba")
    ckpt_dir = Path("/home/toru2/Amara/Deep_learning/checkpoints")
    log_dir = None

    epochs = 30
    batch_size = 128
    lr = 1e-3
    early_stop_patience = 5
    early_stop_min_delta = 1e-4
    save_every_n_epochs = 5
    num_workers = 8
    seed = 42

    torch.manual_seed(seed)

    # Paths
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resolved_log_dir = Path(log_dir) if log_dir else ckpt_dir / "logs"
    log_file = setup_logging(resolved_log_dir)
    logging.info(f"Log file saved at: {log_file}")

    # Dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.PILToTensor(),
    ])
    full_dataset = CelebALandmarksDataset(landmarks_path, img_dir, transform=transform)

    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    logging.info(
        f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")

    model = InceptionLandmarkModel(num_landmarks=5).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        save_checkpoint(ckpt_dir / "face_landmarks_last.pt", model, optimizer, scheduler, epoch, best_val_loss)

        if save_every_n_epochs > 0 and epoch % save_every_n_epochs == 0:
            periodic_ckpt = ckpt_dir / f"face_landmarks_epoch_{epoch}.pt"
            save_checkpoint(periodic_ckpt, model, optimizer, scheduler, epoch, best_val_loss)
            logging.info(f"  -> Saved periodic checkpoint to {periodic_ckpt}")

        logging.info(f"Epoch {epoch}/{epochs} | Train Loss {train_loss:.5f} | Val Loss {val_loss:.5f}")

        if val_loss < (best_val_loss - early_stop_min_delta):
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            ckpt_path = ckpt_dir / "face_landmarks_best.pt"
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, best_val_loss)
            logging.info(f"  -> Saved best model to {ckpt_path}")
        else:
            epochs_without_improvement += 1
            if early_stop_patience > 0:
                logging.info(
                    "  -> No significant val-loss improvement "
                    f"({epochs_without_improvement}/{early_stop_patience})"
                )
            else:
                logging.info("  -> No significant val-loss improvement")

            if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
                logging.info(
                    "Early stopping triggered at epoch "
                    f"{epoch}. Best val loss: {best_val_loss:.5f}"
                )
                break

    # Save final model as the best observed weights to reduce overfitting risk.
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss = validate(model, test_loader, criterion, device)
    logging.info(f"Final Test Loss: {test_loss:.5f}")

    torch.save(model.state_dict(), ckpt_dir / "face_landmarks_final.pt")
    logging.info(f"Training complete. Final model saved to {ckpt_dir / 'face_landmarks_final.pt'}")


if __name__ == "__main__":
    main()
