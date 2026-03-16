#!/usr/bin/env python3
"""
Full training script for face landmark detection on CelebA dataset.
Based on dl_lab345.ipynb/training_test.ipynb
"""

import argparse
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
    """Configure logging to console and file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

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


def main():
    parser = argparse.ArgumentParser(description="Train face landmark model on CelebA")
    parser.add_argument("--landmarks", type=str,
                        default="/home/toru2/Amara/Deep_learning/dl_lab345.ipynb/dataset/landmarks/list_landmarks_align_celeba.txt",
                        help="Path to landmarks file")
    parser.add_argument("--img-dir", type=str,
                        default="/home/toru2/Amara/Deep_learning/dl_lab345.ipynb/dataset/img_align_celeba",
                        help="Path to image directory")
    parser.add_argument("--checkpoint-dir", type=str, default="/home/toru2/Amara/Deep_learning/checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for log files (default: checkpoint-dir/logs)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Paths
    landmarks_path = Path(args.landmarks)
    img_dir = Path(args.img_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir) if args.log_dir else ckpt_dir / "logs"
    setup_logging(log_dir)

    # Dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.PILToTensor(),
    ])
    full_dataset = CelebALandmarksDataset(landmarks_path, img_dir, transform=transform)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_set, test_set = random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    logging.info(f"Train: {len(train_set)}, Test: {len(test_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")

    model = InceptionLandmarkModel(num_landmarks=5).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, test_loader, criterion, device)
        scheduler.step(val_loss)

        logging.info(f"Epoch {epoch}/{args.epochs} | Train Loss {train_loss:.5f} | Val Loss {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = ckpt_dir / "face_landmarks_best1.pt"
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"  -> Saved best model to {ckpt_path}")

    # Save final model
    torch.save(model.state_dict(), ckpt_dir / "face_landmarks_final1.pt")
    logging.info(f"Training complete. Final model saved to {ckpt_dir / 'face_landmarks_final1.pt'}")


if __name__ == "__main__":
    main()
