import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
        super().__init__()

        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=3, padding=1),
            ConvBlock(out_5x5, out_5x5, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_pool, kernel_size=1),
        )

        total_out = out_1x1 + out_3x3 + out_5x5 + out_pool
        self.conv_linear = nn.Conv2d(total_out, total_out, kernel_size=1)

        self.shortcut = nn.Identity()
        if in_channels != total_out:
            self.shortcut = nn.Conv2d(in_channels, total_out, kernel_size=1)

    def forward(self, x):
        out = torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1
        )
        out = self.conv_linear(out)
        return torch.relu(out + self.shortcut(x))


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
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DROPOUT = 0.0
model = InceptionModel(num_classes=10).to(device)  
ckpt = torch.load("/home/toru2/Amara/Deep_learning/dl_lab345.ipynb/checkpoints_static/final_model.pt", map_location=device)

state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
model.load_state_dict(state, strict=True)
model.eval()

video_path = "/home/toru2/Amara/Deep_learning/dl_lab345.ipynb/video/test.mov"

IN_W, IN_H = 178, 218
TARGET_AR = IN_W / IN_H  

def make_bbox_aspect(cx, cy, bw, bh, target_ar, scale=1.35):
    bw *= scale
    bh *= scale

    cur_ar = bw / bh
    if cur_ar > target_ar:
        bh = bw / target_ar
    else:
        bw = bh * target_ar

    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    return x1, y1, x2, y2

def clip_box(x1, y1, x2, y2, W, H):
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(W, int(round(x2)))
    y2 = min(H, int(round(y2)))
    return x1, y1, x2, y2

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

plt.figure(figsize=(8, 5))

while True:
    ok, frame = cap.read()
    if not ok:
        break

    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) > 0:
        x, y, fw, fh = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]

        cx = x + fw / 2.0
        cy = y + fh / 2.0

        fx1, fy1, fx2, fy2 = make_bbox_aspect(cx, cy, fw, fh, TARGET_AR, scale=1.35)
        x1, y1, x2, y2 = clip_box(fx1, fy1, fx2, fy2, W, H)

        face = frame[y1:y2, x1:x2]

        if face.size > 0 and (x2 - x1) > 10 and (y2 - y1) > 10:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)

            inp = torch.from_numpy(face_resized).permute(2, 0, 1).float() / 255.0
            inp = inp.unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(inp)[0].detach().cpu().numpy()  # (10,)

            pred = np.clip(pred, 0.0, 1.0).reshape(-1, 2)

            crop_w = (x2 - x1)
            crop_h = (y2 - y1)

            for px, py in pred:
                gx = int(x1 + px * crop_w)
                gy = int(y1 + py * crop_h)
                cv2.circle(frame, (gx, gy), 3, (0, 255, 0), -1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    clear_output(wait=True)
    plt.imshow(rgb)
    plt.axis("off")
    display(plt.gcf())

cap.release()
clear_output(wait=True)
print("Done.")