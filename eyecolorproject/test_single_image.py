import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import models, transforms
import torch.nn as nn

# ---------------- CONFIG ----------------
MODEL_PATH = "models/eye_color.pth"
IMAGE_PATH = "darkbrown.jpg"   # combined-eye image
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.6
# ----------------------------------------

classes = [
    "Black", "Blue", "Dark_Brown",
    "Gray", "Green", "Hazel", "Light_Brown"
]

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- SAME MODEL AS TRAINING --------
class EyeColorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.efficientnet_b2(weights=None)
        self.net.classifier = nn.Linear(1408, 7)

    def forward(self, x):
        return self.net(x)

model = EyeColorModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -------- IRIS EXTRACTION (MATCH TRAINING) --------
def extract_iris_from_eye_crop(img):
    h, w, _ = img.shape
    cx, cy = w // 2, h // 2
    r = int(min(h, w) * 0.35)

    iris = img[cy-r:cy+r, cx-r:cx+r]
    iris = cv2.resize(iris, (IMG_SIZE, IMG_SIZE))
    return iris

# -------- GLARE REMOVAL --------
def remove_glare(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    _, s, v = cv2.split(hsv)

    glare = cv2.inRange(v, 200, 255) & cv2.inRange(s, 0, 80)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, _, _ = cv2.split(lab)
    highlight = cv2.inRange(l, 220, 255)

    mask = cv2.bitwise_or(glare, highlight)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)

# -------- SOFT NORMALIZATION --------
def normalize(img):
    img = img.astype(np.float32)
    mean = img.mean(axis=(0,1), keepdims=True)
    global_mean = mean.mean()
    img = img * 0.9 + global_mean * 0.1
    return np.clip(img, 0, 255).astype(np.uint8)

# -------- LOAD & PREPROCESS IMAGE --------
raw = cv2.imread(IMAGE_PATH)
if raw is None:
    raise ValueError("❌ Could not read image. Check IMAGE_PATH.")

raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

iris = extract_iris_from_eye_crop(raw)
iris = remove_glare(iris)
iris = normalize(iris)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

img = transform(iris).unsqueeze(0).to(device)

# -------- INFERENCE --------
with torch.no_grad():
    probs = F.softmax(model(img), dim=1)[0].cpu().numpy()

top3 = probs.argsort()[-3:][::-1]
top1 = top3[0]

print("\n🧠 Eye Color Prediction:")
for i in top3:
    print(f"{classes[i]} : {probs[i]:.3f}")

# if probs[top1] < CONFIDENCE_THRESHOLD:
#     print("\n⚠️ LOW CONFIDENCE → Reject prediction, re-capture image")
# else:
print(f"\n✅ Final Eye Color: {classes[top1]} (confidence {probs[top1]:.2f})")

