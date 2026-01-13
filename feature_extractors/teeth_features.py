import os
import json
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50

# ----------------------------
# CONFIG
# ----------------------------
SIZE_MODEL_PATH = "resnet_teeth_size.pth"
IRREG_MODEL_PATH = "teeth_irregularity_resnet18.pth"

SIZE_CLASSES = ["Small", "Medium", "Large"]
REG_CLASSES = ["Regular", "Irregular"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Model Loader
# ----------------------------
def load_model(arch, num_classes, weight_path):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Model weight file not found: {weight_path}")

    if arch == "resnet50":
        model = resnet50(weights=None)
    else:
        model = resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# ----------------------------
# Transforms (same as app.py)
# ----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict(model, image_bgr, classes):
    # model expects RGB-like ordering through PIL conversion; OpenCV is BGR, but for this classifier it's okay.
    # If needed: image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = transform(image_bgr).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(img)
        prob = torch.softmax(out, dim=1)
        conf, idx = torch.max(prob, 1)

    return classes[idx.item()], float(conf.item())


# ----------------------------
# Main Extraction
# ----------------------------
def extract_teeth_features(session_dir):
    roi_dir = os.path.join(session_dir, "analysis", "teeth_roi")
    if not os.path.exists(roi_dir):
        raise RuntimeError(f"ROI folder not found: {roi_dir}")

    smile_roi_path = os.path.join(roi_dir, "TEETH_SMILE_ROI.jpg")
    open_roi_path = os.path.join(roi_dir, "TEETH_OPEN_ROI.jpg")

    if not os.path.exists(smile_roi_path):
        raise RuntimeError(f"Missing ROI: {smile_roi_path}")
    if not os.path.exists(open_roi_path):
        raise RuntimeError(f"Missing ROI: {open_roi_path}")

    # Load models
    print("🦷 Loading teeth models...")
    size_model = load_model("resnet50", 3, SIZE_MODEL_PATH)
    reg_model = load_model("resnet18", 2, IRREG_MODEL_PATH)

    # Read ROI images
    smile_roi = cv2.imread(smile_roi_path)
    open_roi = cv2.imread(open_roi_path)

    if smile_roi is None:
        raise RuntimeError(f"Could not read: {smile_roi_path}")
    if open_roi is None:
        raise RuntimeError(f"Could not read: {open_roi_path}")

    # Predictions
    print("🧠 Running inference...")
    smile_size, smile_size_conf = predict(size_model, smile_roi, SIZE_CLASSES)
    smile_reg, smile_reg_conf = predict(reg_model, smile_roi, REG_CLASSES)

    open_size, open_size_conf = predict(size_model, open_roi, SIZE_CLASSES)
    open_reg, open_reg_conf = predict(reg_model, open_roi, REG_CLASSES)

    # Final JSON output
    teeth_result = {
        "teeth": {
            "smile": {
                "roi_path": smile_roi_path,
                "size": {"label": smile_size, "confidence": smile_size_conf},
                "regularity": {"label": smile_reg, "confidence": smile_reg_conf},
            },
            "open": {
                "roi_path": open_roi_path,
                "size": {"label": open_size, "confidence": open_size_conf},
                "regularity": {"label": open_reg, "confidence": open_reg_conf},
            }
        }
    }

    out_path = os.path.join(session_dir, "analysis", "teeth_analysis.json")
    with open(out_path, "w") as f:
        json.dump(teeth_result, f, indent=2)

    print(f"✅ Teeth analysis saved: {out_path}")

    print("\n📌 Summary:")
    print("SMILE ->", smile_size, f"({smile_size_conf:.2f}) |", smile_reg, f"({smile_reg_conf:.2f})")
    print("OPEN  ->", open_size, f"({open_size_conf:.2f}) |", open_reg, f"({open_reg_conf:.2f})")

    return teeth_result["teeth"]



if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python feature_extractors/teeth_features.py <SESSION_DIR>")
        sys.exit(1)

    extract_teeth_features(sys.argv[1])
