import os
import sys
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps
from model import MNISTCNN

MODEL_PATH = "mnist_cnn.pth"
class_names = [str(i) for i in range(10)]

transform_pipeline = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def predict_digit(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image missing: '{image_path}'")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTCNN().to(device)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    try:
        img = Image.open(image_path).convert("L")
        img_np = np.array(img)
        corners = [img_np[0,0], img_np[0,-1], img_np[-1,0], img_np[-1,-1]]
        if np.mean(corners) > 127:
            img = ImageOps.invert(img)
        tensor = transform_pipeline(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"❌ Failed to process image: {e}")
        return

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = output.argmax(dim=1).item()
        confidence = probs[0][pred_idx].item() * 100

    print(f"\n🎯 Identified: {class_names[pred_idx]} | Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n❌ Missing image path.")
        print("Usage: python predict.py <path_to_digit_image.png>")
    else:
        predict_digit(sys.argv[1])