import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# =========================
# LOAD MODEL
# =========================
model = models.resnet18()

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model.load_state_dict(torch.load("retinal_model.pth"))
model.eval()

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# PREDICT FUNCTION
# =========================
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    if predicted.item() == 0:
        return "Low Dementia Risk"
    else:
        return "High Dementia Risk"

# =========================
# TEST
# =========================
if __name__ == "__main__":
    result = predict_image("test.jpg")  # put any retinal image here
    print("Prediction:", result)