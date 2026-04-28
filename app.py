from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
model = models.resnet18()
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model.load_state_dict(torch.load("retinal_model.pth"))
model.eval()

# =========================
# TRANSFORM
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
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    confidence = confidence.item() * 100

    if predicted.item() == 0:
        return "✅ No Significant Cognitive Risk", confidence
    else:
        return "⚠️ High Cognitive Risk Detected", confidence

# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            os.makedirs("static", exist_ok=True)

            filename = secure_filename(file.filename)
            filepath = os.path.join("static", filename)

            file.save(filepath)

            result, confidence = predict_image(filepath)
            image_path = filepath

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_path=image_path
    )

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)