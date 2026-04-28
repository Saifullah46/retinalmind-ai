import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# =========================
# PATH
# =========================
data_dir = "data/train"

# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# LOAD DATA
# =========================
dataset = datasets.ImageFolder(data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# =========================
# MODEL (ResNet18)
# =========================
model = models.resnet18(pretrained=True)

# Modify final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # Binary classification

# =========================
# LOSS & OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# TRAINING LOOP
# =========================
epochs = 5

for epoch in range(epochs):
    total_loss = 0
    
    for images, labels in loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "retinal_model.pth")

print("✅ Model trained and saved!")