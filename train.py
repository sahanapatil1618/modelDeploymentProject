import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
   
])

dataset = datasets.ImageFolder("dataset", transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))

import os

for c in dataset.classes:
    path = f"dataset/{c}"
    print(c, "->", len(os.listdir(path)))

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


epochs = 15

for epoch in range(epochs):
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")


torch.save(model.state_dict(), "model.pth")
print("Model saved successfully!")

import json

with open("classes.json", "w") as f:
    json.dump(dataset.classes, f)

print("Classes saved!")