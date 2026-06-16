from torchvision import datasets

dataset = datasets.ImageFolder("dataset")

print("Classes:", dataset.classes)
print("Total images:", len(dataset))