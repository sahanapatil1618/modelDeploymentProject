import torch
from torchvision import models, transforms

model = models.resnet18(weights=None)

model.load_state_dict(
    torch.load("model.pth", map_location="cpu")
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(image):
    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)

    prediction = torch.argmax(output, dim=1)

    return prediction.item()