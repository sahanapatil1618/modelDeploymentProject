from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


app = FastAPI()


# Load model
model = models.resnet18(weights=None)

model.load_state_dict(
    torch.load("model.pth", map_location=torch.device("cpu"))
)

model.eval()


# Load class names
weights = ResNet18_Weights.DEFAULT
categories = weights.meta["categories"]


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


@app.get("/")
def home():
    return {
        "message": "Image Classification API is Running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(file.file).convert("RGB")

    input_tensor = transform(image)

    input_batch = input_tensor.unsqueeze(0)


    with torch.no_grad():
        output = model(input_batch)


    predicted_class = torch.argmax(output, dim=1).item()

    prediction = categories[predicted_class]


    return {
        "prediction": prediction
    }