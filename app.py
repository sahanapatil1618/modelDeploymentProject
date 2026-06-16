from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io


 
app = FastAPI(
    title="Image Classification API",
    description="Deep Learning Model Deployment using FastAPI and Docker",
    version="1.0"
)


 
model = models.resnet18(weights=None)

 
model.fc = nn.Linear(512, 10)


 
model.load_state_dict(
    torch.load(
        "model.pth",
        map_location=torch.device("cpu")
    )
)

 
model.eval()


 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


 
classes = [
    "cat",
    "dog",
    "tiger",
    "lion",
    "elephant",
    "horse",
    "bird",
    "car",
    "truck",
    "bus"
]


 
@app.get("/")
def home():
    return {
        "message": "Image Classification API is running"
    }


 
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

     
    image = Image.open(
        io.BytesIO(await file.read())
    ).convert("RGB")


     
    image_tensor = transform(image)

     
    image_tensor = image_tensor.unsqueeze(0)


     
    with torch.no_grad():

        output = model(image_tensor)

        prediction = torch.argmax(
            output,
            dim=1
        ).item()


    predicted_class = classes[prediction]


    return {
        "filename": file.filename,
        "prediction": predicted_class
    } 
