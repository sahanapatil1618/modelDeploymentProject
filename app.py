from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    num1: float
    num2: float

@app.get("/")
def home():
    return {"message": "Model Deployment API is Running"}

@app.post("/predict")
def predict(data: InputData):
    result = data.num1 + data.num2
    return {"prediction": result}