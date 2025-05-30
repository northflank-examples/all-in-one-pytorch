# Loosely based on https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import List
import base64
import io
from PIL import Image
import numpy as np

app = FastAPI(title="PyTorch Fashion MNIST API", version="1.0.0")

model = None
device = None
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

class PredictRequest(BaseModel):
    image_base64: str

class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: List[float]

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def initialize_model():
    global model, device
    
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    
    model = NeuralNetwork().to(device)
    
    if os.path.exists("/home/jovyan/shared/model.pth"):
        model.load_state_dict(torch.load("/home/jovyan/shared/model.pth", weights_only=True, map_location=device))
        print("Loaded existing model from /home/jovyan/shared/model.pth")

def preprocess_image(image_base64: str):
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'L':
            image = image.convert('L')
        
        if image.size != (28, 28):
            image = image.resize((28, 28))
        
        image_array = np.array(image) / 255.0
        tensor = torch.from_numpy(image_array).float().unsqueeze(0).unsqueeze(0)
        
        return tensor.to(device)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.on_event("startup")
async def startup_event():
    initialize_model()

@app.get("/")
async def root():
    return {"message": "PyTorch Fashion MNIST API is running", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    global model
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        image_tensor = preprocess_image(request.image_base64)
        
        model.eval()
        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class_idx = logits.argmax(1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        return PredictResponse(
            predicted_class=classes[predicted_class_idx],
            confidence=confidence,
            probabilities=probabilities[0].tolist()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    global model, device
    
    if model is None:
        return {"status": "Model not initialized"}
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "status": "Model initialized",
        "device": device,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_file_exists": os.path.exists("/shared/model.pth"),
        "classes": classes
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
