# service.py
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI()

# Set the cache directory for PyTorch
os.environ['TORCH_HOME'] = '/root/.cache/torch'

# Load the ConvNeXt model with updated 'weights' argument
weights = models.ConvNeXt_Base_Weights.DEFAULT
model = models.convnext_base(weights=weights)
model.classifier = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(1024, 14, bias=True)  # 14 classes
)

# Load checkpoint
checkpoint = torch.load('convnext_DEWB.pth', map_location=torch.device('cpu'))
state_dict = checkpoint['model']
model.load_state_dict(state_dict)
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Load the image
        img = Image.open(image.file)
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_batch = input_batch.to(device)

        # Perform model inference
        with torch.no_grad():
            output = model(input_batch)

        # Process output
        pred_class = torch.argmax(output).item()
        year_pred = 1930 + np.floor(0.5 + ((1999 - 1930) / 13) * pred_class)

        return JSONResponse(content={"predicted_year": year_pred})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
