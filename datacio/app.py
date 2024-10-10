import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# Load the ConvNeXt model with updated 'weights' argument
weights = models.ConvNeXt_Base_Weights.DEFAULT
model = models.convnext_base(weights=weights)

# Modify the model's classifier
model.classifier = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(1024, 14, bias=True)  # Assuming you have 14 classes
)

# Load checkpoint
checkpoint = torch.load('convnext_DEWB.pth', map_location=torch.device('cpu'))
state_dict = checkpoint['model']  # Adjust as necessary
model.load_state_dict(state_dict)
model.eval()  # Set model to evaluation mode

# Define the image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize FastAPI
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load and preprocess the image
    image = Image.open(file.file).convert("RGB")
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Create batch dimension

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_batch = input_batch.to(device)

    # Forward pass
    with torch.no_grad():
        output = model(input_batch)

    # Process output
    pred_class = torch.argmax(output).item()
    year_pred = 1930 + np.floor(0.5 + ((1999 - 1930) / 13) * pred_class)  # Maps class to year range 1930-1999

    return JSONResponse(content={"predicted_year": int(year_pred)})

# Run the app using Uvicorn
# Command to run: uvicorn filename:app --reload
