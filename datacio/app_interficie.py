import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi import Request
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the ConvNeXt model with updated 'weights' argument
weights = models.ConvNeXt_Base_Weights.DEFAULT
model = models.convnext_base(weights=weights)

# Modify the model's classifier
model.classifier = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(1024, 14, bias=True)  # Assuming you have 14 classes
)

# Load checkpoint
checkpoint_path = 'convnext_DEWB.pth'
try:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['model']  # Adjust as necessary
    model.load_state_dict(state_dict)
except Exception as e:
    logging.error(f"Error loading model checkpoint: {e}")
    raise RuntimeError("Failed to load model checkpoint.") from e

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

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# Pydantic model for response
class YearPredictionResponse(BaseModel):
    predicted_year: int


def predict_year(image: Image.Image) -> int:
    """Predict the year based on the input image."""
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
    year_pred = 1930 + int(np.floor(0.5 + ((1999 - 1930) / 13) * pred_class))  # Maps class to year range 1930-1999

    return year_pred


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the HTML upload form."""
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/predict/", response_model=YearPredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Endpoint to predict year from uploaded image."""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image = Image.open(file.file).convert("RGB")
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail="Error processing image. Please upload a valid image.")

    try:
        year_pred = predict_year(image)
        return YearPredictionResponse(predicted_year=year_pred)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction.")

# Run the app using Uvicorn
# Command to run: uvicorn filename:app --reload
