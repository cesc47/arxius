# service.py
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import zipfile
from io import BytesIO
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Union
import os

# Initialize the FastAPI app
app = FastAPI(
    title="Image Prediction Service",
    description="This service predicts the approximate year for images provided, either individually or as part of a zip file. The model uses a ConvNeXt neural network to infer the year from image data.",
    version="1.0"
)

# Uncomment for Docker if needed: Set the cache directory for PyTorch
# os.environ['TORCH_HOME'] = '/root/.cache/torch'

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


# Define the Pydantic models for structured response
class PredictionResult(BaseModel):
    filename: str
    predicted_year: Union[int, None] = None
    error: Union[str, None] = None


class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]


class SinglePredictionResponse(BaseModel):
    predicted_year: int


def process_image(img: Image.Image):
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

    return int(year_pred)


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint that provides an overview of the available endpoints and their purpose.
    """
    return {
        "message": "Welcome to the Image Prediction Service API!",
        "endpoints": {
            "/predict": "POST endpoint to upload an image or zip file for year prediction"
        }
    }


@app.post("/predict", response_model=Union[SinglePredictionResponse, PredictionResponse], tags=["Prediction"])
async def predict(image: UploadFile = File(...)):
    """
    Predict the year for one or more images provided in a single request.

    **Request**:
    - Accepts either a single image file (jpg, jpeg, png, bmp, gif) or a zip file containing multiple images.

    **Response**:
    - If a single image is provided: returns the predicted year as an integer.
    - If a zip file is provided: returns a list of predictions or errors for each image.

    **File Requirements**:
    - Supported image formats: JPG, JPEG, PNG, BMP, GIF.
    - Zip files should contain only valid image files.

    **Response Codes**:
    - 200: Success
    - 500: Internal server error
    """
    try:
        if image.content_type == "application/octet-stream":
            results = []
            with zipfile.ZipFile(BytesIO(await image.read())) as zip_file:
                # List contents of the zip file for debugging
                file_list = zip_file.namelist()
                print("Files in ZIP:", file_list)  # Print files in zip for debugging

                for filename in file_list:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        with zip_file.open(filename) as img_file:
                            try:
                                # Validate if the file is a valid image
                                img = Image.open(img_file)
                                img.verify()  # Check image validity

                                # Reopen image after verify for processing
                                img = Image.open(img_file)
                                img = img.convert("RGB")
                                year_pred = process_image(img)
                                results.append({"filename": filename, "predicted_year": year_pred})
                            except Exception as e:
                                results.append({"filename": filename, "error": f"Failed to process image: {str(e)}"})
            return JSONResponse(content={"predictions": results})

        else:
            img = Image.open(image.file)
            img.verify()  # Validate single image file
            img = Image.open(image.file)  # Re-open for processing after verification
            img = img.convert("RGB")
            year_pred = process_image(img)
            return JSONResponse(content={"predicted_year": year_pred})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Run the FastAPI app with Uvicorn for development purposes
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
