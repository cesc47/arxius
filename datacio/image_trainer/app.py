import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


# Define a Pydantic model to validate the incoming request data
class TrainRequest(BaseModel):
    data_path: str
    model: str
    batch_size: int
    epochs: int
    lr: float


@app.post("/train")
async def train_model(request: TrainRequest):
    try:
        # Prepare the command to run your training script
        command = [
            "python", "train.py",
            "--data-path", request.data_path,
            "--model", request.model,
            "--batch-size", str(request.batch_size),
            "--epochs", str(request.epochs),
            "--lr", str(request.lr)
        ]

        # Run the command using subprocess
        result = subprocess.run(command, capture_output=True, text=True)

        # Return the output of the training process
        if result.returncode == 0:
            return {"message": "Training completed successfully", "output": result.stdout}
        else:
            raise HTTPException(status_code=500, detail=result.stderr)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

