import gradio as gr
import requests

# Define the endpoint of the service
SERVICE_URL = "http://localhost:5000/predict"

def relay_to_service(file):
    # Determine the type of file (either an image or a zip file)
    file_type = "application/octet-stream" if file.name.endswith(".zip") else "image/jpeg"

    # Send the file to the service using a POST request
    with open(file.name, "rb") as f:
        response = requests.post(
            SERVICE_URL,
            files={"image": (file.name, f, file_type)}
        )

    # Process the response from the service
    if response.status_code == 200:
        result = response.json()
        if "predicted_year" in result:
            return f"Predicted Year: {result['predicted_year']}"
        elif "predictions" in result:
            predictions = []
            for pred in result['predictions']:
                predictions.append(f"Predicted Year for {pred['filename']}: {pred['predicted_year']}")
            return '\n'.join(predictions)
    else:
        return f"Error: {response.text}"

# Define the Gradio interface
interface = gr.Interface(
    fn=relay_to_service,
    inputs=gr.File(type="filepath", label="Upload an Image or Zip File"),
    outputs="text",
    title="Image Year Prediction Relay",
    description="Upload a single image or a zip file with multiple images, and this tool will relay the file to the prediction service to determine the approximate year of each image."
)

if __name__ == "__main__":
    interface.launch()
