import gradio as gr
import requests
from PIL import Image
import zipfile
import os
import tempfile

# Define the endpoint of the service
SERVICE_URL = "http://localhost:5000/predict"  # Adjust as per your setup

def relay_to_service(file_path):
    # Handle the uploaded file
    previews = []
    captions = []  # To store filenames
    annotations = []  # To store annotations
    temp_dir = tempfile.mkdtemp()
    image_count = 0  # Count the number of images

    # If the file is a zip file, extract it and prepare previews
    if file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            for filename in zip_ref.namelist():
                if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                    image_count += 1  # Increment image count
                    if image_count > 10000:  # Check for limit
                        return ([], "", "❌ **Error:** Too many images uploaded. Maximum allowed is 10,000.")
                    image_path = os.path.join(temp_dir, filename)
                    previews.append(image_path)
                    captions.append(filename)  # Add filename as caption
    else:
        # Single image file
        previews.append(file_path)
        captions.append(os.path.basename(file_path))  # Add filename as caption
        image_count = 1

    # Prepare the file to send to the prediction service
    file_type = "application/octet-stream" if file_path.endswith(".zip") else "image/jpeg"
    with open(file_path, "rb") as f:
        response = requests.post(
            SERVICE_URL,
            files={"image": (os.path.basename(file_path), f, file_type)}
        )

    # Process the response
    if response.status_code == 200:
        result = response.json()
        if "predicted_year" in result:
            annotations = [f"{captions[0]}: {result['predicted_year']}"]
        elif "predictions" in result:
            annotations = [
                f"{pred['filename']}: {pred['predicted_year']}"
                for pred in result['predictions']
            ]
    else:
        return ([], "", f"❌ **Error:** {response.text}")

    # Save annotations to a file
    annotations_file = os.path.join(temp_dir, "annotations.txt")
    with open(annotations_file, "w") as f:
        for annotation in annotations:
            f.write(f"{annotation}\n")

    # Return annotations text as a single string and file path for download
    return [(img, caption) for img, caption in zip(previews, captions)], annotations_file, "\n".join(annotations)

# Define the Gradio interface with a beautiful design
with gr.Blocks() as interface:
    # Header
    gr.HTML("""
    <div style="text-align: center; padding: 10px; background-color: #4CAF50; color: white; font-size: 20px;">
        <h1>🖼️ Image Year Prediction Tool</h1>
        <p>Analyze images to predict their approximate year of creation</p>
    </div>
    """)

    # Main content
    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.File(
                type="filepath",
                label="📂 Upload an Image or Zip File",
                file_types=[".jpg", ".jpeg", ".png", ".zip"]
            )
        with gr.Column(scale=2):
            gallery = gr.Gallery(
                label="Preview of Uploaded Images with Filenames",
                elem_id="image-preview"
            )

    # Action buttons
    predict_button = gr.Button(value="🔍 Predict", elem_id="predict-btn")
    download_button = gr.File(label="Download Annotations")

    # Results
    with gr.Row():
        annotations_text = gr.Textbox(
            label="📊 Prediction Results",
            interactive=False
        )

    # Interactivity
    upload.change(
        lambda file_path: relay_to_service(file_path) if file_path else ([], "", "❌ No file uploaded."),
        inputs=upload,
        outputs=[gallery, download_button, annotations_text]
    )
    predict_button.click(
        relay_to_service, inputs=upload, outputs=[gallery, download_button, annotations_text]
    )

    # Footer
    gr.HTML("""
    <div style="text-align: center; margin-top: 20px; font-size: 12px; color: #888;">
        <p>Built with ❤️ using Gradio</p>
    </div>
    """)

# Launch the interface
if __name__ == "__main__":
    interface.launch(favicon_path="favicon.ico", server_port=7860, debug=True)
