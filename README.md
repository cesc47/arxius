
# 🌟 Project Documentation

## 🚀 Overview

This project consists of two main modules: `categoritzacio` and `datacio`. Each module focuses on a distinct task and functionality:

1. **Categoritzacio**: 📂 Implements a multi-label classification system to categorize images based on predefined labels.
2. **Datacio**: 🕰️ Implements a model that predicts the year of an image.

Additionally, there is a submodule `image_predictor_2` that combines these functionalities into a deployable service using Docker. 🐋

---

## 🏗️ Project Structure

### Categoritzacio

- **Description**: Implements a multi-label classification system to categorize images.
- **Directory**: `categoritzacio`
- **Key Files**:
  - `src/dataset.py`: 📄
    - Defines the dataset handling and preprocessing methods.
  - `src/train.py`: 🤖
    - Main script for training the classification model.
  - `src/transforms.py`: 🎛️
    - Contains data augmentation and transformation utilities.
  - `src/visualizations.py`: 📊
    - Visualization utilities.

### Datacio

- **Description**: Implements a model to predict the year of an image.
- **Directory**: `datacio`
- **Key Files**:
  - `image_predictor_2/`: 🖼️
    - Includes Gradio-based servers for hosting and testing the predictor. Takes an input file or a ZIP archive containing images. Outputs the corresponding predicted years for the images.
    - Submodules:
      - `gradio_relay_server/`: 🌐
        - Implements a Gradio-based web interface.
      - `prediction_server/`: ⚙️
        - Server for running the prediction model.
    - **Key Scripts**:
      - `gradio_relay.py`: 🌈 Defines the Gradio interface and functionalities.
      - `service.py`: 🔧 Handles backend operations for prediction.

---

## 🛠️ How to Use

### Categoritzacio

1. Navigate to the `categoritzacio` directory.
2. Use `train.py` to train the multi-label classification model with the following parameters:
   ```bash
   --data-path /your/data/path \
   --model convnext_base \
   --batch-size 256 \
   --opt adamw \
   --lr 1e-4 \
   --lr-scheduler cosineannealinglr \
   --auto-augment ta_wide \
   --epochs 25 \
   --weight-decay 0.05 \
   --workers 16 \
   --norm-weight-decay 0.0 \
   --ra-sampler \
   --weights ConvNeXt_Base_Weights.DEFAULT \
   --output-dir models \
   --log_wandb
   ```

### Datacio

1. If you want to train the image predictor, navigate to `datacio/image_trainer`. 
2. Train the prediction model with the following parameters:
   ```bash
   --data-path /your/data/path \
   --model convnext_base \
   --batch-size 256 \
   --opt adamw \
   --lr 1e-5 \
   --lr-scheduler cosineannealinglr \
   --auto-augment ta_wide \
   --epochs 70 \
   --weight-decay 0.05 \
   --norm-weight-decay 0.0 \
   --train-crop-size 176 \
   --val-resize-size 232 \
   --ra-sampler \
   --weights ConvNeXt_Base_Weights.DEFAULT \
   ```

### Docker Deployment (Datacio) for the Image Prediction Service

1. Navigate to the `datacio/image_predictor_2` directory.
2. To build the image prediction backend, navigate to `datacio/image_predictor_2/prediction_server` and build the Docker image:
   ```bash
   docker build -t image-predictor .
   ```
3. Run the container:
   ```bash
   docker run -p <port>:<port> image-predictor
   ```
4. Access the service and upload images or ZIP files for year prediction.
5. To build the front end, navigate to `datacio/image_predictor_2/gradio_relay_server` and build the Docker image:
   ```bash
   docker build -t gradio_service .
   ```
6. Run the container:
   ```bash
   docker run -p <port>:<port> gradio_service
   ```

---

## 🔧 Requirements

### General Dependencies

- Python 3.8+ 🐍
- Required Python libraries are listed in each module's `requirements.txt`.

### Docker 🐳

- Ensure Docker is installed and running on your system.

---