# Defect Detection System (OpenCV + PyTorch)

This repository hosts a prototype system designed to detect scratches and other defects on metal surfaces. It leverages OpenCV for efficient image preprocessing and PyTorch for building and deploying a robust deep learning model. The system is designed for ease of use with a Streamlit-based graphical interface for interactive demonstrations and analysis.

## Features

-   **Synthetic Data Generation:** Tools to create synthetic datasets for training and testing, enabling rapid prototyping without extensive real-world data.
-   **PyTorch UNetTiny Model:** A compact yet effective U-Net architecture specifically tailored for segmentation tasks, identifying defects at a pixel level.
-   **Training & Inference Scripts:** Comprehensive scripts for training the UNetTiny model and performing defect inference on new images.
-   **ONNX Export & Quantization:** Capabilities to export the trained PyTorch model to ONNX format and quantize it for optimized performance and reduced size, crucial for deployment.
-   **FastAPI Inference Endpoint:** A ready-to-use API endpoint built with FastAPI for serving model inferences, allowing integration with other applications.
-   **Streamlit Dashboard:** An interactive web dashboard for real-time monitoring, visualization, and comparison of defect detection results, facilitating model evaluation and user interaction.

## Technical Details & Algorithms

The system employs several key algorithms and components:

### UNetTiny Model

-   **Purpose:** The core of the defect detection system is a UNetTiny model, a convolutional neural network (CNN) specifically designed for image segmentation. This means it's trained to classify each pixel in an image as either "defect" or "non-defect."
-   **How it Works (Simplified):** A U-Net has an encoder (downsampling path) that captures context and a decoder (upsampling path) that enables precise localization. The "Tiny" variant signifies a smaller, more efficient architecture suitable for prototypes and faster inference while maintaining good performance for the task.

### Preprocessing (`preprocess_image`)

-   **Purpose:** Prepares input images to be in a format suitable for the UNetTiny model.
-   **How it Works:** This typically involves resizing images to a standard dimension required by the model (e.g., 256x256 pixels) and normalizing pixel values (e.g., scaling them from 0-255 to 0-1 and potentially subtracting a mean and dividing by a standard deviation) to help the neural network learn effectively.

### Heatmap (`create_heatmap`)

-   **Purpose:** Visualizes the raw output of the UNetTiny model.
-   **How it Works:** The model's final layer often outputs a probability map, where each pixel's value indicates the likelihood of it being part of a defect. A heatmap function takes this probability map and applies a color gradient (e.g., warmer colors for higher probability) to make these predictions easily interpretable.

### Dice Coefficient

-   **Purpose:** A statistical metric used to gauge the similarity between two samples; in this case, the predicted defect mask and the ground truth defect mask.
-   **How it Works (Simplified):** It measures the overlap between the predicted defect area and the actual defect area. A value of 1 indicates perfect overlap, while 0 indicates no overlap. It's calculated as `(2 * Intersection) / (Total Pixels in Both Masks)`.

### Intersection over Union (IoU)

-   **Purpose:** Another common metric for evaluating the accuracy of object detection or segmentation systems.
-   **How it Works (Simplified):** IoU quantifies the overlap between the predicted mask and the ground truth mask divided by the area of union between them. Like Dice, a higher IoU value signifies better performance, with 1 being perfect overlap. It's calculated as `Intersection / Union`.
