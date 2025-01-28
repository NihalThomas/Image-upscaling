# Image Upscale with ESRGAN

This is a Streamlit web application that enhances image resolution using the ESRGAN (Enhanced Super-Resolution Generative Adversarial Network) model.

### Features:
- Upload an image to enhance its resolution.
- Uses TensorFlow and TensorFlow Hub for ESRGAN.
- Supports image formats: JPG, JPEG, PNG.

## Prerequisites

Before running the application, ensure that you have the following installed:

- **Docker**: To build and run the application container.
- **Git**: To clone the repository (if necessary).

## Setup

### Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/NihalThomas/Image-upscaling.git
cd Image-upscaling
```

## Docker Setup

### Build the Docker Image

```bash
docker build -t Image-upscaling .
```

### Run the Docker Container

```bash
docker run -p 8501:8501 Image-upscaling
```
