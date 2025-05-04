#!/bin/bash
set -e

echo "Setting up zero-shot vehicle detection benchmark environment..."

# Create necessary directories
mkdir -p model_weights
mkdir -p results

echo "Installing GLIP..."
if [ ! -d "GLIP" ]; then
    git clone https://github.com/microsoft/GLIP.git
    cd GLIP
    git checkout c663d9db8a503e04c6b76cd2e14152bab775d28a
    pip install yacs
    python setup.py build develop 
    cd ..
else
    echo "GLIP already installed, only setup.py..."
    cd GLIP
    pip install yacs
    python setup.py build develop
    cd ..
fi

# Download GLIP model weights
GLIP_MODEL_PATH="GLIP/MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"
if [ ! -f "$GLIP_MODEL_PATH" ]; then
    echo "Downloading GLIP model weights..."
    mkdir -p GLIP/MODEL
    wget -q https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_a_tiny_o365.pth -O "$GLIP_MODEL_PATH"
else
    echo "GLIP model weights already exist, skipping..."
fi

# Install YOLO-World
echo "Installing YOLO-World..."
if [ ! -d "YOLO-World" ]; then
    git clone https://github.com/AILab-CVC/YOLO-World.git
    cd YOLO-World
    pip install -v -e .
    cd ..
else
    echo "YOLO-World already installed, only setup.py..."
    cd YOLO-World
    pip install -v -e .
    cd ..
fi

# Download YOLO-World model weights
YOLO_MODEL_PATH="YOLO-World/pretrained/yolo_world_s.pt"
if [ ! -f "$YOLO_MODEL_PATH" ]; then
    echo "Downloading YOLO-World model weights..."
    mkdir -p YOLO-World/pretrained
    wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt -O "$YOLO_MODEL_PATH"
else
    echo "YOLO-World model weights already exist, skipping..."
fi

# Download YOLOv8 model for object detection
echo "Downloading YOLOv8 model..."
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt');"

echo "Setup complete!"