#!/bin/bash
set -e

echo "Setting up zero-shot vehicle detection benchmark environment..."

# Create necessary directories
mkdir -p model_weights
mkdir -p results

# Download YOLOv8 model for object detection
echo "Downloading YOLOv8 model..."
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt');"

echo "Setup complete!"