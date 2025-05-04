# Zero-Shot Vehicle Detection and Classification Benchmark

This project benchmarks various combinations of zero-shot object detectors and classifiers for detecting and classifying vehicles in CCTV images. It also evaluates end-to-end vision-language detection models.

## Overview

The benchmark evaluates the following:

1. Detector + Classifier combinations:
   - YOLOv8 + CLIP
   - YOLOv8 + OpenCLIP
   - YOLOv8 + ViT
   - Supervision + CLIP
   - Supervision + OpenCLIP
   - Supervision + ViT

2. End-to-end zero-shot vision-language models:
   - GLIP (Grounded Language-Image Pretraining)
   - YOLO-World

## Requirements

- Python 3.8+
- Docker (for containerized usage)
- CUDA-compatible GPU (optional, but recommended)
- Apple Silicon compatibility via MPS (Metal Performance Shaders)

## Project Structure

```
zero-shot-vehicle-benchmark/
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose configuration
├── requirements.txt           # Python dependencies
├── setup.sh                   # Setup script for models
├── README.md                  # This file
├── models/                    # Model implementations
│   ├── detectors/             # Object detectors
│   ├── classifiers/           # Zero-shot classifiers
│   └── end_to_end/            # End-to-end models
├── utils/                     # Utility functions
│   ├── data_loader.py         # Dataset loading
│   ├── inference.py           # Inference pipeline
│   ├── evaluation.py          # Evaluation metrics
│   └── visualization.py       # Visualization utilities
├── config.py                  # Configuration settings
└── main.py                    # Main script
```

## Data Format

The benchmark expects the following data structure:

```
data/
├── images/                    # Images for benchmarking
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── annotations/
    └── annotations.json       # COCO format annotations
```

## Vehicle Classes

The benchmark classifies vehicles into the following classes:
- car
- van
- truck
- bus
- emergency
- non-vehicle

## Running the Benchmark

### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/zero-shot-vehicle-benchmark.git
   cd zero-shot-vehicle-benchmark
   ```

2. Place your data in the `data/` directory.

3. Run using Docker Compose:
   ```bash
   docker-compose up
   ```

### Manual Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/zero-shot-vehicle-benchmark.git
   cd zero-shot-vehicle-benchmark
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

4. Run the benchmark:
   ```bash
   python main.py
   ```

## Command Line Arguments

The main script accepts the following arguments:

- `--data_dir`: Path to data directory (default: 'data')
- `--results_dir`: Path to results directory (default: 'results')
- `--limit`: Limit the number of images to process
- `--save_visualizations`: Save detection visualizations
- `--visualize_only`: Only generate visualizations from existing results

Example:
```bash
python main.py --data_dir data --results_dir results --limit 100 --save_visualizations
```

## Evaluation Metrics

The benchmark evaluates models based on:
- F1-score
- Accuracy
- Prediction confidence
- Inference time (detection, classification, and total)

## Apple Silicon Compatibility

The project is designed to be compatible with Apple Silicon (M1/M2/M3) Macs using MPS (Metal Performance Shaders). For optimal performance on Apple Silicon:

1. In `docker-compose.yml`, uncomment the appropriate platform configuration.
2. Ensure you have PyTorch 2.0+ installed with MPS support.

## Adding New Models

### Adding a New Detector

1. Create a new file in `models/detectors/`.
2. Implement a class with `detect()` and `find_main_vehicle()` methods.
3. Update `DETECTOR_CONFIGS` in `config.py`.

### Adding a New Classifier

1. Create a new file in `models/classifiers/`.
2. Implement a class with a `classify()` method.
3. Update `CLASSIFIER_CONFIGS` in `config.py`.

### Adding a New End-to-End Model

1. Create a new file in `models/end_to_end/`.
2. Implement a class with a `detect_and_classify()` method.
3. Update `END_TO_END_CONFIGS` in `config.py`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- GLIP: [microsoft/GLIP](https://github.com/microsoft/GLIP)
- YOLO-World: [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World)
- YOLOv8: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- CLIP: [openai/CLIP](https://github.com/openai/CLIP)
- OpenCLIP: [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
- Supervision: [roboflow/supervision](https://github.com/roboflow/supervision)