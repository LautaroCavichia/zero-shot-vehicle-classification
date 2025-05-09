"""
Configuration settings for the zero-shot vehicle detection benchmark
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_FILE = DATA_DIR / "annotations" / "annotations.json"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)

# Vehicle classes for zero-shot classification
VEHICLE_CLASSES = ["car", "van", "truck", "bus", "emergency", "non-vehicle"]

# CLIP text templates for better zero-shot performance
CLIP_TEMPLATES = [
    "a photo of a {}.",
    "a picture of a {}.",
    "a close-up photo of a {}.",
    "a side view photo of a {}.",
    "a security camera image of a {}.",
    "a cctv image of a {}.",
    "a traffic camera image of a {}.",
]

# Class-specific text descriptions
CLASS_DESCRIPTIONS = {
    "car": [
        "a small personal automobile",
        "a sedan vehicle",
        "a compact car",
        "a family car",
    ],
    "van": [
        "a delivery van",
        "a commercial van",
        "a minivan",
        "a panel van for deliveries",
    ],
    "truck": [
        "a pickup truck",
        "a heavy-duty truck",
        "a cargo truck",
        "a utility truck",
        "a box truck",
        "a garbage truck",
    ],
    "bus": [
        "a public transit bus",
        "a large passenger bus",
        "a school bus",
        "a coach bus",
        "a double-decker bus",
    ],
    "emergency": [
        "an emergency vehicle with flashing lights",
        "an ambulance",
        "a fire truck",
        "a police car",
        "an emergency response vehicle",
        "a paramedic vehicle",
    ],
    "non-vehicle": [
        "a person walking",
        "a bicycle",
        "a motorcycle",
        "a pedestrian",
        "a road with no vehicles",
    ],
}

# Models configuration
DETECTOR_CONFIGS = {
    "yolov12": {
        "model_path": "yolo12m.pt",
        "conf_threshold": 0.3,
        "vehicle_classes": [1, 2, 3, 5, 7],  
    },
    "supervision": {
        "model_size": "medium",
        "conf_threshold": 0.3,
        "vehicle_classes": [1, 2, 3, 5, 7],  
    },
       "ssd": {
        "model_size": "medium",
        "conf_threshold": 0.3,
        "vehicle_classes": [1, 2, 3, 5, 7],  
    },
}

CLASSIFIER_CONFIGS = {
    "clip": {
        "model_size": "medium",
    },
    "openclip": {
        "model_name": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
    },
}

END_TO_END_CONFIGS = {
    "yolo_world": {
        "model_size": "medium",
        "confidence_threshold": 0.3,
    },
    "owlv2": {
        "model_size": "base",
        "confidence_threshold": 0.3,
    },
    'dino': {
        "model_size": "tiny",
        'confidence_threshold': 0.4,
        'text_threshold': 0.3,      
        }
}


NUM_WORKERS = os.cpu_count() or 4
BATCH_SIZE = 4