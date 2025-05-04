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
    "an image of a {}.",
    "a picture of a {}.",
    "a close-up photo of a {}.",
    "a cropped photo of a {}.",
    "a side view photo of a {}.",
    "a security camera image of a {}.",
    "a cctv image of a {}.",
    "a surveillance camera photo of a {}.",
    "a traffic camera image of a {}.",
]

# Class-specific text descriptions (for better CLIP/OpenCLIP performance)
CLASS_DESCRIPTIONS = {
    "car": [
        "a small personal automobile",
        "a sedan vehicle",
        "a compact car",
        "a family car",
        "a hatchback car",
    ],
    "van": [
        "a delivery van",
        "a minivan vehicle",
        "a commercial van",
        "a minivan",
        "a panel van for deliveries",
    ],
    "truck": [
        "a large commercial truck",
        "a pickup truck",
        "a heavy-duty truck",
        "a cargo truck",
        "a utility truck",
        "a flatbed truck",
        "a box truck",
        "a garage truck",
    ],
    "bus": [
        "a public transit bus",
        "a large passenger bus",
        "a city bus with passengers",
        "a school bus",
        "a coach bus",
        "a double-decker bus",
        "a transit bus on the road",
    ],
    "emergency": [
        "an emergency vehicle with flashing lights",
        "an ambulance",
        "a fire truck",
        "a police car",
        "an emergency response vehicle",
        "a police truck",
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
        "vehicle_classes": [0, 2, 3, 5, 7],  # COCO class IDs for vehicles
    },
    "supervision": {
        "model_size": "medium",
        "conf_threshold": 0.3,
        "vehicle_classes": [0, 2, 3, 5, 7],  # COCO class IDs for vehicles
    },
       "ssd": {
        "model_size": "medium",
        "conf_threshold": 0.3,
        "vehicle_classes": [0, 2, 3, 5, 7],  # COCO class IDs for vehicles
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
    "vilt": {
    },
    "blip2": {
        "model_size": "medium",
    },
}

END_TO_END_CONFIGS = {
    "glip": {
        "config_file": "GLIP/configs/pretrain/glip_Swin_T_O365_GoldG.yaml",
        "weight_file": "GLIP/MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth",
        "confidence_threshold": 0.7,
    },
    "yolo_world": {
        "model_size": "small",
        "confidence_threshold": 0.3,
    },
}

# Benchmark settings
NUM_WORKERS = os.cpu_count() or 4
BATCH_SIZE = 4