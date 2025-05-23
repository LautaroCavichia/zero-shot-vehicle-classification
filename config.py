"""
Configuration settings for the zero-shot vehicle detection benchmark.

This file contains all configuration parameters for the benchmark, including
dataset paths, model configurations, and evaluation parameters.
"""
import os
from pathlib import Path
from typing import Dict, List, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_FILE = DATA_DIR / "annotations" / "annotations.json"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)

# Vehicle classes for zero-shot classification (matching CVAT export format)
# These should match the categories in your CVAT COCO export exactly
VEHICLE_CLASSES = ["car", "van", "truck", "bus", "non-vehicle"]

# COCO category mapping for annotations (based on CVAT export)
# Maps COCO category IDs to standardized vehicle class names
COCO_CATEGORY_MAPPING = {
    1: "car",           # Car -> car
    2: "van",           # Van -> van
    3: "truck",         # Truck -> truck
    4: "bus",           # Bus -> bus
    5: "non-vehicle"    # Non-vehicle -> non-vehicle
}

# Reverse mapping for validation
VEHICLE_CLASS_TO_COCO_ID = {v: k for k, v in COCO_CATEGORY_MAPPING.items()}

# CLIP text templates for enhanced zero-shot performance
CLIP_TEMPLATES = [
    "a photo of a {}.",
    "a picture of a {}.", 
    "a close-up photo of a {}.",
    "a side view photo of a {}.",
    "a cctv image of a {}.",
    "a traffic camera image of a {}.",
    "an image showing a {}.",
]

# Class-specific text descriptions for enhanced classification
CLASS_DESCRIPTIONS = {
    "car": [
        "a sedan vehicle",
        "a compact car",
        "a family car",
        "a passenger vehicle",
        "an automobile",
        "a private car",
        "a regular car",
    ],
    "van": [
        "a delivery van",
        "a commercial van", 
        "a minivan",
        "a panel van for deliveries",
        "a cargo van",
        "a utility van",
        "a work van",
    ],
    "truck": [
        "a pickup truck",
        "a heavy-duty truck",
        "a cargo truck",
        "a utility truck", 
        "a box truck",
        "a freight truck",
        "a commercial truck",
        "a delivery truck",
    ],
    "bus": [
        "a public transit bus",
        "a large passenger bus",
        "a school bus",
        "a coach bus",
        "a city bus",
        "a public bus",
        "a passenger bus",
    ],
    "non-vehicle": [
        "a person walking",
        "a pedestrian",
        "people on the street",
        "no vehicles present",
        "an empty road",
        "pedestrians",
        "people",
    ],
}

# Detection model configurations
DETECTOR_CONFIGS = {
    "yolov12": {
        "model_sizes": {
            "small": "yolo11s.pt",
            "medium": "yolo11m.pt", 
            "large": "yolo11l.pt"
        },
        "default_model_size": "medium",
        "confidence_threshold": 0.25,
        "device_preference": ["cuda", "mps", "cpu"],
        # COCO class IDs for vehicles (car, motorcycle, bus, truck, train)
        "vehicle_class_ids": [2, 3, 5, 7, 8],
    },
    "supervision": {
        "model_sizes": {
            "small": "yolov8s.pt",
            "medium": "yolov8m.pt",
            "large": "yolov8l.pt",
        },
        "default_model_size": "medium", 
        "confidence_threshold": 0.25,
        "device_preference": ["cuda", "mps", "cpu"],
        "vehicle_class_ids": [2, 3, 5, 7, 8],
    },
    "ssd": {
        "model_sizes": {
            "small": "ssd_mobilenet_v3",
            "medium": "ssd_vgg16",
        },
        "default_model_size": "medium",
        "confidence_threshold": 0.25, 
        "device_preference": ["cuda", "mps", "cpu"],
        "vehicle_class_ids": [2, 3, 5, 7, 8],
    },
}

# Classification model configurations  
CLASSIFIER_CONFIGS = {
    "clip": {
        "model_sizes": {
            "small": "ViT-B/16",
            "medium": "ViT-B/32", 
            "large": "ViT-L/14"
        },
        "default_model_size": "medium",
        "device_preference": ["cuda", "mps", "cpu"],
    },
    "openclip": {
        "model_sizes": {
            "small": "ViT-B-16",
            "medium": "ViT-B-32",
            "large": "ViT-L-14"
        },
        "pretrained_weights": {
            "small": "laion2b_s34b_b88k",
            "medium": "laion2b_s34b_b79k", 
            "large": "laion2b_s32b_b82k"
        },
        "default_model_size": "medium",
        "device_preference": ["cuda", "mps", "cpu"],
    },
    "siglip": {
        "model_sizes": {
            "medium": "google/siglip-base-patch16-224",
            "large": "google/siglip-large-patch16-384"
        },
        "default_model_size": "medium",
        "device_preference": ["cuda", "mps", "cpu"],
    },
    "git": {
        "model_name": "microsoft/git-base",
        "similarity_model": "all-MiniLM-L6-v2", 
        "max_caption_length": 40,
        "device_preference": ["cuda", "mps", "cpu"],
    }
}

# End-to-end model configurations
END_TO_END_CONFIGS = {
    "yolo_world": {
        "model_sizes": {
            "small": "yolov8s-worldv2.pt",
            "medium": "yolov8m-worldv2.pt",
            "large": "yolov8l-worldv2.pt"
        },
        "default_model_size": "medium",
        "confidence_threshold": 0.25,
        "device_preference": ["cuda", "mps", "cpu"],
    },
    "owlv2": {
        "model_sizes": {
            "base": "google/owlv2-base-patch16",
            "large": "google/owlv2-large-patch14",
            "base-ensemble": "google/owlv2-base-patch16-ensemble"
        },
        "default_model_size": "base",
        "confidence_threshold": 0.2,
        "device_preference": ["cuda", "mps", "cpu"],
    },
    "dino": {
        "model_sizes": {
            "tiny": "IDEA-Research/grounding-dino-tiny",
            "base": "IDEA-Research/grounding-dino-base", 
            "large": "IDEA-Research/grounding-dino-large"
        },
        "default_model_size": "tiny",
        "confidence_threshold": 0.3,
        "text_threshold": 0.25,
        "device_preference": ["cuda", "mps", "cpu"],
    },
    "grounding_dino": {  # Alias for dino
        "model_sizes": {
            "tiny": "IDEA-Research/grounding-dino-tiny",
            "base": "IDEA-Research/grounding-dino-base", 
            "large": "IDEA-Research/grounding-dino-large"
        },
        "default_model_size": "tiny",
        "confidence_threshold": 0.3,
        "text_threshold": 0.25,
        "device_preference": ["cuda", "mps", "cpu"],
    }
}

# Main vehicle scoring parameters (consistent across all models)
MAIN_VEHICLE_SCORING = {
    "centrality_weight": 0.7,      # Weight for distance from center
    "size_weight": 0.3,            # Weight for object size
    "min_area_threshold": 100,     # Minimum bounding box area in pixels
    "max_distance_threshold": 0.8, # Maximum normalized distance from center
}

# Image preprocessing configuration
PREPROCESSING_CONFIG = {
    "clahe_clip_limit": 2.0,
    "clahe_grid_size": (8, 8),
    "sharpen_amount": 0.3,         # Reduced for better compatibility
    "denoise_strength": 3,         # Reduced for better compatibility
    "enhance_contrast": True,
    "enhance_sharpness": True,
    "reduce_noise": True,
}

# Performance and processing settings
NUM_WORKERS = os.cpu_count() or 4
BATCH_SIZE = 1  # Process one image at a time for better memory management
DEFAULT_IMAGE_SIZE = (640, 640)

# Logging configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
VERBOSE_INFERENCE = os.environ.get("VERBOSE_INFERENCE", "False").lower() == "true"

# Validation settings
VALIDATION_CONFIG = {
    "validate_annotations": True,
    "validate_categories": True,
    "strict_mode": False,  # If True, stops on validation errors
    "min_bbox_area": 1,    # Minimum valid bounding box area
    "max_bbox_ratio": 100, # Maximum width/height ratio for valid bbox
}

# Export and results configuration
RESULTS_CONFIG = {
    "save_detailed_results": False,
    "save_per_image_results": False,  # Can be memory intensive
    "save_visualizations": False,     # Controlled by command line
    "export_formats": ["json", "csv"],
}