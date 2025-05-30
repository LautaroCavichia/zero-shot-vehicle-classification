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

# Vehicle classes for zero-shot classification (updated with new classes)
VEHICLE_CLASSES = ["city_car", "large_suv", "van", "truck", "bus", "motorcycle", "non-vehicle"]

# COCO category mapping for annotations (updated for new classes)
# Maps COCO category IDs to standardized vehicle class names
COCO_CATEGORY_MAPPING = {
    1: "city_car",      # City Car (small cars)
    2: "large_suv",     # Large SUV 
    3: "van",           # Van -> van
    4: "truck",         # Truck -> truck
    5: "bus",           # Bus -> bus
    6: "motorcycle",    # Motorcycle -> motorcycle
    7: "non-vehicle"    # Non-vehicle -> non-vehicle
}

# Reverse mapping for validation
VEHICLE_CLASS_TO_COCO_ID = {v: k for k, v in COCO_CATEGORY_MAPPING.items()}

YOLO_WORLD_ENHANCED_PROMPTS = {
    "city_car": [
        "car", "sedan", "hatchback", "small car", "passenger car", 
        "compact car", "city car", "automobile"
    ],
    "large_suv": [
        "SUV", "large SUV", "pickup truck", "4x4", "sport utility vehicle",
        "crossover", "large car", "big car"
    ],
    "van": [
        "van", "minivan", "delivery van", "panel van", "cargo van"
    ],
    "truck": [
        "truck", "large truck", "freight truck", "cargo truck", 
        "semi truck", "heavy truck"
    ],
    "bus": [
        "bus", "school bus", "public bus", "coach", "transit bus"
    ],
    "motorcycle": [
        "motorcycle", "motorbike", "scooter", "bike", "moped"
    ],
    "non-vehicle": [
        "person", "pedestrian", "people", "human", "man", "woman"
    ]
}

# CLIP text templates for enhanced zero-shot performance
CLIP_TEMPLATES = [
    "a photo of a {}.",
    "a side view of a {}.",
    "a cctv image of a {}.",
    "a traffic camera photo of a {}.",
    "a road surveillance image of a {}.",
    "an image showing a {}.",
]

# Class-specific text descriptions for enhanced classification
CLASS_DESCRIPTIONS = {
    "city_car": [
        "a small city car on the road",
        "a compact car in traffic",
        "a small sedan driving",
        "a hatchback vehicle",
        "a station wagon car",
        "a small passenger car",
        "a compact automobile on street",
        "a small family car",
        "an economy car on road",
        "a subcompact vehicle",
        "a city hatchback",
        "a small station wagon",
    ],
    "large_suv": [
        "a large SUV on the road",
        "a big sport utility vehicle driving", 
        "a full-size SUV in traffic",
        "a luxury SUV on street",
        "a large 4x4 vehicle",
        "a big family SUV",
        "a premium SUV driving",
        "a large crossover vehicle",
        "a spacious SUV on road",
        "a heavy-duty SUV",
        "a pickup truck vehicle",
        "a pickup SUV driving",
    ],
    "van": [
        "a van on the road",
        "a delivery van driving",
        "a commercial van in traffic", 
        "a minivan on street",
        "a panel van vehicle",
        "a cargo van driving",
        "a utility van on road",
        "a work van in traffic",
        "a passenger van driving",
        "a service van vehicle",
    ],
    "truck": [
        "a truck on the road",
        "a heavy-duty truck driving",
        "a cargo truck in traffic",
        "a freight truck on street", 
        "a box truck vehicle",
        "a commercial truck driving",
        "a delivery truck on road",
        "a work truck in traffic",
        "a large truck vehicle",
        "a transport truck driving",
    ],
    "bus": [
        "a bus on the road",
        "a public transit bus driving",
        "a large passenger bus in traffic",
        "a school bus on street",
        "a coach bus vehicle",
        "a city bus driving",
        "a public bus on road",
        "a passenger bus in traffic",
        "a tour bus vehicle",
        "a transit bus driving",
    ],
    "motorcycle": [
        "a motorcycle on the road",
        "a motorbike driving",
        "a bike with motor in traffic",
        "a two-wheeled vehicle on street",
        "a motor scooter driving",
        "a sport bike on road",
        "a touring motorcycle in traffic",
        "a street bike driving",
        "a cruiser motorcycle on road",
        "a scooter vehicle",
        "a motorbike with person driving",
        "a person riding a motorcycle",
        "a motorcycle and rider in traffic",
        "a motor scooter with driver",
    ],
    "non-vehicle": [
        "a person walking on the road",
        "a pedestrian on the street",
        "people walking in the area",
        "a person standing on road",
        "pedestrians on the street",
        "people in the scene",
        "a person on foot",
        "a cyclist with bicycle",
        "a person with bike",
        "pedestrians in traffic area",
        "people near vehicles",
        "a person in the image",
    ],
}

GROUNDING_DINO_ENHANCED_PROMPTS = {
    "city_car": [
        "small passenger car on road",
        "compact sedan vehicle", 
        "city car in traffic",
        "small automobile driving",
        "passenger car with four doors",
        "hatchback car on street"
    ],
    "large_suv": [
        "large SUV vehicle on road",
        "sport utility vehicle driving", 
        "big family SUV in traffic",
        "pickup truck vehicle",
        "large crossover car",
        "heavy duty SUV on street"
    ],
    "van": [
        "delivery van on road",
        "commercial van vehicle",
        "passenger van driving",
        "cargo van in traffic", 
        "minivan on street",
        "utility van vehicle"
    ],
    "truck": [
        "large truck on road",
        "freight truck vehicle",
        "cargo truck driving",
        "commercial truck in traffic",
        "heavy duty truck on street", 
        "delivery truck vehicle"
    ],
    "bus": [
        "public bus on road",
        "passenger bus vehicle",
        "transit bus driving",
        "city bus in traffic",
        "school bus on street",
        "coach bus vehicle"
    ],
    "motorcycle": [
        "motorcycle on road",
        "motorbike vehicle driving",
        "person riding motorcycle",
        "motorcycle with rider",
        "motorbike in traffic",
        "two-wheeled motor vehicle"
    ],
    "non-vehicle": [
        "person walking on road",
        "pedestrian on street",
        "person standing near vehicles",
        "people in traffic area",
        "human figure on road",
        "pedestrian crossing street"
    ]
}

# GIT ENHANCED PROMPTS - Optimized for generative classification
GIT_ENHANCED_PROMPTS = {
    "city_car": [
        "This image shows a small car",
        "A compact passenger vehicle",
        "A city car for daily commuting", 
        "A small family automobile",
        "A sedan for urban driving",
        "A hatchback passenger car",
        "A small personal vehicle",
        "A compact city automobile",
        "A regular passenger car",
        "A standard sized vehicle"
    ],
    "large_suv": [
        "This image shows a large SUV",
        "A big sport utility vehicle",
        "A family SUV for multiple passengers",
        "A large crossover vehicle", 
        "A heavy duty SUV",
        "A pickup truck style vehicle",
        "A spacious family vehicle",
        "A large passenger SUV",
        "A big recreational vehicle",
        "A large off-road capable vehicle"
    ],
    "van": [
        "This image shows a van",
        "A delivery van for cargo",
        "A passenger van for groups",
        "A commercial utility van",
        "A minivan for families",
        "A cargo transport van",
        "A service van vehicle",
        "A multipurpose van",
        "A work van for business",
        "A utility transport vehicle"
    ],
    "truck": [
        "This image shows a truck",
        "A large freight truck",
        "A cargo transport truck",
        "A commercial delivery truck",
        "A heavy duty truck",
        "A goods transport vehicle",
        "A large cargo truck", 
        "A freight delivery vehicle",
        "A commercial transport truck",
        "A logistics truck vehicle"
    ],
    "bus": [
        "This image shows a bus",
        "A public transportation bus",
        "A passenger bus for many people",
        "A city transit bus",
        "A school bus for students",
        "A public transport vehicle",
        "A mass transit bus",
        "A passenger coach bus",
        "A group transportation vehicle",
        "A public service bus"
    ],
    "motorcycle": [
        "This image shows a motorcycle",
        "A person riding a motorcycle", 
        "A motorbike with a rider",
        "A two-wheeled motor vehicle",
        "A motorcycle on the road",
        "A person on a motorbike",
        "A rider on a motorcycle",
        "A motorized two-wheeler",
        "A motorcycle being driven",
        "A person using a motorcycle"
    ],
    "non-vehicle": [
        "This image shows a person",
        "A pedestrian walking",
        "A human figure standing",
        "People in the scene",
        "A person on foot",
        "Pedestrians in the area",
        "A human being present",
        "A person without a vehicle",
        "People walking or standing",
        "Human subjects in the image"
    ]
}

# DYNAMIC PROMPT SELECTION STRATEGIES
PROMPT_SELECTION_STRATEGIES = {
    "grounding_dino": {
        "use_contextual_descriptors": True,      # "car on road" vs "car"
        "combine_multiple_prompts": True,        # Use multiple prompts per class
        "separator": " . ",                      # Period separation as recommended
        "max_prompts_per_class": 3,             # Limit for performance
        "prioritize_specific_terms": True,       # Avoid generic "vehicle"
        "include_scene_context": True           # Add road/traffic context
    },
    "git": {
        "use_natural_language": True,           # Full sentences work better
        "include_demonstrative_pronouns": True, # "This image shows..."
        "vary_sentence_structure": True,        # Different sentence patterns
        "use_descriptive_adjectives": True,     # Size, function descriptors
        "include_use_case_context": True       # Purpose-based descriptions
    }
}

# ADAPTIVE CONFIDENCE THRESHOLDS - Based on model characteristics
ADAPTIVE_THRESHOLDS = {
    "grounding_dino": {
        "base_confidence": 0.25,
        "text_threshold": 0.20,
        "class_specific_adjustments": {
            "city_car": 0.30,      # Higher threshold for most common class
            "large_suv": 0.25,     # Standard threshold
            "van": 0.25,           # Standard threshold  
            "truck": 0.22,         # Lower for less common
            "bus": 0.20,           # Lower for rare class
            "motorcycle": 0.20,    # Lower for rare class
            "non-vehicle": 0.35    # Higher for non-vehicle detection
        }
    },
    "git": {
        "similarity_threshold": 0.4,
        "caption_length_preference": "medium",  # 20-40 tokens work best
        "use_ensemble_scoring": True,           # Average multiple similarities
        "boost_exact_matches": True            # Higher weight for exact term matches
    }
}

# PERFORMANCE OPTIMIZATION SETTINGS
OPTIMIZATION_CONFIGS = {
    "grounding_dino": {
        "enable_fast_inference": True,
        "use_efficient_prompts": True,         # Shorter, focused prompts
        "batch_text_processing": True,         # Process prompts together
        "cache_text_features": True,           # Cache encoded text features
        "use_smart_nms": True                  # Improved non-max suppression
    },
    "git": {
        "enable_caption_caching": True,        # Cache generated captions
        "use_beam_search": False,              # Faster greedy decoding
        "optimize_similarity_computation": True, # Vectorized similarity
        "precompute_class_embeddings": True,   # Pre-encode class descriptions
        "use_lightweight_similarity": True     # Faster similarity models
    }
}

# Detection model configurations - UPDATED TO INCLUDE PERSON CLASS
DETECTOR_CONFIGS = {
    "yolov12": {
        "model_sizes": {
            "small": "yolo12s.pt",
            "medium": "yolo12m.pt", 
            "large": "yolo12l.pt"
        },
        "default_model_size": "medium",
        "confidence_threshold": 0.25,
        "device_preference": ["cuda", "mps", "cpu"],
        # COCO class IDs: person, bicycle, car, motorcycle, bus, truck, train
        "vehicle_class_ids": [0, 1, 2, 3, 5, 7],
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
        # COCO class IDs: person, bicycle, car, motorcycle, bus, truck, train
        "vehicle_class_ids": [0, 1, 2, 3, 5, 7],
    },
    "ssd": {
        "model_sizes": {
            "small": "ssd_mobilenet_v3",
            "medium": "ssd_vgg16",
        },
        "default_model_size": "medium",
        "confidence_threshold": 0.25, 
        "device_preference": ["cuda", "mps", "cpu"],
        "vehicle_class_ids": [1, 2, 3, 4, 6, 7, 8],
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

# Main vehicle scoring parameters - UPDATED WITH VERTICAL/HORIZONTAL WEIGHTING
MAIN_VEHICLE_SCORING = {
    "centrality_weight": 0.7,      # Weight for distance from center
    "size_weight": 0.3,            # Weight for object size
    "min_area_threshold": 200,     # Minimum bounding box area in pixels
    "max_distance_threshold": 0.75, # Maximum normalized distance from center
    "vertical_centrality_weight": 0.65,    
    "horizontal_centrality_weight": 0.35,  
}

# Image preprocessing configuration (DISABLED)
PREPROCESSING_CONFIG = {
    "clahe_clip_limit": 2.0,
    "clahe_grid_size": (8, 8),
    "sharpen_amount": 0.3,  
    "denoise_strength": 3,
    "enhance_contrast": False,     # DISABLED
    "enhance_sharpness": False,    # DISABLED
    "reduce_noise": False,         # DISABLED
}

YOLO_WORLD_PROMPT_STRATEGIES = {
    "use_specific_descriptors": True,      # "small passenger car" vs "car"
    "include_secondary_objects": True,     # Add competing classes to reduce false positives
    "use_contextual_prompts": True,        # "car on road" vs "car"
    "multiple_prompts_per_class": True,    # Use 3-6 prompts per class
    "avoid_generic_terms": True,           # Avoid "vehicle", use specific types
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