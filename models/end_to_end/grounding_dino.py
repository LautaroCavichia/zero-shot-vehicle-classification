"""
GroundingDINO model implementation for zero-shot object detection and classification
"""
import time
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image

from config import END_TO_END_CONFIGS, VEHICLE_CLASSES

class GroundingDINODetector:
    """GroundingDINO detector for zero-shot object detection and classification"""
    
    MODEL_SIZES = {
        "tiny": "IDEA-Research/grounding-dino-tiny",
        "base": "IDEA-Research/grounding-dino-base",
        "large": "IDEA-Research/grounding-dino-large"
    }
    
    def __init__(self, model_size: str = "tiny", 
                 confidence_threshold: float = None, 
                 custom_model_path: Optional[str] = None,
                 vehicle_classes_override: Optional[List[str]] = None):
        """
        Initialize GroundingDINO detector
        
        Args:
            model_size: Size of the model (tiny, base, large)
            confidence_threshold: Confidence threshold for detections
            custom_model_path: Path to custom model weights (overrides model_size if provided)
            vehicle_classes_override: Optionally override the VEHICLE_CLASSES from config.
        """
        # Use config values if not provided
        # Ensure END_TO_END_CONFIGS is treated as a dictionary
        config_source = END_TO_END_CONFIGS if isinstance(END_TO_END_CONFIGS, dict) else {}
        g_dino_config = config_source.get('dino', {})
        
        self._model_size = model_size.lower()
        
        self.confidence_threshold = confidence_threshold or g_dino_config.get('confidence_threshold', 0.35)
        self.text_threshold = g_dino_config.get('text_threshold', 0.25) # Get text_threshold from config
        
        # Set up vehicle classes list
        if vehicle_classes_override is not None:
            self.vehicle_classes_list = vehicle_classes_override
            print(f"Using provided vehicle_classes_override: {self.vehicle_classes_list}")
        elif 'VEHICLE_CLASSES' in globals() and VEHICLE_CLASSES is not None:
            self.vehicle_classes_list = VEHICLE_CLASSES
            print(f"Using VEHICLE_CLASSES from config: {self.vehicle_classes_list}")
        else:
            self.vehicle_classes_list = ["car", "truck", "bus", "van", "person"] 
            print(f"Warning: VEHICLE_CLASSES not found in config or override. Using default: {self.vehicle_classes_list}")

        # Set up model path
        if custom_model_path:
            self.model_path = custom_model_path
        else:
            if self._model_size not in self.MODEL_SIZES:
                raise ValueError(f"Invalid model size: {self._model_size}. "
                              f"Choose from {list(self.MODEL_SIZES.keys())}")
            self.model_path = self.MODEL_SIZES[self._model_size]
        
        # Device configuration
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"GroundingDINO detector using device: {self.device}")
        self._init_model()
        
    def _init_model(self):
        """Initialize GroundingDINO model"""
        try:
            # Import using exactly the same imports as in the example
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_path).to(self.device)
            
            # Create text prompt from self.vehicle_classes_list
            # The structure [self.vehicle_classes_list] results in List[List[str]]
            # e.g., [["car", "truck", "bus"]] which matches the example's text_labels format for a single image.
            self.text_prompt = [self.vehicle_classes_list]
            print(f"GroundingDINO model initialized with model size: {self._model_size} ")
            
        except Exception as e:
            print(f"Error initializing GroundingDINO model: {e}")
            raise
    
    def detect_and_classify(self, image: np.ndarray) -> Tuple[Dict, float]:
        """
        Detect and classify objects in the image using GroundingDINO
        
        Args:
            image: Input image (RGB format, numpy array)
            
        Returns:
            Tuple of (detection/classification results, inference time)
        """
        # Start timing
        start_time = time.time()
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image).convert("RGB")
        
        # Run GroundingDINO detection
        try:
            # Prepare inputs
            inputs = self.processor(images=pil_image, text=self.text_prompt, return_tensors="pt").to(self.device)
            
            # Run model inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process results
            # Removed inputs.input_ids from the call to align with the Hugging Face example.
            # The example's `threshold` parameter maps to `box_threshold`.
            results_list = self.processor.post_process_grounded_object_detection(
                outputs,
                # input_ids=inputs.input_ids, # This was removed
                box_threshold=self.confidence_threshold, 
                text_threshold=self.text_threshold,
                target_sizes=[pil_image.size[::-1]]  # [height, width]
            )
            
            # Ensure results_list is not None and is a list with at least one element
            if results_list is None or not isinstance(results_list, list) or not results_list:
                 print("Warning: GroundingDINO post_process_grounded_object_detection returned None or empty list.")
                 return {'detections': [], 'main_vehicle': None}, time.time() - start_time

            result = results_list[0]  # First batch result (assuming batch size of 1)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            # Return empty results on error
            return {'detections': [], 'main_vehicle': None}, time.time() - start_time
        
        # Process results into the desired format
        processed_detections = []
        
        # Ensure all necessary keys exist in the result dictionary before processing.
        # Use .get() with default empty tensors/lists to prevent KeyError.
        boxes_tensor = result.get("boxes", torch.empty(0))
        scores_tensor = result.get("scores", torch.empty(0))
        # Use "text_labels" as per the Hugging Face example for string labels
        text_labels_list = result.get("text_labels", []) 

        # Check if lengths of boxes, scores, and labels match.
        if not (len(boxes_tensor) == len(scores_tensor) == len(text_labels_list)):
            print(f"Warning: Mismatch in lengths of detected components: "
                  f"boxes ({len(boxes_tensor)}), scores ({len(scores_tensor)}), text_labels ({len(text_labels_list)}). "
                  f"Skipping processing for this image.")
        else:
            for box_coords, score_val, text_label_str in zip(boxes_tensor, scores_tensor, text_labels_list):
                # Convert box format: [x1, y1, x2, y2]
                x1, y1, x2, y2 = box_coords.tolist()
                
                # Ensure class name is standardized using the mapping function
                class_name = self._map_to_vehicle_class(text_label_str) 
                
                # Calculate area
                area = (x2 - x1) * (y2 - y1)
                
                processed_detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'class': class_name,
                    'original_class': text_label_str, # Store the original label from the model
                    'score': float(score_val.item()),
                    'area': float(area),
                })
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Find main vehicle
        main_vehicle = self.find_main_vehicle(processed_detections, image.shape)
        
        return {
            'detections': processed_detections,
            'main_vehicle': main_vehicle,
        }, inference_time
    
    def _map_to_vehicle_class(self, label: str) -> str:
        """Map the model's output label to standard vehicle classes"""
        label_lower = label.lower()
        
        # This mapping logic is specific to your application.
        if "person" in label_lower or "non-vehicle" in label_lower or "pedestrian" in label_lower:
            return "non-vehicle"
        elif any(term in label_lower for term in ["car", "sedan", "coupe"]): 
            return "car"
        elif any(term in label_lower for term in ["van", "minivan", "delivery van", "panel van", "commercial van"]):
            return "van"
        elif "truck" in label_lower:
            return "truck"
        elif "bus" in label_lower:
            return "bus"
        elif any(term in label_lower for term in ["ambulance", "police", "emergency", "fire"]):
            return "emergency"
        else:
            # Try to find the best match from the instance's vehicle_classes_list
            if self.vehicle_classes_list: # Check if list is not None or empty
                for cls in self.vehicle_classes_list:
                    if cls.lower() in label_lower:
                        return cls # Return the matched class from the list
            # If no specific vehicle class matches, classify as 'non-vehicle'.
            return "non-vehicle" 
    
    def find_main_vehicle(self, detections: List[Dict], image_shape: Tuple[int, int, int]) -> Optional[Dict]:
        """
        Find the main vehicle in the image based on centrality and size.
        
        Args:
            detections: List of detections from the model.
            image_shape: Image shape (height, width, channels).
            
        Returns:
            Dict containing the main vehicle detection, or None if no suitable vehicle is found.
        """
        if not detections:
            return None
        
        img_height, img_width = image_shape[0], image_shape[1]
        img_center_x, img_center_y = img_width / 2, img_height / 2
        
        center_weight = 2.0
        size_weight = 1.1
        emergency_bonus = 1.5
        
        scored_detections = []
        for det in detections:
            if 'bbox' not in det or 'area' not in det or 'class' not in det: # Ensure necessary keys exist
                continue

            bbox = det['bbox']
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2
            
            distance = np.sqrt((bbox_center_x - img_center_x)**2 + (bbox_center_y - img_center_y)**2)
            
            img_diagonal = np.sqrt(img_width**2 + img_height**2)
            normalized_distance = distance / img_diagonal if img_diagonal > 0 else 1.0 # Avoid division by zero
            
            centrality = 1 - normalized_distance
            
            img_area = img_width * img_height
            size_score = det['area'] / img_area if img_area > 0 else 0.0 # Avoid division by zero
            
            emergency_factor = emergency_bonus if det['class'] == 'emergency' else 1.0
            
            det['combined_score'] = (center_weight * centrality + size_weight * size_score) * emergency_factor
            scored_detections.append(det)
        
        if not scored_detections: # If all detections lacked necessary keys
            return None

        scored_detections.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        # Filter for vehicles only
        vehicles = [det for det in scored_detections if det.get('class') != 'non-vehicle']
        
        # Return the highest-scored vehicle if any, otherwise None.
        return vehicles[0] if vehicles else None
    
    def crop_bbox(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Crop image to the specified bounding box.
        
        Args:
            image: Input image (RGB format, numpy array).
            bbox: Bounding box coordinates [x1, y1, x2, y2].
            
        Returns:
            Cropped image as a numpy array. Returns an empty array if the bbox is invalid.
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        h, w = image.shape[:2]
        # Clamp coordinates to be within image bounds
        x1_c = max(0, x1)
        y1_c = max(0, y1)
        x2_c = min(w, x2)
        y2_c = min(h, y2)
        
        # Check if the clamped bounding box is valid (width and height are positive)
        if x1_c >= x2_c or y1_c >= y2_c:
            # Return an empty array with the correct number of channels if the box is invalid
            channels = image.shape[2] if image.ndim == 3 else 0
            if channels > 0:
                return np.array([], dtype=image.dtype).reshape(0, 0, channels)
            else: # Grayscale or other
                return np.array([], dtype=image.dtype).reshape(0, 0)

        cropped_image = image[y1_c:y2_c, x1_c:x2_c]
        return cropped_image

