"""
Improved OWLv2 model implementation for zero-shot object detection and classification
"""
import time
import os
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
from PIL import Image
import cv2

from config import END_TO_END_CONFIGS, VEHICLE_CLASSES


class OWLv2Detector:
    """OWLv2 detector for zero-shot object detection and classification"""
    
    MODEL_SIZES = {
        "base": "google/owlv2-base-patch16",
        "large": "google/owlv2-large-patch14",
        "base-ensemble": "google/owlv2-base-patch16-ensemble"
    }
    
    def __init__(self, model_size: str = "base", confidence_threshold: float = None, custom_model_path: Optional[str] = None):
        """
        Initialize OWLv2 detector
        
        Args:
            model_size: Size of the model (base, large, base-ensemble)
            confidence_threshold: Confidence threshold for detections
            custom_model_path: Path to custom model weights (overrides model_size if provided)
        """
        # Use config values if not provided
        config = END_TO_END_CONFIGS.get('owlv2', {'confidence_threshold': 0.25})
        self._model_size = model_size.lower()
        self.confidence_threshold = confidence_threshold or config.get('confidence_threshold', 0.25)
        
        # Set up model path
        if custom_model_path:
            self.model_path = custom_model_path
        else:
            if self._model_size not in self.MODEL_SIZES:
                raise ValueError(f"Invalid model size: {self._model_size}. "
                              f"Choose from {list(self.MODEL_SIZES.keys())}")
            self.model_path = self.MODEL_SIZES[self._model_size]
        
        # Device configuration
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"OWLv2 detector using device: {self.device}")
        
        # Store classes for detection
        self.vehicle_classes = ["car", "minivan", "truck", "bus", "person"]
        
        self._init_model()
        
    def _init_model(self):
        """Initialize OWLv2 model - improved with direct model component access"""
        try:
            # Import model components directly instead of using pipeline
            from transformers import Owlv2Processor, Owlv2ForObjectDetection
            
            # Initialize processor and model
            self.processor = Owlv2Processor.from_pretrained(self.model_path)
            self.model = Owlv2ForObjectDetection.from_pretrained(self.model_path)
            
            # Move model to appropriate device
            self.model = self.model.to(self.device)
            
            # For backward compatibility
            self.detector = None
            
            print(f"OWLv2 model initialized with classes: {self.vehicle_classes}")
            
        except Exception as e:
            print(f"Error initializing OWLv2 model: {e}")
            raise
    
    def detect_and_classify(self, image: np.ndarray) -> Tuple[Dict, float]:
        """
        Detect and classify objects in the image using OWLv2
        
        Args:
            image: Input image (RGB format, numpy array)
            
        Returns:
            Tuple of (detection/classification results, inference time)
        """
        # Start timing
        start_time = time.time()
        
        # Convert numpy array to PIL Image if it's not already
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Prepare text labels in the format expected by the model
        # Convert vehicle classes to prompt format
        text_queries = [[f"a photo of a {cls}" for cls in self.vehicle_classes]]
        
        # Run OWLv2 detection
        try:
            # Process inputs
            inputs = self.processor(
                text=text_queries,
                images=pil_image,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run model
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Target image sizes for rescaling boxes
            target_sizes = torch.tensor([(pil_image.height, pil_image.width)]).to(self.device)
            
            # Post-process results
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.confidence_threshold,
                text_labels=text_queries
            )
            
            # Extract results for the first image
            result = results[0]
            boxes, scores, labels = result["boxes"], result["scores"], result["text_labels"]
            
            # Convert to CPU and numpy for processing
            boxes_np = boxes.cpu().numpy()
            scores_np = scores.cpu().numpy()
            
        except Exception as e:
            print(f"Prediction error: {e}")
            # Return empty results on error
            return {'detections': [], 'main_vehicle': None}, time.time() - start_time
        
        # Process results into the same format as before for compatibility
        processed_detections = []
        
        # Extract the original class names from the text prompts
        for i, (box, score, label) in enumerate(zip(boxes_np, scores_np, labels)):
            # Extract class name from the prompt format "a photo of a {class}"
            class_name = label.replace("a photo of a ", "")
            
            # Get box coordinates (already in x1, y1, x2, y2 format)
            x1, y1, x2, y2 = box.tolist()
            
            # Standardize class name
            standard_class = self._map_to_vehicle_class(class_name)
            
            # Calculate area
            area = (x2 - x1) * (y2 - y1)
            
            processed_detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'class': standard_class,
                'original_class': class_name,
                'score': float(score),
                'area': float(area),
            })
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Find main vehicle
        main_vehicle = self.find_main_vehicle(processed_detections, np.array(pil_image).shape)
        
        return {
            'detections': processed_detections,
            'main_vehicle': main_vehicle,
        }, inference_time
    
    def batch_detect(self, images: List[Union[np.ndarray, Image.Image]], batch_size: int = 4) -> List[Dict]:
        """
        Batch detection for multiple images
        
        Args:
            images: List of images (either numpy arrays or PIL Images)
            batch_size: Batch size for processing
            
        Returns:
            List of detection results for each image
        """
        results = []
        total_time = 0
        
        # Process images in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_results, batch_time = self._process_batch(batch_images)
            results.extend(batch_results)
            total_time += batch_time
            
        print(f"Processed {len(images)} images in {total_time:.2f}s (avg: {total_time/len(images):.2f}s per image)")
        return results
    
    def _process_batch(self, batch_images: List[Union[np.ndarray, Image.Image]]) -> Tuple[List[Dict], float]:
        """
        Process a batch of images
        
        Args:
            batch_images: List of images to process
            
        Returns:
            Tuple of (list of detection results, inference time)
        """
        start_time = time.time()
        
        # Convert all images to PIL
        pil_images = []
        for img in batch_images:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img).convert("RGB"))
            elif isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
        
        # Prepare text labels
        text_queries = [[f"a photo of a {cls}" for cls in self.vehicle_classes]]
        
        try:
            # Process inputs for batch
            inputs = self.processor(
                text=text_queries * len(pil_images),  # Repeat for each image
                images=pil_images,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run model
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Target image sizes for rescaling boxes
            target_sizes = torch.tensor([(img.height, img.width) for img in pil_images]).to(self.device)
            
            # Post-process results
            batch_results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.confidence_threshold,
                text_labels=text_queries * len(pil_images)
            )
            
        except Exception as e:
            print(f"Batch prediction error: {e}")
            # Return empty results on error
            return [{'detections': [], 'main_vehicle': None} for _ in batch_images], time.time() - start_time
        
        # Process each image's results
        processed_results = []
        for idx, (result, pil_img) in enumerate(zip(batch_results, pil_images)):
            boxes, scores, labels = result["boxes"], result["scores"], result["text_labels"]
            
            # Convert to CPU and numpy
            boxes_np = boxes.cpu().numpy()
            scores_np = scores.cpu().numpy()
            
            # Process detections for this image
            processed_detections = []
            for box, score, label in zip(boxes_np, scores_np, labels):
                # Extract class name from prompt
                class_name = label.replace("a photo of a ", "")
                
                # Get box coordinates
                x1, y1, x2, y2 = box.tolist()
                
                # Standardize class
                standard_class = self._map_to_vehicle_class(class_name)
                
                # Calculate area
                area = (x2 - x1) * (y2 - y1)
                
                processed_detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'class': standard_class,
                    'original_class': class_name,
                    'score': float(score),
                    'area': float(area),
                })
            
            # Find main vehicle for this image
            main_vehicle = self.find_main_vehicle(processed_detections, np.array(pil_img).shape)
            
            processed_results.append({
                'detections': processed_detections,
                'main_vehicle': main_vehicle,
            })
        
        inference_time = time.time() - start_time
        return processed_results, inference_time
    
    def _map_to_vehicle_class(self, label: str) -> str:
        """Map the model's output label to standard vehicle classes"""
        
        # But let's still do some standardization just in case
        label_lower = label.lower()
        
        if "person" in label_lower or "non-vehicle" in label_lower:
            return "non-vehicle"
        elif any(x in label_lower for x in ["car", "sedan", "coupe", "suv"]):
            return "car"
        elif "van" in label_lower or "minivan" in label_lower:
            return "van"
        elif "truck" in label_lower:
            return "truck"
        elif "bus" in label_lower:
            return "bus"
        elif any(x in label_lower for x in ["ambulance", "police", "emergency", "fire"]):
            return "emergency"
        
        # The label should already be in our vehicle_classes list, so return as is
        if label in self.vehicle_classes:
            return label
            
        # Fallback
        return "non-vehicle"
    
    def find_main_vehicle(self, detections: List[Dict], image_shape: Tuple[int, int, int]) -> Dict:
        """
        Find the main vehicle in the image based on centrality and size
        
        Args:
            detections: List of detections from the model
            image_shape: Image shape (height, width, channels)
            
        Returns:
            Dict containing the main vehicle detection
        """
        if not detections:
            return None
        
        # Get image dimensions
        img_height, img_width = image_shape[0], image_shape[1]
        img_center = (img_width / 2, img_height / 2)
        
        # Define scoring parameters
        center_weight = 2.0    # Weight for centrality
        size_weight = 1.1      # Weight for size
        emergency_bonus = 1.5  # Bonus for emergency vehicles
        
        # Calculate score for each detection
        for det in detections:
            bbox = det['bbox']
            # Calculate bbox center
            bbox_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            # Calculate distance from image center
            distance = np.sqrt((bbox_center[0] - img_center[0])**2 + 
                               (bbox_center[1] - img_center[1])**2)
            
            # Normalize distance by image diagonal
            img_diagonal = np.sqrt(img_width**2 + img_height**2)
            normalized_distance = distance / img_diagonal
            
            # Calculate centrality score (1 - normalized_distance)
            centrality = 1 - normalized_distance
            
            # Calculate size score (normalized by image area)
            size_score = det['area'] / (img_width * img_height)
            
            # Apply emergency vehicle bonus
            emergency_factor = emergency_bonus if det['class'] == 'emergency' else 1.0
            
            # Combined score (balance of centrality and size with emergency bonus)
            det['combined_score'] = (center_weight * centrality + size_weight * size_score) * emergency_factor
        
        # Sort by combined score (descending)
        detections.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Filter out non-vehicles
        vehicles = [det for det in detections if det['class'] != 'non-vehicle']
        
        # Return the highest scored vehicle, or highest scored detection if no vehicles
        return vehicles[0] if vehicles else (detections[0] if detections else None)
    
    def crop_bbox(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Crop image to the specified bounding box
        
        Args:
            image: Input image (RGB format, numpy array)
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            Cropped image
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Crop image
        cropped = image[y1:y2, x1:x2]
        
        return cropped
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict], 
                             highlight_main: bool = True, 
                             font_scale: float = 0.5,
                             thickness: int = 2) -> np.ndarray:
        """
        Visualize detections on the image
        
        Args:
            image: Input image (RGB format, numpy array)
            detections: List of detections
            highlight_main: Whether to highlight the main vehicle
            font_scale: Font scale for text
            thickness: Line thickness
            
        Returns:
            Image with visualized detections
        """
        # Clone image to avoid modifying the original
        vis_image = image.copy()
        
        # Define colors for different classes (BGR format for OpenCV)
        class_colors = {
            'car': (0, 255, 0),       # Green
            'van': (255, 0, 0),       # Blue
            'truck': (0, 0, 255),     # Red
            'bus': (255, 255, 0),     # Cyan
            'emergency': (0, 165, 255),  # Orange
            'non-vehicle': (128, 128, 128)  # Gray
        }
        
        # Default color for unknown classes
        default_color = (255, 0, 255)  # Magenta
        
        # Find main vehicle for highlighting
        main_vehicle = None
        if highlight_main:
            main_vehicles = [d for d in detections if d.get('combined_score') is not None]
            if main_vehicles:
                main_vehicles.sort(key=lambda x: x['combined_score'], reverse=True)
                main_vehicle = main_vehicles[0]
        
        # Draw each detection
        for det in detections:
            # Get bounding box
            x1, y1, x2, y2 = [int(c) for c in det['bbox']]
            
            # Get class and score
            class_name = det['class']
            score = det['score']
            
            # Determine color
            color = class_colors.get(class_name, default_color)
            
            # Highlight main vehicle with thicker lines
            line_thickness = thickness * 2 if main_vehicle is det else thickness
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, line_thickness)
            
            # Prepare label text
            label = f"{class_name}: {score:.2f}"
            
            # Add 'MAIN' to label if this is the main vehicle
            if main_vehicle is det:
                label = "MAIN: " + label
            
            # Get text size
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            
            # Draw label background
            cv2.rectangle(
                vis_image, 
                (x1, y1 - label_height - baseline), 
                (x1 + label_width, y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                vis_image, 
                label, 
                (x1, y1 - baseline), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (255, 255, 255), 
                1, 
                cv2.LINE_AA
            )
        
        return vis_image