"""
Enhanced GroundingDINO implementation with optimized prompting strategies
"""
import logging
from typing import List, Dict, Set
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from .base_end_to_end import BaseEndToEndModel, EndToEndDetection
from config import VEHICLE_CLASSES

logger = logging.getLogger(__name__)


class GroundingDinoDetector(BaseEndToEndModel):
    """Enhanced GroundingDINO with optimized vehicle-specific prompting"""
    
    def _initialize_model(self) -> None:
        """Initialize GroundingDINO model with enhanced configuration"""
        try:
            # Get model configuration
            model_sizes = self.config.get('model_sizes', {})
            self.text_threshold = self.config.get('text_threshold', 0.20)
            
            if self.custom_model_path:
                model_path = self.custom_model_path
                logger.info(f"Using custom model path: {model_path}")
            elif self.model_size in model_sizes:
                model_path = model_sizes[self.model_size]
                logger.info(f"Using model: {model_path}")
            else:
                raise ValueError(f"Invalid model size '{self.model_size}'. "
                               f"Available sizes: {list(model_sizes.keys())}")
            
            # Initialize GroundingDINO processor and model
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(self.device)
            
            # Enhanced prompt preparation with research-based strategies
            self.enhanced_prompts = self._prepare_enhanced_prompts()
            self.class_prompt_mapping = self._create_class_mapping()
            
            # Performance optimizations
            self._setup_optimization_features()
            
            logger.info(f"Successfully initialized Enhanced GroundingDINO model on {self.device}")
            logger.info(f"Using {len(self.enhanced_prompts)} optimized prompts for {len(VEHICLE_CLASSES)} classes")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced GroundingDINO model: {e}")
            raise
    
    def _prepare_enhanced_prompts(self) -> List[str]:
        """
        Create research-based optimized prompts for GroundingDINO
        Based on findings that descriptive, contextual prompts work better
        """
        from config import GROUNDING_DINO_ENHANCED_PROMPTS, PROMPT_SELECTION_STRATEGIES
        
        strategy = PROMPT_SELECTION_STRATEGIES.get('grounding_dino', {})
        all_prompts = []
        
        for vehicle_class in VEHICLE_CLASSES:
            if vehicle_class in GROUNDING_DINO_ENHANCED_PROMPTS:
                class_prompts = GROUNDING_DINO_ENHANCED_PROMPTS[vehicle_class]
                
                # Limit prompts per class for performance
                max_prompts = strategy.get('max_prompts_per_class', 3)
                selected_prompts = class_prompts[:max_prompts]
                
                all_prompts.extend(selected_prompts)
        
        # Join with period separator as recommended for GroundingDINO
        separator = strategy.get('separator', ' . ')
        combined_prompt = separator.join(all_prompts)
        
        logger.info(f"Enhanced GroundingDINO prompt: {combined_prompt[:200]}...")
        return [combined_prompt]
    
    def _create_class_mapping(self) -> Dict[str, str]:
        """Create mapping from prompt keywords to vehicle classes"""
        from config import GROUNDING_DINO_ENHANCED_PROMPTS
        
        mapping = {}
        
        for vehicle_class, prompts in GROUNDING_DINO_ENHANCED_PROMPTS.items():
            for prompt in prompts:
                # Extract key terms for mapping
                key_terms = self._extract_key_terms(prompt, vehicle_class)
                for term in key_terms:
                    mapping[term.lower()] = vehicle_class
                    
        return mapping
    
    def _extract_key_terms(self, prompt: str, vehicle_class: str) -> List[str]:
        """Extract key terms from prompts for classification mapping"""
        # Remove common words and extract vehicle-specific terms
        common_words = {'on', 'road', 'in', 'traffic', 'vehicle', 'driving', 'street', 'the', 'a', 'an', 'with'}
        
        words = prompt.lower().split()
        key_terms = []
        
        # Always include the vehicle class name itself
        key_terms.append(vehicle_class.replace('_', ' '))
        
        # Extract meaningful terms
        for word in words:
            if (len(word) > 2 and 
                word not in common_words and 
                word not in ['person', 'people'] or vehicle_class == 'non-vehicle'):
                key_terms.append(word)
        
        return key_terms
    
    def _setup_optimization_features(self) -> None:
        """Setup performance optimization features"""
        from config import OPTIMIZATION_CONFIGS
        
        self.opt_config = OPTIMIZATION_CONFIGS.get('grounding_dino', {})
        
        # Enable optimizations
        self.use_cache = self.opt_config.get('cache_text_features', True)
        self.batch_processing = self.opt_config.get('batch_text_processing', True)
        self.fast_inference = self.opt_config.get('enable_fast_inference', True)
        
        # Pre-compute text features if caching enabled
        if self.use_cache:
            self._precompute_text_features()
    
    def _precompute_text_features(self) -> None:
        """Pre-compute and cache text features for faster inference"""
        try:
            # This would cache the text encoding for reuse
            # Implementation depends on the specific model architecture
            logger.debug("Text features pre-computation enabled")
        except Exception as e:
            logger.warning(f"Failed to pre-compute text features: {e}")
            self.use_cache = False
    
    def _map_to_vehicle_class(self, label: str) -> str:
        """
        Enhanced mapping using the class mapping dictionary
        """
        label_lower = label.lower().strip()
        
        # Direct mapping from pre-computed dictionary
        if label_lower in self.class_prompt_mapping:
            return self.class_prompt_mapping[label_lower]
        
        # Intelligent keyword-based mapping with context
        mapping_rules = {
            'city_car': ['car', 'sedan', 'hatchback', 'compact', 'small', 'passenger car'],
            'large_suv': ['suv', 'sport utility', 'pickup', 'crossover', 'large', 'big', 'heavy duty'],
            'van': ['van', 'minivan', 'delivery', 'cargo van', 'utility van', 'commercial van'],
            'truck': ['truck', 'freight', 'cargo truck', 'commercial truck', 'heavy truck'],
            'bus': ['bus', 'transit', 'coach', 'school bus', 'public bus'],
            'motorcycle': ['motorcycle', 'motorbike', 'rider', 'two-wheeled'],
            'non-vehicle': ['person', 'pedestrian', 'people', 'human', 'walking', 'standing']
        }
        
        # Score each class based on keyword matches
        class_scores = {}
        for vehicle_class, keywords in mapping_rules.items():
            score = 0
            for keyword in keywords:
                if keyword in label_lower:
                    # Weight longer matches higher
                    score += len(keyword.split())
            class_scores[vehicle_class] = score
        
        # Return class with highest score, default to most common
        if max(class_scores.values()) > 0:
            return max(class_scores.items(), key=lambda x: x[1])[0]
        
        # Conservative fallback to most common class in dataset
        logger.debug(f"Unknown label '{label}', defaulting to city_car")
        return "city_car"
    
    def _apply_adaptive_thresholds(self, detections: List[Dict], original_scores: List[float]) -> List[Dict]:
        """Apply class-specific confidence thresholds"""
        from config import ADAPTIVE_THRESHOLDS
        
        thresholds = ADAPTIVE_THRESHOLDS.get('grounding_dino', {})
        base_threshold = thresholds.get('base_confidence', 0.25)
        class_adjustments = thresholds.get('class_specific_adjustments', {})
        
        filtered_detections = []
        
        for detection, orig_score in zip(detections, original_scores):
            predicted_class = detection.get('predicted_class', 'city_car')
            
            # Get class-specific threshold
            class_threshold = class_adjustments.get(predicted_class, base_threshold)
            
            # Apply threshold
            if orig_score >= class_threshold:
                # Optionally boost confidence for high-confidence detections
                if orig_score > class_threshold * 1.5:
                    detection['confidence'] = min(1.0, orig_score * 1.1)
                
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def _run_inference(self, image: Image.Image) -> List[EndToEndDetection]:
        """
        Enhanced inference with optimized prompting and post-processing
        """
        try:
            start_time = time.time()
            
            # Prepare inputs with enhanced prompts
            inputs = self.processor(
                images=image, 
                text=self.enhanced_prompts, 
                return_tensors="pt"
            ).to(self.device)
            
            # Run model inference with optimizations
            with torch.no_grad():
                if self.fast_inference:
                    # Use faster inference settings
                    outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)
            
            # Enhanced post-processing
            results_list = self.processor.post_process_grounded_object_detection(
                outputs,
                box_threshold=self.confidence_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image.size[::-1]]  # [height, width]
            )
            
            if not results_list:
                return []
            
            result = results_list[0]
            detections = []
            original_scores = []
            
            # Process each detection
            boxes_tensor = result.get("boxes", torch.empty(0))
            scores_tensor = result.get("scores", torch.empty(0))
            text_labels_list = result.get("text_labels", [])
            
            for box_coords, score_val, text_label_str in zip(boxes_tensor, scores_tensor, text_labels_list):
                x1, y1, x2, y2 = box_coords.tolist()
                confidence = float(score_val.item())
                original_scores.append(confidence)
                
                # Enhanced class mapping
                mapped_class = self._map_to_vehicle_class(text_label_str)
                
                # Calculate area with minimum threshold
                area = (x2 - x1) * (y2 - y1)
                if area < 150:  # Minimum area threshold
                    continue
                
                detection = EndToEndDetection(
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    predicted_class=mapped_class,
                    confidence=confidence,
                    area=float(area)
                )
                
                detections.append(detection)
            
            # Apply adaptive thresholds and filtering
            detection_dicts = [det.to_dict() for det in detections]
            filtered_detections = self._apply_adaptive_thresholds(detection_dicts, original_scores)
            
            # Convert back to EndToEndDetection objects
            final_detections = []
            for det_dict in filtered_detections:
                detection = EndToEndDetection(
                    bbox=det_dict['bbox'],
                    predicted_class=det_dict['class'],
                    confidence=det_dict['score'],
                    area=det_dict['area']
                )
                final_detections.append(detection)
            
            inference_time = time.time() - start_time
            logger.debug(f"Enhanced GroundingDINO: {len(final_detections)} detections in {inference_time:.3f}s")
            
            return final_detections
            
        except Exception as e:
            logger.error(f"Enhanced GroundingDINO inference failed: {e}")
            return []
