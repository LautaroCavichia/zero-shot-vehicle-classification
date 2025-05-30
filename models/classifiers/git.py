"""
Enhanced GIT classifier with optimized generative prompting strategies
"""
import logging
from typing import List, Dict, Tuple
import time
import warnings

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

from .base_classifier import BaseClassifier, ClassificationResult
from config import VEHICLE_CLASSES

logger = logging.getLogger(__name__)


class GitClassifier(BaseClassifier):
    """Enhanced GIT classifier with optimized caption-based classification"""
    
    def _initialize_model(self) -> None:
        """Initialize enhanced GIT model with optimized configuration"""
        try:
            # Get model configuration
            git_model_name = self.config.get('model_name', 'microsoft/git-base')
            similarity_model_name = self.config.get('similarity_model', 'all-MiniLM-L6-v2')
            self.max_caption_length = self.config.get('max_caption_length', 50)
            
            # Initialize GIT model for captioning
            self.git_processor = AutoProcessor.from_pretrained(git_model_name)
            self.git_model = AutoModelForCausalLM.from_pretrained(git_model_name).to(self.device)
            
            # Initialize sentence transformer for similarity
            self.similarity_model = SentenceTransformer(similarity_model_name, device=self.device)
            
            # Enhanced class descriptions and embeddings
            self._prepare_enhanced_class_embeddings()
            
            # Setup optimization features
            self._setup_optimization_features()
            
            logger.info(f"Successfully initialized Enhanced GIT model '{git_model_name}' and "
                       f"similarity model '{similarity_model_name}' on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced GIT classifier: {e}")
            raise
    
    def _prepare_enhanced_class_embeddings(self) -> None:
        """Prepare enhanced class descriptions with multiple strategies"""
        from config import GIT_ENHANCED_PROMPTS
        
        # Enhanced descriptions with varied sentence structures
        self.enhanced_descriptions = {}
        self.description_to_class = {}
        self.all_descriptions = []
        
        for cls in VEHICLE_CLASSES:
            if cls in GIT_ENHANCED_PROMPTS:
                cls_descriptions = GIT_ENHANCED_PROMPTS[cls]
            else:
                # Fallback to basic descriptions
                cls_descriptions = self._generate_fallback_descriptions(cls)
            
            self.enhanced_descriptions[cls] = cls_descriptions
            self.all_descriptions.extend(cls_descriptions)
            
            for desc in cls_descriptions:
                self.description_to_class[desc] = cls
        
        # Pre-compute embeddings for all class descriptions
        logger.debug("Pre-computing enhanced embeddings for class descriptions...")
        self.description_embeddings = self.similarity_model.encode(
            self.all_descriptions, 
            convert_to_tensor=True, 
            device=self.device,
            show_progress_bar=False
        )
        
        logger.debug(f"Pre-computed {len(self.all_descriptions)} enhanced description embeddings")
    
    def _generate_fallback_descriptions(self, vehicle_class: str) -> List[str]:
        """Generate fallback descriptions for classes not in config"""
        class_name = vehicle_class.replace('_', ' ')
        
        return [
            f"This image shows a {class_name}",
            f"A {class_name} vehicle",
            f"A {class_name} on the road",
            f"A {class_name} in traffic",
            f"A {class_name} driving"
        ]
    
    def _setup_optimization_features(self) -> None:
        """Setup performance optimization features"""
        from config import OPTIMIZATION_CONFIGS
        
        self.opt_config = OPTIMIZATION_CONFIGS.get('git', {})
        
        # Optimization settings
        self.use_caption_cache = self.opt_config.get('enable_caption_caching', True)
        self.use_beam_search = self.opt_config.get('use_beam_search', False)
        self.optimize_similarity = self.opt_config.get('optimize_similarity_computation', True)
        self.use_ensemble_scoring = self.opt_config.get('use_ensemble_scoring', True)
        
        # Caption cache for repeated images (if enabled)
        if self.use_caption_cache:
            self.caption_cache = {}
    
    def _generate_enhanced_caption(self, image_inputs: Dict) -> str:
        """
        Generate enhanced caption with optimization strategies
        """
        try:
            with torch.no_grad():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    if self.use_beam_search:
                        # Higher quality but slower
                        generated_ids = self.git_model.generate(
                            pixel_values=image_inputs["pixel_values"],
                            max_length=self.max_caption_length,
                            num_beams=3,
                            do_sample=False,
                            early_stopping=True,
                            pad_token_id=self.git_processor.tokenizer.eos_token_id
                        )
                    else:
                        # Faster greedy decoding
                        generated_ids = self.git_model.generate(
                            pixel_values=image_inputs["pixel_values"],
                            max_length=self.max_caption_length,
                            do_sample=False,
                            pad_token_id=self.git_processor.tokenizer.eos_token_id,
                            temperature=1.0
                        )
            
            # Decode the generated caption
            caption = self.git_processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            # Enhanced caption validation and cleaning
            caption = self._clean_and_validate_caption(caption)
            
            return caption
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced caption: {e}")
            return "a vehicle"  # Safe fallback
    
    def _clean_and_validate_caption(self, caption: str) -> str:
        """Clean and validate generated caption"""
        if not caption or len(caption.split()) < 2:
            return "a vehicle in the image"
        
        # Remove redundant phrases and clean up
        caption = caption.lower().strip()
        
        # Remove common GIT artifacts
        unwanted_phrases = ["<unk>", "[unk]", "unk", "<pad>", "[pad]"]
        for phrase in unwanted_phrases:
            caption = caption.replace(phrase, "")
        
        # Ensure minimum meaningful content
        words = caption.split()
        if len(words) < 3:
            caption = f"a vehicle {caption}".strip()
        
        return caption
    
    def _compute_enhanced_similarity(self, caption: str) -> Dict[str, float]:
        """
        Compute enhanced similarity scores with multiple strategies
        """
        try:
            # Encode the generated caption
            caption_embedding = self.similarity_model.encode(
                caption, 
                convert_to_tensor=True, 
                device=self.device
            )
            
            # Calculate cosine similarity with all pre-computed embeddings
            if self.optimize_similarity:
                # Vectorized computation for speed
                cosine_scores = util.pytorch_cos_sim(caption_embedding, self.description_embeddings)[0]
            else:
                # Standard computation
                cosine_scores = torch.cosine_similarity(
                    caption_embedding.unsqueeze(0), 
                    self.description_embeddings,
                    dim=1
                )
            
            cosine_scores = cosine_scores.cpu().numpy()
            
            # Enhanced aggregation strategies
            if self.use_ensemble_scoring:
                class_scores = self._compute_ensemble_scores(caption, cosine_scores)
            else:
                class_scores = self._compute_basic_scores(cosine_scores)
            
            return class_scores
            
        except Exception as e:
            logger.error(f"Enhanced similarity computation failed: {e}")
            # Return uniform distribution as fallback
            return {cls: 1.0/len(VEHICLE_CLASSES) for cls in VEHICLE_CLASSES}
    
    def _compute_ensemble_scores(self, caption: str, cosine_scores: np.ndarray) -> Dict[str, float]:
        """
        Compute ensemble scores using multiple similarity strategies
        """
        from config import ADAPTIVE_THRESHOLDS
        
        # Strategy 1: Maximum similarity per class
        max_scores = {}
        for cls in VEHICLE_CLASSES:
            cls_indices = [i for i, desc in enumerate(self.all_descriptions) 
                          if self.description_to_class[desc] == cls]
            if cls_indices:
                max_scores[cls] = float(np.max(cosine_scores[cls_indices]))
            else:
                max_scores[cls] = 0.0
        
        # Strategy 2: Average similarity per class
        avg_scores = {}
        for cls in VEHICLE_CLASSES:
            cls_indices = [i for i, desc in enumerate(self.all_descriptions) 
                          if self.description_to_class[desc] == cls]
            if cls_indices:
                avg_scores[cls] = float(np.mean(cosine_scores[cls_indices]))
            else:
                avg_scores[cls] = 0.0
        
        # Strategy 3: Keyword boosting
        keyword_scores = self._compute_keyword_boost_scores(caption)
        
        # Ensemble combination with weights
        ensemble_scores = {}
        for cls in VEHICLE_CLASSES:
            ensemble_scores[cls] = (
                0.5 * max_scores[cls] +     # Maximum similarity weight
                0.3 * avg_scores[cls] +     # Average similarity weight  
                0.2 * keyword_scores[cls]   # Keyword boost weight
            )
        
        return ensemble_scores
    
    def _compute_basic_scores(self, cosine_scores: np.ndarray) -> Dict[str, float]:
        """
        Compute basic similarity scores (fallback method)
        """
        class_scores = {}
        
        for cls in VEHICLE_CLASSES:
            cls_indices = [i for i, desc in enumerate(self.all_descriptions) 
                          if self.description_to_class[desc] == cls]
            if cls_indices:
                # Use maximum similarity for each class
                class_scores[cls] = float(np.max(cosine_scores[cls_indices]))
            else:
                class_scores[cls] = 0.0
        
        return class_scores
    
    def _compute_keyword_boost_scores(self, caption: str) -> Dict[str, float]:
        """
        Compute keyword-based boosting scores for vehicle classes
        """
        caption_lower = caption.lower()
        
        # Define strong keywords for each class
        class_keywords = {
            'city_car': ['car', 'sedan', 'hatchback', 'compact', 'small car', 'passenger'],
            'large_suv': ['suv', 'large', 'big', 'pickup', 'truck', 'sport utility', 'crossover'],
            'van': ['van', 'minivan', 'delivery', 'cargo', 'utility'],
            'truck': ['truck', 'freight', 'cargo', 'commercial', 'heavy'],
            'bus': ['bus', 'transit', 'coach', 'school bus', 'public'],
            'motorcycle': ['motorcycle', 'motorbike', 'bike', 'rider', 'riding'],
            'non-vehicle': ['person', 'people', 'pedestrian', 'walking', 'standing', 'human']
        }
        
        keyword_scores = {}
        
        for cls in VEHICLE_CLASSES:
            score = 0.0
            keywords = class_keywords.get(cls, [])
            
            for keyword in keywords:
                if keyword in caption_lower:
                    # Weight longer keywords more heavily
                    weight = len(keyword.split()) * 0.3
                    score += weight
            
            # Normalize by number of keywords
            if keywords:
                score = score / len(keywords)
            
            keyword_scores[cls] = min(1.0, score)  # Cap at 1.0
        
        return keyword_scores
    
    def _apply_adaptive_confidence_boosting(self, class_scores: Dict[str, float], 
                                          caption: str) -> Dict[str, float]:
        """
        Apply adaptive confidence boosting based on caption content
        """
        from config import ADAPTIVE_THRESHOLDS
        
        git_thresholds = ADAPTIVE_THRESHOLDS.get('git', {})
        boost_exact_matches = git_thresholds.get('boost_exact_matches', True)
        
        if not boost_exact_matches:
            return class_scores
        
        boosted_scores = class_scores.copy()
        caption_lower = caption.lower()
        
        # Boost scores for exact class name matches
        class_name_mapping = {
            'city_car': ['car', 'sedan'],
            'large_suv': ['suv', 'pickup'],
            'van': ['van'],
            'truck': ['truck'],
            'bus': ['bus'],
            'motorcycle': ['motorcycle', 'motorbike'],
            'non-vehicle': ['person', 'people']
        }
        
        for cls, names in class_name_mapping.items():
            for name in names:
                if name in caption_lower:
                    # Apply boosting factor
                    boosted_scores[cls] = min(1.0, boosted_scores[cls] * 1.2)
                    break
        
        return boosted_scores
    
    def _normalize_scores_with_softmax(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Apply softmax normalization to convert scores to probabilities
        """
        import math
        
        # Convert to log space to avoid overflow
        max_score = max(scores.values()) if scores.values() else 0
        exp_scores = {}
        
        for cls, score in scores.items():
            exp_scores[cls] = math.exp(score - max_score)
        
        # Normalize
        total = sum(exp_scores.values())
        if total > 0:
            normalized_scores = {cls: score/total for cls, score in exp_scores.items()}
        else:
            # Uniform distribution fallback
            normalized_scores = {cls: 1.0/len(VEHICLE_CLASSES) for cls in VEHICLE_CLASSES}
        
        return normalized_scores
    
    def _run_inference(self, image: Image.Image) -> ClassificationResult:
        """
        Enhanced GIT inference with optimized caption generation and classification
        """
        try:
            start_time = time.time()
            
            # Check caption cache first (if enabled)
            cache_key = None
            if self.use_caption_cache:
                # Simple cache key based on image hash (in production, use proper image hashing)
                cache_key = str(hash(str(image.size) + str(image.mode)))
                if cache_key in self.caption_cache:
                    caption = self.caption_cache[cache_key]
                    logger.debug(f"Using cached caption: {caption}")
                else:
                    # Process image for GIT
                    image_inputs = self.git_processor(images=image, return_tensors="pt").to(self.device)
                    caption = self._generate_enhanced_caption(image_inputs)
                    self.caption_cache[cache_key] = caption
            else:
                # Process image for GIT
                image_inputs = self.git_processor(images=image, return_tensors="pt").to(self.device)
                caption = self._generate_enhanced_caption(image_inputs)
            
            logger.debug(f"Enhanced GIT caption: '{caption}'")
            
            # Compute enhanced similarity scores
            class_scores = self._compute_enhanced_similarity(caption)
            
            # Apply adaptive confidence boosting
            boosted_scores = self._apply_adaptive_confidence_boosting(class_scores, caption)
            
            # Normalize scores to probabilities
            normalized_scores = self._normalize_scores_with_softmax(boosted_scores)
            
            # Get the class with highest score
            top_class = max(normalized_scores.items(), key=lambda x: x[1])[0]
            top_score = normalized_scores[top_class]
            
            inference_time = time.time() - start_time
            
            logger.debug(f"Enhanced GIT classification: {top_class} ({top_score:.3f}) in {inference_time:.3f}s")
            
            return ClassificationResult(
                predicted_class=top_class,
                confidence=top_score,
                all_scores=normalized_scores
            )
            
        except Exception as e:
            logger.error(f"Enhanced GIT inference failed: {e}")
            raise


# Performance monitoring and cache management
class GitPerformanceMonitor:
    """Monitor and optimize GIT classifier performance"""
    
    def __init__(self):
        self.inference_times = []
        self.cache_hits = 0
        self.cache_misses = 0
    
    def log_inference_time(self, time_ms: float):
        self.inference_times.append(time_ms)
        
        # Keep only recent measurements
        if len(self.inference_times) > 100:
            self.inference_times = self.inference_times[-100:]
    
    def log_cache_hit(self):
        self.cache_hits += 1
    
    def log_cache_miss(self):
        self.cache_misses += 1
    
    def get_stats(self) -> Dict:
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'total_inferences': len(self.inference_times)
        }
