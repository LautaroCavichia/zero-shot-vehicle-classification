"""
Data loading utilities for the zero-shot vehicle benchmark with COCO format annotations.

This module provides robust COCO dataset loading with proper category mapping,
ground truth validation, and minimal preprocessing integration.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
from pycocotools.coco import COCO
from PIL import Image

from config import ANNOTATIONS_FILE, IMAGES_DIR, VEHICLE_CLASSES, COCO_CATEGORY_MAPPING
from utils.image_preprocessing import VehicleImageEnhancer

logger = logging.getLogger(__name__)


class AnnotationValidator:
    """Validates COCO annotations against expected format."""
    
    @staticmethod
    def validate_categories(coco_data: Dict) -> bool:
        """
        Validate that COCO categories match expected vehicle classes.
        
        Args:
            coco_data: Raw COCO annotation data
            
        Returns:
            True if categories are valid, False otherwise
        """
        try:
            categories = coco_data.get('categories', [])
            expected_names = set(cls.title() for cls in VEHICLE_CLASSES)  # Car, Van, etc.
            actual_names = set(cat['name'] for cat in categories)
            
            if not expected_names.issubset(actual_names):
                missing = expected_names - actual_names
                logger.warning(f"Missing expected categories: {missing}")
                return False
                
            logger.info(f"Categories validated successfully: {actual_names}")
            return True
            
        except Exception as e:
            logger.error(f"Category validation failed: {e}")
            return False
    
    @staticmethod
    def validate_image_annotations(image_info: Dict, annotations: List[Dict]) -> bool:
        """
        Validate annotations for a single image.
        
        Args:
            image_info: COCO image information
            annotations: List of annotations for the image
            
        Returns:
            True if annotations are valid, False otherwise
        """
        try:
            img_width = image_info['width']
            img_height = image_info['height']
            
            for ann in annotations:
                # Validate bounding box
                bbox = ann.get('bbox', [])
                if len(bbox) != 4:
                    logger.warning(f"Invalid bbox format in annotation {ann.get('id')}: {bbox}")
                    return False
                
                x, y, w, h = bbox
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    logger.warning(f"Invalid bbox values in annotation {ann.get('id')}: {bbox}")
                    return False
                
                if x + w > img_width or y + h > img_height:
                    logger.warning(f"Bbox extends beyond image boundaries in annotation {ann.get('id')}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Annotation validation failed: {e}")
            return False


class GroundTruthProcessor:
    """Processes ground truth annotations for evaluation."""
    
    def __init__(self, category_mapping: Dict[int, str]):
        """
        Initialize with category mapping.
        
        Args:
            category_mapping: Mapping from COCO category IDs to vehicle class names
        """
        self.category_mapping = category_mapping
    
    def process_annotations(self, annotations: List[Dict], image_info: Dict) -> Dict[str, Any]:
        """
        Process raw COCO annotations into evaluation format.
        
        Args:
            annotations: Raw COCO annotations
            image_info: Image metadata
            
        Returns:
            Processed ground truth data
        """
        try:
            processed_annotations = []
            
            for ann in annotations:
                # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
                x, y, w, h = ann['bbox']
                bbox = [x, y, x + w, y + h]
                
                # Map category ID to vehicle class
                category_id = ann['category_id']
                vehicle_class = self.category_mapping.get(category_id, 'non-vehicle')
                
                processed_ann = {
                    'bbox': bbox,
                    'category': vehicle_class.lower(),  # Ensure lowercase for consistency
                    'area': ann.get('area', w * h),
                    'id': ann['id'],
                    'confidence': 1.0,  # Ground truth has perfect confidence
                }
                
                processed_annotations.append(processed_ann)
            
            # Find main vehicle using centrality and size scoring
            main_vehicle = self._find_main_vehicle(processed_annotations, image_info)
            
            return {
                'image_id': image_info['id'],
                'annotations': processed_annotations,
                'main_vehicle': main_vehicle,
                'image_info': image_info
            }
            
        except Exception as e:
            logger.error(f"Failed to process annotations: {e}")
            return {
                'image_id': image_info.get('id', -1),
                'annotations': [],
                'main_vehicle': None,
                'image_info': image_info
            }
    
    def _find_main_vehicle(self, annotations: List[Dict], image_info: Dict) -> Optional[Dict]:
        """
        Find the main vehicle based on centrality and size.
        
        Args:
            annotations: Processed annotations
            image_info: Image metadata
            
        Returns:
            Main vehicle annotation or None
        """
        if not annotations:
            return None
        
        # Filter vehicle annotations (exclude non-vehicle)
        vehicle_annotations = [ann for ann in annotations if ann['category'] != 'non-vehicle']
        
        if not vehicle_annotations:
            return None
        
        img_width = image_info['width']
        img_height = image_info['height']
        img_center = (img_width / 2, img_height / 2)
        img_area = img_width * img_height
        img_diagonal = np.sqrt(img_width**2 + img_height**2)
        
        # Score each vehicle annotation
        for ann in vehicle_annotations:
            bbox = ann['bbox']
            x1, y1, x2, y2 = bbox
            
            # Calculate bbox center
            bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # Calculate distance from image center
            distance = np.sqrt((bbox_center[0] - img_center[0])**2 + 
                             (bbox_center[1] - img_center[1])**2)
            
            # Normalize distance by image diagonal
            normalized_distance = distance / img_diagonal
            
            # Calculate centrality score (1 - normalized_distance)
            centrality = 1 - normalized_distance
            
            # Calculate size score (normalized by image area)
            size_score = ann['area'] / img_area
            
            # Combined score (weighted by centrality and size)
            ann['score'] = 0.7 * centrality + 0.3 * size_score
        
        # Sort by score and return the highest
        vehicle_annotations.sort(key=lambda x: x['score'], reverse=True)
        main_vehicle = vehicle_annotations[0]
        
        logger.debug(f"Main vehicle: {main_vehicle['category']} with score {main_vehicle['score']:.3f}")
        return main_vehicle


class COCOVehicleDataset:
    """
    COCO dataset loader specifically designed for vehicle detection and classification.
    
    This class handles COCO format annotations with proper category mapping,
    validation, and preprocessing integration.
    """
    
    def __init__(self, 
                 annotations_file: Path, 
                 images_dir: Path,
                 enable_preprocessing: bool = True,
                 preprocessing_config: Optional[Dict] = None):
        """
        Initialize the COCO vehicle dataset.
        
        Args:
            annotations_file: Path to COCO format annotations
            images_dir: Path to images directory
            enable_preprocessing: Whether to apply image preprocessing
            preprocessing_config: Configuration for image preprocessing
        """
        self.annotations_file = Path(annotations_file)
        self.images_dir = Path(images_dir)
        self.enable_preprocessing = enable_preprocessing
        
        # Validate paths
        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # Initialize components
        self.validator = AnnotationValidator()
        self.gt_processor = GroundTruthProcessor(COCO_CATEGORY_MAPPING)
        
        # Initialize preprocessing
        if self.enable_preprocessing:
            preprocessing_config = preprocessing_config or {}
            self.preprocessor = VehicleImageEnhancer(**preprocessing_config)
        else:
            self.preprocessor = None
        
        # Load and validate COCO data
        self._load_coco_data()
        
        logger.info(f"Initialized COCOVehicleDataset with {len(self.image_ids)} images")
    
    def _load_coco_data(self) -> None:
        """Load and validate COCO annotation data."""
        try:
            # Load raw COCO data for validation
            with open(self.annotations_file, 'r') as f:
                coco_data = json.load(f)
            
            # Validate categories
            if not self.validator.validate_categories(coco_data):
                logger.warning("Category validation failed, proceeding with available categories")
            
            # Initialize COCO API
            self.coco = COCO(str(self.annotations_file))
            self.image_ids = self.coco.getImgIds()
            
            # Log dataset statistics
            total_annotations = len(self.coco.getAnnIds())
            categories = self.coco.loadCats(self.coco.getCatIds())
            
            logger.info(f"Loaded COCO dataset:")
            logger.info(f"  - Images: {len(self.image_ids)}")
            logger.info(f"  - Annotations: {total_annotations}")
            logger.info(f"  - Categories: {[cat['name'] for cat in categories]}")
            
        except Exception as e:
            logger.error(f"Failed to load COCO data: {e}")
            raise
    
    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_ids)
    
    def get_image_info(self, idx: int) -> Dict[str, Any]:
        """
        Get image metadata by index.
        
        Args:
            idx: Image index
            
        Returns:
            Image metadata dictionary
        """
        if idx < 0 or idx >= len(self.image_ids):
            raise IndexError(f"Image index {idx} out of range [0, {len(self.image_ids)})")
        
        image_id = self.image_ids[idx]
        return self.coco.loadImgs(image_id)[0]
    
    def load_image(self, idx: int) -> Tuple[np.ndarray, str]:
        """
        Load and optionally preprocess an image by index.
        
        Args:
            idx: Image index
            
        Returns:
            Tuple of (preprocessed image array in RGB format, image path)
        """
        try:
            image_info = self.get_image_info(idx)
            image_path = self.images_dir / image_info['file_name']
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Load image using OpenCV (BGR format)
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply preprocessing if enabled
            if self.preprocessor is not None:
                image = self.preprocessor.preprocess(image)
            
            return image, str(image_path)
            
        except Exception as e:
            logger.error(f"Failed to load image at index {idx}: {e}")
            raise
    
    def load_image_pil(self, idx: int) -> Tuple[Image.Image, str]:
        """
        Load an image as PIL Image.
        
        Args:
            idx: Image index
            
        Returns:
            Tuple of (PIL Image in RGB format, image path)
        """
        try:
            image_info = self.get_image_info(idx)
            image_path = self.images_dir / image_info['file_name']
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Load image using PIL
            image = Image.open(image_path).convert('RGB')
            
            return image, str(image_path)
            
        except Exception as e:
            logger.error(f"Failed to load PIL image at index {idx}: {e}")
            raise
    
    def get_annotations(self, idx: int) -> Dict[str, Any]:
        """
        Get processed ground truth annotations for an image.
        
        Args:
            idx: Image index
            
        Returns:
            Processed ground truth annotations
        """
        try:
            image_info = self.get_image_info(idx)
            image_id = image_info['id']
            
            # Get raw annotations
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            raw_annotations = self.coco.loadAnns(ann_ids)
            
            # Validate annotations
            if not self.validator.validate_image_annotations(image_info, raw_annotations):
                logger.warning(f"Annotation validation failed for image {idx}")
            
            # Process annotations
            processed_gt = self.gt_processor.process_annotations(raw_annotations, image_info)
            
            return processed_gt
            
        except Exception as e:
            logger.error(f"Failed to get annotations for image {idx}: {e}")
            return {
                'image_id': -1,
                'annotations': [],
                'main_vehicle': None,
                'image_info': {}
            }
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive dataset statistics.
        
        Returns:
            Dataset statistics dictionary
        """
        try:
            stats = {
                'total_images': len(self.image_ids),
                'total_annotations': 0,
                'class_distribution': {cls: 0 for cls in VEHICLE_CLASSES},
                'images_with_vehicles': 0,
                'images_without_vehicles': 0,
                'avg_annotations_per_image': 0.0,
                'bbox_size_stats': {
                    'min_area': float('inf'),
                    'max_area': 0.0,
                    'avg_area': 0.0
                }
            }
            
            total_area = 0.0
            area_count = 0
            
            for idx in range(len(self.image_ids)):
                annotations = self.get_annotations(idx)
                image_annotations = annotations['annotations']
                
                stats['total_annotations'] += len(image_annotations)
                
                has_vehicle = False
                for ann in image_annotations:
                    category = ann['category']
                    if category in stats['class_distribution']:
                        stats['class_distribution'][category] += 1
                    
                    if category != 'non-vehicle':
                        has_vehicle = True
                    
                    # Update bbox statistics
                    area = ann['area']
                    total_area += area
                    area_count += 1
                    stats['bbox_size_stats']['min_area'] = min(stats['bbox_size_stats']['min_area'], area)
                    stats['bbox_size_stats']['max_area'] = max(stats['bbox_size_stats']['max_area'], area)
                
                if has_vehicle:
                    stats['images_with_vehicles'] += 1
                else:
                    stats['images_without_vehicles'] += 1
            
            # Calculate averages
            if stats['total_images'] > 0:
                stats['avg_annotations_per_image'] = stats['total_annotations'] / stats['total_images']
            
            if area_count > 0:
                stats['bbox_size_stats']['avg_area'] = total_area / area_count
            
            # Handle edge case where no bboxes were found
            if stats['bbox_size_stats']['min_area'] == float('inf'):
                stats['bbox_size_stats']['min_area'] = 0.0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to compute dataset statistics: {e}")
            return {}


def create_dataloader(annotations_file: Optional[Path] = None,
                     images_dir: Optional[Path] = None,
                     enable_preprocessing: bool = False,  
                     preprocessing_config: Optional[Dict] = None) -> COCOVehicleDataset:
    """
    Factory function to create a COCO vehicle dataset loader.
    
    Args:
        annotations_file: Path to annotations file (defaults to config)
        images_dir: Path to images directory (defaults to config)
        enable_preprocessing: Whether to enable image preprocessing (now defaults to False)
        preprocessing_config: Configuration for preprocessing
        
    Returns:
        COCOVehicleDataset instance
    """
    annotations_file = annotations_file or ANNOTATIONS_FILE
    images_dir = images_dir or IMAGES_DIR
    
    return COCOVehicleDataset(
        annotations_file=annotations_file,
        images_dir=images_dir,
        enable_preprocessing=enable_preprocessing,
        preprocessing_config=preprocessing_config
    )