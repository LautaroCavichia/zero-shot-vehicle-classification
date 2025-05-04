"""
Data loading utilities for the zero-shot vehicle benchmark
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from pycocotools.coco import COCO
from PIL import Image

from config import ANNOTATIONS_FILE, IMAGES_DIR, VEHICLE_CLASSES


class COCODataset:
    """Dataset class for loading COCO format annotations"""

    def __init__(self, annotations_file: Path, images_dir: Path):
        """
        Initialize the COCO dataset
        
        Args:
            annotations_file: Path to the COCO format annotations file
            images_dir: Path to the directory containing images
        """
        self.annotations_file = annotations_file
        self.images_dir = images_dir
        
        # Load COCO annotations
        self.coco = COCO(annotations_file)
        
        # Get image IDs and annotations
        self.image_ids = self.coco.getImgIds()
        print(f"Loaded {len(self.image_ids)} images from COCO dataset")
        
        # Map category IDs to our vehicle classes
        self.category_mapping = self._create_category_mapping()
        
    def _create_category_mapping(self) -> Dict[int, str]:
        """
        Create a mapping from COCO category IDs to our vehicle classes
        
        Returns:
            Dict mapping category IDs to vehicle class names
        """
        categories = self.coco.loadCats(self.coco.getCatIds())
        mapping = {}
        
        # Map COCO categories to our vehicle classes
        for cat in categories:
            cat_name = cat['name'].lower()
            if 'car' in cat_name or 'sedan' in cat_name or 'hatchback' in cat_name:
                mapping[cat['id']] = 'car'
            elif 'van' in cat_name or 'minivan' in cat_name:
                mapping[cat['id']] = 'van'
            elif 'truck' in cat_name or 'pickup' in cat_name:
                mapping[cat['id']] = 'truck'
            elif 'bus' in cat_name or 'coach' in cat_name:
                mapping[cat['id']] = 'bus'
            elif 'ambulance' in cat_name or 'police' in cat_name or 'fire' in cat_name or 'emergency' in cat_name:
                mapping[cat['id']] = 'emergency'
            else:
                mapping[cat['id']] = 'non-vehicle'
                
        return mapping
    
    def __len__(self) -> int:
        """Return the number of images in the dataset"""
        return len(self.image_ids)
    
    def get_image_info(self, idx: int) -> Dict[str, Any]:
        """
        Get image info by index
        
        Args:
            idx: Index of the image
            
        Returns:
            Dict containing image information
        """
        image_id = self.image_ids[idx]
        return self.coco.loadImgs(image_id)[0]
    
    def load_image(self, idx: int) -> Tuple[np.ndarray, str]:
        """
        Load an image by index
        
        Args:
            idx: Index of the image
            
        Returns:
            Tuple of (image array, image path)
        """
        image_info = self.get_image_info(idx)
        image_path = os.path.join(self.images_dir, image_info['file_name'])
        
        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image, image_path
    
    def load_image_pil(self, idx: int) -> Tuple[Image.Image, str]:
        """
        Load an image by index as PIL Image
        
        Args:
            idx: Index of the image
            
        Returns:
            Tuple of (PIL Image, image path)
        """
        image_info = self.get_image_info(idx)
        image_path = os.path.join(self.images_dir, image_info['file_name'])
        
        # Load image using PIL
        image = Image.open(image_path).convert('RGB')
        
        return image, image_path
    
    def get_annotations(self, idx: int) -> Dict[str, Any]:
        """
        Get annotations for an image by index
        
        Args:
            idx: Index of the image
            
        Returns:
            Dict containing annotations
        """
        image_id = self.image_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Process annotations to our format
        processed_anns = []
        for ann in anns:
            # Get bounding box
            x, y, w, h = ann['bbox']
            # Convert to [x1, y1, x2, y2] format
            bbox = [x, y, x + w, y + h]
            
            # Get category
            category_id = ann['category_id']
            category = self.category_mapping.get(category_id, 'non-vehicle')
            
            processed_anns.append({
                'bbox': bbox,
                'category': category,
                'area': ann['area'],
                'id': ann['id'],
            })
        
        # Sort annotations by area (descending) to find the main vehicle
        processed_anns.sort(key=lambda x: x['area'], reverse=True)
        
        # Find the main vehicle (the one closest to the center and largest)
        main_vehicle = self._find_main_vehicle(processed_anns, self.get_image_info(idx))
        
        return {
            'image_id': image_id,
            'annotations': processed_anns,
            'main_vehicle': main_vehicle,
        }
    
    def _find_main_vehicle(self, annotations: List[Dict], image_info: Dict) -> Dict:
        """
        Find the main vehicle in the image based on centrality and size
        
        Args:
            annotations: List of processed annotations
            image_info: Image information
            
        Returns:
            Dict containing main vehicle annotation
        """
        if not annotations:
            return None
        
        # Get image dimensions
        img_width = image_info['width']
        img_height = image_info['height']
        img_center = (img_width / 2, img_height / 2)
        
        # Calculate centrality score for each annotation
        for ann in annotations:
            bbox = ann['bbox']
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
            size_score = ann['area'] / (img_width * img_height)
            
            # Combined score (balance of centrality and size)
            ann['score'] = 0.6 * centrality + 0.4 * size_score
            
        # Sort by combined score (descending)
        annotations.sort(key=lambda x: x['score'], reverse=True)
        
        # Return the annotation with the highest score
        # Only consider vehicles (filter out non-vehicles)
        for ann in annotations:
            if ann['category'] != 'non-vehicle':
                return ann
                
        # If no vehicle is found, return the highest scored annotation
        return annotations[0] if annotations else None


def create_dataloader() -> COCODataset:
    """
    Create and return the COCO dataset loader
    
    Returns:
        COCODataset instance
    """
    return COCODataset(ANNOTATIONS_FILE, IMAGES_DIR)