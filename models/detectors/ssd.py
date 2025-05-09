"""
SSD detector implementation
"""
import os
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torchvision
import cv2

from config import DETECTOR_CONFIGS


class SSDDetector:
    """Robust SSD-based detector for object detection"""

    MODEL_SIZES = {
        "small": "ssd_mobilenet_v3",
        "medium": "ssd_vgg16",
    }

    def __init__(self, model_size="medium", conf_threshold=None, custom_model_path=None):
        config = DETECTOR_CONFIGS.get('ssd', {
            'model_size': 'medium',
            'conf_threshold': 0.3,
            'vehicle_classes': [1, 2, 3, 5, 7]  # COCO class IDs for vehicles
        })

        self._model_size = model_size
        self.conf_threshold = conf_threshold or config.get('conf_threshold', 0.3)
        self.vehicle_classes = config.get('vehicle_classes', [0, 2, 3, 5, 7])

        # Load model
        if custom_model_path and os.path.exists(custom_model_path):
            self.model = torch.load(custom_model_path)
        else:
            model_name = self.MODEL_SIZES.get(self._model_size)
            if not model_name:
                raise ValueError(f"Invalid model size: {model_size}")

        if model_name == "ssd_mobilenet_v3":
            self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
        elif model_name == "ssd_vgg16":
            self.model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
        else:
            raise ValueError(f"Unsupported SSD model: {model_name}")

        self.model.eval()
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"SSD detector using device: {self.device}")

    def detect(self, image: np.ndarray) -> Tuple[List[Dict], float]:
        try:
            start_time = time.time()
            image_tensor = torchvision.transforms.functional.to_tensor(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                predictions = self.model(image_tensor)

            pred = predictions[0] if predictions else {}
            boxes = pred.get('boxes', torch.empty(0)).cpu().numpy()
            scores = pred.get('scores', torch.empty(0)).cpu().numpy()
            labels = pred.get('labels', torch.empty(0)).cpu().numpy()

            detections = []
            for i in range(len(boxes)):
                if scores[i] < self.conf_threshold:
                    continue

                class_id = int(labels[i])
                if class_id in self.vehicle_classes:
                    x1, y1, x2, y2 = boxes[i]
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(scores[i]),
                        'class_id': class_id,
                        'area': float((x2 - x1) * (y2 - y1)),
                    })

            inference_time = time.time() - start_time
            return detections, inference_time

        except Exception as e:
            print(f"[SSDDetector] Detection error: {e}")
            return [], 0.0  # Safe fallback

    def find_main_vehicle(self, detections: List[Dict], image_shape: Tuple[int, int, int]) -> Dict:
        if not detections:
            # Return a default dummy prediction if no detections found
            return {
                'bbox': [0, 0, image_shape[1], image_shape[0]],  # full image
                'confidence': 0.0,
                'class_id': 0,
                'area': float(image_shape[0] * image_shape[1]),
                'score': 0.0
            }

        img_height, img_width = image_shape[:2]
        img_center = (img_width / 2, img_height / 2)
        img_area = img_width * img_height
        img_diagonal = np.sqrt(img_width**2 + img_height**2)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            distance = np.sqrt((bbox_center[0] - img_center[0])**2 + (bbox_center[1] - img_center[1])**2)
            normalized_distance = distance / img_diagonal
            centrality = 1 - normalized_distance
            size_score = det['area'] / img_area
            det['score'] = 0.7 * centrality + 0.3 * size_score

        detections.sort(key=lambda x: x['score'], reverse=True)
        return detections[0]

    def crop_bbox(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            # Fallback to center crop
            return image[h//4:h*3//4, w//4:w*3//4]

        return image[y1:y2, x1:x2]
