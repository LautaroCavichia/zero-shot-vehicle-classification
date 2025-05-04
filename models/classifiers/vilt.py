import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import ViltProcessor, ViltForImageAndTextRetrieval

from config import CLASSIFIER_CONFIGS, CLIP_TEMPLATES, CLASS_DESCRIPTIONS, VEHICLE_CLASSES


class ViLTClassifier:
    """ViLT-based zero-shot image-text retrieval classifier"""

    def __init__(self):
        """
        Initialize ViLT classifier
        """
        config = CLASSIFIER_CONFIGS.get('vilt', {})
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"ViLT classifier using device: {self.device}")

        # Load processor and model fine-tuned for image-text retrieval (ITR)
        self.model_name = "dandelin/vilt-b32-finetuned-coco"
        self.processor = ViltProcessor.from_pretrained(self.model_name)
        self.model = ViltForImageAndTextRetrieval.from_pretrained(self.model_name).to(self.device)

        # Prepare class descriptions
        self.class_descriptions = self._prepare_class_descriptions()

    def _prepare_class_descriptions(self) -> Dict[str, List[str]]:
        """
        Prepare text descriptions for each class
        """
        class_descriptions = {}
        for cls in VEHICLE_CLASSES:
            descriptions = []
            for template in CLIP_TEMPLATES:
                descriptions.append(template.format(cls))
            if cls in CLASS_DESCRIPTIONS:
                descriptions.extend(CLASS_DESCRIPTIONS[cls])
            class_descriptions[cls] = descriptions
        return class_descriptions

    def classify(self, image: np.ndarray) -> Tuple[Dict, float]:
        """
        Classify image using ViLT with normalized confidence scores
        """
        start_time = time.time()
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        
        class_scores = {}
        for cls, descriptions in self.class_descriptions.items():
            cls_scores = []
            for text in descriptions:
                encoding = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**encoding)
                    score = outputs.logits[0, :].item()  # single similarity score
                cls_scores.append(score)
            class_scores[cls] = sum(cls_scores) / len(cls_scores) if cls_scores else 0.0
        
        # Normalize scores using softmax
        scores_list = list(class_scores.values())
        scores_tensor = torch.tensor(scores_list)
        normalized_scores = torch.nn.functional.softmax(scores_tensor, dim=0).tolist()
        
        # Update the dictionary with normalized scores
        normalized_class_scores = {}
        for i, (cls, _) in enumerate(class_scores.items()):
            normalized_class_scores[cls] = normalized_scores[i]
        
        # Find top class based on normalized scores
        top_class = max(normalized_class_scores.items(), key=lambda x: x[1])[0]
        top_score = normalized_class_scores[top_class]
        
        inference_time = time.time() - start_time
        return {
            'class': top_class,
            'score': top_score,
            'all_scores': normalized_class_scores,
        }, inference_time