import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor as AutoProcessorVision, AutoModelForCausalLM # Git uses CausalLM for generation
from sentence_transformers import SentenceTransformer, util # For text similarity
import time
from config import VEHICLE_CLASSES, CLASS_DESCRIPTIONS, CLIP_TEMPLATES

class GitClassifier:
    """
    Zero-shot classifier using GIT for captioning and SentenceTransformer
    for comparing caption similarity to class descriptions.
    """
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"GIT+SBERT classifier using device: {self.device}")

        # --- GIT Model for Captioning ---
        self.git_model_name = "microsoft/git-base"
        # Use the processor associated with GIT
        self.git_processor = AutoProcessorVision.from_pretrained(self.git_model_name)
        # Use the correct AutoModel class for GIT's generation task
        self.git_model = AutoModelForCausalLM.from_pretrained(self.git_model_name).to(self.device)
        # --- ---

        # --- Sentence Transformer for Similarity ---
        # Using a common, efficient model. Others exist.
        self.similarity_model_name = 'all-MiniLM-L6-v2'
        self.similarity_model = SentenceTransformer(self.similarity_model_name, device=self.device)
        # --- ---

        # Prepare class descriptions (same as before)
        self.class_descriptions_map = self._prepare_class_descriptions()
        # Create a flat list of all descriptions for embedding
        self.all_descriptions = []
        self.description_to_class = {}
        for cls, descriptions in self.class_descriptions_map.items():
            self.all_descriptions.extend(descriptions)
            for desc in descriptions:
                self.description_to_class[desc] = cls

        # Pre-compute embeddings for all class descriptions for efficiency
        print("Pre-computing embeddings for class descriptions...")
        self.description_embeddings = self.similarity_model.encode(
            self.all_descriptions, convert_to_tensor=True, device=self.device
        )
        print("Embeddings computed.")


    def _prepare_class_descriptions(self):
        # This function remains the same as your original
        class_descriptions = {}
        for cls in VEHICLE_CLASSES:
            descriptions = []
            for template in CLIP_TEMPLATES:
                descriptions.append(template.format(cls))
            if cls in CLASS_DESCRIPTIONS:
                descriptions.extend(CLASS_DESCRIPTIONS[cls])
            class_descriptions[cls] = list(set(descriptions)) # Use set to remove duplicates
        return class_descriptions

    def _generate_caption(self, processed_image_inputs):
        """Generates a caption using the GIT model."""
        with torch.no_grad():
            generated_ids = self.git_model.generate(
                pixel_values=processed_image_inputs["pixel_values"],
                max_length=40 # Adjust max length as needed
            )
        # Decode the generated IDs, skipping special tokens
        caption = self.git_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        # Sometimes GIT might output only the prompt if it was included; handle this
        # (This depends on how the processor is configured/used; might not be needed)
        # A basic check:
        if not caption or len(caption.split()) < 2:
             caption = "an image" # Fallback caption
        return caption

    def classify(self, image: np.ndarray):
        start_time = time.time()

        if isinstance(image, np.ndarray):
             # Ensure image is in RGB if it comes from OpenCV (BGR)
            if image.shape[2] == 3: # Basic check for 3 channels
                 image = image[:, :, ::-1] # Convert BGR to RGB
            image = Image.fromarray(image.astype("uint8"))

        # 1. Process image for GIT
        inputs = self.git_processor(images=image, return_tensors="pt").to(self.device)

        # 2. Generate Caption using GIT
        caption = self._generate_caption(inputs)
        # print(f"Generated Caption: {caption}") # For debugging

        # 3. Embed the generated caption
        caption_embedding = self.similarity_model.encode(
            caption, convert_to_tensor=True, device=self.device
        )

        # 4. Calculate Cosine Similarity between caption and all pre-computed description embeddings
        # Shape: [1, num_descriptions]
        cosine_scores = util.pytorch_cos_sim(caption_embedding, self.description_embeddings)[0]
        cosine_scores = cosine_scores.cpu() # Move scores to CPU for easier handling

        # 5. Aggregate scores per class (e.g., max similarity for any description in that class)
        class_scores = {cls: 0.0 for cls in VEHICLE_CLASSES}
        for i, description in enumerate(self.all_descriptions):
            cls = self.description_to_class[description]
            # Update class score with the highest similarity found so far for that class
            class_scores[cls] = max(class_scores[cls], cosine_scores[i].item())

        # 6. Normalize the aggregated scores (using softmax for probabilities)
        # Filter out classes with 0 score before softmax if necessary, though max aggregation helps
        valid_classes = [cls for cls, score in class_scores.items() if score > 0]
        if not valid_classes: # Handle case where no description had positive similarity (unlikely)
             normalized_class_scores = {cls: 1.0/len(VEHICLE_CLASSES) for cls in VEHICLE_CLASSES} # Uniform distribution
             top_class = VEHICLE_CLASSES[0] if VEHICLE_CLASSES else None
             top_score = normalized_class_scores.get(top_class, 0.0)
        else:
            scores_tensor = torch.tensor([class_scores[cls] for cls in valid_classes])
            normalized_scores = torch.nn.functional.softmax(scores_tensor, dim=0).tolist()
            normalized_class_scores = {cls: 0.0 for cls in VEHICLE_CLASSES} # Initialize all classes
            for i, cls in enumerate(valid_classes):
                normalized_class_scores[cls] = normalized_scores[i]

            top_class = max(normalized_class_scores.items(), key=lambda x: x[1])[0]
            top_score = normalized_class_scores[top_class]


        inference_time = time.time() - start_time

        return {
            "class": top_class,
            "score": top_score,
            "all_scores": normalized_class_scores,
            "caption": caption # Optionally return the caption for inspection
        }, inference_time