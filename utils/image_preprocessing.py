import cv2
import numpy as np
from typing import Union, List, Tuple, Optional


class VehicleImageEnhancer:
    """
    A utility class to enhance vehicle images without changing dimensions or color space.
    Compatible with zero-shot vehicle classification pipelines using YOLO, SSD, CLIP, etc.
    """
    
    def __init__(self, 
                 clahe_clip_limit: float = 2.0,
                 clahe_grid_size: Tuple[int, int] = (8, 8),
                 sharpen_amount: float = 0.5,
                 denoise_strength: int = 5,
                 enhance_contrast: bool = True,
                 enhance_sharpness: bool = True,
                 reduce_noise: bool = True):
        """
        Initialize the image enhancer with customizable parameters.
        
        Args:
            clahe_clip_limit: Clip limit for CLAHE algorithm
            clahe_grid_size: Grid size for CLAHE algorithm
            sharpen_amount: Amount of sharpening to apply (0.0 to 1.0)
            denoise_strength: Strength of denoising (higher = more smoothing)
            enhance_contrast: Whether to enhance contrast
            enhance_sharpness: Whether to enhance sharpness
            reduce_noise: Whether to apply noise reduction
        """
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
        self.sharpen_amount = sharpen_amount
        self.denoise_strength = denoise_strength
        self.enhance_contrast = enhance_contrast
        self.enhance_sharpness = enhance_sharpness
        self.reduce_noise = reduce_noise
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE, applied to luminance only.
        Preserves color information.
        """
        # Convert to LAB color space temporarily (keeps original RGB/BGR intact after conversion back)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # Apply CLAHE to L channel only
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        # Merge channels and convert back to original color space
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced
    
    def _enhance_sharpness(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image sharpness with unsharp masking.
        """
        # Create a blurred version of the image
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        # Apply unsharp masking
        sharpened = cv2.addWeighted(
            image, 1.0 + self.sharpen_amount, 
            blurred, -self.sharpen_amount, 
            0
        )
        return sharpened
    
    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction while preserving edges.
        """
        # Use edge-preserving filter
        denoised = cv2.edgePreservingFilter(
            image, 
            flags=cv2.NORMCONV_FILTER,
            sigma_s=self.denoise_strength,
            sigma_r=0.15
        )
        return denoised
    
    def preprocess(self, image: Union[np.ndarray, str], 
                  roi: Optional[List[Tuple[int, int, int, int]]] = None) -> np.ndarray:
        """
        Enhance image for better vehicle detection and classification.
        Does not change dimensions or color space of the input image.
        
        Args:
            image: Input image (numpy array) or path to image file
            roi: Optional list of ROIs (x, y, w, h) to focus enhancement on
                 specific regions only
        
        Returns:
            Enhanced image with same dimensions and color space as input
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not load image from {image}")
        
        # Make a copy to avoid modifying the original
        result = image.copy()
        
        # If ROIs are provided, only enhance those regions
        if roi:
            for x, y, w, h in roi:
                region = result[y:y+h, x:x+w]
                # Apply enhancements to the region
                if self.enhance_contrast:
                    region = self._enhance_contrast(region)
                if self.reduce_noise:
                    region = self._reduce_noise(region)
                if self.enhance_sharpness:
                    region = self._enhance_sharpness(region)
                # Put enhanced region back
                result[y:y+h, x:x+w] = region
            return result
        
        # Otherwise enhance the entire image
        if self.enhance_contrast:
            result = self._enhance_contrast(result)
        if self.reduce_noise:
            result = self._reduce_noise(result)
        if self.enhance_sharpness:
            result = self._enhance_sharpness(result)
        
        return result
    
    def batch_preprocess(self, 
                        images: List[Union[np.ndarray, str]], 
                        rois: Optional[List[List[Tuple[int, int, int, int]]]] = None) -> List[np.ndarray]:
        """
        Process a batch of images.
        
        Args:
            images: List of input images or paths
            rois: Optional list of ROIs for each image
        
        Returns:
            List of enhanced images
        """
        results = []
        for i, img in enumerate(images):
            roi_list = rois[i] if rois and i < len(rois) else None
            results.append(self.preprocess(img, roi_list))
        return results


def enhance_image(image: Union[np.ndarray, str], 
                 roi: Optional[List[Tuple[int, int, int, int]]] = None,
                 clahe_clip_limit: float = 2.0,
                 clahe_grid_size: Tuple[int, int] = (8, 8),
                 sharpen_amount: float = 0.5,
                 denoise_strength: int = 5) -> np.ndarray:
    """
    Convenience function to enhance a single image without creating an enhancer object.
    
    Args:
        image: Input image (numpy array) or path to image file
        roi: Optional list of ROIs (x, y, w, h) to focus enhancement on
        clahe_clip_limit: Clip limit for contrast enhancement
        clahe_grid_size: Grid size for contrast enhancement
        sharpen_amount: Amount of sharpening (0.0 to 1.0)
        denoise_strength: Strength of denoising
    
    Returns:
        Enhanced image with same dimensions and color space as input
    """
    enhancer = VehicleImageEnhancer(
        clahe_clip_limit=clahe_clip_limit,
        clahe_grid_size=clahe_grid_size,
        sharpen_amount=sharpen_amount,
        denoise_strength=denoise_strength
    )
    return enhancer.preprocess(image, roi)
