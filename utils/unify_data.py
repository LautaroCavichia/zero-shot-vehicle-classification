#!/usr/bin/env python3
"""
Image Unification Script for Zero-Shot Vehicle Detection Benchmark

This script unifies all images in a folder to a consistent format:
- Resizes to 640x640 pixels
- Converts to RGB format
- Saves as JPG format
- Renames with increasing numbers (001.jpg, 002.jpg, etc.)
- Maintains aspect ratio with padding if needed
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import json

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageUnifier:
    """Handles image unification tasks."""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (640, 640),
                 output_format: str = 'JPEG',
                 quality: int = 95,
                 pad_color: Tuple[int, int, int] = (114, 114, 114),
                 crop_percentage: float = 0.1,
                 enhance_contrast: bool = True,
                 enhance_sharpness: bool = True,
                 contrast_factor: float = 1.2,
                 sharpness_factor: float = 1.3):
        """
        Initialize the image unifier.
        
        Args:
            target_size: Target image size (width, height)
            output_format: Output image format ('JPEG', 'PNG')
            quality: JPEG quality (1-100)
            pad_color: Padding color for aspect ratio preservation (R, G, B) - default gray
            crop_percentage: Percentage to crop from each side for very elongated images (0.1 = 10%)
            enhance_contrast: Whether to apply adaptive contrast enhancement
            enhance_sharpness: Whether to apply edge sharpening
            contrast_factor: Contrast enhancement factor (1.0 = no change, >1.0 = more contrast)
            sharpness_factor: Sharpness enhancement factor (1.0 = no change, >1.0 = sharper)
        """
        self.target_size = target_size
        self.output_format = output_format
        self.quality = quality
        self.pad_color = pad_color
        self.crop_percentage = crop_percentage
        self.enhance_contrast = enhance_contrast
        self.enhance_sharpness = enhance_sharpness
        self.contrast_factor = contrast_factor
        self.sharpness_factor = sharpness_factor
        
        # Supported input formats
        self.supported_formats = {
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', 
            '.webp', '.gif', '.ppm', '.pgm', '.pbm'
        }
        
        logger.info(f"Initialized ImageUnifier:")
        logger.info(f"  Target size: {target_size}")
        logger.info(f"  Output format: {output_format}")
        logger.info(f"  Quality: {quality}")
        logger.info(f"  Padding color: {pad_color}")
        logger.info(f"  Center crop percentage: {crop_percentage * 100}%")
        logger.info(f"  Enhance contrast: {enhance_contrast} (factor: {contrast_factor})")
        logger.info(f"  Enhance sharpness: {enhance_sharpness} (factor: {sharpness_factor})")
    
    def get_image_files(self, input_dir: Path, recursive: bool = True) -> List[Path]:
        """
        Get all supported image files from the input directory.
        
        Args:
            input_dir: Input directory path
            recursive: Whether to search subdirectories
            
        Returns:
            List of image file paths
        """
        image_files = []
        
        if recursive:
            pattern = '**/*'
        else:
            pattern = '*'
            
        for file_path in input_dir.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_files.append(file_path)
        
        # Sort files for consistent ordering
        image_files.sort()
        
        logger.info(f"Found {len(image_files)} image files in {input_dir}")
        return image_files
    
    def center_crop_and_resize(self, image: Image.Image, image_name: str = "", crop_percentage: float = 0.1) -> Image.Image:
        """
        Center crop image (if needed) and resize with letterboxing to maintain aspect ratio.
        If the image is too elongated, apply smart cropping based on image name:
        - "ingresso" in name: crop 20% from left side only
        - "uscita" in name: crop 20% from right side only  
        - Otherwise: crop 10% from each side (center crop)
        
        Args:
            image: Input PIL Image
            image_name: Image filename for smart cropping logic
            crop_percentage: Percentage to crop from each side (0.1 = 10%)
            
        Returns:
            Cropped, resized and padded PIL Image
        """
        # Get original dimensions
        orig_width, orig_height = image.size
        target_width, target_height = self.target_size
        
        # Calculate aspect ratios
        orig_aspect = orig_width / orig_height
        target_aspect = target_width / target_height
        
        # Determine if we need to crop
        # If image is much wider or taller than target, apply crop
        aspect_ratio_threshold = 1.2  
        
        cropped_image = image
        image_name_lower = image_name.lower()
        
        if orig_aspect / target_aspect > aspect_ratio_threshold:
            # Image is too wide - crop horizontally with smart logic
            
            if "ingresso" in image_name_lower:
                # Entrance/entry - crop from left side only (20%)
                crop_amount = int(orig_width * 0.2)
                left = crop_amount
                right = orig_width
                top = 0
                bottom = orig_height
                logger.debug(f"Applied ingresso crop: {crop_amount}px from left side")
                
            elif "uscita" in image_name_lower:
                # Exit - crop from right side only (20%)
                crop_amount = int(orig_width * 0.2)
                left = 0
                right = orig_width - crop_amount
                top = 0
                bottom = orig_height
                logger.debug(f"Applied uscita crop: {crop_amount}px from right side")
                
            else:
                # Default center crop - crop from both sides (10% each)
                crop_amount = int(orig_width * crop_percentage)
                left = crop_amount
                right = orig_width - crop_amount
                top = 0
                bottom = orig_height
                logger.debug(f"Applied center crop: {crop_amount}px from each side")
            
            cropped_image = image.crop((left, top, right, bottom))
            
        elif target_aspect / orig_aspect > aspect_ratio_threshold:
            # Image is too tall - crop vertically (always center crop)
            crop_amount = int(orig_height * crop_percentage)
            left = 0
            right = orig_width
            top = crop_amount
            bottom = orig_height - crop_amount
            
            cropped_image = image.crop((left, top, right, bottom))
            logger.debug(f"Applied vertical center crop: {crop_amount}px from top and bottom")
        
        # Now apply letterbox resize to the cropped image
        crop_width, crop_height = cropped_image.size
        
        # Calculate scaling factor
        scale = min(target_width / crop_width, target_height / crop_height)
        
        # Calculate new dimensions
        new_width = int(crop_width * scale)
        new_height = int(crop_height * scale)
        
        # Resize image
        resized_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create padded image
        padded_image = Image.new('RGB', self.target_size, self.pad_color)
        
        # Calculate padding offsets to center the image
        pad_x = (target_width - new_width) // 2
        pad_y = (target_height - new_height) // 2
        
        # Paste resized image onto padded background
        padded_image.paste(resized_image, (pad_x, pad_y))
        
        return padded_image
    
    def enhance_image_quality(self, image: Image.Image) -> Image.Image:
        """
        Apply intelligent contrast and sharpness enhancement for better classification.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Enhanced PIL Image
        """
        enhanced_image = image
        
        if self.enhance_contrast:
            # Apply adaptive contrast enhancement
            enhanced_image = self._enhance_contrast_adaptive(enhanced_image)
        
        if self.enhance_sharpness:
            # Apply intelligent edge sharpening
            enhanced_image = self._enhance_sharpness_smart(enhanced_image)
        
        return enhanced_image
    
    def _enhance_contrast_adaptive(self, image: Image.Image) -> Image.Image:
        """
        Apply adaptive contrast enhancement that works well for vehicle images.
        Uses a combination of histogram equalization and contrast stretching.
        """
        # Convert to numpy for OpenCV processing
        img_array = np.array(image)
        
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        enhanced_image = Image.fromarray(enhanced_array)
        
        # Apply additional contrast enhancement using PIL
        contrast_enhancer = ImageEnhance.Contrast(enhanced_image)
        enhanced_image = contrast_enhancer.enhance(self.contrast_factor)
        
        return enhanced_image
    
    def _enhance_sharpness_smart(self, image: Image.Image) -> Image.Image:
        """
        Apply intelligent sharpness enhancement that emphasizes edges without over-sharpening.
        """
        # First apply unsharp masking using PIL
        sharpness_enhancer = ImageEnhance.Sharpness(image)
        sharpened = sharpness_enhancer.enhance(self.sharpness_factor)
        
        # Apply additional edge enhancement using OpenCV for better vehicle detection
        img_array = np.array(sharpened)
        
        # Create edge enhancement kernel
        kernel = np.array([[-0.5, -0.5, -0.5],
                          [-0.5,  5.0, -0.5],
                          [-0.5, -0.5, -0.5]])
        
        # Apply subtle edge enhancement
        edge_enhanced = cv2.filter2D(img_array, -1, kernel * 0.3)  # Reduced intensity
        
        # Blend with original to avoid over-sharpening
        blended = cv2.addWeighted(img_array, 0.7, edge_enhanced, 0.3, 0)
        
        # Ensure values are in valid range
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        return Image.fromarray(blended)
    
    def validate_image(self, image_path: Path) -> bool:
        """
        Validate if image can be loaded and processed.
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            with Image.open(image_path) as img:
                # Check if image has valid dimensions
                if img.size[0] == 0 or img.size[1] == 0:
                    return False
                # Try to load image data
                img.load()
                return True
        except Exception:
            return False
    
    def process_image(self, input_path: Path, output_path: Path) -> bool:
        """
        Process a single image: load, resize, and save.
        
        Args:
            input_path: Input image path
            output_path: Output image path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate image first
            if not self.validate_image(input_path):
                logger.warning(f"Invalid or corrupted image: {input_path}")
                return False
            
            # Load image
            with Image.open(input_path) as image:
                # Convert to RGB (handles RGBA, grayscale, etc.)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Apply center crop and letterbox resize with smart cropping
                processed_image = self.center_crop_and_resize(image, input_path.name)
                
                # Apply image quality enhancement for better classification
                if self.enhance_contrast or self.enhance_sharpness:
                    processed_image = self.enhance_image_quality(processed_image)
                
                # Create output directory if needed
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save image
                save_kwargs = {
                    'format': self.output_format,
                    'optimize': True
                }
                
                if self.output_format == 'JPEG':
                    save_kwargs['quality'] = self.quality
                
                processed_image.save(output_path, **save_kwargs)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to process {input_path}: {e}")
            return False
    
    def get_image_info(self, image_path: Path) -> dict:
        """Get basic image information."""
        try:
            with Image.open(image_path) as img:
                return {
                    'size': img.size,
                    'mode': img.mode,
                    'format': img.format
                }
        except:
            return {
                'size': (0, 0),
                'mode': 'unknown',
                'format': 'unknown'
            }
    
    def unify_images(self, 
                    input_dir: Path, 
                    output_dir: Path,
                    start_number: int = 1,
                    name_prefix: str = "",
                    recursive: bool = True,
                    create_mapping: bool = True,
                    skip_existing: bool = False) -> Tuple[int, int, int]:
        """
        Unify all images in the input directory.
        
        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for unified images
            start_number: Starting number for file naming
            name_prefix: Prefix for output filenames
            recursive: Whether to search subdirectories
            create_mapping: Whether to create a mapping file
            skip_existing: Whether to skip existing output files
            
        Returns:
            Tuple of (successful_count, failed_count, skipped_count)
        """
        # Validate input directory
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        
        # Get all image files
        image_files = self.get_image_files(input_dir, recursive)
        
        if not image_files:
            logger.warning(f"No image files found in {input_dir}")
            return 0, 0, 0
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize counters and mapping
        successful_count = 0
        failed_count = 0
        skipped_count = 0
        file_mapping = {}
        
        # Determine file extension
        file_extension = '.jpg' if self.output_format == 'JPEG' else '.png'
        
        # Process images with progress bar
        with tqdm(image_files, desc="Unifying images", unit="img") as pbar:
            for i, input_path in enumerate(pbar):
                # Generate output filename
                output_number = start_number + successful_count
                output_filename = f"{name_prefix}{output_number:03d}{file_extension}"
                output_path = output_dir / output_filename
                
                # Update progress bar
                pbar.set_postfix({
                    'current': input_path.name,
                    'success': successful_count,
                    'failed': failed_count
                })
                
                # Skip if file exists and skip_existing is True
                if skip_existing and output_path.exists():
                    skipped_count += 1
                    logger.debug(f"Skipping existing file: {output_path}")
                    continue
                
                # Process image
                if self.process_image(input_path, output_path):
                    successful_count += 1
                    
                    # Add to mapping
                    if create_mapping:
                        input_info = self.get_image_info(input_path)
                        file_mapping[output_filename] = {
                            'original_path': str(input_path),
                            'original_name': input_path.name,
                            'original_size': input_info['size'],
                            'original_mode': input_info['mode'],
                            'original_format': input_info['format'],
                            'processed_size': self.target_size,
                            'relative_path': str(input_path.relative_to(input_dir)) if recursive else input_path.name
                        }
                else:
                    failed_count += 1
                    logger.warning(f"Failed to process: {input_path}")
        
        # Save mapping file if requested
        if create_mapping and file_mapping:
            mapping_file = output_dir / 'image_mapping.json'
            mapping_data = {
                'info': {
                    'total_processed': successful_count,
                    'total_failed': failed_count,
                    'total_skipped': skipped_count,
                    'target_size': self.target_size,
                    'output_format': self.output_format,
                    'quality': self.quality if self.output_format == 'JPEG' else None,
                    'input_directory': str(input_dir),
                    'output_directory': str(output_dir)
                },
                'mappings': file_mapping
            }
            
            with open(mapping_file, 'w') as f:
                json.dump(mapping_data, f, indent=2)
            logger.info(f"Saved mapping file: {mapping_file}")
        
        # Log final results
        total_attempted = successful_count + failed_count
        logger.info(f"Processing complete:")
        logger.info(f"  Successfully processed: {successful_count}")
        logger.info(f"  Failed: {failed_count}")
        logger.info(f"  Skipped: {skipped_count}")
        logger.info(f"  Success rate: {successful_count/total_attempted*100:.1f}%" if total_attempted > 0 else "  Success rate: N/A")
        
        return successful_count, failed_count, skipped_count


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {output_dir}")
    
    if args.target_size <= 0:
        raise ValueError(f"Target size must be positive: {args.target_size}")
    
    if not (1 <= args.quality <= 100):
        raise ValueError(f"Quality must be between 1 and 100: {args.quality}")
    
    if not (0.0 <= args.crop_percentage <= 0.5):
        raise ValueError(f"Crop percentage must be between 0.0 and 0.5: {args.crop_percentage}")
    
    if not (0.5 <= args.contrast_factor <= 3.0):
        raise ValueError(f"Contrast factor must be between 0.5 and 3.0: {args.contrast_factor}")
        
    if not (0.5 <= args.sharpness_factor <= 3.0):
        raise ValueError(f"Sharpness factor must be between 0.5 and 3.0: {args.sharpness_factor}")
    
    if args.start_number < 0:
        raise ValueError(f"Start number must be non-negative: {args.start_number}")


def parse_color(color_str: str) -> Tuple[int, int, int]:
    """Parse color string to RGB tuple."""
    try:
        values = [int(x.strip()) for x in color_str.split(',')]
        if len(values) != 3:
            raise ValueError("Color must have exactly 3 values")
        if not all(0 <= v <= 255 for v in values):
            raise ValueError("Color values must be between 0 and 255")
        return tuple(values)
    except Exception as e:
        raise ValueError(f"Invalid color format '{color_str}': {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Unify images to consistent size and format for vehicle detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default enhancements
  python unify_images.py input_folder output_folder
  
  # Custom enhancement settings
  python unify_images.py raw_images unified_images --contrast-factor 1.4 --sharpness-factor 1.5
  
  # Disable enhancements
  python unify_images.py images processed --no-enhance-contrast --no-enhance-sharpness
  
  # Full custom processing
  python unify_images.py input output --target-size 640 --crop-percentage 0.15 --contrast-factor 1.3
        """
    )
    
    parser.add_argument(
        'input_dir',
        type=str,
        help='Input directory containing images to unify'
    )
    
    parser.add_argument(
        'output_dir', 
        type=str,
        help='Output directory for unified images'
    )
    
    parser.add_argument(
        '--target-size',
        type=int,
        default=640,
        help='Target image size (creates square: size x size)'
    )
    
    parser.add_argument(
        '--format',
        choices=['JPEG', 'PNG'],
        default='JPEG',
        help='Output image format'
    )
    
    parser.add_argument(
        '--quality',
        type=int,
        default=95,
        help='JPEG quality (1-100, only applies to JPEG format)'
    )
    
    parser.add_argument(
        '--start-number',
        type=int,
        default=1,
        help='Starting number for file naming'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='',
        help='Prefix for output filenames (e.g., "img_" -> img_001.jpg)'
    )
    
    parser.add_argument(
        '--crop-percentage',
        type=float,
        default=0.1,
        help='Percentage to crop from each side for very elongated images (0.1 = 10%%)'
    )
    
    parser.add_argument(
        '--enhance-contrast',
        action='store_true',
        default=True,
        help='Apply adaptive contrast enhancement for better classification'
    )
    
    parser.add_argument(
        '--no-enhance-contrast',
        action='store_true',
        help='Disable contrast enhancement'
    )
    
    parser.add_argument(
        '--enhance-sharpness',
        action='store_true', 
        default=True,
        help='Apply edge sharpening for better classification'
    )
    
    parser.add_argument(
        '--no-enhance-sharpness',
        action='store_true',
        help='Disable sharpness enhancement'
    )
    
    parser.add_argument(
        '--contrast-factor',
        type=float,
        default=1.2,
        help='Contrast enhancement factor (1.0 = no change, >1.0 = more contrast)'
    )
    
    parser.add_argument(
        '--sharpness-factor',
        type=float,
        default=1.3,
        help='Sharpness enhancement factor (1.0 = no change, >1.0 = sharper)'
    )
    
    parser.add_argument(
        '--pad-color',
        type=str,
        default='114,114,114',
        help='Padding color as R,G,B values (default: gray 114,114,114)'
    )
    
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not search subdirectories recursively'
    )
    
    parser.add_argument(
        '--no-mapping',
        action='store_true',
        help='Do not create image mapping file'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip processing if output file already exists'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Validate arguments
        validate_arguments(args)
        
        # Parse padding color
        pad_color = parse_color(args.pad_color)
        
        # Handle enhancement flags
        enhance_contrast = args.enhance_contrast and not args.no_enhance_contrast
        enhance_sharpness = args.enhance_sharpness and not args.no_enhance_sharpness
        
        # Initialize image unifier
        unifier = ImageUnifier(
            target_size=(args.target_size, args.target_size),
            output_format=args.format,
            quality=args.quality,
            pad_color=pad_color,
            crop_percentage=args.crop_percentage,
            enhance_contrast=enhance_contrast,
            enhance_sharpness=enhance_sharpness,
            contrast_factor=args.contrast_factor,
            sharpness_factor=args.sharpness_factor
        )
        
        logger.info(f"Starting image unification:")
        logger.info(f"  Input: {args.input_dir}")
        logger.info(f"  Output: {args.output_dir}")
        logger.info(f"  Recursive: {not args.no_recursive}")
        
        # Unify images
        successful, failed, skipped = unifier.unify_images(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            start_number=args.start_number,
            name_prefix=args.prefix,
            recursive=not args.no_recursive,
            create_mapping=not args.no_mapping,
            skip_existing=args.skip_existing
        )
        
        # Final status
        total_processed = successful + failed
        if failed == 0:
            logger.info("✅ Image unification completed successfully!")
        elif successful > 0:
            logger.warning(f"⚠️  Completed with {failed} failures out of {total_processed} images")
        else:
            logger.error("❌ No images were processed successfully")
            return 1
            
        # Show summary
        if successful > 0:
            output_dir = Path(args.output_dir)
            logger.info(f"📁 Results saved to: {output_dir}")
            if not args.no_mapping:
                logger.info(f"📄 Mapping file: {output_dir / 'image_mapping.json'}")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Script failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())