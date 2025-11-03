#!/usr/bin/env python3
"""
Transform images in the plume_image_dataset.

This script applies random translations (0-100 pixels in random direction) and
increases contrast by 20% to images while maintaining the original image size.
Out-of-bounds pixels are dropped and empty areas are filled with 0.
"""

import cv2
import numpy as np
import json
import argparse
import math
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional


def transform_bbox(
    bbox: List[int],
    translation: Tuple[int, int]
) -> List[int]:
    """
    Transform bbox coordinates accounting for translation only.

    Args:
        bbox: Bounding box in xywh format [x, y, width, height]
        translation: Translation (tx, ty) in pixels

    Returns:
        Transformed bbox in xywh format (clamped to image bounds)
    """
    x, y, w, h = bbox
    tx, ty = translation
    
    # Apply translation to bbox
    new_x = x + tx
    new_y = y + ty
    
    # Clamp to valid bounds (assuming image size doesn't change)
    # We'll clamp later after getting image dimensions
    return [new_x, new_y, w, h]


def enhance_contrast(image: np.ndarray, contrast_factor: float = 1.2) -> np.ndarray:
    """
    Enhance image contrast using sigmoid curve for smoother contrast enhancement.
    
    Args:
        image: Input image (grayscale or color BGR) as numpy array
        contrast_factor: Contrast enhancement factor (1.2 = 20% increase)
                         Maps to sigmoid steepness (higher = more contrast)
    Returns:
        Contrast-enhanced image (same dtype and channels as input)
    """
    # Determine image properties
    if image.dtype == np.uint8:
        max_val = 255.0
        midpoint = 128.0
        dtype = np.uint8
    elif image.dtype == np.uint16:
        max_val = 65535.0
        midpoint = 32767.5
        dtype = np.uint16
    else:
        max_val = 255.0
        midpoint = 128.0
        dtype = image.dtype
    
    # Map contrast_factor to sigmoid steepness, base steepness chosen heuristically
    base_steepness = 0.03
    steepness = base_steepness * contrast_factor
    
    # Convert to float for calculations
    img_array = image.astype(np.float32)
    
    # Apply sigmoid contrast curve
    # Formula: output = max_val / (1 + exp(-steepness * (input - midpoint)))
    # Normalize steepness for different bit depths
    normalized_steepness = steepness / (max_val / 255.0)
    img_array = max_val / (1 + np.exp(-normalized_steepness * (img_array - midpoint)))
    
    # Clip and convert back to original dtype
    img_array = np.clip(img_array, 0, max_val)
    enhanced_img = img_array.astype(dtype)
    
    return enhanced_img


def translate_image(
    image: np.ndarray,
    translation: Tuple[int, int]
) -> np.ndarray:
    """
    Apply translation to an image while maintaining original size.
    Out-of-bounds pixels are dropped and empty areas are filled with 0.

    Args:
        image: Input image as numpy array
        translation: Translation (tx, ty) in pixels

    Returns:
        Translated image (same size as input)
    """
    height, width = image.shape[:2]
    tx, ty = translation
    
    # Create translation matrix
    # Translation matrix: [1 0 tx; 0 1 ty]
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Calculate new canvas size needed (larger to accommodate translation)
    # If translating right/down, we need extra space
    # If translating left/up, we need to shift origin
    min_x = min(0, tx)
    min_y = min(0, ty)
    max_x = max(width, width + tx)
    max_y = max(height, height + ty)
    
    canvas_width = int(max_x - min_x)
    canvas_height = int(max_y - min_y)
    
    # Adjust translation matrix to account for canvas offset
    if min_x < 0:
        translation_matrix[0, 2] -= min_x
    if min_y < 0:
        translation_matrix[1, 2] -= min_y
    
    # Apply translation to expanded canvas
    translated_canvas = cv2.warpAffine(
        image,
        translation_matrix,
        (canvas_width, canvas_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # Calculate where to extract the original-sized window
    # We want to extract at the original image position (0,0 in original coords)
    # After adjusting for canvas offset
    offset_x = -min_x if min_x < 0 else 0
    offset_y = -min_y if min_y < 0 else 0
    
    # Extract original-sized region at original position
    crop_x = offset_x
    crop_y = offset_y
    
    # Create output image filled with zeros
    output = np.zeros((height, width, *image.shape[2:]), dtype=image.dtype)
    
    # Calculate valid source region in canvas
    src_x_start = int(max(0, crop_x))
    src_y_start = int(max(0, crop_y))
    src_x_end = int(min(canvas_width, crop_x + width))
    src_y_end = int(min(canvas_height, crop_y + height))
    
    # Calculate corresponding destination region in output
    dst_x_start = int(max(0, -crop_x))
    dst_y_start = int(max(0, -crop_y))
    dst_x_end = int(dst_x_start + (src_x_end - src_x_start))
    dst_y_end = int(dst_y_start + (src_y_end - src_y_start))
    
    # Copy valid region from canvas to output (out-of-bounds parts are already zeros)
    if dst_x_end > dst_x_start and dst_y_end > dst_y_start:
        output[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            translated_canvas[src_y_start:src_y_end, src_x_start:src_x_end]
    
    return output


def generate_random_translation(max_pixels: int = 100) -> Tuple[int, int]:
    """
    Generate random translation in a random direction.

    Args:
        max_pixels: Maximum translation distance in pixels

    Returns:
        Tuple of (tx, ty) translation values
    """
    # Random angle in radians
    angle = random.uniform(0, 2 * math.pi)
    # Random distance between 0 and max_pixels
    distance = random.uniform(0, max_pixels)
    
    tx = int(distance * math.cos(angle))
    ty = int(distance * math.sin(angle))
    
    return (tx, ty)


def transform_images(
    source_images_dir: Path,
    source_labels_path: Path,
    output_dir: Path,
    max_translation: int = 100,
    num_transforms: int = 1,
    contrast_factor: float = 1.2,
    seed: Optional[int] = None
) -> None:
    """
    Transform all images with random translation and contrast enhancement.

    Args:
        source_images_dir: Directory containing original image files
        source_labels_path: Path to source labels.json file
        output_dir: Directory to save transformed images and new labels.json
        max_translation: Maximum translation distance in pixels (default: 100)
        num_transforms: Number of random transformations to apply per image (default: 1)
        contrast_factor: Contrast enhancement factor (default: 1.2 = 20% increase)
        seed: Random seed for reproducibility (None = no seed)
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load source labels.json
    if not source_labels_path.exists():
        print(f"Error: labels.json not found at {source_labels_path}")
        return
    
    with open(source_labels_path, 'r') as f:
        source_labels = json.load(f)
    
    print(f"Loaded {len(source_labels)} image labels")
    print(f"Source images directory: {source_images_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max translation: {max_translation} pixels")
    print(f"Contrast enhancement: {(contrast_factor - 1) * 100:.1f}% increase")
    print(f"Transforms per image: {num_transforms}")
    if seed is not None:
        print(f"Random seed: {seed}")
    print()
    
    transformed_labels = []
    transformed_count = 0
    error_count = 0
    total_images = len(source_labels)
    
    # Find project root for path construction
    current = output_dir
    project_root = None
    while current != current.parent:
        if current.name == 'source_localization':
            project_root = current.parent
            break
        current = current.parent
    
    for i, label in enumerate(source_labels, 1):
        if i % 1000 == 0 or i == 1 or i == total_images:
            print(f"Transforming images: {i}/{total_images}")
        image_name = label['image_name']
        source_image_path = source_images_dir / image_name
        
        if not source_image_path.exists():
            print(f"Warning: Image not found: {source_image_path.name}")
            error_count += 1
            continue
        
        try:
            # Load image once
            image = cv2.imread(str(source_image_path), cv2.IMREAD_UNCHANGED)
            
            if image is None:
                print(f"Warning: Could not read image: {source_image_path.name}")
                error_count += 1
                continue
            
            # Enhance contrast first
            image = enhance_contrast(image, contrast_factor)
            
            # Get original image dimensions
            original_height, original_width = image.shape[:2]
            original_bbox = label['bbox']
            
            # Apply multiple transformations to the same image
            for _ in range(num_transforms):
                # Generate random translation
                translation = generate_random_translation(max_translation)
                
                # Apply translation to image
                transformed_image = translate_image(image, translation)
                
                # Transform bbox coordinates
                transformed_bbox = transform_bbox(
                    original_bbox,
                    translation
                )
                
                # Clamp bbox to image bounds
                transformed_bbox[0] = max(0, min(transformed_bbox[0], original_width - 1))
                transformed_bbox[1] = max(0, min(transformed_bbox[1], original_height - 1))
                transformed_bbox[2] = max(1, min(transformed_bbox[2], original_width - transformed_bbox[0]))
                transformed_bbox[3] = max(1, min(transformed_bbox[3], original_height - transformed_bbox[1]))
                
                # Create new image name with transformation parameters
                # Remove .png extension from original name if present
                base_name = image_name.replace('.png', '')
                new_image_name = f"{base_name}_trans{translation[0]}_{translation[1]}.png"
                new_image_path = output_dir / new_image_name
                
                # Save transformed image to output directory
                cv2.imwrite(str(new_image_path), transformed_image)
                
                # Create new label entry
                new_label = label.copy()
                new_label['image_name'] = new_image_name
                new_label['rotation'] = 0
                new_label['translation'] = list(translation)
                new_label['image_size'] = [original_width, original_height]
                new_label['bbox'] = transformed_bbox
                
                # Update image_path to point to new location
                if project_root:
                    relative_path = new_image_path.relative_to(project_root)
                else:
                    # Fallback: construct path manually
                    relative_path = Path('source_localization') / 'dataset' / 'plume_image_dataset' / 'transformed_images' / new_image_name
                
                new_label['image_path'] = str(relative_path).replace('\\', '/')
                
                transformed_labels.append(new_label)
                transformed_count += 1
        
        except Exception as e:
            print(f"Error processing {source_image_path.name}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue
    
    # Save new labels.json in output directory
    output_labels_path = output_dir / 'labels.json'
    print(f"\n\nSaving new labels.json to: {output_labels_path}")
    with open(output_labels_path, 'w') as f:
        json.dump(transformed_labels, f, indent=4)
    
    print(f"\nTransformation complete!")
    print(f"  Transformed: {transformed_count} images")
    print(f"  Errors: {error_count} images")
    print(f"  Output directory: {output_dir}")
    print(f"  New labels.json: {output_labels_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Apply random translation and contrast enhancement to images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform images with default settings (translation + 20% contrast increase)
  python transform_images.py

  # Transform with custom max translation
  python transform_images.py --max-translation 50

  # Apply 3 random transformations to each image
  python transform_images.py --num-transforms 3

  # Use specific random seed for reproducibility
  python transform_images.py --seed 42

  # Custom paths
  python transform_images.py --source-images-dir /path/to/images --source-labels /path/to/labels.json --output-dir /path/to/output
        """
    )

    parser.add_argument('--source-images-dir', type=str, default=None,
                        help='Directory containing original image files (default: source_localization/dataset/plume_image_dataset/all_images)')
    parser.add_argument('--source-labels', type=str, default=None,
                        help='Path to source labels.json (default: source_localization/dataset/plume_image_dataset/all_images/labels.json)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save transformed images and labels.json (default: source_localization/dataset/plume_image_dataset/transformed_images)')
    parser.add_argument('--max-translation', type=int, default=100,
                        help='Maximum translation distance in pixels (default: 100)')
    parser.add_argument('--num-transforms', type=int, default=1,
                        help='Number of random transformations to apply per image (default: 1)')
    parser.add_argument('--contrast-factor', type=float, default=1.2,
                        help='Contrast enhancement factor (default: 1.2 = 20%% increase)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None)')

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent

    if args.source_images_dir:
        source_images_dir = Path(args.source_images_dir)
    else:
        source_images_dir = script_dir / 'plume_image_dataset' / 'all_images'

    if args.source_labels:
        source_labels_path = Path(args.source_labels)
    else:
        source_labels_path = script_dir / 'plume_image_dataset' / 'all_images' / 'labels.json'

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = script_dir / 'plume_image_dataset' / 'transformed_images'

    # Validate paths
    if not source_images_dir.exists():
        print(f"Error: Source images directory not found: {source_images_dir}")
        return 1

    if not source_labels_path.exists():
        print(f"Error: Source labels.json not found: {source_labels_path}")
        return 1

    # Transform images
    try:
        transform_images(
            source_images_dir=source_images_dir,
            source_labels_path=source_labels_path,
            output_dir=output_dir,
            max_translation=args.max_translation,
            num_transforms=args.num_transforms,
            contrast_factor=args.contrast_factor,
            seed=args.seed
        )
        return 0
    except Exception as e:
        print(f"Error transforming images: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

