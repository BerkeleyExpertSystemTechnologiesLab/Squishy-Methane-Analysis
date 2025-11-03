#!/usr/bin/env python3
"""
Create labels.json file with metadata for all images in plume_image_dataset/all_images.

This script reads all PNG images from the all_images directory and creates a JSON file
with image metadata including name, path, size, channels, format, bbox, rotation, and translation.
"""

import cv2
import json
import argparse
from pathlib import Path
from typing import List, Dict


def get_image_properties(image_path: Path) -> Dict:
    """
    Get properties of an image file.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary with image_size, image_channels, and image_format
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    height, width = img.shape[:2]
    
    # Determine number of channels
    if len(img.shape) == 2:
        channels = 1  # Grayscale
    else:
        channels = img.shape[2]
    
    # Get format from file extension
    format_ext = image_path.suffix.lower()
    if format_ext == '.png':
        image_format = 'png'
    elif format_ext == '.jpg' or format_ext == '.jpeg':
        image_format = 'jpg'
    else:
        image_format = format_ext[1:] if format_ext.startswith('.') else 'unknown'
    
    return {
        'image_size': [width, height],
        'image_channels': channels,
        'image_format': image_format
    }


def create_labels(
    images_dir: Path,
    output_path: Path,
    bbox: List[int] = [170, 120, 20, 10]
) -> None:
    """
    Create labels.json file with metadata for all images.

    Args:
        images_dir: Directory containing image files
        output_path: Path to save labels.json file
        bbox: Bounding box in xywh format [x, y, width, height]
    """
    # Find all PNG images
    image_files = sorted(images_dir.glob("*.png"))
    
    if not image_files:
        print(f"Warning: No PNG images found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Creating labels.json at: {output_path}")
    print(f"Using bbox: {bbox} (xywh format)\n")
    
    labels = []
    
    for image_path in image_files:
        print(f"Processing: {image_path.name}", end='\r')
        
        try:
            # Get image properties
            props = get_image_properties(image_path)
            
            # Create path relative to project root
            # Expected format: source_localization/dataset/plume_image_dataset/all_images/...
            # Find project root by looking for source_localization directory
            current = image_path.parent
            project_root = None
            
            while current != current.parent:
                if current.name == 'source_localization':
                    project_root = current.parent
                    break
                current = current.parent
            
            if project_root:
                # Path relative to project root
                relative_path = image_path.relative_to(project_root)
            else:
                # Fallback: construct path manually
                # Assuming structure: .../source_localization/dataset/plume_image_dataset/all_images/image.png
                relative_path = Path('source_localization') / 'dataset' / 'plume_image_dataset' / 'all_images' / image_path.name
            
            image_path_str = str(relative_path).replace('\\', '/')
            
            # Create label entry
            label_entry = {
                "image_name": image_path.name,
                "image_path": image_path_str,
                "image_size": props['image_size'],
                "image_channels": props['image_channels'],
                "image_format": props['image_format'],
                "bbox": bbox,
                "bbox_format": "xywh",
                "rotation": 0,
                "rotation_format": "degrees",
                "translation": [0, 0]
            }
            
            labels.append(label_entry)
            
        except Exception as e:
            print(f"\nError processing {image_path.name}: {e}")
            continue
    
    # Write labels to JSON file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=4)
    
    print(f"\n\nSuccessfully created labels.json")
    print(f"  Total images processed: {len(labels)}")
    print(f"  Output file: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Create labels.json file with metadata for all images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create labels.json with default paths
  python create_labels.py

  # Create labels.json with custom paths
  python create_labels.py --images-dir /path/to/images --output /path/to/labels.json

  # Custom bbox
  python create_labels.py --bbox 100 100 50 50
        """
    )

    parser.add_argument('--images-dir', type=str, default=None,
                        help='Directory containing image files (default: source_localization/dataset/plume_image_dataset/all_images)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save labels.json (default: source_localization/dataset/plume_image_dataset/labels.json)')
    parser.add_argument('--bbox', type=int, nargs=4, default=[170, 120, 20, 10],
                        metavar=('X', 'Y', 'W', 'H'),
                        help='Bounding box in xywh format (default: 170 120 20 10)')

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent

    if args.images_dir:
        images_dir = Path(args.images_dir)
    else:
        images_dir = script_dir / 'plume_image_dataset' / 'all_images'

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = script_dir / 'plume_image_dataset' / 'labels.json'

    # Validate paths
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return 1

    # Create labels
    try:
        create_labels(
            images_dir=images_dir,
            output_path=output_path,
            bbox=args.bbox
        )
        return 0
    except Exception as e:
        print(f"Error creating labels: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

