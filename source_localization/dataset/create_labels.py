#!/usr/bin/env python3
"""
Create labels.json file with metadata for all images in plume_image_dataset/all_images.

This script reads all PNG images from the all_images directory and creates a JSON file
with image metadata including name, path, size, channels, format, bbox, rotation, and translation.
"""

import cv2
import json
import argparse
import csv
import re
from pathlib import Path
from typing import List, Dict, Optional


def parse_source_bboxes(csv_path: Path) -> Dict[int, List[int]]:
    """
    Parse source bounding boxes from CSV file and convert to center coordinates.

    Args:
        csv_path: Path to source_bbox.csv file

    Returns:
        Dictionary mapping video numbers to [center_x, center_y] center coordinates
    """
    centers = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_no = int(row['video_number'])
            x = int(row['x'])
            y = int(row['y'])
            width = int(row['width'])
            height = int(row['height'])
            # Compute center from bbox (x, y, width, height)
            center_x = x + width // 2
            center_y = y + height // 2
            centers[video_no] = [center_x, center_y]
    return centers


def extract_video_number(image_filename: str) -> Optional[int]:
    """
    Extract video number from image filename.
    
    Expected format: MOV_<video_number>_plume_frame_<frame_number>.png
    or: MOV_<video_number>_frame_<frame_number>.png
    
    Args:
        image_filename: Name of the image file
        
    Returns:
        Video number if found, None otherwise
    """
    # Match patterns like MOV_1237_plume_frame_000099.png or MOV_1237_frame_000099.png
    match = re.search(r'MOV_(\d+)', image_filename)
    if match:
        return int(match.group(1))
    return None


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
    bbox_csv_path: Optional[Path] = None,
    default_center: List[int] = [195, 145]
) -> None:
    """
    Create labels.json file with metadata for all images.
    Uses per-video center coordinates from source_bbox.csv if available.

    Args:
        images_dir: Directory containing image files
        output_path: Path to save labels.json file
        bbox_csv_path: Path to source_bbox.csv file (default: metadata/source_bbox.csv)
        default_center: Default center coordinate [center_x, center_y]
                       used for videos not found in CSV (default: [195, 145] which is center of 50x50 box at 170,120)
    """
    # Load center coordinates from CSV
    if bbox_csv_path is None:
        script_dir = Path(__file__).parent
        bbox_csv_path = script_dir / 'metadata' / 'source_bbox.csv'
    
    centers = {}
    if bbox_csv_path.exists():
        print(f"Loading center coordinates from: {bbox_csv_path}")
        centers = parse_source_bboxes(bbox_csv_path)
        print(f"Loaded {len(centers)} video center coordinates")
    else:
        print(f"Warning: Bbox CSV not found at {bbox_csv_path}, using default center for all images")
    
    # Find all PNG images
    image_files = sorted(images_dir.glob("*.png"))
    
    if not image_files:
        print(f"Warning: No PNG images found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Creating labels.json at: {output_path}\n")
    
    labels = []
    videos_with_bbox = set()
    videos_without_bbox = set()
    
    for image_path in image_files:
        print(f"Processing: {image_path.name}", end='\r')
        
        try:
            # Get image properties
            props = get_image_properties(image_path)
            
            # Extract video number from filename
            video_no = extract_video_number(image_path.name)
            
            # Get center coordinate for this video, or use default
            if video_no is not None and video_no in centers:
                center_coord = centers[video_no]
                videos_with_bbox.add(video_no)
            else:
                center_coord = default_center
                if video_no is not None:
                    videos_without_bbox.add(video_no)
            
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
                "center_coord": center_coord,
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
    print(f"  Videos with bbox from CSV: {len(videos_with_bbox)}")
    if videos_without_bbox:
        print(f"  Videos using default bbox: {len(videos_without_bbox)} ({sorted(videos_without_bbox)})")
    print(f"  Output file: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Create labels.json file with metadata for all images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create labels.json with default paths (uses source_bbox.csv for per-video bboxes)
  python create_labels.py

  # Create labels.json with custom paths
  python create_labels.py --images-dir /path/to/images --output /path/to/labels.json

  # Use custom bbox CSV file
  python create_labels.py --bbox-csv /path/to/source_bbox.csv

  # Custom default bbox for videos not in CSV
  python create_labels.py --default-bbox 100 100 50 50
        """
    )

    parser.add_argument('--images-dir', type=str, default=None,
                        help='Directory containing image files (default: source_localization/dataset/plume_image_dataset/all_images)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save labels.json (default: source_localization/dataset/plume_image_dataset/labels.json)')
    parser.add_argument('--bbox-csv', type=str, default=None,
                        help='Path to source_bbox.csv file (default: source_localization/dataset/metadata/source_bbox.csv)')
    parser.add_argument('--default-center', type=int, nargs=2, default=[195, 145],
                        metavar=('CENTER_X', 'CENTER_Y'),
                        help='Default center coordinate for videos not in CSV (default: 195 145)')

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
        output_path = script_dir / 'plume_image_dataset' / 'all_images' / 'labels.json'

    if args.bbox_csv:
        bbox_csv_path = Path(args.bbox_csv)
    else:
        bbox_csv_path = None  # Will use default path in create_labels

    # Validate paths
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return 1

    # Create labels
    try:
        create_labels(
            images_dir=images_dir,
            output_path=output_path,
            bbox_csv_path=bbox_csv_path,
            default_center=args.default_center
        )
        return 0
    except Exception as e:
        print(f"Error creating labels: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

