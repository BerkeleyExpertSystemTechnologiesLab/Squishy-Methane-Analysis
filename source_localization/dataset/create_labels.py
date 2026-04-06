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
from typing import List, Dict, Optional, Tuple

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


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


def predict_bbox_with_model(image_path: Path, model, conf_threshold: float = 0.25) -> Optional[List[int]]:
    """
    Use YOLO model to predict bounding box for an image.
    
    Args:
        image_path: Path to the image file
        model: YOLO model instance
        conf_threshold: Confidence threshold for predictions
        
    Returns:
        Bounding box in [x, y, width, height] format, or None if no detection
    """
    try:
        # Run inference
        results = model.predict(str(image_path), conf=conf_threshold, verbose=False)
        
        if not results or len(results) == 0:
            return None
        
        # Get the first result
        result = results[0]
        boxes = result.boxes
        
        if len(boxes) == 0:
            return None
        
        # Get the box with highest confidence
        best_box = boxes[0]
        
        # Extract coordinates in xyxy format (x1, y1, x2, y2)
        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
        
        # Convert to [x, y, width, height] format
        x = int(x1)
        y = int(y1)
        width = int(x2 - x1)
        height = int(y2 - y1)
        
        return [x, y, width, height]
    
    except Exception as e:
        print(f"\nWarning: Model prediction failed for {image_path.name}: {e}")
        return None


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
    use_model_prediction: bool = False,
    model_path: Optional[Path] = None,
    conf_threshold: float = 0.25
) -> None:
    """
    Create labels.json file with metadata for all images.
    Bounding boxes must be provided via either source_bbox.csv or YOLO model prediction.

    Args:
        images_dir: Directory containing image files
        output_path: Path to save labels.json file
        bbox_csv_path: Path to source_bbox.csv file (default: metadata/source_bbox.csv)
        use_model_prediction: Whether to use YOLO model to predict bounding boxes
        model_path: Path to YOLO model file (e.g., 'best.pt' or 'yolov8s.pt')
        conf_threshold: Confidence threshold for model predictions (default: 0.25)
    """
    # Load YOLO model if requested
    model = None
    if use_model_prediction:
        if not YOLO_AVAILABLE:
            print("Error: ultralytics package is required for model predictions")
            print("Install it with: pip install ultralytics")
            return
        
        if model_path is None:
            print("Error: model_path is required when use_model_prediction=True")
            return
        
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"Error: Model file not found: {model_path}")
            return
        
        print(f"Loading YOLO model from: {model_path}")
        try:
            model = YOLO(str(model_path))
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    
    # Load center coordinates from CSV
    if bbox_csv_path is None:
        script_dir = Path(__file__).parent
        bbox_csv_path = script_dir / 'metadata' / 'source_bbox.csv'
    
    centers = {}
    if not use_model_prediction and bbox_csv_path.exists():
        print(f"Loading center coordinates from: {bbox_csv_path}")
        centers = parse_source_bboxes(bbox_csv_path)
        print(f"Loaded {len(centers)} video center coordinates")
    elif not use_model_prediction:
        print(f"Warning: Bbox CSV not found at {bbox_csv_path}, using default center for all images")
    
    # Find all PNG images
    image_files = sorted(images_dir.glob("*.png"))
    
    if not image_files:
        print(f"Warning: No PNG images found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Creating labels.json at: {output_path}")
    if use_model_prediction:
        print(f"Using model predictions with confidence threshold: {conf_threshold}\n")
    else:
        print()
    
    labels = []
    videos_with_bbox = set()
    model_predictions_count = 0
    images_skipped = 0
    
    for image_path in image_files:
        print(f"Processing: {image_path.name}", end='\r')
        
        try:
            # Get image properties
            props = get_image_properties(image_path)
            
            # Extract video number from filename
            video_no = extract_video_number(image_path.name)
            
            # Determine bounding box source
            bbox = None
            center_coord = None
            
            if use_model_prediction and model is not None:
                # Use model prediction
                bbox = predict_bbox_with_model(image_path, model, conf_threshold)
                if bbox is not None:
                    # Compute center from predicted bbox
                    center_coord = [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2]
                    model_predictions_count += 1
                else:
                    # Model prediction failed - skip this image
                    print(f"\nSkipping {image_path.name}: No bounding box detected by model")
                    images_skipped += 1
                    continue
            else:
                # Use CSV-based center
                if video_no is not None and video_no in centers:
                    center_coord = centers[video_no]
                    videos_with_bbox.add(video_no)
                else:
                    # No center available from CSV - skip this image
                    print(f"\nSkipping {image_path.name}: No bounding box found in CSV for video {video_no}")
                    images_skipped += 1
                    continue
            
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
            
            # Add bbox if available (from model prediction)
            if bbox is not None:
                label_entry["bbox"] = bbox
            
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
    print(f"  Images skipped (no bbox available): {images_skipped}")
    if use_model_prediction:
        print(f"  Images with model predictions: {model_predictions_count}")
    else:
        print(f"  Videos with bbox from CSV: {len(videos_with_bbox)}")
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
  python create_labels.py --default-center 100 100

  # Use YOLO model to predict bounding boxes
  python create_labels.py --use-model --model-path /path/to/best.pt

  # Use model with custom confidence threshold
  python create_labels.py --use-model --model-path /path/to/best.pt --conf 0.5
        """
    )

    parser.add_argument('--images-dir', type=str, default=None,
                        help='Directory containing image files (default: source_localization/dataset/plume_image_dataset/all_images)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save labels.json (default: source_localization/dataset/plume_image_dataset/labels.json)')
    parser.add_argument('--bbox-csv', type=str, default=None,
                        help='Path to source_bbox.csv file (default: source_localization/dataset/metadata/source_bbox.csv)')
    parser.add_argument('--use-model', action='store_true',
                        help='Use YOLO model to predict bounding boxes instead of CSV')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to YOLO model file (e.g., best.pt or yolov8s.pt). Required if --use-model is set')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for model predictions (default: 0.25)')

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent  # source_localization directory

    if args.images_dir:
        images_dir = Path(args.images_dir)
    else:
        images_dir = script_dir / 'plume_image_dataset' / 'all_images'

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = script_dir / 'plume_image_dataset' / 'labels.json'

    if args.bbox_csv:
        bbox_csv_path = Path(args.bbox_csv)
    else:
        bbox_csv_path = None  # Will use default path in create_labels

    # Determine model path
    if args.use_model:
        if args.model_path:
            model_path = Path(args.model_path)
        else:
            # Default to models/yolov8s.pt relative to project root
            model_path = project_root / 'models' / 'yolov8s.pt'
    else:
        model_path = None

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
            use_model_prediction=args.use_model,
            model_path=model_path,
            conf_threshold=args.conf
        )
        return 0
    except Exception as e:
        print(f"Error creating labels: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

