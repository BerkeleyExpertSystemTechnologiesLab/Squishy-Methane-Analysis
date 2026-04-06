#!/usr/bin/env python3
"""
Predict labels for raw images using a YOLO model.

This script reads raw images from a directory and uses a YOLO model to predict
bounding boxes for each image. Creates a labels.json file with image metadata
including predicted bounding boxes and center coordinates.
"""

import cv2
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


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


def predict_labels(
    images_dir: Path,
    output_path: Path,
    model_path: Path,
    conf_threshold: float = 0.25
) -> None:
    """
    Predict labels for raw images using a YOLO model.

    Args:
        images_dir: Directory containing image files
        output_path: Path to save labels.json file
        model_path: Path to YOLO model file (e.g., 'best.pt' or 'yolov8s.pt')
        conf_threshold: Confidence threshold for model predictions (default: 0.25)
    """
    # Load YOLO model
    if not YOLO_AVAILABLE:
        print("Error: ultralytics package is required")
        print("Install it with: pip install ultralytics")
        return
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
    print(f"Loading YOLO model from: {model_path}")
    try:
        model = YOLO(str(model_path))
        print("Model loaded successfully\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Find all image files (PNG, JPG, JPEG)
    image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.jpeg"))
    
    if not image_files:
        print(f"Warning: No image files found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Creating labels.json at: {output_path}")
    print(f"Using confidence threshold: {conf_threshold}\n")
    
    labels = []
    predictions_count = 0
    no_detection_count = 0
    
    for image_path in image_files:
        print(f"Processing: {image_path.name}", end='\r')
        
        try:
            # Get image properties
            props = get_image_properties(image_path)
            
            # Predict bounding box
            bbox = predict_bbox_with_model(image_path, model, conf_threshold)
            
            if bbox is None:
                print(f"\nNo detection: {image_path.name}")
                no_detection_count += 1
                continue
            
            # Compute center from predicted bbox
            center_coord = [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2]
            predictions_count += 1
            
            # Create label entry
            label_entry = {
                "image_name": image_path.name,
                "image_size": props['image_size'],
                "image_channels": props['image_channels'],
                "image_format": props['image_format'],
                "bbox": bbox,
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
    print(f"  Images with predictions: {predictions_count}")
    print(f"  Images with no detection: {no_detection_count}")
    print(f"  Output file: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Predict labels for raw images using a YOLO model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict labels with default model (models/yolov8s.pt)
  python predict_labels.py --images-dir /path/to/images --output /path/to/labels.json

  # Predict labels with custom model
  python predict_labels.py --images-dir /path/to/images --output /path/to/labels.json --model-path /path/to/best.pt

  # Use custom confidence threshold
  python predict_labels.py --images-dir /path/to/images --output /path/to/labels.json --conf 0.5
        """
    )

    parser.add_argument('--images-dir', type=str, required=True,
                        help='Directory containing image files')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save labels.json file')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to YOLO model file (default: ../models/yolov8s.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for model predictions (default: 0.25)')

    args = parser.parse_args()

    # Validate images directory
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return 1

    # Determine model path
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        # Default to models/yolov8s.pt relative to script directory
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        model_path = project_root / 'models' / 'yolov8s.pt'

    output_path = Path(args.output)

    # Predict labels
    try:
        predict_labels(
            images_dir=images_dir,
            output_path=output_path,
            model_path=model_path,
            conf_threshold=args.conf
        )
        return 0
    except Exception as e:
        print(f"Error predicting labels: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
