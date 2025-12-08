#!/usr/bin/env python3
"""
Export dataset from transformed_images to YOLO and COCO formats and push to Hugging Face.

This script:
1. Reads labels.json from transformed_images directory
2. Converts to YOLO format (normalized center x, center y, width, height)
3. Converts to COCO format (JSON with images, annotations, categories)
4. Creates train/val/test splits
5. Pushes both datasets to Hugging Face Hub
"""

import json
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
from collections import defaultdict

try:
    from huggingface_hub import HfApi, create_repo, login
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Install with: pip install huggingface_hub")


def xywh_to_yolo(x: float, y: float, w: float, h: float, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from xywh format to YOLO format (normalized center x, center y, width, height).
    
    Args:
        x, y, w, h: Bounding box in xywh format (top-left corner, width, height)
        img_width, img_height: Image dimensions
        
    Returns:
        Tuple of (center_x, center_y, width, height) all normalized to [0, 1]
    """
    # Calculate center coordinates
    center_x = (x + w / 2.0) / img_width
    center_y = (y + h / 2.0) / img_height
    
    # Normalize width and height
    norm_width = w / img_width
    norm_height = h / img_height
    
    # Clamp to [0, 1]
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))
    norm_width = max(0.0, min(1.0, norm_width))
    norm_height = max(0.0, min(1.0, norm_height))
    
    return (center_x, center_y, norm_width, norm_height)


def xywh_to_coco(x: float, y: float, w: float, h: float) -> List[float]:
    """
    Convert bounding box from xywh format to COCO format (x, y, width, height).
    COCO format is the same as xywh, but we ensure it's a list.
    
    Args:
        x, y, w, h: Bounding box in xywh format
        
    Returns:
        List [x, y, width, height]
    """
    return [float(x), float(y), float(w), float(h)]


def create_yolo_dataset(
    labels: List[Dict],
    images_dir: Path,
    output_dir: Path,
    class_id: int = 0,
    class_name: str = "gas_leak"
) -> None:
    """
    Create YOLO format dataset.
    
    YOLO format structure:
    - images/train/, images/val/, images/test/
    - labels/train/, labels/val/, labels/test/
    - data.yaml (dataset configuration)
    
    Args:
        labels: List of label dictionaries from labels.json
        images_dir: Directory containing source images
        output_dir: Output directory for YOLO dataset
        class_id: Class ID for the object (default: 0)
        class_name: Class name (default: "gas_leak")
    """
    print(f"\nCreating YOLO dataset in: {output_dir}")
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Process labels
    for label in labels:
        image_name = label['image_name']
        split = label.get('split', 'train')  # Default to train if not specified
        
        # Source image path
        source_image = images_dir / image_name
        
        if not source_image.exists():
            print(f"Warning: Image not found: {source_image}")
            continue
        
        # Copy image to YOLO dataset
        dest_image = output_dir / 'images' / split / image_name
        shutil.copy2(source_image, dest_image)
        
        # Create YOLO label file
        label_file = output_dir / 'labels' / split / (image_name.replace('.png', '.txt').replace('.jpg', '.txt'))
        
        # Get bbox and image dimensions
        bbox = label['bbox']  # [x, y, width, height]
        img_size = label['image_size']  # [width, height]
        
        # Convert to YOLO format
        center_x, center_y, norm_width, norm_height = xywh_to_yolo(
            bbox[0], bbox[1], bbox[2], bbox[3],
            img_size[0], img_size[1]
        )
        
        # Write YOLO label file (class_id center_x center_y width height)
        with open(label_file, 'w') as f:
            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
    
    # Create data.yaml
    yaml_content = f"""# YOLO Dataset Configuration
path: {output_dir.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
nc: 1  # number of classes
names:
  0: {class_name}
"""
    
    with open(output_dir / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"YOLO dataset created successfully!")
    print(f"  Images: {output_dir / 'images'}")
    print(f"  Labels: {output_dir / 'labels'}")
    print(f"  Config: {output_dir / 'data.yaml'}")


def create_coco_dataset(
    labels: List[Dict],
    images_dir: Path,
    output_dir: Path,
    class_id: int = 1,
    class_name: str = "gas_leak"
) -> None:
    """
    Create COCO format dataset.
    
    COCO format structure:
    - images/train/, images/val/, images/test/
    - annotations/instances_train.json, instances_val.json, instances_test.json
    
    Args:
        labels: List of label dictionaries from labels.json
        images_dir: Directory containing source images
        output_dir: Output directory for COCO dataset
        class_id: Class ID for the object (default: 1, COCO uses 1-indexed)
        class_name: Class name (default: "gas_leak")
    """
    print(f"\nCreating COCO dataset in: {output_dir}")
    
    # Create directory structure
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'annotations').mkdir(parents=True, exist_ok=True)
    
    # Group labels by split
    labels_by_split = defaultdict(list)
    for label in labels:
        split = label.get('split', 'train')
        labels_by_split[split].append(label)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_labels = labels_by_split[split]
        
        if not split_labels:
            print(f"Warning: No labels for {split} split, skipping...")
            continue
        
        # Create split image directory
        split_image_dir = output_dir / 'images' / split
        split_image_dir.mkdir(parents=True, exist_ok=True)
        
        # COCO format structure
        coco_data = {
            "info": {
                "description": "Gas Leak Source Localization Dataset",
                "version": "1.0",
                "year": 2025
            },
            "licenses": [],
            "categories": [
                {
                    "id": class_id,
                    "name": class_name,
                    "supercategory": "gas"
                }
            ],
            "images": [],
            "annotations": []
        }
        
        image_id = 1
        annotation_id = 1
        
        for label in split_labels:
            image_name = label['image_name']
            source_image = images_dir / image_name
            
            if not source_image.exists():
                print(f"Warning: Image not found: {source_image}")
                continue
            
            # Copy image
            dest_image = split_image_dir / image_name
            shutil.copy2(source_image, dest_image)
            
            # Get image properties
            img_size = label['image_size']  # [width, height]
            bbox = label['bbox']  # [x, y, width, height]
            
            # Add image entry
            image_entry = {
                "id": image_id,
                "width": img_size[0],
                "height": img_size[1],
                "file_name": image_name
            }
            coco_data["images"].append(image_entry)
            
            # Convert bbox to COCO format
            coco_bbox = xywh_to_coco(bbox[0], bbox[1], bbox[2], bbox[3])
            area = bbox[2] * bbox[3]  # width * height
            
            # Add annotation entry
            annotation_entry = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": coco_bbox,
                "area": area,
                "iscrowd": 0
            }
            coco_data["annotations"].append(annotation_entry)
            
            image_id += 1
            annotation_id += 1
        
        # Save COCO annotation file
        annotation_file = output_dir / 'annotations' / f'instances_{split}.json'
        with open(annotation_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"  {split}: {len(split_labels)} images, {len(coco_data['annotations'])} annotations")
    
    print(f"COCO dataset created successfully!")
    print(f"  Images: {output_dir / 'images'}")
    print(f"  Annotations: {output_dir / 'annotations'}")


def split_dataset(
    labels: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: Optional[int] = None
) -> List[Dict]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        labels: List of label dictionaries
        train_ratio: Ratio for training set (default: 0.7)
        val_ratio: Ratio for validation set (default: 0.2)
        test_ratio: Ratio for test set (default: 0.1)
        seed: Random seed for reproducibility
        
    Returns:
        Labels with 'split' field added
    """
    if seed is not None:
        random.seed(seed)
    
    # Shuffle labels
    shuffled_labels = labels.copy()
    random.shuffle(shuffled_labels)
    
    total = len(shuffled_labels)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Assign splits
    for i, label in enumerate(shuffled_labels):
        if i < train_end:
            label['split'] = 'train'
        elif i < val_end:
            label['split'] = 'val'
        else:
            label['split'] = 'test'
    
    train_count = sum(1 for l in shuffled_labels if l['split'] == 'train')
    val_count = sum(1 for l in shuffled_labels if l['split'] == 'val')
    test_count = sum(1 for l in shuffled_labels if l['split'] == 'test')
    
    print(f"\nDataset split:")
    print(f"  Train: {train_count} ({train_count/total*100:.1f}%)")
    print(f"  Val: {val_count} ({val_count/total*100:.1f}%)")
    print(f"  Test: {test_count} ({test_count/total*100:.1f}%)")
    
    return shuffled_labels


def upload_to_huggingface(
    dataset_dir: Path,
    repo_id: str,
    repo_type: str = "dataset",
    token: Optional[str] = None
) -> None:
    """
    Upload dataset to Hugging Face Hub.
    
    Args:
        dataset_dir: Directory containing the dataset
        repo_id: Hugging Face repository ID (e.g., "username/dataset-name")
        repo_type: Type of repository ("dataset" or "model")
        token: Hugging Face token (if None, will try to use cached token)
    """
    if not HF_AVAILABLE:
        raise ImportError("huggingface_hub is not installed. Install with: pip install huggingface_hub")
    
    print(f"\nUploading to Hugging Face: {repo_id}")
    
    # Login if token provided
    if token:
        login(token=token)
    else:
        # Try to use cached token
        try:
            login()
        except Exception as e:
            print(f"Error: Could not authenticate with Hugging Face. Please provide a token or run 'huggingface-cli login'")
            raise
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id, repo_type=repo_type, exist_ok=True)
    except Exception as e:
        print(f"Note: Repository may already exist or there was an issue: {e}")
    
    # Upload dataset
    api = HfApi()
    print(f"Uploading folder: {dataset_dir}")
    print("This may take a while for large datasets...")
    
    try:
        api.upload_folder(
            folder_path=str(dataset_dir),
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message="Upload dataset"
        )
        print(f"\n✓ Successfully uploaded to: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"\n✗ Error uploading: {e}")
        print("For large datasets, consider using 'hf upload-large-folder' command or uploading in batches")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Export dataset to YOLO and COCO formats and push to Hugging Face',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to YOLO and COCO formats (local only)
  python export_to_huggingface.py

  # Export and push to Hugging Face
  python export_to_huggingface.py --push --repo-id username/gas-leak-dataset --hf-token YOUR_TOKEN

  # Custom split ratios
  python export_to_huggingface.py --train-ratio 0.8 --val-ratio 0.15 --test-ratio 0.05

  # Custom output directories
  python export_to_huggingface.py --yolo-dir ./yolo_dataset --coco-dir ./coco_dataset
        """
    )
    
    parser.add_argument('--images-dir', type=str, default=None,
                        help='Directory containing transformed images (default: source_localization/dataset/plume_image_dataset/transformed_images)')
    parser.add_argument('--labels-json', type=str, default=None,
                        help='Path to labels.json (default: images-dir/labels.json)')
    parser.add_argument('--yolo-dir', type=str, default=None,
                        help='Output directory for YOLO dataset (default: ./yolo_dataset)')
    parser.add_argument('--coco-dir', type=str, default=None,
                        help='Output directory for COCO dataset (default: ./coco_dataset)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Validation set ratio (default: 0.2)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for dataset splitting (default: 42)')
    parser.add_argument('--class-name', type=str, default='leak_source',
                        help='Class name for annotations (default: leak_source)')
    parser.add_argument('--push', action='store_true',
                        help='Push datasets to Hugging Face Hub')
    parser.add_argument('--repo-id', type=str, default=None,
                        help='Hugging Face repository ID (e.g., username/dataset-name). Required if --push is used.')
    parser.add_argument('--hf-token', type=str, default=None,
                        help='Hugging Face token (optional, will use cached token if not provided)')
    parser.add_argument('--yolo-only', action='store_true',
                        help='Only create YOLO dataset')
    parser.add_argument('--coco-only', action='store_true',
                        help='Only create COCO dataset')
    
    args = parser.parse_args()
    
    # Validate split ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print(f"Error: Train, val, and test ratios must sum to 1.0")
        print(f"  Current: {args.train_ratio} + {args.val_ratio} + {args.test_ratio} = {args.train_ratio + args.val_ratio + args.test_ratio}")
        return 1
    
    # Determine paths
    script_dir = Path(__file__).parent
    
    if args.images_dir:
        images_dir = Path(args.images_dir)
    else:
        images_dir = script_dir / 'plume_image_dataset' / 'transformed_images'
    
    if args.labels_json:
        labels_path = Path(args.labels_json)
    else:
        labels_path = images_dir / 'labels.json'
    
    if args.yolo_dir:
        yolo_dir = Path(args.yolo_dir)
    else:
        yolo_dir = Path('./yolo_dataset')
    
    if args.coco_dir:
        coco_dir = Path(args.coco_dir)
    else:
        coco_dir = Path('./coco_dataset')
    
    # Validate paths
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return 1
    
    if not labels_path.exists():
        print(f"Error: Labels file not found: {labels_path}")
        return 1
    
    # Load labels
    print(f"Loading labels from: {labels_path}")
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    print(f"Loaded {len(labels)} labels")
    
    # Split dataset
    labels = split_dataset(
        labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Create datasets
    if not args.coco_only:
        create_yolo_dataset(
            labels,
            images_dir,
            yolo_dir,
            class_id=0,
            class_name=args.class_name
        )
    
    if not args.yolo_only:
        create_coco_dataset(
            labels,
            images_dir,
            coco_dir,
            class_id=1,
            class_name=args.class_name
        )
    
    # Push to Hugging Face if requested
    if args.push:
        if not args.repo_id:
            print("Error: --repo-id is required when using --push")
            return 1
        
        if not args.yolo_only:
            yolo_repo_id = f"{args.repo_id}-yolo"
            print(f"\n{'='*60}")
            print(f"Uploading YOLO dataset to: {yolo_repo_id}")
            print(f"{'='*60}")
            try:
                upload_to_huggingface(
                    yolo_dir,
                    yolo_repo_id,
                    repo_type="dataset",
                    token=args.hf_token
                )
            except Exception as e:
                print(f"Failed to upload YOLO dataset: {e}")
        
        if not args.coco_only:
            coco_repo_id = f"{args.repo_id}-coco"
            print(f"\n{'='*60}")
            print(f"Uploading COCO dataset to: {coco_repo_id}")
            print(f"{'='*60}")
            try:
                upload_to_huggingface(
                    coco_dir,
                    coco_repo_id,
                    repo_type="dataset",
                    token=args.hf_token
                )
            except Exception as e:
                print(f"Failed to upload COCO dataset: {e}")
    
    print("\n✓ Export complete!")
    return 0


if __name__ == '__main__':
    exit(main())
