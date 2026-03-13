import json
from pathlib import Path
from PIL import Image, ImageDraw
from typing import Union, List, Dict


def draw_bbox_on_image(
    label: Dict,
    output_path: Union[str, Path] = None,
    color: str = "green",
    thickness: int = 2,
    base_path: Union[str, Path] = None,
    box_size: int = 50
) -> Image.Image:
    """
    Draw a bounding box on an image based on center coordinate.
    
    Args:
        label: Dictionary containing image data with 'image_path' and 'center_coord' keys
        output_path: Optional path to save the annotated image
        color: Color of the bounding box (default: "green")
        thickness: Thickness of the bounding box lines (default: 2)
        base_path: Base path to resolve relative image paths (default: current working directory)
        box_size: Size of the bounding box (width and height in pixels, default: 50)
    
    Returns:
        PIL Image object with bounding box drawn
    
    Raises:
        FileNotFoundError: If the image file does not exist
        ValueError: If required keys are missing from label
    """
    # Validate label dictionary
    if "image_path" not in label or "center_coord" not in label:
        raise ValueError("Label must contain 'image_path' and 'center_coord' keys")
    
    # Construct full image path
    image_path = Path(label["image_path"])
    image_filename = image_path.name
    if not image_path.is_absolute() and base_path:
        image_path = Path(base_path) / image_filename
    
    # Load image
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    
    # Extract center coordinate and compute bbox
    center_x, center_y = label["center_coord"]
    half_size = box_size // 2
    
    # Create bbox in (x1, y1, x2, y2) format centered on the coordinate
    bbox_coords = (
        center_x - half_size,
        center_y - half_size,
        center_x + half_size,
        center_y + half_size
    )
    
    # Draw bounding box
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox_coords, outline=color, width=thickness)
    
    # Save if output path is provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        print(f"Annotated image saved to: {output_path}")
    
    return image


def draw_bboxes_from_json(
    json_path: Union[str, Path],
    output_dir: Union[str, Path] = None,
    color: str = "green",
    thickness: int = 2,
    base_path: Union[str, Path] = None,
    box_size: int = 50
) -> List[Image.Image]:
    """
    Draw bounding boxes on images from a labels.json file.
    Creates 50x50 (or custom size) bounding boxes centered on coordinates from the JSON.
    
    Args:
        json_path: Path to the labels.json file
        output_dir: Optional directory to save annotated images
        color: Color of the bounding boxes (default: "green")
        thickness: Thickness of the bounding box lines (default: 2)
        base_path: Base path to resolve relative image paths (default: parent of json_path)
        box_size: Size of the bounding box in pixels (default: 50)
    
    Returns:
        List of PIL Image objects with bounding boxes drawn
    """
    json_path = Path(json_path)
    
    # Use json_path's parent as base_path if not provided
    if base_path is None:
        base_path = json_path.parent
    
    # Load labels
    with open(json_path, "r") as f:
        labels = json.load(f)
    
    # Ensure labels is a list
    if isinstance(labels, dict):
        labels = [labels]
    
    images = []
    for i, label in enumerate(labels):
        output_path = None
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            image_name = label.get("image_name", f"image_{i}.png")
            output_path = output_dir / f"annotated_{image_name}"
        
        image = draw_bbox_on_image(
            label,
            output_path=output_path,
            color=color,
            thickness=thickness,
            base_path=base_path,
            box_size=box_size
        )
        images.append(image)
    
    return images

import argparse


if __name__ == "__main__":
    # parse command-line arguments so users can specify input paths
    parser = argparse.ArgumentParser(
        description="Draw bounding boxes on images using a labels JSON file."
    )
    parser.add_argument(
        "--json-file",
        required=True,
        type=Path,
        help="Path to the labels JSON file."
    )
    parser.add_argument(
        "--base-path",
        required=False,
        type=Path,
        help="Base directory to resolve relative image paths. Defaults to parent of --json-file."
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        type=Path,
        default=None,
        help="Directory where annotated images will be saved. If not provided, images are not saved."
    )
    parser.add_argument(
        "--color",
        required=False,
        default="green",
        help="Bounding box color (default: green)."
    )
    parser.add_argument(
        "--thickness",
        required=False,
        type=int,
        default=2,
        help="Bounding box line thickness (default: 2)."
    )
    parser.add_argument(
        "--box-size",
        required=False,
        type=int,
        default=50,
        help="Size of the bounding box in pixels (default: 50)."
    )

    args = parser.parse_args()

    json_file = args.json_file
    base_path = args.base_path or json_file.parent

    if json_file.exists():
        images = draw_bboxes_from_json(
            json_file,
            output_dir=args.output_dir,
            color=args.color,
            thickness=args.thickness,
            base_path=base_path,
            box_size=args.box_size,
        )
        print(f"Processed {len(images)} images")
    else:
        print(f"labels.json not found at {json_file}")