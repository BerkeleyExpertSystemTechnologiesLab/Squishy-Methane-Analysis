import json
from pathlib import Path
from PIL import Image, ImageDraw
from typing import Union, List, Dict


def draw_bbox_on_image(
    label: Dict,
    output_path: Union[str, Path] = None,
    color: str = "green",
    thickness: int = 2,
    base_path: Union[str, Path] = None
) -> Image.Image:
    """
    Draw a bounding box on an image based on label data.
    
    Args:
        label: Dictionary containing image data with 'image_path' and 'bbox' keys
        output_path: Optional path to save the annotated image
        color: Color of the bounding box (default: "red")
        thickness: Thickness of the bounding box lines (default: 2)
        base_path: Base path to resolve relative image paths (default: current working directory)
    
    Returns:
        PIL Image object with bounding box drawn
    
    Raises:
        FileNotFoundError: If the image file does not exist
        ValueError: If required keys are missing from label
    """
    # Validate label dictionary
    if "image_path" not in label or "bbox" not in label:
        raise ValueError("Label must contain 'image_path' and 'bbox' keys")
    
    # Construct full image path
    image_path = Path(label["image_path"])
    image_filename = image_path.name
    if not image_path.is_absolute() and base_path:
        image_path = Path(base_path) / image_filename
    
    # Load image
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    
    # Extract bbox coordinates
    bbox = label["bbox"]
    bbox_format = label.get("bbox_format", "xywh")
    
    # Convert bbox to (x1, y1, x2, y2) format if needed
    if bbox_format == "xywh":
        x, y, w, h = bbox
        bbox_coords = (x, y, x + w, y + h)
    elif bbox_format == "xyxy":
        bbox_coords = tuple(bbox)
    else:
        raise ValueError(f"Unsupported bbox_format: {bbox_format}")
    
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
    base_path: Union[str, Path] = None
) -> List[Image.Image]:
    """
    Draw bounding boxes on images from a labels.json file.
    
    Args:
        json_path: Path to the labels.json file
        output_dir: Optional directory to save annotated images
        color: Color of the bounding boxes (default: "red")
        thickness: Thickness of the bounding box lines (default: 2)
        base_path: Base path to resolve relative image paths (default: parent of json_path)
    
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
            base_path=base_path
        )
        images.append(image)
    
    return images


if __name__ == "__main__":
    # Example usage
    json_file = Path("/Users/valerie/code_practice/urap/labels.json")
    # base_path = Path(__file__).parent.parent
    base_path = Path("/Users/valerie/code_practice/urap/resampled_test_image")
    
    if json_file.exists():
        images = draw_bboxes_from_json(json_file, output_dir=Path("/Users/valerie/code_practice/urap/resampled_test_image"), base_path=base_path)
        print(f"Processed {len(images)} images")
    else:
        print(f"labels.json not found at {json_file}")
