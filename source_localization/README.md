# Gas Leak Detection - Source Localization Dataset Pipeline

This repository implements a complete pipeline for preparing a dataset of semi-transparent gas leak images from infrared video footage. The pipeline processes raw infrared videos through background subtraction, frame extraction, labeling, and data augmentation to create a training-ready dataset for source localization models.

## Overview

The pipeline consists of four main stages that transform raw infrared videos into a labeled, augmented image dataset:

1. **Background Subtraction**: Extract plume videos from original infrared footage
2. **Frame Extraction**: Extract individual leak frames from plume videos
3. **Label Creation**: Generate metadata and bounding box labels for each frame
4. **Data Augmentation**: Apply transformations (translation + contrast enhancement) to augment the dataset

## Dataset Structure

```
source_localization/dataset/
├── original_gasvid_dataset/          # Raw input videos and leak range metadata
│   ├── MOV_*.mp4                    # Original infrared videos
│   └── leak_range.csv               # Leak timing information
├── gasvid_comparison_videos/         # Side-by-side comparison videos
├── plume_video_dataset/              # Background-subtracted plume videos
└── plume_image_dataset/
    ├── all_images/                   # Extracted leak frames
    │   ├── MOV_*_plume_frame_*.png
    │   └── labels.json               # Metadata and bounding box labels
    └── transformed_images/           # Augmented images ready for training
        ├── MOV_*_plume_frame_*_trans*.png
        └── labels.json               # Updated labels with transformations
```

## Pipeline Stages

### Stage 1: Background Subtraction

**Script**: `dataset/background_subtraction.py`

Performs background subtraction on infrared videos using MOG2 algorithm to isolate gas plumes.

**Usage**:
```bash
# Process entire dataset
cd source_localization/dataset
python background_subtraction.py --dataset

# Process single video
python background_subtraction.py video.mp4 -o comparison.mp4 -s plume.mp4
```

**Output**:
- `gasvid_comparison_videos/`: Side-by-side comparison videos
- `plume_video_dataset/`: Background-subtracted plume videos

**Parameters**:
- `--history`: Number of frames for background model (default: 30)
- `--threshold`: Binary threshold for mask creation (default: 40)
- `--morph-kernel`: Morphological closing kernel size (default: 30)

### Stage 2: Frame Extraction

**Script**: `dataset/extract_frames.py`

Extracts individual frames from plume videos, only extracting frames within leak ranges specified in `leak_range.csv`.

**Usage**:
```bash
# Extract frames from all videos (default paths)
python extract_frames.py

# Extract from first N videos only
python extract_frames.py --max-videos 10

# Custom paths
python extract_frames.py --plume-dir /path/to/videos --output-dir /path/to/output
```

**Output**: `plume_image_dataset/all_images/` - Individual PNG frames with leak content

### Stage 3: Label Creation

**Script**: `dataset/create_labels.py`

Generates `labels.json` file containing metadata for each image including:
- Image properties (size, channels, format)
- Fixed bounding box coordinates `[170, 120, 20, 10]` (xywh format)
- Image paths and names

**Usage**:
```bash
# Create labels for all images (default paths)
python create_labels.py

# Custom paths
python create_labels.py --images-dir /path/to/images --output /path/to/labels.json
```

**Output**: `plume_image_dataset/all_images/labels.json` - Complete metadata for all frames

### Stage 4: Data Augmentation

**Script**: `dataset/transform_images.py`

Applies data augmentation to images with:
- **Random Translation**: 0-100 pixels in random direction
- **Contrast Enhancement**: 20% increase using sigmoid curve
- Images maintain original size (out-of-bounds pixels dropped, empty areas zero-padded)

**Usage**:
```bash
# Transform with default settings (1 transform per image, 20% contrast boost)
python transform_images.py

# Apply multiple transformations per image
python transform_images.py --num-transforms 5

# Custom contrast and translation
python transform_images.py --contrast-factor 1.5 --max-translation 150

# Reproducible with seed
python transform_images.py --seed 42
```

**Output**: `plume_image_dataset/transformed_images/` - Augmented images with updated labels

**Parameters**:
- `--num-transforms`: Number of random transformations per image (default: 1)
- `--contrast-factor`: Contrast enhancement factor (default: 1.2 = 20% increase)
- `--max-translation`: Maximum translation distance in pixels (default: 100)
- `--seed`: Random seed for reproducibility

## Requirements

```bash
pip install opencv-python numpy
```

## Complete Pipeline Workflow

Run the full pipeline from start to finish:

```bash
cd source_localization/dataset

# Step 1: Extract plume videos from original footage
python background_subtraction.py --dataset

# Step 2: Extract leak frames from plume videos
python extract_frames.py

# Step 3: Generate labels for all extracted frames
python create_labels.py

# Step 4: Augment dataset with transformations
python transform_images.py --num-transforms 3
```

## Implementation Details

### Background Subtraction

Uses OpenCV's MOG2 (Mixture of Gaussians) algorithm with:
- Short history (30 frames) to avoid false positives from slow-moving objects
- Adaptive enhancement using mean and standard deviation of difference image
- Morphological operations (opening + closing) to refine masks

Based on the paper: **"LangGas: Introducing Language in Selective Zero-Shot Background Subtraction for Semi-Transparent Gas Leak Detection with a New Dataset"** (arXiv:2503.02910v1)

### Contrast Enhancement

The sigmoid-based contrast enhancement uses the formula:
```
output = max_val / (1 + exp(-steepness * (input - midpoint)))
```

This creates an S-curve that:
- Enhances mid-tones while preserving detail
- Smoothly transitions between bright and dark regions
- Prevents harsh clipping of extreme values

### Translation Augmentation

Images are translated on a larger canvas then cropped to original size:
- Cropping centered on original image center position
- Makes translations visible while maintaining image dimensions
- Out-of-bounds pixels are dropped, empty areas zero-padded

## Dataset Statistics

- **Original Videos**: 33 infrared videos with gas leak events
- **Extracted Frames**: ~17,000 leak frames (frames within leak ranges only)
- **Augmented Dataset**: Configurable (default: 1 transformation per image)
- **Bounding Box**: Fixed at `[170, 120, 20, 10]` (xywh format)

## Next Steps: Model Training

The transformed images in `plume_image_dataset/transformed_images/` are ready for model training. This dataset includes:

- **Preprocessed Images**: Background-subtracted, contrast-enhanced, and augmented  
- **Complete Labels**: Bounding box coordinates and metadata in JSON format  
- **Data Augmentation**: Multiple transformed versions per original image  
- **Consistent Format**: All images maintain original size with standardized labels  

### Recommended Training Approaches

1. **Object Detection Models**: Train YOLO, Faster R-CNN, or similar models using the bounding box labels
2. **Source Localization**: Develop models to predict gas leak source coordinates from plume images
3. **Segmentation**: Convert bounding boxes to segmentation masks for pixel-level prediction

### Using the Dataset

The `labels.json` file in `transformed_images/` contains:
- `image_name`: Filename of the transformed image
- `image_path`: Relative path from project root
- `image_size`: `[width, height]`
- `bbox`: Bounding box in xywh format `[x, y, width, height]`
- `rotation`: Rotation applied (0 for translation-only)
- `translation`: Translation vector `[tx, ty]` applied

Example label entry:
```json
{
  "image_name": "MOV_1237_plume_frame_000001_trans50_-30.png",
  "image_path": "source_localization/dataset/plume_image_dataset/transformed_images/MOV_1237_plume_frame_000001_trans50_-30.png",
  "image_size": [320, 240],
  "bbox": [220, 90, 20, 10],
  "rotation": 0,
  "translation": [50, -30]
}
```

## References

**Paper**: Wenqi Marshall Guo, Yiyang Du, Shan Du. "LangGas: Introducing Language in Selective Zero-Shot Background Subtraction for Semi-Transparent Gas Leak Detection with a New Dataset." arXiv:2503.02910v1 [cs.CV], March 2025.

## Troubleshooting

### Video processing issues
- Ensure video files are in correct format (MP4)
- Check that `leak_range.csv` exists and has correct format
- Verify output directories have write permissions

### Frame extraction
- Ensure `plume_video_dataset/` contains processed videos
- Check that video numbers in filenames match CSV entries

### Memory issues
- Process videos in batches using `--max-videos` flag
- Reduce `--num-transforms` if generating too many augmented images

### Label generation
- Verify all images are valid and readable
- Check that image paths are correct relative to project root
