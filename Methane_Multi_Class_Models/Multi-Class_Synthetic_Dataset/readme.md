# Multi-Class Synthetic Dataset Generator

## Overview

This pipeline processes the original GasVid dataset to create a semi-synthetic 2-channel dataset optimized for methane leak detection using machine learning. The dataset generation involves background subtraction, frame extraction, and metadata integration to produce training-ready numpy arrays.

### How It Works

1. **Background Generation**: Creates class-specific background images for each 3-minute section (8 classes per video)
2. **Frame Extraction**: Randomly samples frames from each time window
3. **Background Subtraction**: Isolates the methane gas plume by subtracting the background
4. **PPM Scaling**: Scales pixel values to Parts Per Million (PPM) concentration values
5. **2-Channel Output**: Combines background and gas-only frames into numpy arrays

**Note**: Some frames may contain artifacts from rapidly moving clouds or atmospheric changes. Future improvements will address noise reduction.

---

## Running the Pipeline

The pipeline is run via a single unified command-line interface:

```bash
python run_pipeline.py [OPTIONS]
```

### Basic Usage

```bash
# Run complete pipeline with default settings (double-channel output)
python run_pipeline.py

# Run with single-channel output (frames only, no background channel)
python run_pipeline.py --channels single

# Include artificial backgrounds in dataset
python run_pipeline.py --include-artificial

# Test mode: process only specific videos
python run_pipeline.py --test-videos 1237 1238

# Custom frames per class (default: 200)
python run_pipeline.py --frames-per-class 100
```

### Command-Line Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--channels` | `single`, `double` | `double` | Output format: single (frames only) or double (background + plume) |
| `--include-artificial` | flag | off | Include artificial background samples |
| `--step` | `1`, `2`, `3`, `all` | `all` | Run specific step or complete pipeline |
| `--test-videos` | video codes | none | Process only specified videos (e.g., `1237 1238`) |
| `--frames-per-class` | integer | 200 | Number of frames to extract per class |
| `--skip-step-1` | flag | off | Skip base dataset creation |
| `--skip-step-2` | flag | off | Skip numpy dataset creation |

### Pipeline Steps

The pipeline consists of three steps that run sequentially:

1. **Step 1 - Base Dataset Creation**: Creates directory structure, extracts class frames, generates backgrounds, and populates metadata
2. **Step 2 - Numpy Dataset Creation**: Generates numpy arrays with the configured frame counts
3. **Step 3 - Final Dataset Assembly**: Consolidates all `.npy` and `.json` files into an exportable dataset

**Estimated Runtime**: 15-30 minutes depending on system performance and dataset size.

---

## Project Structure

```
Multi-Class_Synthetic_Dataset/
|-- config.py                 # Central configuration (paths, parameters)
|-- run_pipeline.py           # Main entry point (CLI)
|-- requirements.txt          # Python dependencies
|-- environment.yml           # Conda environment
|
|-- pipeline/                 # Pipeline step modules
|   |-- step_1_create_base_files.py
|   |-- step_2_create_numpy_dataset.py
|   |-- step_3_create_exportable_dataset.py
|
|-- core/                     # Core processing modules
|   |-- artificial_background_handler.py
|   |-- background_generator.py
|   |-- dataset_finalizer.py
|   |-- frame_extract.py
|   |-- metadata_handler.py
|   |-- numpy_creator.py
|
|-- utils/                    # Utility functions
|   |-- directory.py
|   |-- excel.py
|   |-- image_scaling.py
|   |-- json_utils.py
|   |-- verification.py
|
|-- source_data/              # Input data (not tracked in git)
|   |-- GasVid_Dataset/       # Video files and logging spreadsheet
|   |-- Metadata/             # Plume models and consolidated metadata
|   |-- BackGrounds/          # Background images (including artificial)
|
|-- export_datasets/          # Generated output (not tracked in git)
    |-- Processed_Dataset_single_channel/
    |-- Processed_Dataset_double_channel/
    |-- Final_Dataset_single_channel/
    |-- Final_Dataset_double_channel/
```

---

## Configuration

All paths and parameters are centralized in `config.py`:

```python
# Input paths
EXCEL_PATH = "source_data/GasVid_Dataset/GasVid Logging File.xlsx"
MOV_PATH = "source_data/GasVid_Dataset/Videos/"
PLUME_MODELING_PATH = "source_data/Metadata/Gasvid Plume Models.xlsx"
CONSOLIDATED_METADATA_PATH = "source_data/Metadata/consolidated_metadata.json"
CLASSES_JSON_PATH = "source_data/Metadata/classes.json"

# Background paths
BACKGROUNDS_PATH = "source_data/BackGrounds"
ARTIFICIAL_BACKGROUNDS_PATH = "source_data/BackGrounds/Artificial_Backgrounds"

# Output paths (automatically suffixed with channel mode)
PROCESSED_DATASET_BASE = "export_datasets/Processed_Dataset"
FINAL_DATASET_BASE = "export_datasets/Final_Dataset"

# Processing parameters
NUM_CLASSES = 8
CLASS_DURATION_MINUTES = 3
FRAMES_PER_CLASS_DEFAULT = 200  # ~44,000 data points total
```

---

## Data Sources

### GasVid Dataset (Required)
The GasVid video files are **not included** in this repository due to size constraints.

**Download Options:**

1. **BEST Lab Google Drive**:
   - Navigate to: `BEST Lab / Squishy Robotics URAP Root -> URAP Fall 2025 - ML/Software Team -> Datasets`

2. **Direct Download**:
   - [GasVid Dataset on Google Drive](https://drive.google.com/drive/folders/1JKEMtCGPSq2IqGk4uXZP9A6gr8fyJsGC)
   - (Link verified as of November 10th, 2025)

Place downloaded videos in: `source_data/GasVid_Dataset/Videos/`
(All paths are configured in `config.py`)

### Metadata Files
- `source_data/GasVid_Dataset/GasVid Logging File.xlsx`: Distance measurements and class information per video
- `source_data/Metadata/Gasvid Plume Models.xlsx`: Squishy Robotics plume modeling data (PPM values)
- `source_data/Metadata/consolidated_metadata.json`: Pre-compiled metadata (optional, speeds up processing)

---

## Artificial Background Generation

To increase dataset diversity and robustness, you can generate augmented backgrounds using AI image generation tools. These artificial backgrounds are processed separately and kept isolated from normal backgrounds throughout the pipeline.
This work was done by: Zakaria Al-Alie zakaria.al-alie@berkeley.edu

### Tool
Use **[Google Imagen Whisk](https://labs.google/fx/tools/whisk)** to generate or modify backgrounds.

### Workflow

#### Strategy Overview
1. Generate a base Class 0 background by replacing the smokestack with industrial/oil extraction equipment
2. Create minor augmentations (birds, planes) for additional diversity
3. Copy augmented backgrounds to all classes (0-7)
4. Resize and letterbox to match GasVid dimensions (320×240)

---

#### Step 1: Smokestack Replacement

**Goal**: Replace the smokestack entirely with industrial equipment/oil extraction equipment

**Prompt**:
```
Replace the central smokestack with a station venting unit, keeping the tip 
aligned with the original smokestack location. Blend the equipment naturally 
with the rest of the scene.
```

---

#### Step 2: Minor Augmentation (Birds/Planes)

**Goal**: Add small realistic variations for more diversity without modifying the main background

**Prompt**:
```
Using this existing background, do not modify or alter any structures, equipment, 
smokestack, lighting, sky, ground, or thermal palette. Only add small, realistic 
variations for augmentation:

- Birds: Add 0–3 birds flying at varying distances, appearing as bright thermal 
  signatures, varying size for depth.
- Planes: Add 0–1 distant plane or jet trail, optionally from a slightly different 
  angle or direction.

Keep all other aspects of the background exactly as in the original.
```

---

#### Step 3: Classes

**Goal**: Generate the dataset for all leak classes

**Method**: Copy the augmented background from Step 2 for each class (0-7)

---

#### Step 4: Resize/Rescale

**Goal**: Match GasVid frame size (320×240) and aspect ratio

**Method**: Use OpenCV to resize and letterbox images (aspect ratios don't match, so black bars will be added)

**Python Example**:
```python
import cv2
import numpy as np

def resize_with_letterbox(image_path, target_size=(320, 240)):
    """Resize image to target size with letterboxing"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(img, (new_w, new_h))
    
    # Create black canvas
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    
    # Center the resized image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas
```

### File Organization
Place generated artificial backgrounds in:
```
source_data/BackGrounds/Artificial_Backgrounds/XXXX/Class_Y/
```

Then run the pipeline with the `--include-artificial` flag:
```bash
python run_pipeline.py --include-artificial
```

The pipeline will automatically:
- Copy them to `export_datasets/Processed_Dataset_*/XXXX_ARTIF/Class_Y/`
- Generate numpy files with `_artif.npy` suffix
- Keep artificial data completely separate from normal data in the final dataset

---

## Output Structure

```
export_datasets/
├── Processed_Dataset_double_channel/   # or _single_channel
│   ├── XXXX/                           # Video code (e.g., 1237, 1238)
│   │   ├── Class_0/
│   │   │   ├── XXXX_class_0.json       # Class metadata (distance, leak rate, PPM)
│   │   │   ├── XXXX_class_0_background_cv2.png
│   │   │   ├── XXXX_class_0_background_moving_avg.png
│   │   │   └── processed_data/
│   │   │       ├── XXXX_frame_XX_class_0.npy
│   │   │       └── ...
│   │   ├── Class_1/
│   │   └── ... (Class_2 through Class_7)
│   └── XXXX_ARTIF/                     # Artificial backgrounds (if enabled)
│
└── Final_Dataset_double_channel/       # Consolidated exportable dataset
    ├── *.npy                           # All numpy arrays
    └── *.json                          # All metadata files
```

### Numpy Array Format

**Double-channel mode** (`--channels double`):
Each `.npy` file contains a 2-channel array with shape `(2, Height, Width)`:
- **Channel 0**: Background image (grayscale)
- **Channel 1**: PPM-scaled gas plume (background-subtracted)

**Single-channel mode** (`--channels single`):
Each `.npy` file contains the raw frame with shape `(Height, Width)`

---

## Configuration Options

### Frame Count Per Class
Use the `--frames-per-class` argument:
```bash
python run_pipeline.py --frames-per-class 200
```

**Total Dataset Size**: `frames_per_class x 8 classes x 28 videos`
- Default (200): 200 x 8 x 28 = **44,800 samples**
- Example (50): 50 x 8 x 28 = **11,200 samples**

Good validation accuracy was achieved with ~40,000+ numpy arrays with corresponding metadata.

### Test Mode
Process specific videos only for faster iteration:
```bash
python run_pipeline.py --test-videos 1237 1238 1239
```

### Step Control
Run individual pipeline steps or skip completed ones:
```bash
# Run only step 1 (base dataset creation)
python run_pipeline.py --step 1

# Run only step 3 (final assembly) - useful after manual edits
python run_pipeline.py --step 3

# Skip step 1 if already completed
python run_pipeline.py --skip-step-1

# Skip steps 1 and 2, only run final assembly
python run_pipeline.py --skip-step-1 --skip-step-2
```

---

## References

### Original GasVid Paper
**Title**: "Machine vision for natural gas methane emissions detection using an infrared camera"

**Link**: [ScienceDirect Article](https://www.sciencedirect.com/science/article/pii/S030626191931685X)

**Citation**:
```
Jingfan Wang, Lyne P. Tchapmi, Arvind P. Ravikumar, Mike McGuire, Clay S. Bell, Daniel Zimmerle, Silvio Savarese, Adam R. Brandt,
Machine vision for natural gas methane emissions detection using an infrared camera,
Applied Energy,
Volume 257,
2020,
113998,
ISSN 0306-2619,
https://doi.org/10.1016/j.apenergy.2019.113998.
(https://www.sciencedirect.com/science/article/pii/S030626191931685X)
Abstract: In a climate-constrained world, it is crucial to reduce natural gas methane emissions, which can potentially offset the climate benefits of replacing coal with gas. Optical gas imaging (OGI) is a widely-used method to detect methane leaks, but is labor-intensive and cannot provide leak detection results without operators’ judgment. In this paper, we develop a computer vision approach for OGI-based leak detection using convolutional neural networks (CNN) trained on methane leak images to enable automatic detection. First, we collect ∼1 M frames of labeled videos of methane leaks from different leaking equipment, covering a wide range of leak sizes (5.3–2051.6 g CH4/h) and imaging distances (4.6–15.6 m). Second, we examine different background subtraction methods to extract the methane plume in the foreground. Third, we then test three CNN model variants, collectively called GasNet, to detect plumes in videos. We assess the ability of GasNet to perform leak detection by comparing it to a baseline method that uses an optical-flow based change detection algorithm. We explore the sensitivity of results to the CNN structure, with a moderate-complexity variant performing best across distances. The generated detection probability curves show that the detection accuracy (fraction of leak and non-leak images correctly identified by the algorithm) can reach as high as 99%, the overall detection accuracy can exceed 95% across all leak sizes and imaging distances. Binary detection accuracy exceeds 97% for large leaks (∼710 g CH4/h) imaged closely (∼5–7 m). The GasNet-based computer vision approach could be deployed in OGI surveys for automatic vigilance of methane leak detection with high accuracy in the real world.
Keywords: Natural gas; Methane emission; Deep learning; Convolutional neural network; Computer vision; Optical gas imaging

```

---

## Troubleshooting

### Common Issues

**"Video file not found"**
- Verify videos are in `source_data/GasVid_Dataset/Videos/`
- Check file extensions (`.mp4` or `.mov`)
- Confirm paths in `config.py` match your directory structure

**"Missing metadata"**
- Ensure `source_data/Metadata/consolidated_metadata.json` exists, OR
- Verify Excel files are present in `source_data/`

**"Background generation failed"**
- Check video duration (must be at least 24 minutes)
- Verify video codec compatibility with OpenCV

**Slow processing**
- Use consolidated metadata for faster processing
- Process subset of videos with `--test-videos`
- Check available disk space

**Import errors**
- Ensure you're in the correct directory: `Multi-Class_Synthetic_Dataset/`
- Install dependencies: `pip install -r requirements.txt`

---

## Dataset Statistics

- **Videos**: 28 total 
- **Classes per Video**: 8 (representing different leak rates)
- **Time per Class**: 3 minutes
- **Distance Ranges**: 4.6m, 6.9m, 9.8m, 12.6m, 15.6m, 18.6m
- **Leak Rate Range**: 0.3 - 124.3 SCFH (Standard Cubic Feet per Hour)

---

## License & Attribution

This dataset processing pipeline is developed by the BEST Lab at UC Berkeley for the Squishy Robotics project. When using this dataset, please cite both the original GasVid paper and acknowledge the BEST Lab's contributions.


