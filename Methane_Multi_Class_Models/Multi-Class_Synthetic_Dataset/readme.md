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

## Pipeline Scripts

Run these scripts sequentially in the following order:

### 1. `1_base_dataset_creation.py`
Creates directory structure, extracts class frames, generates backgrounds, and populates metadata.

### 2. `2_numpy_dataset_creation.py`
Generates final numpy arrays with configurable frame counts per class.

### 3. `3_final_dataset.py`
Consolidates all `.npy` and `.json` files into a final dataset directory.

**Estimated Runtime**: 15-30 minutes depending on system performance and dataset size.

---

## Recent Updates

### Consolidated Metadata System
- **Smart Metadata Loading**: Automatically uses `Metadata/consolidated_metadata.json` if available
- **Fallback Support**: Falls back to individual Excel/CSV/JSON sources if consolidated metadata is missing
- **Faster Processing**: Single file read instead of multiple Excel file operations
- **Path Constants**: All file paths now defined at module-level for easy configuration

### Configuration
All file paths are now defined as constants at the top of `1_base_dataset_creation.py`:
```python
EXCEL_PATH = "Original_Dataset/GasVid Logging File.xlsx"
MOV_PATH = "Original_Dataset/Videos-20251002T175522Z-1-001/Videos"
PROCESSED_DATASET_PATH = "Processed_Dataset"
CLASSES_JSON_PATH = "classes.json"
PLUME_MODELING_PATH = "Plume_Modeling/Gasvid Plume Models.xlsx"
CONSOLIDATED_METADATA_PATH = "Metadata/consolidated_metadata.json"
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

Place downloaded videos in: `Original_Dataset/`
(If confused about the correct path check the paths at the top of 1_base_dataset_creation.py)

### Metadata Files
- `GasVid Logging File.xlsx`: Distance measurements and class information per video
- `Plume_Modeling/Gasvid Plume Models.csv`: Squishy Robotics plume modeling data (PPM values)
- `Metadata/consolidated_metadata.json`: Pre-compiled metadata combining the two files mentioned above (optional, speeds up processing)

---

## Output Structure

```
Processed_Dataset/
├── XXXX/                          # Video code (e.g., 1237, 1238)
│   ├── Class_0/
│   │   ├── XXXX_class_0.json      # Class metadata (distance, leak rate, PPM)
│   │   ├── XXXX_class_0_background_cv2.png
│   │   ├── XXXX_class_0_background_moving_avg.png
│   │   └── processed_data/
│   │       ├── XXXX_frame_XX_class_0.npy  # 2-channel numpy array
│   │       └── ...                         # Multiple frame samples
│   ├── Class_1/
│   └── ... (Class_2 through Class_7)
```

### Numpy Array Format
Each `.npy` file contains a 2-channel array with shape `(2, Height, Width)`:
- **Channel 0**: Background image (grayscale)
- **Channel 1**: PPM-scaled gas plume (background-subtracted)

---

## Configuration Options

### Frame Count Per Class
Modify in `2_numpy_dataset_creation.py`:
```python
frames_per_class = 50  # Default: 50 frames per class per video
```

**Total Dataset Size**: `frames_per_class × 8 classes × 28 videos`
- Example: 50 × 8 × 28 = **11,200 samples**
Good validation accuracy was acheived once the dataset was ~40,000 numpy arrays with corresponding metadata.
Models tested were two modal: a combination of the two channel images mentioned and the 

### Test Mode
Process specific videos only:
```python
test_videos = ["1237", "1238", "1239"]
success = create_dataset_from_scratch(test_videos=test_videos)
```

### Step Control
Enable/disable specific pipeline steps:
```python
create_dataset_from_scratch(
    step_1=True,  # Load video files
    step_2=True,  # Extract frames & backgrounds
    step_3=True,  # Add metadata
    step_4=True,  # Add leak rates
    step_5=True,  # Add PPM data
    step_6=True   # Create example scaled images
)
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
- Verify videos are in `Original_Dataset/Videos-20251002T175522Z-1-001/Videos/`
- Check file extensions (`.mp4` or `.mov`)

**"Missing metadata"**
- Ensure `Metadata/consolidated_metadata.json` exists, OR
- Verify Excel files are present: `GasVid Logging File.xlsx` and `Gasvid Plume Models.xlsx`

**"Background generation failed"**
- Check video duration (must be at least 24 minutes)
- Verify video codec compatibility with OpenCV

**Slow processing**
- Use consolidated metadata for faster processing
- Process subset of videos in test mode
- Check available disk space

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


