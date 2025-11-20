# Multi-Class Quantification Models

## Goal

Recognize and classify methane leaks using multi-modal deep learning models.

## Input Data

### Modal 1: 2-Channel Images

- **Channel 1**: Greyscale image (background)
- **Channel 2**: Heatmap of methane gas detected via Optical Gas Imaging (OGI)

### Modal 2: Environmental Vector Data

- Distance from leak
- Maximum PPM detected using OGI
- Wind speed (TO BE ADDED LATER)

## Output

Classification of methane leaks into **8 distinct categories**.

## Training Dataset

**GasVid Dataset** - Collected at METEC facility in Colorado
This dataset has been synthetically altered from it's original .mp4 format to 2 channel numpy arrays combined with metadata to create the two modalities. 
In addition we have experimented with creating backgrounds altered using AI image generators, feeding in GasVid frames and adding additional objects in the foreground and background. 

### Data Processing Pipeline

1. Original format: `.mp4` video files
2. Applied background subtraction technique
3. Extracted two channels:
   - Background (greyscale) -> Channel 1
   - Gas plume (heatmap) -> Channel 2

## Repository Structure

- **Root directory**: Latest model versions (`.ipynb` files)
- **old_models/**: Archived previous model versions
- **Documentation**: `.csv` and `.xlsx` files contain process documentation and accuracy metrics 