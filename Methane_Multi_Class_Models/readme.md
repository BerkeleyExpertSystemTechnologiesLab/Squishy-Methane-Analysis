# Methane Multi-Class Models

## Overview

This repository contains two interconnected directories that work together for multi-class methane quantification:

1. **Multi-Class_Synthetic_Dataset** - Dataset generation
2. **Multi-Class_Quantification_Models** - Model training and experimentation

## Dataset Generation

The `Multi-Class_Synthetic_Dataset` directory contains all the code necessary to generate 2-channel numpy arrays.

**What are 2-channel arrays?**
- "Channels" refer to the layers in an image
- Standard RGB images have 3 channels
- Greyscale images have 1 channel
- Our datasets use 2 channels

## Model Training

The `Multi-Class_Quantification_Models` directory contains prototype models for experimentation. Running the dataset generation code will produce 2-channel numpy arrays that can be fed directly into these models.

### Model Versioning

- Higher model numbers indicate more recent experiments
- Older models are retained for reference and comparison of past experiments

## Contact

For further questions, please reach out:

**Joseph G. Berry**
- joseph.g.berry@berkeley.edu
- joseph.g.berry@gmail.com