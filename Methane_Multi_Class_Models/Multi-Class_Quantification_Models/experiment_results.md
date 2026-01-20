# Squishy Robot Quantification Model Experiments

## Overview

This document describes a series of machine learning experiments conducted to determine the optimal data format for the Squishy Robot Quantification models.

## Baseline Models (Two-Channel with Metadata)

The experiments in `Squish_Robot_Quant_Model_v5-1_mm_+_image_transforms.ipynb` and `Squish_Robot_Quant_Model_v5-2_artif_data_exp.ipynb` utilized two-channel NumPy arrays (background-only and plume-only channels) combined with image metadata (distance and ppm). This multimodal approach achieved high accuracy, exceeding 90%.

## Ablation Studies

To evaluate the contribution of each data component, additional experiments tested single-channel inputs and non-multimodal configurations.

### Image-Only Model (Single Channel, Non-Multimodal)

**Notebook:** `Quant_Model_1_Chan_NOT_Multi_Mode.ipynb`

Using only the combined background and plume images without metadata, the highest accuracy across 30 trials was approximately 20%. This result indicates that images alone are insufficient to accurately quantify methane leaks.

### Metadata-Only Model

**Notebook:** `Quant_Model_Metadata_Only.ipynb`

Using only the metadata (distance and ppm), the highest accuracy across 30 trials was 62%. This experiment was limited to 224 metadata samples; additional data may yield improved accuracy.

### Single-Channel Multimodal Model

**Notebook:** `Quant_Model_1_Chan_and_Multi_Mode.ipynb`

This experiment combined single-channel images with metadata. Due to a connectivity interruption while running on google colab, only 29 of the planned 30 trials were completed. The highest observed accuracy was 95%.

## Summary

| Configuration | Highest Accuracy | Trials Completed |
|---------------|------------------|------------------|
| Two-channel + metadata | >90% | 30 |
| Single-channel + metadata | 95% | 29 |
| Metadata only | 62% | 30 |
| Single-channel image only | ~20% | 30 |

These results demonstrate that multimodal approaches combining image data with metadata significantly outperform unimodal methods for methane leak quantification.