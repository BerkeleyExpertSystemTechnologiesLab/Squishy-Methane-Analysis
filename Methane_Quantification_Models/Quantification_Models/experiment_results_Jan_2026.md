# Squishy Robot Quantification Model Experiments

## Overview

This document describes a series of machine learning experiments conducted to determine the optimal data format for the Squishy Robot Quantification models. These are the results as of 20 January 2026. 

## Baseline Models (Two-Channel with Metadata)

The experiments in `Squish_Robot_Quant_Model_v5-1_mm_+_image_transforms.ipynb` and `Squish_Robot_Quant_Model_v5-2_artif_data_exp.ipynb` utilized two-channel NumPy arrays (background-only and plume-only channels) combined with image metadata (distance and ppm). This multimodal approach achieved high accuracy, exceeding 90%.
The difference between the two models is their datasets, 5-1 was trained and tested on a modified GasVid dataset, and 5-2 was trained on that dataset and tested on an artificially generated version, using Generative LLM's to modify the images. 

## Ablation Studies

To evaluate the contribution of each data modality, additional experiments tested single-channel inputs and non-multimodal configurations.

### Image-Only Model (Single Channel, Non-Multimodal)

**Notebook:** `Quant_Model_1_Chan_NOT_Multi_Mode.ipynb`

Using only the combined background and plume images without metadata, the highest accuracy across 30 trials was approximately 20%. This result indicates that images alone are insufficient to accurately quantify methane leaks.

### Metadata-Only Model

**Notebook:** `Quant_Model_Metadata_Only.ipynb`

Using only the metadata (distance and ppm), the highest accuracy across 30 trials was 62%. This experiment was limited to 224 metadata samples; additional data may improve accuracy.

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

These results demonstrate that multimodal approaches combining image data with metadata (ppm and distance) significantly outperform unimodal methods for methane leak quantification. One possible issue is the calculation for ppm using Squishy Robotics plume modeling, the model used takes in the leak rate to estimate ppm that should be detected, since the models used here classify based on what the estimated leak rate is, it's possible that the ppm model is simply a stand in for leak rate, and we are giving the ML models here the answer with ppm. Further tests should be done with data where ppm is detected using equipment, and other factors like distance and wind speed are fed into the ML models here. 

These are the results as of 20 January 2026, by Joseph G. Berry, contact me with any questions.

joseph.g.berry@gmail.com