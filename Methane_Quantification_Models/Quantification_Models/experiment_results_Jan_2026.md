# Squishy Robot Quantification Model Experiments

## Overview

This document describes a series of machine learning experiments conducted to determine the optimal data format for the Squishy Robot Quantification models. These are the results as of 20 January 2026. 

## Baseline Models (CNN Two-Channel with Metadata)

The experiments in `Squish_Robot_Quant_Model_v5-1_mm_+_image_transforms.ipynb` and `Squish_Robot_Quant_Model_v5-2_artif_data_exp.ipynb` are CNN based ML models. They are a copy of the model described in the VideoGasNet paper, recreated. They consist of 3 basic blocks, CNN + Norm + Activation + Pool + Dropout, then combined with a smaller neural network for metadata, and then fed into a final block that combines the two and ends with classification. The models utilized two-channel NumPy arrays (background-only and plume-only channels) combined with image metadata (distance and ppm). This multimodal approach achieved high accuracy, exceeding 90%.
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

### ViT Multimodal Model

**Notebook:** `Quant_Model_ViT.ipynb`

This experiment attempts to use Vision Transformers instead of CNN's for classification and still consists of a smaller neural netword for metadata. After 20 trials I ended the test early since results were poor, the highest validation accuracy was ~65% after upwards of 36 hours of testing. ViT's are known for requiring more data (GB's or millions of images) before their performance is better than CNN's. 


## Summary

| Model | Configuration | Highest Accuracy | Trials Completed |
|-------|---------------|------------------|------------------|
| CNN   | Two-channel + metadata    | >90% | 30 |
| CNN   | Single-channel + metadata | 95%  | 29 |
| CNN   | Metadata only             | 62%  | 30 |
| CNN   | Single-channel image only | ~20% | 30 |
| ViT   | Two-channel + metadata    | ~60% | 20 |

These results demonstrate that multimodal approaches combining image data with metadata (ppm and distance) significantly outperform unimodal methods for methane leak quantification. One possible issue is the calculation for ppm using Squishy Robotics plume modeling, the model used takes in the leak rate to estimate ppm that should be detected, since the models used here classify based on what the estimated leak rate is, it's possible that the ppm model is simply a stand in for leak rate, and we are giving the ML models here the answer with ppm. 

Further tests should be done with data where ppm is detected using equipment, and other factors like distance and wind speed are fed into the ML models here. 

ViT based models take much longer to test and experiment with than CNN's, and since the performance for some trials of CNN's is approximately 95% this indicates that we can safely stick to simpler architectures for quantification models, and only move to ViT's if we gain massive amounts of data (GB's more or millions of images) and CNN's underperform on new datasets. 

These are the results as of 20 January 2026, by Joseph G. Berry, contact me with any questions.

joseph.g.berry@gmail.com