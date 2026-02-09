# Squishy Robot Quantification Model Experiments

## Overview

This document describes a series of machine learning experiments conducted to determine the optimal data format for the Squishy Robot Quantification models. These are the results as of 20 January 2026. 

## Baseline Models (CNN Two-Channel with Metadata)

The experiments in `Squish_Robot_Quant_Model_v5-1_mm_+_image_transforms.ipynb` and `Squish_Robot_Quant_Model_v5-2_artif_data_exp.ipynb` are CNN based ML models. They are a copy of the model described in the VideoGasNet paper, recreated. They consist of 3 basic blocks, CNN + Norm + Activation + Pool + Dropout, then combined with a smaller neural network for metadata, and then fed into a final block that combines the two and ends with classification. The models utilized two-channel NumPy arrays (background-only and plume-only channels) combined with image metadata (distance and ppm). This multimodal approach achieved high accuracy, exceeding 90%.
The difference between the two models is their datasets, 5-1 was trained and tested on a modified GasVid dataset, and 5-2 was trained on that dataset and tested on an artificially generated version, using Generative LLM's to modify the images. 

The experiment `Squish_Robot_Quant_Model_v6-1_Five_Classes+image_transform.ipynb` in Old_Models/ was experiments with reducing from 8 classes to 5 classes by merging classes 1 & 2, 3 & 4, and 5 & 6, leaving classes 0 and 7 the same. This improved accuracy, but since other changes to the 8 class model got high validation accuracy, I stopped testing 5 class models. Revisiting this may help if there are issues with new datasets. 

The synthetic dataset creator can be used to make single channel datasets, since the hardware team informed me that double channel separating backgrounds from plumes is unrealistic, in the future models should be tested on single channel images. 

## Metadata and Image Contribution Experiments

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

## Vision Transformer Experiments

### ViT Multimodal Model

**Notebook:** `Quant_Model_ViT.ipynb`

This experiment attempts to use Vision Transformers instead of CNN's for classification and still consists of a smaller neural netword for metadata. After 20 trials I ended the test early since results were poor, the highest validation accuracy was ~65% after upwards of 36 hours of testing. ViT's are known for requiring more data (GB's or millions of images) before their performance is better than CNN's. 

### CNN with ViT Multimodal Model

**Notebook:** `Quant_Model_CNN_w_ViT.ipynb`

This was an attempt at combining CNN's with ViT in sequential blocks [CNN -> ViT -> GELU] -> [Another Block], it includes the pytorch autocast which reduces the models weights from float 32's to float 16's, then converts back for the backward pass. This model experienced the classic high training accuracy low validation overtraining symptom that is common with ViTs. Further attempts might be attempted with other CNN + ViT architectures but this model was not efficient for training and underperformed. 

### Imported ViT models (Huggingface, Pytorch)

**Notebooks:** `Quant_Model_Swin_ViT.ipynb` `Quant_Model_MaxViT.ipynb` `Quant_Model_DeiT_ViT.ipynb`

I imported predesigned but not pretrained models from timm (Pytorch models) and Huggingface. Performance was middleing, and seems like they suffered the same issues with ViT's, low accuracy when not presented with a massive amount of data available. 


## Summary

| Model | Configuration | Highest Val Accuracy | Trials Completed |
|-------|---------------|------------------|------------------|
| CNN + NN    | Two-channel + metadata    | >90%  | >40 |
| CNN + NN    | Single-channel + metadata | 95%   | 29 |
| CNN + NN    | Metadata only             | 62%   | 30 |
| CNN + NN    | Single-channel image only | 21%   | 30 |
| ViT + NN        | Two-channel + metadata    | 65%   | 20 |
| CNN w ViT + NN  | Two-channel + metadata    | 52%   | 7  |
| DeiT + NN      | Single-Channel + metadata | 81%   | 5  | 
| MaxViT + NN     | Single-Channel + metadata | 52%   | 10 | 
| SwinViT + NN    | Single-Channel + metadata | 79%   | 5  | 

These results demonstrate that multimodal approaches combining image data with metadata (ppm and distance) significantly outperform single mode methods for methane leak quantification such as only metadata or only images. 

One possible issue with the models is the calculation for ppm using Squishy Robotics plume modeling, the plume estimation model (https://docs.google.com/spreadsheets/d/19fZelUZhGEwCyux0knJEHTg4rHD8eXcsizefphtPDFY/edit?gid=0#gid=0) takes in the leak rate to estimate ppm that should be detected given that leak rate. Since the models used here classify based on what the estimated leak rate is, it's possible that the ppm model is simply a stand in for leak rate, and we are giving the ML models here the answer to leak rate class by giving them our estimated ppm, further testing on different ppm algorithmic estimations should be performed. Including further tests done with data where ppm is verified using equipment, and other factors like distance and wind speed are fed into the ML models here. 

ViT based models take much longer to test and experiment with than CNN's and since the ViT models here underperformed the CNN's by a wide margin of 30% accuracy, this indicates that we can safely stick to simpler architectures for quantification models, and only move to ViT's if we gain massive amounts of data (GB's more or millions of images) and simultaneuously our CNN models underperform on future datasets. 

Unfortunately my trials for the Two-Channel + Metadata were interrupted and I had to restart. I witnessed accuracies above 90% but lost that documentation. 

These are the results as of 30 January 2026, by Joseph G. Berry, contact me with any questions.

joseph.g.berry@gmail.com