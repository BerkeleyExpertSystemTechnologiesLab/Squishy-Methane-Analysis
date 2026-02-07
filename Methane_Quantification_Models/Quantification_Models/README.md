# Multi-Class Quantification Models

## Goal

Recognize and classify methane leaks using multi-modal deep learning models.

## Input Data

### Modal 1: 1-Channel or 2-Channel Images

If 1-channel, then standard greyscale images
If 2-channel:

- **Channel 1**: Greyscale image (background)
- **Channel 2**: Heatmap of methane gas detected via Optical Gas Imaging (OGI)

### Modal 2: Environmental Vector Data

- Distance from leak
- Maximum PPM detected using OGI
(TO BE ADDED LATER)
- Wind speed 
- Cross Section of Plume
- etc...

## Output

Classification of methane leaks into **8 distinct categories**.

## Training Dataset

**GasVid Dataset** - Collected at METEC facility in Colorado
The GasVid Dataset consists of 28 .mp4 videos of the METEC facility, with accurate measured disance to the leak and leak rate. We use the distance to the leak and an B.E.S.T. Lab / Squishy Robotics algorithm for modeling the ppm that should be detected (ppm is not recorded by METEC)

The METEC dataset has been synthetically altered from it's original .mp4 format to 2 channel numpy arrays combined with metadata to create a two modal dataset. 

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

### Old Models

The models inside this directory were kept for documentation, they are result and don't need to be kept. The only result from the old models that should be recorded is that reducing the classes from 8 to 5 can boost validation accuracy. 

### How to run these models

Using the instructions in readme.md in the Synthetic_Dataset_Creator/ Directory generate a final dataset (Final_Dataset_... directory)

Turn that directory into a zip file and then upload to a Google Colab Instance (either upload directly or access your google drive from colab)

Open the .ipynb files here using google colab. 

Run the model, you will have to change the name of the zip file when unzipping, but the rest of the code will automatically run. The default setting is 30 trials, on google colab running an A100 or TPU this takes approximatley 30 hours or more. Trials can be shortened, this is a setting in the final code section of the .ipynb files. 

I've run these models to completion, going to the last section of the ipynb files to find the best combination of parameters can save you time if you don't want to run all trials. The trials search for best combination of parameters. 