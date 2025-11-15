# Squishy_Robots_Quant_Models


The goal of these models is to be able to recognize a methane leak using 2-Channel images. First channel is a greyscale image, second channel is a heatmap of believed methane gas (Using Optical Gas Imaging)

These models are two modal, 1st Modal is 2 Channel Images as stated, 2nd Modal is vector of data collected (Wind Speed, distance from leak, max ppm detected using OGI)

End goal is accurate classification of leak into one of 8 categories. 

These models were trained on GasVid dataset, collected at METEC in Colorado. The dataset was modified from .mp4 videos to the 2 channel images using background subtraction, then outputting the background and the gas plume into the two channels previously mentioned.

Older versions of the models are stored in the old_models directory. Any ipynb files not in this directory are the latest version. 

The .csv and .xlsx files are documentation of the processes and accuracy achieved. 