# Squishy_Robots_Quant_Models


The goal of this model is to be able to recognize a methane leak using 2-Channel images. First channel is a greyscale image, second channel is a heatmap of believed methane gas (Using Optical Gas Imaging)

This model will be two modal, 1st Modal is 2 Channel Images as stated, 2nd Modal is vector of data collected (Wind Speed, distance from leak, max ppm detected using OGI)

End goal is accurate classification of leak into one of 8 categories. 

This model was trained on GasVid dataset, collected at METEC in Colorado. 