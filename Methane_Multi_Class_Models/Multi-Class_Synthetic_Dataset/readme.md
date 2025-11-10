This code takes the original GasVid dataset and makes a semi-synthetic 2 channel dataset. The Original videos are used to make a background image of every 3 minute section for each of the 8 classes of methane leak per video. Then random frames are taken from each of the 8 sections per video and the background image is subtracted from that frame, creating a second frame in which should leave only the gas leak visible. 

(This method does not work completely, some frames have rapid moving clouds get stuck, I will attempt to clean this up but for now some of the images will be noisy)

Once we have two frames, the background image and the only gas frame are combined into a numpy matrix and exported to "Processed_Dataset/" directory.

This code is meant to be run one at a time:


1_base_dataset_creation.py
2_numpy_dataset_creation.py
3_final_dataset.py

Running 1_base_dataset_creation.py will search for the original gasvid dataset under /Original_Dataset and will create background images for each class for each .mp4

The GasVid dataset is not included in these files/repository since it is consists of video files and is too large to upload to github. You can find the GasVid dataset on the BEST lab google drive under:
 BEST Lab / Squishy Robotics URAP Root -> URAP Fall 2025 - ML/Software Team -> Datasets

Or you can download the dataset yourself at:
https://drive.google.com/drive/folders/1JKEMtCGPSq2IqGk4uXZP9A6gr8fyJsGC 
(This link worked as of November 10th 2025)


It will also create .json files containing data from the gasvid "GasVid Logging File.xlsx" detailing distance measurements and classes for each video, and it will also copy data from 
"/Plume_Modeling/Gasvid Plume Models.csv" which is the Squishy Robots Plume modeling of the GasVid dataset. 

The number of frames exportable can be set in the file 2_numpy_dataset_creation.py

3_final_dataset.py will copy all npy (numpy) and .json files to a folder called final dataset

These three python scripts can take up to 15 minutes to run, but should not take significantly longer.


Here is a link to the original GasVid paper:
https://www.sciencedirect.com/science/article/pii/S030626191931685X

Paper Title:
"Machine vision for natural gas methane emissions detection using an
infrared camera"


