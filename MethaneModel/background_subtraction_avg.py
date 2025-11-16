# Imports
import os
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import multiprocessing
import uuid
import matplotlib.pyplot as plt
import pandas as pd
import re

# Helper Functions

def calc_median(frames):
    median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
    return median_frame

def doMovingAverageBGS(image, prev_frames):
    median_img = calc_median(prev_frames)
    image = cv2.absdiff(image, median_img)
    return image

def calc_avg(frames):
  average_frame = np.mean(frames).astype(dtype=np.uint8)
  return average_frame

dir_path = os.path.dirname(os.path.realpath("__file__"))
data_dir = os.path.join(dir_path, 'data')
train_data_dir = os.path.join(data_dir, 'train')
test_data_dir = os.path.join(data_dir, 'test')

# # get all raw video data directories
# vid_path = '/home/bestlab/Desktop/Squishy-Methane-URAP-New/AngelineLee/MethaneModel/data/train/MOV_2559.mp4'
background_path = '/home/bestlab/Desktop/Squishy-Methane-URAP-New/AngelineLee/MethaneModel/background_sub_testing_movingavg'
os.makedirs(background_path, exist_ok=True)

def get_frames(vid_path, out_path, med_count):

    before_path = os.path.join(out_path, 'before')
    after_path = os.path.join(out_path, 'after')
    median_path = os.path.join(out_path, 'median')

    print("Before path" + before_path, flush = True)
    print("After path" + after_path, flush = True)
    print("Median path" + median_path, flush = True)

    os.makedirs(before_path, exist_ok=True)
    os.makedirs(after_path, exist_ok=True)
    os.makedirs(median_path, exist_ok=True)

    cap = cv2.VideoCapture(vid_path)

    cap.set(cv2.CAP_PROP_POS_MSEC, 0)

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) 
    print("Num frames: %d" % num_frames, flush = True)
    print("Frames per second: %d" % fps, flush = True)

    background = []
    times = []

    for i in range(med_count):
        success, image = cap.read()
        background.append(image) 

    cap.set(cv2.CAP_PROP_POS_MSEC, 0)

    for i in range(num_frames):
        success, image = cap.read()
        time = cap.get(cv2.CAP_PROP_POS_MSEC)
        times.append(time)
        cv2.imwrite(os.path.join(before_path, 'test%d.jpg' % i), image)
        median_background = np.median(background, axis = 0)
        cv2.imwrite(os.path.join(median_path, 'test%d.jpg' % i), median_background)
        removed_image = image-median_background
        cv2.imwrite(os.path.join(after_path, 'test%d.jpg' % i), removed_image)
        if (i>=med_count):
            background.pop(0)
            background.append(image)

    cap.release()
    cv2.destroyAllWindows()

    df_time = pd.DataFrame(times)
    df_time.to_csv(out_path + '/df_time.csv')

for file in os.listdir(train_data_dir):
    vid_path = os.path.join(train_data_dir, file)
    vid_id = int(re.findall("_(\d{4}).mp4",os.path.basename(vid_path))[0])
    print("Extracting vid_id: %d" % vid_id, flush = True)
    output_path = os.path.join(background_path, str(vid_id))
    os.makedirs(output_path, exist_ok=True)
    get_frames(vid_path, output_path, 210)

print("Completed", flush = True)