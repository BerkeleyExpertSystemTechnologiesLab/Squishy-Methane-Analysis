# Imports
import os
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import multiprocessing
import uuid

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

# Try Fixed background substraction
def doFixedBGS(image, init_frames):
  init_img = calc_avg(init_frames)
  image = cv2.absdiff(image, init_img)
  return image

# Try Mixture of Gaussian-Based (MOG) background substraction
def doMOGBGS(image, all_frames):
  gmm = GaussianMixture(n_components = 2)
  gmm_background = np.zeros(shape=(all_frames.shape[1:]))
  for i in range(all_frames.shape[1]):
      for j in range(all_frames.shape[2]):
          for k in range(all_frames.shape[3]):
              X = all_frames[:, i, j, k]
              X = X.reshape(X.shape[0], 1)
              gmm.fit(X)
              means = gmm.means_
              covars = gmm.covariances_
              weights = gmm.weights_
              idx = np.argmax(weights)
              gmm_background[i][j][k] = int(means[idx])
  image = cv2.absdiff(image, gmm_background)
  return image

#Updated design for extract_images and its much easier to read

def extract_images(args):
    '''
    Input:
        extract_images(path_in, path_out, leak_range, nonleak_range, curr_count_leak, curr_count_nonleak)

        path_in (str): The path of the video.
        path_out (str): The path of the folder where data is being stored for testing or training.
        leak_range (tuple): Range of leak frames from the video.
        nonleak_range (tuple): Range of nonleak frames from the video.
        curr_count_leak (int): Current count of leak frames.
        curr_count_nonleak (int): Current count of nonleak frames.

    Output:
        (curr_count_nonleak, curr_count_leak)

    Description:
        Given the path of the video, `path_in`, this function will perform moving average background subtraction
        on the entire video. Based on `leak_range` and `nonleak_range`, it will save those frames into
        `path_out/Leak` and `path_out/Nonleak` folders respectively. The function will output the updated
        `curr_count_leak` and `curr_count_nonleak`.
    '''
     # this is for parallelization
    path_in, path_out, leak_range, nonleak_range, curr_count_leak, curr_count_nonleak = args
    leak_path = os.path.join(path_out, "Leak")
    nonleak_path = os.path.join(path_out, "Nonleaks")

    os.makedirs(leak_path, exist_ok=True)
    os.makedirs(nonleak_path, exist_ok=True)

    prev_imgs = []
    prev_limit = 210 

    cap = cv2.VideoCapture(path_in)
  
    success = True
    count = 0

    if cap.isOpened():
        while success: # Iterate through entire video
            success, image = cap.read()
            if success:
                curr_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                prev_imgs.append(gray_image)
                if len(prev_imgs) > prev_limit:
                    prev_imgs.pop(0)

                processed_img = doMovingAverageBGS(gray_image, prev_imgs) 

                if nonleak_range[0] * 1000 <= curr_frame_time <= nonleak_range[1] * 1000: # If frame in nonleak range, save it
                    cv2.imwrite(os.path.join(nonleak_path, "nonleak.frame%d.jpg" % uuid.uuid4()), processed_img)
                    curr_count_nonleak += 1
                elif leak_range[0] * 1000 <= curr_frame_time <= leak_range[1] * 1000: # If frame in leak range, save it
                    # reason for using uuid now is because calling this function in parallel - so can't have sequential data from
                    # previous calls
                    cv2.imwrite(os.path.join(leak_path, "leak.frame%d.jpg" % uuid.uuid4()), processed_img) 
                    curr_count_leak += 1
              
                count += 1
            else:
                if count == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                    print()
                    print("Extracted All Frames!!!")
                else:
                    print()
                    print("Last frame viewed", count)
                    print("Last mms that we grabbed", curr_frame_time)
                break
        cap.release()
    cv2.destroyAllWindows()
    return curr_count_nonleak, curr_count_leak


def read_frames_from_dir(dir_path, output_path, ranges, max_vids=None):
    cur_count = 1
    currNonLeakCount = 0
    currLeakCount = 0

    total_leak = 0
    total_nonleak = 0

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())  # Use all available CPU cores
    print(f'Using {multiprocessing.cpu_count()} CPU Cores')

    # Create a list of arguments for the worker function
    args_list = []
    for file in os.listdir(dir_path):
        if max_vids and cur_count > max_vids:
            break
        vid_path = os.path.join(dir_path, file)
        vid_id = int(re.findall("_(\d{4}).mp4",os.path.basename(vid_path))[0])
        # vid_id = int(os.path.basename(vid_path)[4:8])
        if vid_id not in ranges.keys():
            continue

        nonleak_start = ranges[vid_id][0][0]
        nonleak_end = ranges[vid_id][0][1]
        leak_start = ranges[vid_id][1][0]
        leak_end = ranges[vid_id][1][1]
        args_list.append((vid_path, output_path, (leak_start, leak_end), (nonleak_start, nonleak_end), currLeakCount, currNonLeakCount))
        cur_count += 1
    
    # Use tqdm to create a progress bar
    # But whole process works
    progress_bar = tqdm(total=len(args_list), desc="Processing Videos")

    # Use the pool to map the arguments to the worker function with tqdm progress tracking
    results = pool.imap(extract_images, args_list)
    # Update the progress bar as processes complete
    # results should equal to the two ouputs from extract_images
    # The pool.imap function in Python's multiprocessing module is lazy, meaning it starts processing the items 
    # in the iterable (args_list in this case) only when you start iterating over the results 
    # The function returns an iterator, and the actual computation is performed on-demand during the iteration

    for result in results:
        nonleak_count, leak_count = result
        total_nonleak += nonleak_count
        total_leak += leak_count
        progress_bar.update(1) # Increment the progress bar

    pool.close()
    pool.join()

    progress_bar.close()  # Close the progress bar when done
    return total_nonleak, total_leak

def main():
   # get generic path to directory
    print('Start extraction')
    dir_path = os.path.dirname(os.path.realpath("__file__"))

    # get all raw video data directories
    data_dir = os.path.join("/home/bestlab/Desktop/Squishy-Methane-Analysis/MethaneModel", 'data')
    
    train_data_dir = os.path.join(data_dir, 'train')
    test_data_dir = os.path.join(data_dir, 'test')

    frame_data_dir = os.path.join(dir_path, 'frame_data_movingAvgC1C2')
    frame_train_data_dir = os.path.join(frame_data_dir, 'train')
    frame_test_data_dir = os.path.join(frame_data_dir, 'test')

    raw_data = np.loadtxt(os.path.join(dir_path, 'GasVid_Ranges_C1C2.csv'), skiprows=1, delimiter=',', dtype=int)

    ranges = list(zip(raw_data[:, 0], raw_data[:, 1:3], raw_data[:, 3:5])) #need to upload new ranges
    ranges = {ranges[i][0] : (ranges[i][1], ranges[i][2]) for i in range(len(ranges))}

    vid_count = 15 # Smaller on local computer to limit computer resources #max => 15

    test_count = 10 #Smaller on local computer to limit computer resources #max => 10

    total_train_NonLeak, total_train_Leak = read_frames_from_dir(train_data_dir, frame_train_data_dir, ranges, vid_count)
    print("Done with Training Data")
    total_test_NonLeak, total_test_Leak = read_frames_from_dir(test_data_dir, frame_test_data_dir, ranges, test_count)
    print("Done with Testing Data")

if __name__ == "__main__":
    main()
