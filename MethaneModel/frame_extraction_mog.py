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

def extractImages(args):

  '''
  Input:
    String: pathIn should be the path of the video
    String: pathOut should be the path of the folder where data is being stored for testing or training
    Tuple: range of leak frames from video
    Tuple: range of nonleak frames from video

  Output:
    creates two subfolders in pathOut called Leaks and Nonleaks
      Leaks folder contains the frames where there are leaks
      Nonleaks folder contains the frames where there are noleaks
  '''
  # this is for parallelization
  pathIn, pathOut, leakRange, nonleakRange, currCountLeak, currCountNonLeak = args

  leakPath = os.path.join(pathOut, "Leak")
  nonleakPath = os.path.join(pathOut, "Nonleaks")

  os.makedirs(leakPath, exist_ok=True)
  os.makedirs(nonleakPath, exist_ok=True)

  def helper(pathIn, pathOut, range, isLeak, currCountLeak, currCountNonLeak):
    '''
    Might need to clean this up, but this was extracted from the original extractImages from the previous implementation

    '''
    #setting up moving average list

    start = range[0] * 1000 # converting seconds to milliseconds
    end = range[1] * 1000
    cap = cv2.VideoCapture(pathIn)
    cap.set(cv2.CAP_PROP_POS_MSEC, start)
    success = True
    fgbg = cv2.createBackgroundSubtractorMOG2()

    if cap.isOpened():
      while success and start < end:
          success, image = cap.read()
          if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            start = cap.get(cv2.CAP_PROP_POS_MSEC)

            fgmask = fgbg.apply(image)

            processed_img = cv2.erode(fgmask, np.ones((3,3), dtype=np.uint8))

            if isLeak:
                cv2.imwrite(os.path.join(pathOut, "leak.frame%d.jpg" % uuid.uuid4()), processed_img)     # save frame as JPEG file
                currCountLeak += 1
            else:
                cv2.imwrite(os.path.join(pathOut, "nonleak.frame%d.jpg" % uuid.uuid4()), processed_img)
                currCountNonLeak += 1
          else:
            break
      cap.release()
    cv2.destroyAllWindows()
    if isLeak:
       return currCountLeak
    else:
       return currCountNonLeak
  # call helper for both nonLeak and leak and get updated counts
  updated_currCountNonLeak = helper(pathIn, nonleakPath, nonleakRange, isLeak=False, currCountLeak=currCountLeak, currCountNonLeak=currCountNonLeak)
  updated_currCountLeak = helper(pathIn, leakPath, leakRange, isLeak=True,currCountLeak=currCountLeak, currCountNonLeak=currCountNonLeak)

  return updated_currCountNonLeak, updated_currCountLeak


def read_frames_from_dir(dir_path, output_path, ranges, max_vids=None):
    cur_count = 1
    currNonLeakCount = 0
    currLeakCount = 0

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2)  # Use all available CPU cores
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

        # currNonLeakCount, currLeakCount = extractImages(vid_path, output_path, (leak_start, leak_end), (nonleak_start, nonleak_end), currLeakCount, currNonLeakCount)
        args_list.append((vid_path, output_path, (leak_start, leak_end), (nonleak_start, nonleak_end), currLeakCount, currNonLeakCount))
        # print("Video", vid_id)
        # print("Current NonLeak Count", currNonLeakCount)
        # print("Current Leak Count", currLeakCount)

        # print('Done with', cur_count, "video(s)")
        cur_count += 1
      
    # Use tqdm to create a progress bar
    progress_bar = tqdm(total=len(args_list), desc="Processing Videos")

    # Use the pool to map the arguments to the worker function with tqdm progress tracking
    results = list(pool.imap(extractImages, args_list))

    # Update the progress bar as processes complete
    for nonleak_count, leak_count in results:
        currNonLeakCount += nonleak_count
        currLeakCount += leak_count
        progress_bar.update(1)  # Increment the progress bar

    pool.close()
    pool.join()

    progress_bar.close()  # Close the progress bar when done
    return currNonLeakCount, currLeakCount

def main():
   # get generic path to directory
    print('Start extraction')
    dir_path = os.path.dirname(os.path.realpath("__file__"))

    # get all raw video data directories
    data_dir = os.path.join(dir_path, 'data')

    train_data_dir = os.path.join(data_dir, 'train')
    test_data_dir = os.path.join(data_dir, 'test')

    frame_data_dir = os.path.join(dir_path, 'frame_data_mog')
    frame_train_data_dir = os.path.join(frame_data_dir, 'train')
    frame_test_data_dir = os.path.join(frame_data_dir, 'test')

    raw_data = np.loadtxt(os.path.join(dir_path, 'GasVid_Ranges_Seconds.csv'), skiprows=1, delimiter=',', dtype=int)

    ranges = list(zip(raw_data[:, 0], raw_data[:, 1:3], raw_data[:, 3:5])) #need to upload new ranges
    ranges = {ranges[i][0] : (ranges[i][1], ranges[i][2]) for i in range(len(ranges))}

    vid_count = None # Smaller on local computer to limit computer resources #max =>15

    test_count = None #Smaller on local computer to limit computer resources #max =>10

    total_train_NonLeak, total_train_Leak = read_frames_from_dir(train_data_dir, frame_train_data_dir, ranges, vid_count)
    print("Done with Training Data")
    total_test_NonLeak, total_test_Leak = read_frames_from_dir(test_data_dir, frame_test_data_dir, ranges, test_count)
    print("Done with Testing Data")

if __name__ == "__main__":
    main()
