from directory_functions import get_directory_files, extract_video_code
from frame_extract import subtract_background
from image_scaling import scale_jpg_to_ppm

import os
import json
import numpy as np
import cv2
import random

# =============================================================================
# FULL DATASET CREATION - 50 RANDOM FRAMES PER CLASS PER VIDEO
# =============================================================================

def create_full_dataset(test_videos=None, frames_per_class=50):
    """
    Create a full dataset by selecting 50 random frames from each of the 8 classes 
    from each video, then processing them through the complete pipeline.
    Assumes all metadata and background images already exist.
    
    Args:
        test_videos (list, optional): List of 4-digit video codes to process. 
                                    If None, processes all videos found.
        frames_per_class (int): Number of random frames to extract per class (default: 50)
    """
    print("\n" + "="*80)
    print("STARTING FULL DATASET CREATION")
    print("="*80)
    print(f"Target: {frames_per_class} random frames per class per video")
    print("Assumes: JSON metadata and background images already exist")
    
    # Define paths
    mov_path = "Original_Dataset/Videos-20251002T175522Z-1-001/Videos"
    processed_dataset_path = "Processed_Dataset"
    
    try:
        # Step 1: Get video files
        print("\n" + "="*50)
        print("STEP 1: LOADING VIDEO FILES")
        print("="*50)
        
        print("Getting video files...")
        video_files = get_directory_files(mov_path, ['.mp4', '.mov'])
        if video_files is None:
            print("ERROR: Failed to get video files")
            return False
        print(f"Found {video_files['total_files']} video files")
        
        # Determine which videos to process
        if test_videos:
            print(f"TEST MODE: Processing only specified videos: {test_videos}")
            video_codes = test_videos
        else:
            print("Getting video codes from all videos...")
            video_codes = []
            for file_info in video_files['files']:
                video_code = extract_video_code(file_info['filename'])
                if video_code:
                    video_codes.append(video_code)
        
        if not video_codes:
            print("ERROR: No valid video codes found")
            return False
        
        print(f"Processing {len(video_codes)} videos: {video_codes}")
        
        # Step 2: Process each video
        print("\n" + "="*50)
        print("STEP 2: PROCESSING VIDEOS")
        print("="*50)
        
        total_processed = 0
        total_successful = 0
        total_failed = 0
        
        for video_code in video_codes:
            print(f"\nProcessing video {video_code}...")
            print("-" * 40)
            
            # Find the video file
            video_file = None
            for file_info in video_files['files']:
                if video_code in file_info['filename']:
                    video_file = file_info['file_path']
                    break
            
            if not video_file:
                print(f"  Video file not found for code {video_code}")
                total_failed += 1
                continue
            
            # Process this video
            video_result = process_single_video_full_dataset(
                video_file, video_code, processed_dataset_path, frames_per_class
            )
            
            if video_result:
                total_successful += video_result['successful']
                total_failed += video_result['failed']
                total_processed += video_result['total']
                print(f"  Video {video_code}: {video_result['successful']}/{video_result['total']} samples created")
            else:
                print(f"  Video {video_code}: Processing failed")
                total_failed += 1
        
        # Final summary
        print("\n" + "="*80)
        print("FULL DATASET CREATION COMPLETED!")
        print("="*80)
        print(f"Total samples processed: {total_processed}")
        print(f"Successful: {total_successful}")
        print(f"Failed: {total_failed}")
        print(f"Target frames per class: {frames_per_class}")
        print(f"Expected total samples: {len(video_codes) * 8 * frames_per_class}")
        
        if total_successful > 0:
            print(f"\nSuccessfully created {total_successful} samples!")
            print("File structure:")
            print("  Processed_Dataset/")
            print("  ├── XXXX/")
            print("  │   ├── XXXX_background_cv2.png  # Video-level background")
            print("  │   ├── XXXX_background_moving_avg.png  # Video-level background")
            print("  │   ├── Class_0/")
            print("  │   │   ├── processed_data/")
            print("  │   │   │   ├── XXXX_frame_15-15_class_0.npy")
            print("  │   │   │   ├── XXXX_frame_23-23_class_0.npy")
            print("  │   │   │   └── ... (50 frames)")
            print("  │   │   └── XXXX_class_0.json  # Shared metadata")
            print("  │   └── Class_1/...")
            print("  └── ...")
            print("\nEach numpy file contains:")
            print("  - 2-channel array: (2, H, W)")
            print("  - Channel 0: Background image")
            print("  - Channel 1: PPM-scaled subtracted image (gas leak)")
            print("  - Metadata shared via class JSON file")
        else:
            print("\nNo samples were successfully created")
        
        return total_successful > 0
        
    except Exception as e:
        print(f"\nError in full dataset creation: {e}")
        return False

def process_single_video_full_dataset(video_path, video_code, processed_dataset_path, frames_per_class=50):
    """
    Process a single video to extract 50 random frames from each class and create
    the complete dataset samples.
    
    Args:
        video_path (str): Path to the video file
        video_code (str): 4-digit video code
        processed_dataset_path (str): Path to processed dataset directory
        frames_per_class (int): Number of random frames per class
        
    Returns:
        dict: Summary of processing results
    """
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"  Error: Could not open video file {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_minutes = total_frames / (fps * 60)
        
        # Check if video is at least 24 minutes
        if duration_minutes < 24:
            print(f"  Error: Video {video_code} is only {duration_minutes:.2f} minutes long. Need at least 24 minutes.")
            cap.release()
            return None
        
        print(f"  Video {video_code}: {duration_minutes:.2f} minutes, {fps:.2f} FPS")
        
        # Video directory
        video_dir = os.path.join(processed_dataset_path, video_code)
        
        # Use class-specific background images from class directories
        print(f"  Loading class-specific background images...")
        
        # Look for class-specific background files for each class
        class_backgrounds = {}
        for class_num in range(8):
            class_dir = os.path.join(video_dir, f"Class_{class_num}")
            
            cv2_background_file = os.path.join(class_dir, f"{video_code}_class_{class_num}_background_cv2.png")
            moving_avg_background_file = os.path.join(class_dir, f"{video_code}_class_{class_num}_background_moving_avg.png")
            
            # Prefer CV2, fallback to moving average
            # if os.path.exists(cv2_background_file):
            #     class_backgrounds[class_num] = cv2_background_file
            #     print(f"    Class_{class_num}: Using CV2 background")
            # elif os.path.exists(moving_avg_background_file):

            # Swapped to prefer average background using frame averaging not cv2 method
            if os.path.exists(moving_avg_background_file):
                class_backgrounds[class_num] = moving_avg_background_file
                print(f"    Class_{class_num}: Using Moving Average background")
            else:
                print(f"  Error: No background found for Class_{class_num}")
                print(f"    Expected: {cv2_background_file} or {moving_avg_background_file}")
                cap.release()
                return None
        
        print(f"  Successfully loaded class-specific backgrounds for all {len(class_backgrounds)} classes")
        
        # Process each class
        total_processed = 0
        total_successful = 0
        total_failed = 0
        
        for class_num in range(8):
            print(f"  Processing Class_{class_num}...")
            
            # Calculate time window for this class
            start_minutes = class_num * 3
            end_minutes = (class_num + 1) * 3
            start_frame = int(start_minutes * 60 * fps)
            end_frame = int(end_minutes * 60 * fps)
            
            # Class directory
            class_dir = os.path.join(video_dir, f"Class_{class_num}")
            
            # Create processed data subfolder
            processed_data_dir = os.path.join(class_dir, "processed_data")
            os.makedirs(processed_data_dir, exist_ok=True)
            
            # Count existing files to avoid duplicates
            existing_files = [f for f in os.listdir(processed_data_dir) if f.endswith('.npy')]
            if existing_files:
                print(f"    Class_{class_num}: Found {len(existing_files)} existing files, skipping...")
                continue
            
            # Load class metadata
            class_json_file = os.path.join(class_dir, f"{video_code}_class_{class_num}.json")
            if not os.path.exists(class_json_file):
                print(f"    Class_{class_num}: No JSON file found")
                total_failed += frames_per_class
                continue
            
            try:
                with open(class_json_file, 'r') as f:
                    class_data = json.load(f)
                
                # Extract metadata
                distance_m = class_data.get('distance_m')
                leak_rate_scfh = class_data.get('leak_rate_scfh')
                ppm_value = class_data.get('ppm')
                
                if any(x is None for x in [distance_m, leak_rate_scfh, ppm_value]):
                    print(f"    Class_{class_num}: Missing metadata")
                    total_failed += frames_per_class
                    continue
                
                # Extract random frames from this class time window
                print(f"    Class_{class_num}: Extracting {frames_per_class} random frames...")
                
                class_successful = 0
                class_failed = 0
                
                for frame_idx in range(frames_per_class):
                    try:
                        # Select random frame from this class time window
                        random_frame = random.randint(start_frame, end_frame - 1)
                        
                        # Extract the frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
                        ret, frame = cap.read()
                        
                        if not ret:
                            print(f"      Frame {frame_idx}: Failed to extract")
                            class_failed += 1
                            continue
                        
                        # Save original frame (temporary, for processing)
                        frame_filename = f"temp_frame_{frame_idx:03d}.jpg"
                        frame_path = os.path.join(processed_data_dir, frame_filename)
                        cv2.imwrite(frame_path, frame)
                        
                        # Apply background subtraction
                        background_path = class_backgrounds[class_num]
                        subtracted_filename = f"temp_subtracted_{frame_idx:03d}.jpg"
                        subtracted_path = os.path.join(processed_data_dir, subtracted_filename)
                        
                        subtract_success = subtract_background(background_path, frame_path, subtracted_path)
                        if not subtract_success:
                            print(f"      Frame {frame_idx}: Background subtraction failed")
                            class_failed += 1
                            continue
                        
                        # Scale to PPM (temporary file)
                        scaled_filename = f"temp_scaled_{frame_idx:03d}.npy"
                        scaled_path = os.path.join(processed_data_dir, scaled_filename)
                        
                        scaled_array = scale_jpg_to_ppm(subtracted_path, ppm_value, scaled_path, grayscale=True)
                        if scaled_array is None:
                            print(f"      Frame {frame_idx}: PPM scaling failed")
                            class_failed += 1
                            continue
                        
                        # Load background as numpy array
                        background_array = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
                        if background_array is None:
                            print(f"      Frame {frame_idx}: Failed to load background")
                            class_failed += 1
                            continue
                        
                        # Create combined 2-channel array
                        image_array = np.stack([
                            background_array.astype(np.float32),      # Channel 0: Background
                            scaled_array.astype(np.float32),          # Channel 1: Gas leak
                        ], axis=0)
                        
                        # Save combined array in processed_data subfolder
                        # Format: XXXX_frame_XX-XX_class_X.npy
                        # where XX-XX is the frame range from original movie
                        frame = f"{random_frame:02d}"  # Single frame range
                        combined_filename = f"{video_code}_frame_{frame}_class_{class_num}.npy"
                        combined_path = os.path.join(processed_data_dir, combined_filename)
                        np.save(combined_path, image_array)
                        
                        # Clean up temporary files
                        os.remove(frame_path)
                        os.remove(subtracted_path)
                        os.remove(scaled_path)
                        
                        class_successful += 1
                        total_successful += 1
                        total_processed += 1
                        
                        if (frame_idx + 1) % 10 == 0:
                            print(f"      Processed {frame_idx + 1}/{frames_per_class} frames")
                    
                    except Exception as e:
                        print(f"      Frame {frame_idx}: Error - {e}")
                        class_failed += 1
                        continue
                
                print(f"    Class_{class_num}: {class_successful} samples created")
                
            except Exception as e:
                print(f"    Class_{class_num}: Error processing - {e}")
                total_failed += frames_per_class
                continue
        
        cap.release()
        
        return {
            'total': total_processed,
            'successful': total_successful,
            'failed': total_failed
        }
        
    except Exception as e:
        print(f"  Error processing video {video_code}: {e}")
        return None

def create_dataset_summary(processed_dataset_path="Processed_Dataset"):
    """
    Create a summary of the generated dataset.
    
    Args:
        processed_dataset_path (str): Path to the processed dataset directory
    """
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    total_samples = 0
    video_summaries = {}
    
    # Get all video directories
    video_dirs = [d for d in os.listdir(processed_dataset_path) 
                  if os.path.isdir(os.path.join(processed_dataset_path, d)) and d.isdigit() and len(d) == 4]
    
    for video_code in sorted(video_dirs):
        video_dir = os.path.join(processed_dataset_path, video_code)
        video_samples = 0
        class_samples = {}
        
        # Count samples in each class
        for class_num in range(8):
            class_dir = os.path.join(video_dir, f"Class_{class_num}")
            processed_data_dir = os.path.join(class_dir, "processed_data")
            if os.path.exists(processed_data_dir):
                # Count numpy files in processed_data subfolder
                numpy_files = [f for f in os.listdir(processed_data_dir) if f.endswith('.npy')]
                class_samples[class_num] = len(numpy_files)
                video_samples += len(numpy_files)
        
        video_summaries[video_code] = {
            'total_samples': video_samples,
            'class_samples': class_samples
        }
        total_samples += video_samples
    
    print(f"Total samples in dataset: {total_samples}")
    print(f"Videos processed: {len(video_summaries)}")
    print("\nPer-video breakdown:")
    print("-" * 40)
    
    for video_code, summary in video_summaries.items():
        print(f"Video {video_code}: {summary['total_samples']} samples")
        for class_num in range(8):
            count = summary['class_samples'].get(class_num, 0)
            print(f"  Class_{class_num}: {count} samples")
    
    return {
        'total_samples': total_samples,
        'video_count': len(video_summaries),
        'video_summaries': video_summaries
    }

if __name__ == "__main__":
    # Example usage
    print("Full Dataset Creation Tool")
    print("="*40)
    
    # Create full dataset with 50 frames per class per video
    # For testing, you can specify specific videos:
    # test_videos = ["1237", "1238", "1239"]
    # success = create_full_dataset(test_videos=test_videos, frames_per_class=50)

    # Set the total number of frames per class per video 
    # frames_per_class X 8 classes X 28 Videos (50 X 8 X 28 = 11,200 frames)

    
    # Or process all videos:
    success = create_full_dataset(frames_per_class=100)
    
    if success:
        print("\nCreating dataset summary...")
        summary = create_dataset_summary()
        print(f"\nDataset creation completed successfully!")
        print(f"Total samples created: {summary['total_samples']}")
    else:
        print("\nDataset creation failed!")
