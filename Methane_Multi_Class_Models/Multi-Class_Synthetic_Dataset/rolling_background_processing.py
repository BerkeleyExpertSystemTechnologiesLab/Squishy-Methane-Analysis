import cv2
import numpy as np
import os
import json
from collections import deque
from pathlib import Path


def compute_adaptive_alpha(difference_image, default_alpha=15):
    """
    Calculate adaptive enhancement factor to maximize visibility without clipping.
    Based on the method from gas detection research.
    
    Args:
        difference_image (numpy.ndarray): The difference image (background - current frame)
        default_alpha (float): Default enhancement factor (default: 15)
        
    Returns:
        float: Adaptive alpha value
    """
    mean_diff = np.mean(difference_image)
    std_diff = np.std(difference_image)
    
    # Prevent division by zero
    denominator = mean_diff + std_diff + 1e-6
    
    # Ensure one standard deviation above mean doesn't exceed 255
    alpha = min(default_alpha, 255.0 / denominator)
    
    return alpha


def enhance_difference_image(difference_image, alpha):
    """
    Apply enhancement factor to difference image and clip values.
    
    Args:
        difference_image (numpy.ndarray): The difference image
        alpha (float): Enhancement factor
        
    Returns:
        numpy.ndarray: Enhanced and clipped image (uint8)
    """
    # Enhance the difference
    enhanced = alpha * difference_image.astype(np.float32)
    
    # Clip values between 0 and 255
    enhanced = np.clip(enhanced, 0, 255)
    
    # Convert to uint8
    enhanced = enhanced.astype(np.uint8)
    
    return enhanced


def compute_rolling_background(frame_buffer):
    """
    Compute background image by averaging frames in the buffer.
    
    Args:
        frame_buffer (deque): Deque containing frame images
        
    Returns:
        numpy.ndarray: Averaged background image (uint8)
    """
    if len(frame_buffer) == 0:
        return None
    
    # Convert buffer to numpy array and compute mean
    frames_array = np.array(frame_buffer, dtype=np.float32)
    background = np.mean(frames_array, axis=0)
    
    # Convert back to uint8
    background = background.astype(np.uint8)
    
    return background


def process_single_frame_sample(current_frame, frame_buffer, default_alpha=15):
    """
    Process a single frame: compute background, difference, and enhancement.
    
    Args:
        current_frame (numpy.ndarray): Current frame to process
        frame_buffer (deque): Buffer containing previous frames for background
        default_alpha (float): Default enhancement factor
        
    Returns:
        dict: Dictionary containing 'background', 'enhanced_difference', and 'alpha'
               Returns None if buffer is not full enough
    """
    # Check if we have enough frames for background
    if len(frame_buffer) == 0:
        return None
    
    # Compute background from buffer
    background = compute_rolling_background(frame_buffer)
    
    if background is None:
        return None
    
    # Ensure current frame is grayscale
    if len(current_frame.shape) == 3:
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Compute absolute difference
    difference = cv2.absdiff(current_frame, background)
    
    # Compute adaptive alpha
    alpha = compute_adaptive_alpha(difference, default_alpha)
    
    # Enhance difference
    enhanced_difference = enhance_difference_image(difference, alpha)
    
    return {
        'background': background,
        'enhanced_difference': enhanced_difference,
        'alpha': alpha,
        'raw_difference': difference
    }


def process_video_sequential_sampling(
    video_path,
    video_code,
    processed_dataset_path,
    sampling_interval=100,
    history_length=30,
    default_alpha=15,
    frames_per_class=None
):
    """
    Process a video using sequential sampling with rolling background method.
    
    Args:
        video_path (str): Path to the video file
        video_code (str): 4-digit video code
        processed_dataset_path (str): Path to processed dataset directory
        sampling_interval (int): Sample every Nth frame (default: 100)
        history_length (int): Number of frames to keep in rolling buffer (default: 30)
        default_alpha (float): Default enhancement factor (default: 15)
        frames_per_class (int): Optional limit on frames per class (default: None, process all)
        
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
        print(f"  Sampling every {sampling_interval} frames with {history_length}-frame rolling background")
        
        # Video directory
        video_dir = os.path.join(processed_dataset_path, video_code)
        os.makedirs(video_dir, exist_ok=True)
        
        # Initialize frame buffer
        frame_buffer = deque(maxlen=history_length)
        
        # Track samples per class
        class_sample_counts = {i: 0 for i in range(8)}
        total_samples = 0
        skipped_samples = 0
        
        # Process frames sequentially
        for frame_idx in range(0, total_frames, sampling_interval):
            # Determine which class this frame belongs to
            time_minutes = frame_idx / (fps * 60)
            class_num = int(time_minutes // 3)
            
            # Skip if beyond class 7
            if class_num >= 8:
                break
            
            # Check if we've reached the limit for this class
            if frames_per_class is not None and class_sample_counts[class_num] >= frames_per_class:
                skipped_samples += 1
                continue
            
            # Read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"    Warning: Could not read frame {frame_idx}")
                continue
            
            # Convert to grayscale
            if len(frame.shape) == 3:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame_gray = frame
            
            # Process frame if we have enough history
            if len(frame_buffer) == history_length:
                # Process this frame
                result = process_single_frame_sample(frame_gray, frame_buffer, default_alpha)
                
                if result is not None:
                    # Create class directory and processed_data subfolder
                    class_dir = os.path.join(video_dir, f"Class_{class_num}")
                    processed_data_dir = os.path.join(class_dir, "processed_data")
                    os.makedirs(processed_data_dir, exist_ok=True)
                    
                    # Create 2-channel array: [background, enhanced_difference]
                    combined_array = np.stack([
                        result['background'].astype(np.float32),
                        result['enhanced_difference'].astype(np.float32)
                    ], axis=0)
                    
                    # Save numpy file
                    sample_filename = f"{video_code}_frame_{frame_idx}_class_{class_num}.npy"
                    sample_path = os.path.join(processed_data_dir, sample_filename)
                    np.save(sample_path, combined_array)
                    
                    # Update counts
                    class_sample_counts[class_num] += 1
                    total_samples += 1
                    
                    # Periodic progress update
                    if total_samples % 50 == 0:
                        print(f"    Processed {total_samples} samples...")
            
            # Add current frame to buffer for next iteration
            frame_buffer.append(frame_gray)
        
        cap.release()
        
        # Print summary
        print(f"  Video {video_code} complete: {total_samples} samples created")
        for class_num, count in class_sample_counts.items():
            if count > 0:
                print(f"    Class_{class_num}: {count} samples")
        
        return {
            'video_code': video_code,
            'total_samples': total_samples,
            'class_samples': class_sample_counts,
            'skipped_samples': skipped_samples
        }
        
    except Exception as e:
        print(f"  Error processing video {video_code}: {e}")
        import traceback
        print(f"  Full error: {traceback.format_exc()}")
        return None


def load_class_metadata(video_dir, video_code, class_num):
    """
    Load existing metadata for a class.
    
    Args:
        video_dir (str): Path to video directory
        video_code (str): 4-digit video code
        class_num (int): Class number (0-7)
        
    Returns:
        dict: Class metadata if it exists, empty dict otherwise
    """
    class_dir = os.path.join(video_dir, f"Class_{class_num}")
    class_json_file = os.path.join(class_dir, f"{video_code}_class_{class_num}.json")
    
    if os.path.exists(class_json_file):
        try:
            with open(class_json_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    return {}


def save_class_metadata(video_dir, video_code, class_num, metadata):
    """
    Save metadata for a class.
    
    Args:
        video_dir (str): Path to video directory
        video_code (str): 4-digit video code
        class_num (int): Class number (0-7)
        metadata (dict): Metadata to save
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        class_dir = os.path.join(video_dir, f"Class_{class_num}")
        os.makedirs(class_dir, exist_ok=True)
        
        class_json_file = os.path.join(class_dir, f"{video_code}_class_{class_num}.json")
        
        with open(class_json_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"  Error saving class metadata: {e}")
        return False





