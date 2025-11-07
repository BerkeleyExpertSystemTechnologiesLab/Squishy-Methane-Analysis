import cv2
import numpy as np
import os
import json

def analyze_numpy_array(numpy_array, array_name="Array"):
    """
    Analyze a numpy array and return statistics.
    
    Args:
        numpy_array (numpy.ndarray): The numpy array to analyze
        array_name (str): Name for display purposes
        
    Returns:
        dict: Dictionary with statistics or None if failed
    """
    try:
        if numpy_array is None:
            print(f"Error: Array is None")
            return None
        
        # Calculate statistics
        max_value = np.max(numpy_array)
        min_value = np.min(numpy_array)
        mean_value = np.mean(numpy_array)
        std_value = np.std(numpy_array)
        shape = numpy_array.shape
        dtype = numpy_array.dtype
        
        # Count non-zero pixels
        non_zero_pixels = np.count_nonzero(numpy_array)
        total_pixels = numpy_array.size
        non_zero_percentage = (non_zero_pixels / total_pixels) * 100
        
        # Print results
        print(f"{array_name} Statistics:")
        print(f"  Shape: {shape}")
        print(f"  Data type: {dtype}")
        print(f"  Min value: {min_value}")
        print(f"  Max value: {max_value}")
        print(f"  Mean value: {mean_value:.2f}")
        print(f"  Std deviation: {std_value:.2f}")
        print(f"  Value range: {max_value - min_value}")
        print(f"  Non-zero pixels: {non_zero_pixels:,} / {total_pixels:,} ({non_zero_percentage:.1f}%)")
        
        return {
            "max_value": max_value,
            "min_value": min_value,
            "mean_value": mean_value,
            "std_value": std_value,
            "shape": shape,
            "dtype": dtype,
            "non_zero_pixels": non_zero_pixels,
            "total_pixels": total_pixels,
            "non_zero_percentage": non_zero_percentage,
            "value_range": max_value - min_value
        }
        
    except Exception as e:
        print(f"Error analyzing array: {str(e)}")
        return None

def check_jpg_pixel_values(jpg_path):
    """
    Load a JPG image, convert to numpy array, and return statistics.
    
    Args:
        jpg_path (str): Path to the JPG file
        
    Returns:
        dict: Dictionary with statistics or None if failed
    """
    try:
        # Check if file exists
        if not os.path.exists(jpg_path):
            print(f"Error: File not found: {jpg_path}")
            return None
        
        # Load the JPG image
        print(f"Loading JPG: {jpg_path}")
        image = cv2.imread(jpg_path)
        
        if image is None:
            print(f"Error: Could not load image from {jpg_path}")
            return None
        
        # Convert to numpy array
        numpy_array = np.array(image)
        
        # Use shared analysis function
        return analyze_numpy_array(numpy_array, "Image")
        
    except Exception as e:
        print(f"Error processing {jpg_path}: {str(e)}")
        return None

def check_multiple_jpgs(jpg_paths):
    """
    Check pixel values for multiple JPG files.
    
    Args:
        jpg_paths (list): List of paths to JPG files
        
    Returns:
        dict: Dictionary with file paths as keys and statistics as values
    """
    results = {}
    
    print(f"Checking {len(jpg_paths)} JPG files...")
    print("="*60)
    
    for i, jpg_path in enumerate(jpg_paths, 1):
        print(f"\n[{i}/{len(jpg_paths)}] Checking: {os.path.basename(jpg_path)}")
        print("-" * 40)
        
        stats = check_jpg_pixel_values(jpg_path)
        results[jpg_path] = stats
        
        if stats is not None:
            print(f"Success")
        else:
            print(f"Failed")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful = sum(1 for stats in results.values() if stats is not None)
    print(f"Successfully processed: {successful}/{len(jpg_paths)} files")
    
    if successful > 0:
        # Get all max values from successful files
        max_values = [stats["max_value"] for stats in results.values() if stats is not None]
        min_values = [stats["min_value"] for stats in results.values() if stats is not None]
        
        print(f"\nOverall Statistics:")
        print(f"  Max value across all files: {max(max_values)}")
        print(f"  Min value across all files: {min(min_values)}")
        print(f"  Average max value: {np.mean(max_values):.2f}")
        print(f"  Average min value: {np.mean(min_values):.2f}")
        
        # Check if values are in expected range (0-255 for 8-bit images)
        if all(v <= 255 for v in max_values) and all(v >= 0 for v in min_values):
            print(f"All values are in valid 8-bit range (0-255)")
        else:
            print(f"Some values are outside 8-bit range (0-255)")
    
    return results

def find_scaled_jpgs(directory_path):
    """
    Find all scaled JPG files in a directory and subdirectories.
    
    Args:
        directory_path (str): Path to search for scaled JPG files
        
    Returns:
        list: List of paths to scaled JPG files
    """
    scaled_jpgs = []
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('_ppm_scaled.jpg'):
                full_path = os.path.join(root, file)
                scaled_jpgs.append(full_path)
    
    return scaled_jpgs

# Example usage function
def verify_scaling_in_dataset(processed_dataset_path="Processed_Dataset"):
    """
    Verify scaling across all scaled JPG files in the processed dataset.
    
    Args:
        processed_dataset_path (str): Path to the processed dataset directory
    """
    print("Finding all scaled JPG files in the dataset...")
    scaled_jpgs = find_scaled_jpgs(processed_dataset_path)
    
    if not scaled_jpgs:
        print(f"No scaled JPG files found in {processed_dataset_path}")
        return
    
    print(f"Found {len(scaled_jpgs)} scaled JPG files")
    
    # Check all files
    results = check_multiple_jpgs(scaled_jpgs)
    
    return results

def verify_specific_videos(video_codes, processed_dataset_path="Processed_Dataset"):
    """
    Verify scaling for specific video directories across all 8 classes.
    
    Args:
        video_codes (list): List of 4-digit video codes to check
        processed_dataset_path (str): Path to the processed dataset directory
    """
    print(f"Verifying scaling for videos: {video_codes}")
    print("="*60)
    
    all_scaled_jpgs = []
    
    for video_code in video_codes:
        video_dir = os.path.join(processed_dataset_path, video_code)
        
        if not os.path.exists(video_dir):
            print(f"Video directory not found: {video_dir}")
            continue
        
        print(f"\nChecking video {video_code}:")
        print("-" * 30)
        
        video_scaled_jpgs = []
        
        # Check all 8 classes for this video
        for class_num in range(8):
            class_dir = os.path.join(video_dir, f"Class_{class_num}")
            
            if not os.path.exists(class_dir):
                print(f"  Class_{class_num}: Directory not found")
                continue
            
            # Find scaled JPG in this class
            scaled_jpg = None
            for file in os.listdir(class_dir):
                if file.endswith('_ppm_scaled.jpg'):
                    scaled_jpg = os.path.join(class_dir, file)
                    break
            
            if scaled_jpg:
                video_scaled_jpgs.append(scaled_jpg)
                print(f"  Class_{class_num}: Found scaled JPG")
            else:
                print(f"  Class_{class_num}: No scaled JPG found")
        
        print(f"  Total scaled JPGs found for video {video_code}: {len(video_scaled_jpgs)}")
        all_scaled_jpgs.extend(video_scaled_jpgs)
    
    if not all_scaled_jpgs:
        print(f"\nNo scaled JPG files found for the specified videos")
        return {}
    
    print(f"\nTotal scaled JPGs to verify: {len(all_scaled_jpgs)}")
    print("="*60)
    
    # Check all found files
    results = check_multiple_jpgs(all_scaled_jpgs)
    
    return results

def check_numpy_array_values(npy_path):
    """
    Load a numpy array file and check the values.
    
    Args:
        npy_path (str): Path to the .npy file
        
    Returns:
        dict: Dictionary with statistics or None if failed
    """
    try:
        # Check if file exists
        if not os.path.exists(npy_path):
            print(f"Error: File not found: {npy_path}")
            return None
        
        # Load the numpy array
        print(f"Loading numpy array: {npy_path}")
        numpy_array = np.load(npy_path)
        
        if numpy_array is None:
            print(f"Error: Could not load numpy array from {npy_path}")
            return None
        
        # Check if it's a single-channel PPM array (2D format)
        if len(numpy_array.shape) != 2:
            print(f"Warning: Expected single-channel PPM array (H, W), got shape {numpy_array.shape}")
            return None
        
        # Use shared analysis function
        stats = analyze_numpy_array(numpy_array, "PPM Values")
        
        if stats is not None:
            # Add PPM-specific analysis
            max_val = stats["max_value"]
            
            # Check if values look like actual PPM values (not 0-255 normalized)
            if max_val <= 255 and max_val > 150:
                print(f"WARNING: Values look like normalized 0-255 range, not PPM values")
            elif max_val < 150:
                print(f"Values look like actual PPM values (reasonable range)")
            else:
                print(f"? Unusual value range - verify this is correct")
            
            # Wrap in ppm_channel for compatibility with existing code
            return {"ppm_channel": stats}
        
        return None
        
    except Exception as e:
        print(f"Error processing {npy_path}: {str(e)}")
        return None

def find_numpy_arrays(directory_path):
    """
    Find all numpy array files in a directory and subdirectories.
    
    Args:
        directory_path (str): Path to search for numpy array files
        
    Returns:
        list: List of paths to numpy array files
    """
    numpy_arrays = []
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.npy') and 'combined' in file:
                full_path = os.path.join(root, file)
                numpy_arrays.append(full_path)
    
    return numpy_arrays

def verify_numpy_arrays_specific_videos(video_codes, processed_dataset_path="Processed_Dataset"):
    """
    Verify numpy arrays for specific video directories across all 8 classes.
    
    Args:
        video_codes (list): List of 4-digit video codes to check
        processed_dataset_path (str): Path to the processed dataset directory
    """
    print(f"Verifying numpy arrays for videos: {video_codes}")
    print("="*60)
    
    all_numpy_arrays = []
    
    for video_code in video_codes:
        video_dir = os.path.join(processed_dataset_path, video_code)
        
        if not os.path.exists(video_dir):
            print(f"Video directory not found: {video_dir}")
            continue
        
        print(f"\nChecking video {video_code}:")
        print("-" * 30)
        
        video_numpy_arrays = []
        
        # Check all 8 classes for this video
        for class_num in range(8):
            class_dir = os.path.join(video_dir, f"Class_{class_num}")
            
            if not os.path.exists(class_dir):
                print(f"  Class_{class_num}: Directory not found")
                continue
            
            # Find PPM-scaled numpy array in this class
            numpy_array = None
            for file in os.listdir(class_dir):
                if file.endswith('_ppm_scaled.npy'):
                    numpy_array = os.path.join(class_dir, file)
                    break
            
            if numpy_array:
                video_numpy_arrays.append(numpy_array)
                print(f"  Class_{class_num}: Found PPM-scaled numpy array")
            else:
                print(f"  Class_{class_num}: No PPM-scaled numpy array found")
        
        print(f"  Total numpy arrays found for video {video_code}: {len(video_numpy_arrays)}")
        all_numpy_arrays.extend(video_numpy_arrays)
    
    if not all_numpy_arrays:
        print(f"\nNo numpy array files found for the specified videos")
        return {}
    
    print(f"\nTotal numpy arrays to verify: {len(all_numpy_arrays)}")
    print("="*60)
    
    # Check all found files
    results = {}
    
    for i, npy_path in enumerate(all_numpy_arrays, 1):
        print(f"\n[{i}/{len(all_numpy_arrays)}] Checking: {os.path.basename(npy_path)}")
        print("-" * 50)
        
        stats = check_numpy_array_values(npy_path)
        results[npy_path] = stats
        
        if stats is not None:
            print(f"Success")
        else:
            print(f"Failed")
    
    # Summary
    print("\n" + "="*60)
    print("NUMPY ARRAY SUMMARY")
    print("="*60)
    
    successful = sum(1 for stats in results.values() if stats is not None)
    print(f"Successfully processed: {successful}/{len(all_numpy_arrays)} files")
    
    if successful > 0:
        # Get all PPM max values from successful files
        ppm_max_values = []
        ppm_min_values = []
        
        for stats in results.values():
            if stats is not None:
                ppm_max_values.append(stats["ppm_channel"]["max_value"])
                ppm_min_values.append(stats["ppm_channel"]["min_value"])
        
        print(f"\nPPM Values Statistics:")
        print(f"  Max PPM value across all files: {max(ppm_max_values):.2f}")
        print(f"  Min PPM value across all files: {min(ppm_min_values):.2f}")
        print(f"  Average max PPM value: {np.mean(ppm_max_values):.2f}")
        print(f"  Average min PPM value: {np.mean(ppm_min_values):.2f}")
        
        # Check if PPM values look reasonable (not normalized 0-255)
        high_values = [v for v in ppm_max_values if v > 100]
        if len(high_values) > 0:
            print(f"  WARNING: {len(high_values)} files have max values > 100 PPM - may be normalized!")
        else:
            print(f"  All PPM values are in reasonable range (< 100 PPM)")
        
        # Check for very low values (might indicate scaling issues)
        very_low_values = [v for v in ppm_max_values if v < 1]
        if len(very_low_values) > 0:
            print(f"  WARNING: {len(very_low_values)} files have very low max values (< 1 PPM)")
    
    return results

def verify_all_numpy_arrays_in_dataset(processed_dataset_path="Processed_Dataset"):
    """
    Verify ALL numpy arrays in the entire dataset and count high PPM values.
    
    Args:
        processed_dataset_path (str): Path to the processed dataset directory
        
    Returns:
        dict: Summary statistics including count of high PPM values
    """
    print("VERIFYING ALL NUMPY ARRAYS IN DATASET")
    print("="*60)
    
    all_numpy_arrays = []
    high_ppm_files = []  # Files with max PPM > 150
    
    # Find all video directories
    if not os.path.exists(processed_dataset_path):
        print(f"Error: Dataset directory not found: {processed_dataset_path}")
        return {}
    
    video_dirs = [d for d in os.listdir(processed_dataset_path) 
                  if os.path.isdir(os.path.join(processed_dataset_path, d)) and d.isdigit() and len(d) == 4]
    
    print(f"Found {len(video_dirs)} video directories")
    
    for video_code in sorted(video_dirs):
        video_dir = os.path.join(processed_dataset_path, video_code)
        print(f"\nChecking video {video_code}:")
        print("-" * 30)
        
        video_numpy_arrays = []
        video_high_ppm = []
        
        # Check all 8 classes for this video
        for class_num in range(8):
            class_dir = os.path.join(video_dir, f"Class_{class_num}")
            
            if not os.path.exists(class_dir):
                print(f"  Class_{class_num}: Directory not found")
                continue
            
            # Find PPM-scaled numpy array in this class
            numpy_array = None
            for file in os.listdir(class_dir):
                if file.endswith('_ppm_scaled.npy'):
                    numpy_array = os.path.join(class_dir, file)
                    break
            
            if numpy_array:
                video_numpy_arrays.append(numpy_array)
                print(f"  Class_{class_num}: Found PPM-scaled numpy array")
            else:
                print(f"  Class_{class_num}: No PPM-scaled numpy array found")
        
        print(f"  Total numpy arrays found for video {video_code}: {len(video_numpy_arrays)}")
        all_numpy_arrays.extend(video_numpy_arrays)
    
    if not all_numpy_arrays:
        print(f"\nNo numpy array files found in the dataset")
        return {"total_files": 0, "high_ppm_count": 0, "high_ppm_files": []}
    
    print(f"\nTotal numpy arrays to verify: {len(all_numpy_arrays)}")
    print("="*60)
    
    # Check all found files
    results = {}
    ppm_max_values = []
    
    for i, npy_path in enumerate(all_numpy_arrays, 1):
        print(f"\n[{i}/{len(all_numpy_arrays)}] Checking: {os.path.basename(npy_path)}")
        print("-" * 50)
        
        stats = check_numpy_array_values(npy_path)
        results[npy_path] = stats
        
        if stats is not None:
            max_ppm = stats["ppm_channel"]["max_value"]
            ppm_max_values.append(max_ppm)
            
            if max_ppm > 150:
                high_ppm_files.append(npy_path)
                print(f"  HIGH PPM DETECTED: {max_ppm:.2f} PPM")
            else:
                print(f"  Normal PPM range: {max_ppm:.2f} PPM")
        else:
            print(f"Failed")
    
    # Summary
    print("\n" + "="*60)
    print("COMPLETE DATASET SUMMARY")
    print("="*60)
    
    successful = sum(1 for stats in results.values() if stats is not None)
    print(f"Successfully processed: {successful}/{len(all_numpy_arrays)} files")
    
    if successful > 0:
        print(f"\nPPM Values Statistics:")
        print(f"  Max PPM value across all files: {max(ppm_max_values):.2f}")
        print(f"  Min PPM value across all files: {min(ppm_max_values):.2f}")
        print(f"  Average max PPM value: {np.mean(ppm_max_values):.2f}")
        print(f"  Average min PPM value: {np.mean(ppm_max_values):.2f}")
        
        # Count high PPM values
        high_ppm_count = len(high_ppm_files)
        print(f"\nHIGH PPM VALUES (>150 PPM):")
        print(f"  Total files with high PPM: {high_ppm_count}")
        print(f"  Percentage of total files: {(high_ppm_count/len(all_numpy_arrays)*100):.1f}%")
        
        if high_ppm_count > 0:
            print(f"\nFiles with high PPM values:")
            for file_path in high_ppm_files:
                print(f"  - {file_path}")
        else:
            print(f"  No files found with PPM values > 150")
        
        # Check for very low values
        very_low_values = [v for v in ppm_max_values if v < 1]
        if len(very_low_values) > 0:
            print(f"\nWARNING: {len(very_low_values)} files have very low max values (< 1 PPM)")
    
    return {
        "total_files": len(all_numpy_arrays),
        "successful_files": successful,
        "high_ppm_count": len(high_ppm_files),
        "high_ppm_files": high_ppm_files,
        "ppm_max_values": ppm_max_values
    }

if __name__ == "__main__":
    # Example usage
    print("Dataset Verification Tool")
    print("="*40)
    
    # Check ALL files in the dataset for high PPM values
    print("\nVERIFYING ALL NUMPY ARRAYS IN DATASET")
    print("="*50)
    all_results = verify_all_numpy_arrays_in_dataset()
    
    # Alternative: Check specific videos only
    # target_videos = ["2583", "2581", "2580", "2566"]
    # print("\n1. VERIFYING JPG FILES (Final Combined Images)")
    # print("="*50)
    # jpg_results = verify_specific_videos(target_videos)
    # print("\n\n2. VERIFYING NUMPY ARRAYS (Raw Data)")
    # print("="*50)
    # numpy_results = verify_numpy_arrays_specific_videos(target_videos)
    
    # Alternative: You can test with a specific file
    # check_jpg_pixel_values("path/to/your/scaled_image.jpg")
    # check_numpy_array_values("path/to/your/combined_array.npy")
