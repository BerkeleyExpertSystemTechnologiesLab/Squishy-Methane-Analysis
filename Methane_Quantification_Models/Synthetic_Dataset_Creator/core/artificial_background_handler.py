"""
Artificial Background Handler

Handles copying and processing of artificial backgrounds for dataset augmentation.
Extracted from copy_backgrounds.py and artificial_dataset_creation.py.
"""

import json
import os
import re
import shutil

import config
import cv2
import numpy as np
from utils.directory import get_directory_files
from utils.image_scaling import scale_jpg_to_ppm

from core.frame_extract import subtract_background

# ==============================
# COPYING ARTIFICIAL BACKGROUNDS
# ==============================


def get_all_artificial_backgrounds(source_dir):
    """
    Recursively find all PNG files in the Artificial_Backgrounds directory.
    Uses get_directory_files() from directory_functions for each subdirectory.

    Args:
        source_dir: Path to Artificial_Backgrounds directory

    Returns:
        List of full file paths to all PNG files
    """
    if not os.path.exists(source_dir):
        return []

    png_files = []

    for root, dirs, files in os.walk(source_dir):
        files_info = get_directory_files(root, [".png"])

        if files_info and files_info["files"]:
            for file_info in files_info["files"]:
                png_files.append(file_info["file_path"])

    return png_files


def extract_artificial_background_metadata(filename):
    """
    Extract video code and class number from artificial background filename.
    Uses regex pattern similar to extract_video_code() from directory_functions.

    Expected pattern: XXXX_class_Y_*.png
    Example: 1237_class_0_background_artificial_cropped.png

    Args:
        filename: Just the filename (not full path)

    Returns:
        Tuple of (video_code, class_num) as strings, or None if parsing fails
    """
    pattern = r"(\d{4})_class_(\d+)"
    match = re.search(pattern, filename)

    if match:
        video_code = match.group(1)
        class_num = match.group(2)
        return (video_code, class_num)

    return None


def validate_and_create_artificial_destination(dest_base_dir, video_code, class_num):
    """
    Validate that the original video directory exists, then create/return
    the artificial background destination directory path.

    Args:
        dest_base_dir: Base path to Processed_Dataset
        video_code: 4-digit video code
        class_num: Class number as string

    Returns:
        Tuple of (valid: bool, dest_path: str)
        - valid: True if original video dir exists (making this a valid video code)
        - dest_path: Path to XXXX_ARTIF/Class_Y directory
    """
    original_video_dir = os.path.join(dest_base_dir, video_code)
    video_code_valid = os.path.exists(original_video_dir) and os.path.isdir(
        original_video_dir
    )

    artificial_video_dir = os.path.join(dest_base_dir, f"{video_code}_ARTIF")
    dest_path = os.path.join(artificial_video_dir, f"Class_{class_num}")

    if video_code_valid:
        os.makedirs(dest_path, exist_ok=True)

    return (video_code_valid, dest_path)


def copy_artificial_backgrounds_to_processed(source_base_dir, dest_base_dir):
    """
    Copy artificial backgrounds from Artificial_Backgrounds directory
    to their corresponding ARTIF locations in Processed_Dataset.

    Creates separate XXXX_ARTIF directories to keep artificial data separate
    from standard data. Files are placed in XXXX_ARTIF/Class_Y/ directories.

    Args:
        source_base_dir: Path to BackGrounds/Artificial_Backgrounds
        dest_base_dir: Path to Processed_Dataset

    Returns:
        dict: Summary with copied, skipped, and failed counts
    """
    if not os.path.exists(source_base_dir):
        print(f"Source directory does not exist: {source_base_dir}")
        return None

    all_files = get_all_artificial_backgrounds(source_base_dir)

    if not all_files:
        print(f"No PNG files found in {source_base_dir}")
        return None

    # Statistics
    total_found = len(all_files)
    total_copied = 0
    total_skipped_existing = 0
    total_skipped_no_dest = 0
    total_failed_parse = 0
    total_failed_copy = 0

    print(f"\nCopying artificial backgrounds from: {source_base_dir}")
    print(f"Found {total_found} artificial background files")

    for file_path in all_files:
        filename = os.path.basename(file_path)

        metadata = extract_artificial_background_metadata(filename)

        if not metadata:
            total_failed_parse += 1
            continue

        video_code, class_num = metadata

        video_code_valid, dest_dir = validate_and_create_artificial_destination(
            dest_base_dir, video_code, class_num
        )

        if not video_code_valid:
            total_skipped_no_dest += 1
            continue

        dest_file_path = os.path.join(dest_dir, filename)
        if os.path.exists(dest_file_path):
            total_skipped_existing += 1
            continue

        try:
            shutil.copy2(file_path, dest_file_path)
            total_copied += 1
        except Exception:
            total_failed_copy += 1

    print("Artificial Background Copy Summary:")
    print(f"  Total files found: {total_found}")
    print(f"  Successfully copied to XXXX_ARTIF directories: {total_copied}")
    print(f"  Skipped (already exists): {total_skipped_existing}")
    print(f"  Skipped (original video dir not found): {total_skipped_no_dest}")

    return {
        "total_found": total_found,
        "copied": total_copied,
        "skipped_existing": total_skipped_existing,
        "skipped_no_dest": total_skipped_no_dest,
        "failed_parse": total_failed_parse,
        "failed_copy": total_failed_copy,
    }


# =============================
# GENERATING ARTIFICIAL SAMPLES
# =============================


def save_background_selection_metadata(
    class_dir, video_code, class_num, background_filename, available_backgrounds
):
    """
    Save metadata about which artificial background was selected.

    Args:
        class_dir (str): Path to the class directory
        video_code (str): 4-digit video code
        class_num (int): Class number
        background_filename (str): Name of the selected background file
        available_backgrounds (list): List of all available background files
    """
    metadata = {
        "video_code": video_code,
        "class_num": class_num,
        "selected_background": background_filename,
        "available_backgrounds": available_backgrounds,
        "selection_method": "alphabetically_first",
        "total_available": len(available_backgrounds),
    }

    metadata_path = os.path.join(class_dir, "background_selection.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_artificial_backgrounds(processed_dataset_path, video_code):
    """
    Load all artificial background files for a specific video from XXXX_ARTIF directory.
    Backgrounds are selected deterministically (alphabetically first) and metadata is saved.

    Args:
        processed_dataset_path (str): Path to processed dataset directory
        video_code (str): 4-digit video code (e.g., "1237")

    Returns:
        dict: Dictionary mapping class_num to artificial background path
              Returns None if no ARTIF directory exists
    """
    class_backgrounds_artif = {}
    video_dir_artif = os.path.join(processed_dataset_path, f"{video_code}_ARTIF")

    if not os.path.exists(video_dir_artif):
        return None

    for class_num in range(config.NUM_CLASSES):
        class_dir_artif = os.path.join(video_dir_artif, f"Class_{class_num}")

        if os.path.exists(class_dir_artif):
            artif_backgrounds = [
                f
                for f in os.listdir(class_dir_artif)
                if "background" in f.lower() and f.endswith(".png")
            ]

            if artif_backgrounds:
                artif_backgrounds.sort()
                selected_background = artif_backgrounds[0]

                class_backgrounds_artif[class_num] = os.path.join(
                    class_dir_artif, selected_background
                )

                save_background_selection_metadata(
                    class_dir_artif,
                    video_code,
                    class_num,
                    selected_background,
                    artif_backgrounds,
                )

    return class_backgrounds_artif if class_backgrounds_artif else None


def copy_metadata_to_artif(processed_dataset_path, video_code, class_num):
    """
    Copy class metadata JSON file from normal directory to ARTIF directory.

    Args:
        processed_dataset_path (str): Path to processed dataset directory
        video_code (str): 4-digit video code
        class_num (int): Class number (0-7)

    Returns:
        bool: True if successful, False otherwise
    """
    video_dir = os.path.join(processed_dataset_path, video_code)
    class_dir = os.path.join(video_dir, f"Class_{class_num}")
    source_json = os.path.join(class_dir, f"{video_code}_class_{class_num}.json")

    video_dir_artif = os.path.join(processed_dataset_path, f"{video_code}_ARTIF")
    class_dir_artif = os.path.join(video_dir_artif, f"Class_{class_num}")
    os.makedirs(class_dir_artif, exist_ok=True)
    dest_json = os.path.join(class_dir_artif, f"{video_code}_class_{class_num}.json")

    if os.path.exists(source_json):
        if not os.path.exists(dest_json):
            try:
                shutil.copy2(source_json, dest_json)
                return True
            except Exception as e:
                print(f"    Warning: Could not copy metadata: {e}")
                return False
        return True

    return False


def generate_artificial_sample(
    cap,
    video_code,
    class_num,
    random_frame,
    normal_background_path,
    artif_background_path,
    processed_data_dir,
    ppm_value,
):
    """
    Generate a single artificial background numpy sample.

    This function:
    1. Extracts a frame from the video
    2. Subtracts the NORMAL background (to get gas plume)
    3. Scales to PPM values
    4. Combines with ARTIFICIAL background in Channel 0
    5. Saves as .npy file with "_artif" suffix

    Args:
        cap: OpenCV VideoCapture object (already opened)
        video_code (str): 4-digit video code
        class_num (int): Class number (0-7)
        random_frame (int): Frame number to extract
        normal_background_path (str): Path to normal background (for subtraction)
        artif_background_path (str): Path to artificial background (for Channel 0)
        processed_data_dir (str): Directory to save the numpy file
        ppm_value (float): PPM value for scaling

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        ret, frame = cap.read()

        if not ret:
            return False

        # Temporary filenames
        frame_filename = f"temp_artif_frame_{random_frame}.jpg"
        frame_path = os.path.join(processed_data_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

        # Apply background subtraction using NORMAL background
        subtracted_filename = f"temp_artif_subtracted_{random_frame}.jpg"
        subtracted_path = os.path.join(processed_data_dir, subtracted_filename)

        subtract_success = subtract_background(
            normal_background_path, frame_path, subtracted_path
        )

        if not subtract_success:
            os.remove(frame_path)
            return False

        # Scale to PPM
        scaled_filename = f"temp_artif_scaled_{random_frame}.npy"
        scaled_path = os.path.join(processed_data_dir, scaled_filename)

        scaled_array = scale_jpg_to_ppm(
            subtracted_path, ppm_value, scaled_path, grayscale=True
        )

        if scaled_array is None:
            os.remove(frame_path)
            os.remove(subtracted_path)
            return False

        # Load ARTIFICIAL background as numpy array
        artif_background_array = cv2.imread(artif_background_path, cv2.IMREAD_GRAYSCALE)

        if artif_background_array is None:
            os.remove(frame_path)
            os.remove(subtracted_path)
            os.remove(scaled_path)
            return False

        # Create 2-channel array with ARTIFICIAL background
        image_array_artif = np.stack(
            [
                artif_background_array.astype(np.float32),  # Channel 0: Artificial
                scaled_array.astype(np.float32),  # Channel 1: Gas leak
            ],
            axis=0,
        )

        # Save to ARTIF directory with "_artif" suffix for clarity
        frame_str = f"{random_frame:02d}"
        combined_filename = (
            f"{video_code}_frame_{frame_str}_class_{class_num}_artif.npy"
        )
        combined_path = os.path.join(processed_data_dir, combined_filename)
        np.save(combined_path, image_array_artif)

        # Clean up temporary files
        os.remove(frame_path)
        os.remove(subtracted_path)
        os.remove(scaled_path)

        return True

    except Exception as e:
        print(f"      Error: {e}")
        return False
