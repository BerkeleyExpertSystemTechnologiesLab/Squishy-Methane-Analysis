import json
import os
import random

import config
import cv2
import numpy as np
from utils.image_scaling import scale_jpg_to_ppm

from core.background_generator import load_normal_backgrounds
from core.frame_extract import subtract_background


def load_class_metadata(class_dir, video_code, class_num):
    """
    Load metadata for a specific class.

    Returns:
        dict: Class metadata or None if failed
    """
    class_json_file = os.path.join(class_dir, f"{video_code}_class_{class_num}.json")

    if not os.path.exists(class_json_file):
        return None

    try:
        with open(class_json_file, "r") as f:
            class_data = json.load(f)

        distance_m = class_data.get("distance_m")
        leak_rate_scfh = class_data.get("leak_rate_scfh")
        ppm_value = class_data.get("ppm")

        if any(x is None for x in [distance_m, leak_rate_scfh, ppm_value]):
            return None

        return class_data

    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None


def create_single_numpy_sample(
    cap,
    random_frame,
    background_path,
    ppm_value,
    processed_data_dir,
    frame_idx,
    channels="double",
):
    """
    Create a single numpy sample from a video frame.

    Args:
        channels: "single" for frames only, "double" for background + plume

    Returns:
        numpy.ndarray or None: 1-channel or 2-channel array if successful
    """
    try:
        # Extract frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        ret, frame = cap.read()

        if not ret:
            return None

        # Save temporary frame
        frame_filename = f"temp_frame_{frame_idx:03d}.jpg"
        frame_path = os.path.join(processed_data_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

        if channels == "single":
            # Single channel: just convert frame to grayscale numpy array (no PPM scaling)

            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame_gray = frame

            # Create 1-channel array directly from frame (no scaling, no temp files)
            image_array = frame_gray.astype(np.float32)[np.newaxis, :]

            return image_array

        else:
            # Double channel: background + subtracted plume
            # Subtract background
            subtracted_filename = f"temp_subtracted_{frame_idx:03d}.jpg"
            subtracted_path = os.path.join(processed_data_dir, subtracted_filename)

            if not subtract_background(background_path, frame_path, subtracted_path):
                os.remove(frame_path)
                return None

            # Scale to PPM
            scaled_filename = f"temp_scaled_{frame_idx:03d}.npy"
            scaled_path = os.path.join(processed_data_dir, scaled_filename)

            scaled_array = scale_jpg_to_ppm(
                subtracted_path, ppm_value, scaled_path, grayscale=True
            )
            if scaled_array is None:
                os.remove(frame_path)
                os.remove(subtracted_path)
                return None

            # Load background
            background_array = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
            if background_array is None:
                os.remove(frame_path)
                os.remove(subtracted_path)
                os.remove(scaled_path)
                return None

            # Create 2-channel array
            image_array = np.stack(
                [
                    background_array.astype(np.float32),
                    scaled_array.astype(np.float32),
                ],
                axis=0,
            )

            # Cleanup
            os.remove(frame_path)
            os.remove(subtracted_path)
            os.remove(scaled_path)

            return image_array

    except Exception as e:
        print(f"Error creating sample: {e}")
        return None


def process_single_class(
    cap,
    video_code,
    class_num,
    video_dir,
    class_backgrounds,
    frames_per_class,
    fps,
    start_frame,
    end_frame,
    channels="double",
):
    """
    Process a single class to create numpy samples.

    Args:
        channels: "single" for frames only, "double" for background + plume

    Returns:
        dict: Results with successful and failed counts
    """
    class_dir = os.path.join(video_dir, f"Class_{class_num}")
    processed_data_dir = os.path.join(class_dir, "processed_data")
    os.makedirs(processed_data_dir, exist_ok=True)

    # Skip if already processed
    existing_files = [f for f in os.listdir(processed_data_dir) if f.endswith(".npy")]
    if existing_files:
        print(f"    Class_{class_num}: Already processed ({len(existing_files)} files)")
        return {"successful": 0, "failed": 0, "skipped": True}

    # Load metadata
    class_data = load_class_metadata(class_dir, video_code, class_num)
    if class_data is None:
        print(f"    Class_{class_num}: Missing metadata")
        return {"successful": 0, "failed": frames_per_class, "skipped": False}

    ppm_value = class_data["ppm"]
    background_path = class_backgrounds.get(class_num) if class_backgrounds else None

    # For single channel, background path is optional
    if channels == "double" and background_path is None:
        print(f"    Class_{class_num}: Missing background")
        return {"successful": 0, "failed": frames_per_class, "skipped": False}

    # print(f"    Class_{class_num}: Extracting {frames_per_class} frames...")

    successful = 0
    failed = 0

    for frame_idx in range(frames_per_class):
        random_frame = random.randint(start_frame, end_frame - 1)

        image_array = create_single_numpy_sample(
            cap,
            random_frame,
            background_path,
            ppm_value,
            processed_data_dir,
            frame_idx,
            channels,
        )

        if image_array is not None:
            # Save combined array
            combined_filename = (
                f"{video_code}_frame_{random_frame:02d}_class_{class_num}.npy"
            )
            combined_path = os.path.join(processed_data_dir, combined_filename)
            np.save(combined_path, image_array)

            successful += 1
        else:
            failed += 1

    # print(f"    Class_{class_num}: {successful} samples created")

    return {"successful": successful, "failed": failed, "skipped": False}


def process_single_video(
    video_path, video_code, video_dir, frames_per_class, channels="double"
):
    """
    Process a single video to create all numpy samples.

    Args:
        channels: "single" for frames only, "double" for background + plume

    Returns:
        dict: Results summary
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("  Error: Could not open video")
        return None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_minutes = total_frames / (fps * 60)

        if duration_minutes < config.VIDEO_MIN_DURATION_MINUTES:
            print(f"  Error: Video too short ({duration_minutes:.2f} min)")
            return None

        # print(f"  Video: {duration_minutes:.2f} min, {fps:.2f} FPS")

        # Load backgrounds (only needed for double channel)
        class_backgrounds = None
        if channels == "double":
            processed_dataset_path = os.path.dirname(video_dir)
            class_backgrounds = load_normal_backgrounds(
                processed_dataset_path, video_code
            )
            if len(class_backgrounds) < config.NUM_CLASSES:
                print("  Error: Missing backgrounds")
                return None

        total_successful = 0
        total_failed = 0

        # Process each class
        for class_num in range(config.NUM_CLASSES):
            start_frame = int(class_num * config.CLASS_DURATION_MINUTES * 60 * fps)
            end_frame = int((class_num + 1) * config.CLASS_DURATION_MINUTES * 60 * fps)

            result = process_single_class(
                cap,
                video_code,
                class_num,
                video_dir,
                class_backgrounds,
                frames_per_class,
                fps,
                start_frame,
                end_frame,
                channels,
            )

            total_successful += result["successful"]
            total_failed += result["failed"]

        return {
            "total": total_successful + total_failed,
            "successful": total_successful,
            "failed": total_failed,
        }

    finally:
        cap.release()
