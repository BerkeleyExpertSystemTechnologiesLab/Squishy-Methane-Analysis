import json
import os
import random

import config
import cv2
from core.artificial_background_handler import (
    copy_metadata_to_artif,
    generate_artificial_sample,
    load_artificial_backgrounds,
)
from core.background_generator import load_normal_backgrounds
from core.numpy_creator import process_single_video
from utils.directory import extract_video_code, get_directory_files


def count_dataset_samples(processed_dataset_path):
    """Count total samples in dataset."""
    total_samples = 0

    video_dirs = [
        d
        for d in os.listdir(processed_dataset_path)
        if os.path.isdir(os.path.join(processed_dataset_path, d))
        and d.isdigit()
        and len(d) == 4
    ]

    for video_code in video_dirs:
        video_dir = os.path.join(processed_dataset_path, video_code)
        for class_num in range(config.NUM_CLASSES):
            processed_data_dir = os.path.join(
                video_dir, f"Class_{class_num}", "processed_data"
            )
            if os.path.exists(processed_data_dir):
                numpy_files = [
                    f for f in os.listdir(processed_data_dir) if f.endswith(".npy")
                ]
                total_samples += len(numpy_files)

    return total_samples


def process_video_artificial(
    video_path, video_code, processed_dataset_path, frames_per_class
):
    """
    Process a single video to create artificial background numpy files.

    Args:
        video_path (str): Path to video file
        video_code (str): 4-digit video code
        processed_dataset_path (str): Path to processed dataset directory
        frames_per_class (int): Number of samples to create per class

    Returns:
        dict: Results with successful and failed counts, or None if error
    """

    # Load artificial backgrounds
    class_backgrounds_artif = load_artificial_backgrounds(
        processed_dataset_path, video_code
    )

    if not class_backgrounds_artif:
        print("  No artificial backgrounds found")
        return None

    # Load normal backgrounds (for subtraction)
    normal_backgrounds = load_normal_backgrounds(processed_dataset_path, video_code)

    if not normal_backgrounds:
        print("  No normal backgrounds found for subtraction")
        return None

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("  Could not open video")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)

    video_dir_artif = os.path.join(processed_dataset_path, f"{video_code}_ARTIF")
    total_successful = 0
    total_failed = 0

    for class_num in range(config.NUM_CLASSES):
        if (
            class_num not in class_backgrounds_artif
            or class_num not in normal_backgrounds
        ):
            continue

        class_dir_artif = os.path.join(video_dir_artif, f"Class_{class_num}")
        processed_data_dir = os.path.join(class_dir_artif, "processed_data")
        os.makedirs(processed_data_dir, exist_ok=True)

        # Copy metadata
        copy_metadata_to_artif(processed_dataset_path, video_code, class_num)

        # Load class metadata for PPM value
        video_dir_normal = os.path.join(processed_dataset_path, video_code)
        class_json = os.path.join(
            video_dir_normal,
            f"Class_{class_num}",
            f"{video_code}_class_{class_num}.json",
        )

        try:
            with open(class_json, "r") as f:
                class_data = json.load(f)
            ppm_value = class_data.get("ppm")
        except Exception:
            continue

        # Calculate time window
        start_frame = int(class_num * config.CLASS_DURATION_MINUTES * 60 * fps)
        end_frame = int((class_num + 1) * config.CLASS_DURATION_MINUTES * 60 * fps)

        # Generate samples
        for i in range(frames_per_class):
            random_frame = random.randint(start_frame, end_frame - 1)

            success = generate_artificial_sample(
                cap,
                video_code,
                class_num,
                random_frame,
                normal_backgrounds[class_num],
                class_backgrounds_artif[class_num],
                processed_data_dir,
                ppm_value,
            )

            if success:
                total_successful += 1
            else:
                total_failed += 1

    cap.release()

    return {"successful": total_successful, "failed": total_failed}


def run(test_videos=None, pipeline_config=None):
    """
    Main orchestrator for numpy dataset creation.

    Args:
        test_videos (list): Optional specific videos to process
        pipeline_config (PipelineConfig): Configuration object

    Returns:
        bool: True if successful
    """
    if pipeline_config is None:
        pipeline_config = config.PipelineConfig()

    frames_per_class = pipeline_config.frames_per_class

    print("\n" + "=" * 80)
    print("NUMPY DATASET CREATION PIPELINE")
    print("=" * 80)
    print(f"Target: {frames_per_class} frames per class per video")
    print(f"Channel mode: {pipeline_config.channels}")

    # Get video files
    video_files = get_directory_files(str(config.MOV_PATH), [".mp4", ".mov"])
    if video_files is None:
        print("ERROR: Failed to get video files")
        return False

    # Determine videos to process
    if test_videos:
        video_codes = test_videos
    else:
        video_codes = []
        for file_info in video_files["files"]:
            video_code = extract_video_code(file_info["filename"])
            if video_code:
                video_codes.append(video_code)

    print(f"Processing {len(video_codes)} videos")

    # Process each video
    total_successful = 0
    total_failed = 0

    for video_code in video_codes:
        print(f"\nProcessing video {video_code}...")

        # Find video file
        video_file = None
        for file_info in video_files["files"]:
            if video_code in file_info["filename"]:
                video_file = file_info["file_path"]
                break

        if not video_file:
            print("  Video file not found")
            continue

        video_dir = os.path.join(pipeline_config.processed_dataset_path, video_code)

        result = process_single_video(
            video_file,
            video_code,
            video_dir,
            frames_per_class,
            pipeline_config.channels,
        )

        if result:
            total_successful += result["successful"]
            total_failed += result["failed"]
            print(f"  {result['successful']}/{result['total']} samples created")

    # Summary
    print("\n" + "=" * 80)
    print("NUMPY DATASET CREATION COMPLETED!")
    print("=" * 80)
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_failed}")

    total_samples = count_dataset_samples(pipeline_config.processed_dataset_path)
    print(f"Total samples in dataset: {total_samples}")

    # Process artificial backgrounds if requested
    if pipeline_config.include_artificial:
        print("\n" + "=" * 80)
        print("PROCESSING ARTIFICIAL BACKGROUNDS")
        print("=" * 80)

        artif_successful = 0
        artif_failed = 0

        for video_code in video_codes:
            print(f"\nProcessing artificial backgrounds for video {video_code}...")

            # Find video file
            video_file = None
            for file_info in video_files["files"]:
                if video_code in file_info["filename"]:
                    video_file = file_info["file_path"]
                    break

            if not video_file:
                continue

            result = process_video_artificial(
                video_file,
                video_code,
                pipeline_config.processed_dataset_path,
                frames_per_class,
            )

            if result:
                artif_successful += result["successful"]
                artif_failed += result["failed"]
                print(f"  {result['successful']} artificial samples created")

        print(
            f"\nArtificial samples: {artif_successful} created, {artif_failed} failed"
        )
        total_successful += artif_successful
        total_failed += artif_failed

    # Success conditions:
    # 1. If we created new samples successfully
    # 2. If samples already exist (skipped processing)
    if total_successful > 0:
        return True
    elif total_successful == 0 and total_failed == 0 and total_samples > 0:
        print("\nAll numpy files already exist - dataset is ready!")
        return True
    else:
        return False


if __name__ == "__main__":
    success = run(frames_per_class=200)

    if success:
        print("\nReady for step 3: final dataset assembly")
    else:
        print("\nPipeline failed!")
