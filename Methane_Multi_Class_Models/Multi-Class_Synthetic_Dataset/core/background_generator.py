import os

import config
import cv2

from core.frame_extract import (
    generate_average_background,
)


def generate_backgrounds_for_video(video_path, video_code, video_dir, cap=None):
    """
    Generate class-specific backgrounds for a single video.

    Args:
        video_path (str): Path to video file
        video_code (str): 4-digit video code
        video_dir (str): Output directory for this video
        cap: Optional existing VideoCapture object

    Returns:
        dict: Results summary with success/failure counts
    """
    should_release = False
    if cap is None:
        cap = cv2.VideoCapture(video_path)
        should_release = True

        if not cap.isOpened():
            print(f"  Error: Could not open video file {video_path}")
            return {"successful": 0, "failed": config.NUM_CLASSES}

    successful = 0
    failed = 0

    try:
        os.makedirs(video_dir, exist_ok=True)

        for class_num in range(config.NUM_CLASSES):
            class_dir = os.path.join(video_dir, f"Class_{class_num}")
            os.makedirs(class_dir, exist_ok=True)

            start_min = class_num * config.CLASS_DURATION_MINUTES
            end_min = (class_num + 1) * config.CLASS_DURATION_MINUTES

            # print(
            #     f"  Class_{class_num}: Generating backgrounds from {start_min:.1f}-{end_min:.1f} min"
            # )

            # Generate Average background
            avg_background_path = os.path.join(
                class_dir, f"{video_code}_class_{class_num}_background_avg.png"
            )
            avg_success = generate_average_background(
                video_path=video_path,
                output_path=avg_background_path,
                use_entire_video=False,
                start_min=start_min,
                end_min=end_min,
                alpha=config.BACKGROUND_ALPHA,
                cap=cap,
            )

            if avg_success:
                # print("    Avg background saved")
                successful += 1
            else:
                print("    Avg background FAILED")
                failed += 1

    finally:
        if should_release:
            cap.release()

    return {"successful": successful, "failed": failed}


def generate_all_backgrounds(video_codes, video_code_to_file, processed_dataset_path):
    """
    Generate backgrounds for all videos.

    Args:
        video_codes (list): List of video codes to process
        video_code_to_file (dict): Mapping of video codes to file info
        processed_dataset_path (str): Base output directory

    Returns:
        dict: Overall results summary
    """
    print("\n" + "=" * 50)
    print("GENERATING CLASS-SPECIFIC BACKGROUND IMAGES")
    print("=" * 50)

    total_successful = 0
    total_failed = 0

    for i, video_code in enumerate(video_codes):
        file_info = video_code_to_file[video_code]
        video_path = file_info["file_path"]
        # filename = file_info["filename"]

        # print(f"\nVideo {i + 1}/{len(video_codes)}: {filename} (Code: {video_code})")

        video_dir = os.path.join(processed_dataset_path, video_code)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("  Error: Could not open video")
            total_failed += config.NUM_CLASSES
            continue

        try:
            result = generate_backgrounds_for_video(
                video_path, video_code, video_dir, cap
            )
            total_successful += result["successful"]
            total_failed += result["failed"]
        finally:
            cap.release()

    print("\nBackground generation completed:")
    print(f"  Successful: {total_successful}")
    print(f"  Failed: {total_failed}")

    return {"successful": total_successful, "failed": total_failed}


def load_normal_backgrounds(processed_dataset_path, video_code):
    """
    Load average backgrounds for subtraction.

    Args:
        processed_dataset_path (str): Path to processed dataset directory
        video_code (str): 4-digit video code

    Returns:
        dict: Dictionary mapping class_num to normal background path
    """
    normal_backgrounds = {}
    video_dir = os.path.join(processed_dataset_path, video_code)

    for class_num in range(config.NUM_CLASSES):
        class_dir = os.path.join(video_dir, f"Class_{class_num}")
        avg_bg = os.path.join(
            class_dir, f"{video_code}_class_{class_num}_background_avg.png"
        )

        if os.path.exists(avg_bg):
            normal_backgrounds[class_num] = avg_bg

    return normal_backgrounds
