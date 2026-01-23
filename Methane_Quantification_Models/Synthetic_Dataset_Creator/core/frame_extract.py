import json
import os
import random

import cv2
import numpy as np


def extract_random_frame(video_path, output_path, time_min=3, time_max=24):
    """
    Extract a random frame from a video within a specified time range.

    Args:
        video_path (str): Path to the video file
        output_path (str): Path to save the extracted frame
        time_min (float): Minimum time in minutes (default: 3)
        time_max (float): Maximum time in minutes (default: 24)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            # print(f"Error: Could not open video file {video_path}")
            return False

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_minutes = total_frames / (fps * 60)

        # Calculate frame range
        start_frame = int(time_min * 60 * fps)
        end_frame = int(min(time_max, duration_minutes) * 60 * fps)

        if start_frame >= end_frame:
            # print(f"Error: Invalid time range for video {video_path}")
            cap.release()
            return False

        # Select random frame
        random_frame = random.randint(start_frame, end_frame)

        # Extract the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        ret, frame = cap.read()

        cap.release()

        if not ret:
            # print(f"Error: Could not extract frame from {video_path}")
            return False

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save frame as both .npy and .png
        # np.save(output_path, frame)

        # Also save as PNG for visualization
        png_path = output_path.replace(".npy", ".png")
        cv2.imwrite(png_path, frame)

        # print(f"Random frame extracted successfully: {output_path}")
        # print(f"Frame extracted from {random_frame/fps/60:.2f} minutes")

        return True

    except Exception:
        # print(f"Error extracting random frame: {e}")
        return False


def subtract_background(background_path, frame_path, output_path):
    """
    Subtract background from a frame to isolate moving objects/gas leaks.

    Args:
        background_path (str): Path to the background image (.npy)
        frame_path (str): Path to the frame image (.npy)
        output_path (str): Path to save the result
        threshold (int): Threshold for background subtraction (default: 30)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load background and frame from PNG files
        background = cv2.imread(
            background_path.replace(".npy", ".png"), cv2.IMREAD_GRAYSCALE
        )
        frame = cv2.imread(frame_path.replace(".npy", ".png"), cv2.IMREAD_GRAYSCALE)

        # Ensure both images have the same shape
        if background.shape != frame.shape:
            # print(f"Error: Shape mismatch - background: {background.shape}, frame: {frame.shape}")
            return False

        # Subtract background (one-directional: only removes values, no addition)
        # cv2.subtract saturates at 0, preventing negative values
        # This ensures background additions don't create false plumes
        # https://www.geeksforgeeks.org/python/how-to-subtract-two-images-using-python-opencv/
        diff = cv2.subtract(frame, background)

        # Use raw difference for nuanced grayscale result
        mask = diff

        # Apply morphological operations to clean up the mask
        # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        # MORPH_CLOSE closes small holes in the foreground (Fills in small holes in objects)
        # MORPH_OPEN removes small objects from the foreground ro remove likely noise
        # small kernel looks at a 3x3 pixel neighborhood (9 pixels)
        # using 6x6 or 9x9 is too blocky and smoothes out too much
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save result as both .npy and .png
        # np.save(output_path, mask)

        # Also save as PNG for visualization
        png_path = output_path.replace(".npy", ".png")
        cv2.imwrite(png_path, mask)

        # print(f"Background subtraction completed: {output_path}")
        # print(f"Threshold used: {threshold}")

        return True

    except Exception:
        # print(f"Error in background subtraction: {e}")
        return False


def extract_class_frames(
    video_path, processed_dataset_path, video_number, channels="double"
):
    """
    Extract one random frame from each 3-minute interval and organize by leak class.

    Args:
        video_path (str): Path to the video file
        processed_dataset_path (str): Path to the Processed_Dataset directory
        video_number (str): 4-digit video number (e.g., "1237")
        channels (str): "single" or "double" channel mode

    Returns:
        dict: Summary of processed classes and their frame files
    """
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_minutes = total_frames / (fps * 60)

        # Check if video is at least 24 minutes
        if duration_minutes < 24:
            print(
                f"Error: Video {video_number} is only {duration_minutes:.2f} minutes long. Need at least 24 minutes."
            )
            cap.release()
            return None

        # print(
        #     f"Processing video {video_number}: {duration_minutes:.2f} minutes, {fps:.2f} FPS"
        # )

        # Video directory should already exist (created by directory_functions)
        video_dir = os.path.join(processed_dataset_path, video_number)

        # Load class-specific backgrounds only for double channel mode
        class_backgrounds = {}
        if channels == "double":
            # print("Loading class-specific background images...")

            # Look for class-specific background files for each class
            for class_num in range(8):
                class_dir = os.path.join(video_dir, f"Class_{class_num}")

                avg_background_file = os.path.join(
                    class_dir,
                    f"{video_number}_class_{class_num}_background_avg.png",
                )

                if os.path.exists(avg_background_file):
                    avg_bg = avg_background_file
                    # print(f"  Class_{class_num}: Found Average background")
                else:
                    avg_bg = None
                    # print(f"  Class_{class_num}: No Average background found")

                if avg_bg is None:
                    print(f"  Error: No backgrounds found for Class_{class_num}")
                    cap.release()
                    return None

                class_backgrounds[class_num] = {"avg": avg_bg}

            # print(
            #     f"Successfully loaded class-specific backgrounds for all {len(class_backgrounds)} classes"
            # )
        else:
            print("SKIPPING background loading (single channel mode)")

        # Process each class (0-7)
        processed_classes = {}
        class_metadata = {}

        for class_num in range(8):
            # Calculate time window for this class
            start_minutes = class_num * 3
            end_minutes = (class_num + 1) * 3

            # Calculate frame range
            start_frame = int(start_minutes * 60 * fps)
            end_frame = int(end_minutes * 60 * fps)

            # Randomly select one frame from this 3-minute window
            random_frame = random.randint(start_frame, end_frame - 1)

            # Extract the frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
            ret, frame = cap.read()

            if not ret:
                print(f"Error: Could not extract frame for class {class_num}")
                continue

            # Class directory should already exist (created by directory_functions)
            class_dir = os.path.join(video_dir, f"Class_{class_num}")

            # Save original frame
            frame_file = os.path.join(
                class_dir, f"{video_number}_class_{class_num}.jpg"
            )
            cv2.imwrite(frame_file, frame)

            # Round time_minutes to 3 decimal places
            time_minutes = round(random_frame / (fps * 60), 3)

            if channels == "double":
                # Apply background subtraction using class-specific backgrounds (both methods)
                avg_subtracted_file = os.path.join(
                    class_dir,
                    f"{video_number}_class_{class_num}_subtracted_avg.jpg",
                )

                # Use Average background for subtraction
                avg_background_file = class_backgrounds[class_num]["avg"]
                avg_success = subtract_background(
                    avg_background_file, frame_file, avg_subtracted_file
                )

                if avg_success:
                    processed_classes[class_num] = {
                        "frame_file": frame_file,
                        "avg_subtracted_file": avg_subtracted_file,
                        "time_minutes": random_frame / (fps * 60),
                        "frame_number": random_frame,
                    }

                    class_metadata[f"class_{class_num}"] = {
                        "class": class_num,
                        "time_minutes": time_minutes,
                        "frame_number": random_frame,
                    }
                    # print(
                    #     f"Processed Class {class_num}: {random_frame / (fps * 60):.2f} minutes"
                    # )
                else:
                    print(f"Failed to process Class {class_num}  Avg: {avg_success}")
            else:
                # Single channel mode - just save the frame, no background subtraction
                processed_classes[class_num] = {
                    "frame_file": frame_file,
                    "time_minutes": random_frame / (fps * 60),
                    "frame_number": random_frame,
                }

                class_metadata[f"class_{class_num}"] = {
                    "class": class_num,
                    "time_minutes": time_minutes,
                    "frame_number": random_frame,
                }
                print(
                    f"Processed Class {class_num}: {random_frame / (fps * 60):.2f} minutes (single channel)"
                )

        cap.release()

        # Create separate JSON files for each class
        for class_num, class_info in class_metadata.items():
            class_num_int = int(
                class_num.split("_")[1]
            )  # Extract number from "class_0"
            class_dir = os.path.join(video_dir, f"Class_{class_num_int}")

            # Create class-specific JSON file
            class_json_file = os.path.join(
                class_dir, f"{video_number}_class_{class_num_int}.json"
            )

            # Load existing data or create new dict
            existing_data = {}
            if os.path.exists(class_json_file):
                try:
                    with open(class_json_file, "r") as f:
                        existing_data = json.load(f)
                    # print(f" Appending to existing class JSON file: {class_json_file}")
                except (json.JSONDecodeError, FileNotFoundError):
                    print(
                        f" Warning: Could not read existing class JSON file {class_json_file}, creating new one"
                    )
                    existing_data = {}
            # else:
            #     print(f" Creating new class JSON file: {class_json_file}")

            # Add/update class metadata (preserving existing data like distance_m)
            class_data = {
                "video_number": video_number,
                "class": class_num_int,
                "time_minutes": class_info["time_minutes"],
                "frame_number": class_info["frame_number"],
            }

            # Merge new data with existing data
            existing_data.update(class_data)

            # Write updated class-specific JSON file
            with open(class_json_file, "w") as f:
                json.dump(existing_data, f, indent=2)

            # print(f" Updated class JSON file: {class_json_file}")

        # Also update the main video JSON file with just the distance info (if it exists)
        main_json_file = os.path.join(video_dir, f"{video_number}.json")
        if os.path.exists(main_json_file):
            try:
                with open(main_json_file, "r") as f:
                    existing_data = json.load(f)
                # print(f" Main JSON file already exists: {main_json_file}")
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"Warning: Could not read main JSON file {main_json_file}")
        else:
            print(f"Note: No main JSON file found at {main_json_file}")
        print(f"Processed {len(processed_classes)} classes for video {video_number}")

        return {
            "video_number": video_number,
            "processed_classes": processed_classes,
            "total_classes": len(processed_classes),
            "background_files": class_backgrounds,
            "metadata_file": main_json_file,
        }

    except Exception as e:
        print(f"Error processing video {video_number}: {e}")
        return None


def generate_average_background(
    video_path,
    output_path,
    use_entire_video=False,
    start_min=1,
    end_min=3,
    alpha=0.1,
    cap=None,
):
    """
    Generate a background image using running average approach from the entire video.
    This method creates a stable background by averaging pixel values over time.

    Args:
        video_path (str): Path to the video file (used for error messages if cap is None)
        output_path (str): Path to save the background image
        use_entire_video (bool): If True, use entire video; if False, use time range
        start_min (float): Start time in minutes (only used if use_entire_video=False)
        end_min (float): End time in minutes (only used if use_entire_video=False)
        alpha (float): Learning rate for running average (0.0-1.0, default: 0.1)
        cap (cv2.VideoCapture, optional): Pre-opened video capture object

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Use provided cap or open video file
        if cap is None:
            cap = cv2.VideoCapture(video_path)
            should_release = True
        else:
            should_release = False

        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return False

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # duration_minutes = total_frames / (fps * 60)

        # Get first frame to check format
        ret, test_frame = cap.read()
        # if ret:
        #     # print(
        #     #     f"  Video properties: FPS={fps:.2f}, Total frames={total_frames}, Duration={duration_minutes:.2f} minutes"
        #     # )
        #     # print(f"  Frame shape: {test_frame.shape}, dtype: {test_frame.dtype}")
        #     if len(test_frame.shape) == 3:
        #         print(f"  Frame channels: {test_frame.shape[2]} (BGR format)")
        #     else:
        #         print("  Frame is already grayscale")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning

        # Calculate frame range for sampling
        if use_entire_video:
            start_frame = 0
            end_frame = total_frames - 1
            # print(f"  Using entire video: frames 0-{end_frame}")
        else:
            start_frame = int(start_min * 60 * fps)
            end_frame = int(end_min * 60 * fps)
            # print(f"  Time range: {start_min}-{end_min} minutes")
            # print(f"  Frame range: {start_frame}-{end_frame}")

            # Ensure we don't exceed video length
            end_frame = min(end_frame, total_frames - 1)

            if start_frame >= end_frame:
                print(
                    f"Error: Invalid time range for video {video_path} - start_frame={start_frame}, end_frame={end_frame}"
                )
                cap.release()
                return False

        # Running average background generation
        # print(f"  Alpha (learning rate): {alpha}")

        # Initialize running average
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, first_frame = cap.read()

        if not ret:
            print(f"Error: Could not read first frame from {video_path}")
            if should_release:
                cap.release()
            return False

        # Convert first frame to grayscale and initialize running average
        if len(first_frame.shape) == 3:
            first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        else:
            first_frame_gray = first_frame

        # Initialize running average as float32 grayscale
        running_average = np.float32(first_frame_gray)

        frame_count = 0
        frames_processed = 1  # First frame already processed

        # Sample frames to avoid memory issues
        if use_entire_video:
            # Sample every 30th frame (2 FPS for 30 FPS video) to reduce memory usage
            sample_interval = max(1, int(fps // 2))  # Sample at ~2 FPS
            # print(f"  Sampling every {sample_interval} frames to reduce memory usage")
        else:
            sample_interval = 1  # Process every frame for time range

        # Process remaining frames
        while frame_count < (end_frame - start_frame):
            ret, frame = cap.read()

            if not ret:
                break

            # Only process sampled frames
            if frame_count % sample_interval == 0:
                # Convert to grayscale
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Update running average using cv2.accumulateWeighted
                cv2.accumulateWeighted(frame, running_average, alpha)
                frames_processed += 1

            frame_count += 1

        # Don't release here - let the caller handle video capture lifecycle

        if frames_processed == 0:
            print(f"Error: No frames processed from {video_path}")
            return False

        # print(f"  Processed {frames_processed} frames for running average")

        # Convert running average back to uint8
        background = cv2.convertScaleAbs(running_average)

        # print(
        #     f" Average background calculated successfully - shape: {background.shape}, dtype: {background.dtype}"
        # )

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        # print(f"  Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Save as PNG
        if output_path.endswith(".npy"):
            png_path = output_path.replace(".npy", ".png")
        elif output_path.endswith(".png"):
            png_path = output_path  # Already has .png extension
        else:
            png_path = output_path + ".png"

        # print(f"  Saving average background to: {png_path}")
        success = cv2.imwrite(png_path, background)

        if not success:
            print(f"Error: Failed to save average background image to {png_path}")
            return False

        # print(f"  average background generated successfully: {png_path}")
        # if use_entire_video:
        #     print(
        #         f"  Processed {frames_processed} frames from entire video ({duration_minutes:.2f} minutes)"
        #     )
        # else:
        #     print(
        #         f"  Processed {frames_processed} frames from {start_min}-{end_min} minutes"
        #     )

        return True

    except Exception as e:
        print(f"Error generating average background: {e}")
        if should_release:
            cap.release()
        return False


def combine_manual_backgrounds(
    background1_path, background2_path, output_path, blend_ratio=0.5
):
    """
    Combine two manually edited background images by converting them to numpy arrays,
    blending them together, and saving the result.

    Args:
        background1_path (str): Path to the first background JPG image
        background2_path (str): Path to the second background JPG image
        output_path (str): Path to save the combined background
        blend_ratio (float): Ratio for blending (0.0 = only background1, 1.0 = only background2, 0.5 = equal blend)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load both background images
        print(f"Loading background 1: {background1_path}")
        bg1 = cv2.imread(background1_path, cv2.IMREAD_GRAYSCALE)
        if bg1 is None:
            print(f"Error: Could not load background 1 from {background1_path}")
            return False

        print(f"Loading background 2: {background2_path}")
        bg2 = cv2.imread(background2_path, cv2.IMREAD_GRAYSCALE)
        if bg2 is None:
            print(f"Error: Could not load background 2 from {background2_path}")
            return False

        # Ensure both images have the same shape
        if bg1.shape != bg2.shape:
            print(
                f"Error: Shape mismatch - background1: {bg1.shape}, background2: {bg2.shape}"
            )
            return False

        # print(f"Both backgrounds loaded successfully - shape: {bg1.shape}")

        # Convert to numpy arrays (float32 for blending)
        bg1_array = bg1.astype(np.float32)
        bg2_array = bg2.astype(np.float32)

        # Blend the backgrounds
        print(f"Blending backgrounds with ratio {blend_ratio}")
        combined_array = (1 - blend_ratio) * bg1_array + blend_ratio * bg2_array

        # Convert back to uint8
        combined_background = combined_array.astype(np.uint8)

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Save as PNG
        if output_path.endswith(".npy"):
            png_path = output_path.replace(".npy", ".png")
        elif output_path.endswith(".png"):
            png_path = output_path  # Already has .png extension
        else:
            png_path = output_path + ".png"

        print(f"Saving combined background to: {png_path}")
        success = cv2.imwrite(png_path, combined_background)

        if not success:
            print(f"Error: Failed to save combined background to {png_path}")
            return False

        print(f"Combined background saved successfully: {png_path}")
        print(f"Blend ratio: {blend_ratio} (0.0 = only bg1, 1.0 = only bg2)")

        return True

    except Exception as e:
        print(f"Error combining backgrounds: {e}")
        return False


def extract_frames_for_all_videos(
    video_codes, video_code_to_file, processed_dataset_path, channels="double"
):
    """
    Extract class frames for all videos.

    Args:
        video_codes (list): List of video codes
        video_code_to_file (dict): Mapping of codes to file info
        processed_dataset_path (str): Output directory
        channels (str): "single" or "double" channel mode

    Returns:
        dict: Results with successful and failed counts
    """
    print("\n" + "=" * 50)
    print("EXTRACTING CLASS FRAMES")
    print("=" * 50)

    successful_videos = 0
    failed_videos = 0

    for i, video_code in enumerate(video_codes):
        file_info = video_code_to_file[video_code]
        video_path = file_info["file_path"]
        filename = file_info["filename"]

        print(
            f"\nProcessing video {i + 1}/{len(video_codes)}: {filename} (Code: {video_code})"
        )

        try:
            class_result = extract_class_frames(
                video_path=video_path,
                processed_dataset_path=processed_dataset_path,
                video_number=video_code,
                channels=channels,
            )

            if class_result and class_result.get("total_classes", 0) > 0:
                # print(f"Successfully processed {class_result['total_classes']} classes")
                successful_videos += 1
            else:
                # print(f"Failed to process video {video_code}")
                failed_videos += 1

        except Exception as e:
            print(f"Error processing video {video_code}: {e}")
            failed_videos += 1

    print("\nFrame extraction summary:")
    print(f"  Successful: {successful_videos}")
    print(f"  Failed: {failed_videos}")

    return {"successful": successful_videos, "failed": failed_videos}
