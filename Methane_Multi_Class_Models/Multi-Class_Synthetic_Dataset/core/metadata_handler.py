import json
import os


def load_consolidated_metadata(consolidated_metadata_path):
    """
    Load consolidated metadata from JSON file.

    Args:
        consolidated_metadata_path (str): Path to consolidated metadata JSON file

    Returns:
        dict: Consolidated metadata dictionary, or None if failed
    """
    try:
        if not os.path.exists(consolidated_metadata_path):
            return None

        with open(consolidated_metadata_path, "r") as f:
            metadata = json.load(f)

        print(
            f"Successfully loaded consolidated metadata from {consolidated_metadata_path}"
        )
        print(f"  Found metadata for {len(metadata)} videos")
        return metadata

    except Exception as e:
        print(f"Error loading consolidated metadata: {e}")
        return None


def write_class_json_file(class_dir, video_code, class_num, class_metadata):
    """
    Write metadata to individual class JSON file.

    Args:
        class_dir (str): Path to class directory
        video_code (str): 4-digit video code
        class_num (int): Class number (0-7)
        class_metadata (dict): Metadata dictionary for this class

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        class_json_file = os.path.join(
            class_dir, f"{video_code}_class_{class_num}.json"
        )

        with open(class_json_file, "w") as f:
            json.dump(class_metadata, f, indent=2)

        return True

    except Exception as e:
        print(f"    Error writing JSON file for Class_{class_num}: {e}")
        return False


def populate_metadata_from_consolidated(
    processed_dataset_path, consolidated_metadata, video_codes
):
    """
    Populate individual class JSON files from consolidated metadata.

    Args:
        processed_dataset_path (str): Path to processed dataset directory
        consolidated_metadata (dict): Consolidated metadata dictionary
        video_codes (list): List of video codes to process

    Returns:
        dict: Summary of results
    """
    print("\nPopulating class JSON files from consolidated metadata...")

    total_videos = 0
    total_classes = 0
    successful_updates = 0
    failed_updates = 0
    missing_videos = []

    for video_code in video_codes:
        # Check if video exists in consolidated metadata
        if video_code not in consolidated_metadata:
            print(f"  Warning: Video {video_code} not found in consolidated metadata")
            missing_videos.append(video_code)
            continue

        video_dir = os.path.join(processed_dataset_path, video_code)
        if not os.path.exists(video_dir):
            print(f"  Warning: Video directory not found: {video_dir}")
            continue

        total_videos += 1
        video_metadata = consolidated_metadata[video_code]

        # Process each class (0-7)
        for class_num in range(8):
            class_key = f"class_{class_num}"

            if class_key not in video_metadata:
                print(
                    f"  Warning: {video_code} - Class_{class_num} not found in metadata"
                )
                failed_updates += 1
                continue

            class_dir = os.path.join(video_dir, f"Class_{class_num}")
            if not os.path.exists(class_dir):
                print(f"  Warning: Class directory not found: {class_dir}")
                failed_updates += 1
                continue

            total_classes += 1
            class_metadata = video_metadata[class_key]

            # Write class JSON file
            success = write_class_json_file(
                class_dir, video_code, class_num, class_metadata
            )

            if success:
                successful_updates += 1
            else:
                failed_updates += 1

    print("\nConsolidated metadata population completed:")
    print(f"  Videos processed: {total_videos}")
    print(f"  Classes processed: {total_classes}")
    print(f"  Successful updates: {successful_updates}")
    print(f"  Failed updates: {failed_updates}")
    if missing_videos:
        print(f"  Videos not found in metadata: {missing_videos}")

    return {
        "total_videos": total_videos,
        "total_classes": total_classes,
        "successful_updates": successful_updates,
        "failed_updates": failed_updates,
        "missing_videos": missing_videos,
    }
