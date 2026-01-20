import json
import os
import shutil

import config


def create_directory_structure(final_dataset_path):
    """Create final dataset directory structure."""
    os.makedirs(final_dataset_path, exist_ok=True)
    os.makedirs(os.path.join(final_dataset_path, "data"), exist_ok=True)
    os.makedirs(os.path.join(final_dataset_path, "metadata"), exist_ok=True)

    for class_num in range(config.NUM_CLASSES):
        os.makedirs(
            os.path.join(final_dataset_path, "data", f"class_{class_num}"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(final_dataset_path, "metadata", f"class_{class_num}"),
            exist_ok=True,
        )

    print(f"Created directory structure: {final_dataset_path}")
    return final_dataset_path


def copy_files_for_video(
    video_code,
    processed_dataset_path,
    final_dataset_path,
    file_ext,
    dest_subdir,
    src_subdir=None,
):
    """
    Copy files of a specific type from one video to the final dataset.

    Args:
        video_code: 4-digit video code
        processed_dataset_path: Source base directory
        final_dataset_path: Destination base directory
        file_ext: File extension to copy (e.g., ".npy", ".json")
        dest_subdir: Destination subdirectory (e.g., "data", "metadata")
        src_subdir: Optional subdirectory within class folder (e.g., "processed_data")

    Returns:
        dict: Counts per class
    """
    video_dir = os.path.join(processed_dataset_path, video_code)
    dest_dir = os.path.join(final_dataset_path, dest_subdir)
    counts = {f"class_{i}": 0 for i in range(config.NUM_CLASSES)}

    for class_num in range(config.NUM_CLASSES):
        class_dir = os.path.join(video_dir, f"Class_{class_num}")
        source_dir = os.path.join(class_dir, src_subdir) if src_subdir else class_dir

        if not os.path.exists(source_dir):
            continue

        files = [f for f in os.listdir(source_dir) if f.endswith(file_ext)]

        for file_name in files:
            source = os.path.join(source_dir, file_name)
            target = os.path.join(dest_dir, f"class_{class_num}", file_name)
            try:
                shutil.copy2(source, target)
                counts[f"class_{class_num}"] += 1
            except Exception as e:
                print(f"    Error copying {file_name}: {e}")

    return counts


def copy_numpy_files_for_video(video_code, processed_dataset_path, final_dataset_path):
    """Copy numpy files from one video."""
    return copy_files_for_video(
        video_code,
        processed_dataset_path,
        final_dataset_path,
        file_ext=".npy",
        dest_subdir="data",
        src_subdir="processed_data",
    )


def copy_json_files_for_video(video_code, processed_dataset_path, final_dataset_path):
    """Copy JSON files from one video."""
    return copy_files_for_video(
        video_code,
        processed_dataset_path,
        final_dataset_path,
        file_ext=".json",
        dest_subdir="metadata",
    )


def create_dataset_info_file(final_dataset_path):
    """Create dataset_info.json with summary."""
    data_dir = os.path.join(final_dataset_path, "data")
    sample_counts = {}

    for class_num in range(config.NUM_CLASSES):
        class_dir = os.path.join(data_dir, f"class_{class_num}")
        if os.path.exists(class_dir):
            npy_files = [f for f in os.listdir(class_dir) if f.endswith(".npy")]
            sample_counts[f"class_{class_num}"] = len(npy_files)
        else:
            sample_counts[f"class_{class_num}"] = 0

    dataset_info = {
        "dataset_name": "Gas Leak Detection Dataset",
        "description": "Multi-modal dataset with gas leak images and metadata",
        "classes": config.NUM_CLASSES,
        "data_format": {
            "numpy_files": "2-channel arrays: [background, gas_leak_ppm]",
            "json_files": "Metadata with PPM, distance, leak_rate",
        },
        "sample_counts": sample_counts,
        "total_samples": sum(sample_counts.values()),
    }

    info_path = os.path.join(final_dataset_path, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Dataset info saved: {info_path}")
    print(f"Total samples: {dataset_info['total_samples']}")

    return dataset_info
