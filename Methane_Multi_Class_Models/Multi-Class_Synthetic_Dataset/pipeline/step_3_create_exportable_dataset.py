import os

import config
from core.dataset_finalizer import (
    copy_json_files_for_video,
    copy_numpy_files_for_video,
    create_dataset_info_file,
    create_directory_structure,
)


def get_video_directories(processed_dataset_path, pipeline_config=None):
    """
    Get list of video directories.

    Args:
        processed_dataset_path (str): Path to processed dataset
        pipeline_config (PipelineConfig): Configuration object

    Returns:
        list: Sorted list of video directory names
    """
    if pipeline_config is None:
        pipeline_config = config.PipelineConfig()

    if not os.path.exists(processed_dataset_path):
        print(f"Error: Processed dataset not found: {processed_dataset_path}")
        return []

    video_dirs = [
        d
        for d in os.listdir(processed_dataset_path)
        if os.path.isdir(os.path.join(processed_dataset_path, d))
        and d.isdigit()
        and len(d) == 4
    ]

    # Add ARTIF directories if requested
    if pipeline_config.include_artificial:
        artif_dirs = [
            d
            for d in os.listdir(processed_dataset_path)
            if os.path.isdir(os.path.join(processed_dataset_path, d))
            and d.endswith("_ARTIF")
        ]
        video_dirs.extend(artif_dirs)

    return sorted(video_dirs)


def run(processed_dataset_path=None, final_dataset_path=None, pipeline_config=None):
    """
    Main orchestrator for final dataset assembly.

    Args:
        processed_dataset_path (str): Source directory
        final_dataset_path (str): Target directory
        pipeline_config (PipelineConfig): Configuration object

    Returns:
        bool: True if successful
    """
    if pipeline_config is None:
        pipeline_config = config.PipelineConfig()

    if processed_dataset_path is None:
        processed_dataset_path = pipeline_config.processed_dataset_path

    if final_dataset_path is None:
        final_dataset_path = pipeline_config.final_dataset_path

    print("\n" + "=" * 80)
    print("FINAL DATASET ASSEMBLY PIPELINE")
    print("=" * 80)
    print(f"Source: {processed_dataset_path}")
    print(f"Destination: {final_dataset_path}")
    print("=" * 80)

    # Create structure
    final_dataset_path = create_directory_structure(final_dataset_path)

    # Get video directories
    video_dirs = get_video_directories(processed_dataset_path, pipeline_config)
    if not video_dirs:
        print("ERROR: No video directories found")
        return False

    print(f"Found {len(video_dirs)} video directories")
    if pipeline_config.include_artificial:
        artif_count = sum(1 for d in video_dirs if d.endswith("_ARTIF"))
        print(f"  Including {artif_count} artificial background directories")

    # Copy numpy files
    print("\nCopying numpy files...")
    numpy_total = {f"class_{i}": 0 for i in range(config.NUM_CLASSES)}

    for video_code in video_dirs:
        print(f"Processing {video_code}...")
        counts = copy_numpy_files_for_video(
            video_code, processed_dataset_path, final_dataset_path
        )
        for key in counts:
            numpy_total[key] += counts[key]

    print("\nNumpy files copied:")
    for class_name, count in numpy_total.items():
        print(f"  {class_name}: {count} samples")

    # Copy JSON files
    print("\nCopying JSON files...")
    json_total = {f"class_{i}": 0 for i in range(config.NUM_CLASSES)}

    for video_code in video_dirs:
        counts = copy_json_files_for_video(
            video_code, processed_dataset_path, final_dataset_path
        )
        for key in counts:
            json_total[key] += counts[key]

    print("\nJSON files copied:")
    for class_name, count in json_total.items():
        print(f"  {class_name}: {count} samples")

    # Create dataset info
    dataset_info = create_dataset_info_file(final_dataset_path)

    print("\n" + "=" * 80)
    print("FINAL DATASET ASSEMBLY COMPLETED!")
    print("=" * 80)
    print(f"Final dataset: {final_dataset_path}")
    print(f"Total samples: {dataset_info['total_samples']}")

    return True


if __name__ == "__main__":
    success = run()

    if success:
        print("\nDataset ready for use with PyTorch DataLoaders!")
    else:
        print("\nPipeline failed!")
