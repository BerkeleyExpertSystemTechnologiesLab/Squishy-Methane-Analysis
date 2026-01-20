import os

import config
from core.artificial_background_handler import (
    copy_artificial_backgrounds_to_processed,
)
from core.background_generator import generate_all_backgrounds
from core.frame_extract import extract_frames_for_all_videos
from core.metadata_handler import (
    load_consolidated_metadata,
    populate_metadata_from_consolidated,
)
from utils.directory import (
    create_class_directories,
    extract_video_code,
    get_directory_files,
)
from utils.excel import add_distance_descriptors, convert_excel_to_csv, load_excel_data
from utils.json_utils import add_leak_rates_to_classes, add_ppm_data_to_classes


def load_video_files():
    """Load video files and create code-to-file mapping."""
    video_files = get_directory_files(str(config.MOV_PATH), [".mp4", ".mov"])
    if video_files is None:
        print("ERROR: Failed to get video files")
        return None, None

    print(f"Found {video_files['total_files']} video files")

    video_code_to_file = {}
    for file_info in video_files["files"]:
        video_code = extract_video_code(file_info["filename"])
        if video_code:
            video_code_to_file[video_code] = file_info

    return video_files, video_code_to_file


def determine_videos_to_process(video_code_to_file, test_videos=None):
    """Determine which videos to process."""
    if test_videos:
        print(f"TEST MODE: Processing only specified videos: {test_videos}")
        video_codes = [code for code in test_videos if code in video_code_to_file]
    else:
        print("Processing all videos...")
        video_codes = list(video_code_to_file.keys())

    if not video_codes:
        print("ERROR: No valid video codes found")
        return None

    print(f"Found {len(video_codes)} videos to process")
    return video_codes


def setup_directories(video_codes, processed_dataset_path):
    """Create directory structure for all videos."""
    print("\n" + "=" * 50)
    print("CREATING DIRECTORY STRUCTURE")
    print("=" * 50)

    dir_result = create_class_directories(
        video_codes, num_classes=config.NUM_CLASSES, output_path=processed_dataset_path
    )
    if dir_result is None:
        print("ERROR: Failed to create directory structure")
        return False

    print(f"Successfully created {dir_result['total_created']} directories")
    if dir_result["total_failed"] > 0:
        print(f"Warning: {dir_result['total_failed']} directories failed")

    return True


def add_metadata_consolidated(video_codes, processed_dataset_path):
    """Add metadata using consolidated metadata file."""
    consolidated_metadata = load_consolidated_metadata(
        str(config.CONSOLIDATED_METADATA_PATH)
    )

    if consolidated_metadata is None:
        return False, None

    print("\nUsing consolidated metadata file")

    metadata_result = populate_metadata_from_consolidated(
        processed_dataset_path, consolidated_metadata, video_codes
    )

    if metadata_result["successful_updates"] == 0:
        print("ERROR: Failed to populate any metadata")
        return False, None

    print(
        f"\nMetadata populated: {metadata_result['successful_updates']} class JSON files"
    )
    return True, metadata_result


def add_metadata_individual_sources(step_3, step_4, step_5, processed_dataset_path):
    """Add metadata using individual Excel/CSV/JSON sources (fallback)."""
    print("\nUsing individual metadata sources...")

    if step_3:
        print("\nAdding distance descriptors...")
        distance_result = add_distance_descriptors(
            str(config.EXCEL_PATH), processed_dataset_path
        )
        if distance_result is None:
            print("ERROR: Failed to add distance descriptors")
            return False

    if step_4:
        print("\nAdding leak rates...")
        leak_rate_result = add_leak_rates_to_classes(
            processed_dataset_path, str(config.CLASSES_JSON_PATH)
        )
        if not leak_rate_result:
            print("Leak rate addition failed!")
            return False

    if step_5:
        print("\nAdding PPM data...")
        csv_path = str(config.PLUME_MODELING_PATH).replace(".xlsx", ".csv")
        if not os.path.exists(csv_path):
            print("Converting Excel to CSV...")
            if not convert_excel_to_csv(str(config.PLUME_MODELING_PATH), csv_path):
                print("ERROR: Failed to convert Excel to CSV")
                return False

        ppm_result = add_ppm_data_to_classes(
            processed_dataset_path, str(config.PLUME_MODELING_PATH)
        )
        if not ppm_result:
            print("PPM data addition failed!")
            return False

    return True


def run(
    test_videos=None,
    pipeline_config=None,
    step_1=True,
    step_2=True,
    step_3=True,
    step_4=True,
    step_5=True,
):
    """
    Main orchestrator for base dataset creation.

    Args:
        test_videos (list): Optional list of video codes to process
        pipeline_config (PipelineConfig): Configuration object
        step_1-5 (bool): Enable/disable specific steps

    Returns:
        bool: True if successful
    """
    if pipeline_config is None:
        pipeline_config = config.PipelineConfig()

    print("\n" + "=" * 80)
    print("BASE DATASET CREATION PIPELINE")
    print("=" * 80)
    print(f"Output directory: {pipeline_config.processed_dataset_path}")
    print("=" * 80)

    try:
        # Step 1: Load and setup
        if step_1:
            df = load_excel_data(str(config.EXCEL_PATH))
            if df is None:
                return False

            video_files, video_code_to_file = load_video_files()
            if video_files is None:
                return False

            video_codes = determine_videos_to_process(video_code_to_file, test_videos)
            if video_codes is None:
                return False

            if not setup_directories(
                video_codes, pipeline_config.processed_dataset_path
            ):
                return False

        # Step 2: Generate backgrounds and extract frames
        if step_2:
            # Only generate backgrounds for double channel mode
            if pipeline_config.channels == "double":
                bg_result = generate_all_backgrounds(
                    video_codes,
                    video_code_to_file,
                    pipeline_config.processed_dataset_path,
                )

                if bg_result["successful"] == 0:
                    print("ERROR: No backgrounds generated")
                    return False
            else:
                print("\nSKIPPING background generation (single channel mode)")

            frame_result = extract_frames_for_all_videos(
                video_codes,
                video_code_to_file,
                pipeline_config.processed_dataset_path,
                pipeline_config.channels,
            )

            if frame_result["successful"] == 0:
                print("ERROR: No frames extracted")
                return False

        # Steps 3-5: Add metadata
        if step_3 or step_4 or step_5:
            success, _ = add_metadata_consolidated(
                video_codes, pipeline_config.processed_dataset_path
            )

            if not success:
                # Fallback to individual sources
                if not add_metadata_individual_sources(
                    step_3, step_4, step_5, pipeline_config.processed_dataset_path
                ):
                    return False

        # Copy artificial backgrounds if requested
        if pipeline_config.include_artificial:
            print("\n" + "=" * 50)
            print("COPYING ARTIFICIAL BACKGROUNDS")
            print("=" * 50)

            artif_source = config.ARTIFICIAL_BACKGROUNDS_PATH

            result = copy_artificial_backgrounds_to_processed(
                artif_source, pipeline_config.processed_dataset_path
            )

            if result and result["copied"] > 0:
                print(f"\nCopied {result['copied']} artificial backgrounds")
            elif result and result["copied"] == 0:
                print("\nNo new artificial backgrounds to copy (already exist)")
            else:
                print("\nWarning: No artificial backgrounds found or copied")

        print("\n" + "=" * 80)
        print("BASE DATASET CREATION COMPLETED!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        return False


if __name__ == "__main__":
    test_mode = False
    test_video_codes = ["2583", "2581", "2580"]

    if test_mode:
        success = run(test_videos=test_video_codes)
    else:
        success = run()

    if success:
        print("\nReady for step 2: numpy dataset creation")
    else:
        print("\nPipeline failed!")
