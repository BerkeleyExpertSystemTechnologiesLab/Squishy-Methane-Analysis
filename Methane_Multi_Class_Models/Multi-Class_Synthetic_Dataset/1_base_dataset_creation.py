from excel_functions import load_excel_data, add_distance_descriptors, convert_excel_to_csv
from directory_functions import get_directory_files, extract_video_code, create_class_directories
from frame_extract import extract_class_frames, generate_cv2_background, generate_moving_average_background
from image_scaling import jpg_to_numpy, scale_jpg_to_ppm, numpy_to_jpg
from json_functions import add_leak_rates_to_classes, add_ppm_data_to_classes

import re
import os
import json
import pandas as pd
import numpy as np


# =============================================================================
# PATH CONSTANTS
# =============================================================================

EXCEL_PATH = "Original_Dataset/GasVid Logging File.xlsx"
MOV_PATH = "Original_Dataset/Videos-20251002T175522Z-1-001/Videos"
PROCESSED_DATASET_PATH = "Processed_Dataset"
CLASSES_JSON_PATH = "classes.json"
PLUME_MODELING_PATH = "Plume_Modeling/Gasvid Plume Models.xlsx"
CONSOLIDATED_METADATA_PATH = "Metadata/consolidated_metadata.json"


# =============================================================================
# CONSOLIDATED METADATA FUNCTIONS
# =============================================================================

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
            
        with open(consolidated_metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Successfully loaded consolidated metadata from {consolidated_metadata_path}")
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
        class_json_file = os.path.join(class_dir, f"{video_code}_class_{class_num}.json")
        
        with open(class_json_file, 'w') as f:
            json.dump(class_metadata, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"    Error writing JSON file for Class_{class_num}: {e}")
        return False


def populate_metadata_from_consolidated(processed_dataset_path, consolidated_metadata, video_codes):
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
                print(f"  Warning: {video_code} - Class_{class_num} not found in metadata")
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
            success = write_class_json_file(class_dir, video_code, class_num, class_metadata)
            
            if success:
                successful_updates += 1
            else:
                failed_updates += 1
    
    print(f"\nConsolidated metadata population completed:")
    print(f"  Videos processed: {total_videos}")
    print(f"  Classes processed: {total_classes}")
    print(f"  Successful updates: {successful_updates}")
    print(f"  Failed updates: {failed_updates}")
    if missing_videos:
        print(f"  Videos not found in metadata: {missing_videos}")
    
    return {
        'total_videos': total_videos,
        'total_classes': total_classes,
        'successful_updates': successful_updates,
        'failed_updates': failed_updates,
        'missing_videos': missing_videos
    }


# =============================================================================
# COMPLETE DATASET CREATION PIPELINE
# =============================================================================

def create_dataset_from_scratch(test_videos=None, step_1=True, step_2=True, step_3=True, step_4=True, step_5=True):
    """
    Create a complete dataset from scratch by running all processing steps.
    
    Args:
        test_videos (list, optional): List of 4-digit video codes to process. 
                                    If None, processes all videos found.
        step_1 (bool): Load Excel data and video files 
        step_2 (bool): Extract class frames and generate backgrounds 
        step_3 (bool): Add distance descriptors 
        step_4 (bool): Add leak rates 
        step_5 (bool): Add PPM data 
    """
    print("\n" + "="*80)
    print("STARTING COMPLETE DATASET CREATION PIPELINE")
    print("="*80)
    
    try:
        ##########################################################################
        # Step 1: Load Excel data, get video files, and create directory structure
        ##########################################################################
        if step_1:
            print("\n" + "="*50)
            print("STEP 1: LOADING EXCEL DATA AND VIDEO FILES")
            print("="*50)
            
            df = load_excel_data(EXCEL_PATH)
            if df is None:
                print("ERROR: Failed to load Excel data")
                return False
            print("Excel data loaded successfully")
            
            video_files = get_directory_files(MOV_PATH, ['.mp4', '.mov'])
            if video_files is None:
                print("ERROR: Failed to get video files")
                return False
            print(f"Found {video_files['total_files']} video files")
            
            # Create mapping of video codes to file info (extract once, use many times)
            video_code_to_file = {}
            for file_info in video_files['files']:
                video_code = extract_video_code(file_info['filename'])
                if video_code:
                    video_code_to_file[video_code] = file_info
            
            # Determine which videos to process
            if test_videos:
                print(f"TEST MODE: Processing only specified videos: {test_videos}")
                video_codes = [code for code in test_videos if code in video_code_to_file]
            else:
                print("Processing all videos...")
                video_codes = list(video_code_to_file.keys())
            
            if not video_codes:
                print("ERROR: No valid video codes found")
                return False
            
            print(f"Found {len(video_codes)} videos to process")
            
            # Create directories for all videos
            print("\n" + "="*50)
            print("STEP 1.5: CREATING DIRECTORY STRUCTURE")
            print("="*50)
            
            dir_result = create_class_directories(video_codes, num_classes=8)
            if dir_result is None:
                print("ERROR: Failed to create directory structure")
                return False
            
            print(f"Successfully created {dir_result['total_created']} directories")
            if dir_result['total_failed'] > 0:
                print(f"Warning: {dir_result['total_failed']} directories failed to create")
        
        ##########################################################################
        # Step 2: Process each video file to extract class frames
        ##########################################################################
        if step_2:
            # Generate class-specific background images
            print("\n" + "="*50)
            print("STEP 2: GENERATING CLASS-SPECIFIC BACKGROUND IMAGES")
            print("="*50)
            
            for i, video_code in enumerate(video_codes):
                file_info = video_code_to_file[video_code]
                video_path = file_info['file_path']
                filename = file_info['filename']
                
                print(f"\nGenerating class-specific backgrounds for video {i+1}/{len(video_codes)}: {filename} (Code: {video_code})")
                print(f"  Each class will get a background from OTHER classes' time windows...")
                
                # Open video once for all background generation
                import cv2
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"  Error: Could not open video file {video_path}")
                    continue
                
                try:
                    # Create video-level output directory
                    video_dir = os.path.join(PROCESSED_DATASET_PATH, video_code)
                    os.makedirs(video_dir, exist_ok=True)
                    
                    # Generate background for EACH class (8 classes)
                    for class_num in range(8):
                        class_dir = os.path.join(video_dir, f"Class_{class_num}")
                        os.makedirs(class_dir, exist_ok=True)
                        
                        # Use frames from WITHIN this class's time window
                        # The gas plumes are transient, so averaging will reduce them
                        # while preserving the static background scene
                        start_min = class_num * 3
                        end_min = (class_num + 1) * 3
                        print(f"  Class_{class_num}: Using frames {start_min:.1f}-{end_min:.1f} min (within class)")
                        
                        # Generate CV2 background for this class
                        cv2_background_path = os.path.join(class_dir, f"{video_code}_class_{class_num}_background_cv2.png")
                        cv2_success = generate_cv2_background(
                            video_path=video_path,
                            output_path=cv2_background_path,
                            use_entire_video=False,  # Use specific time range
                            start_min=start_min,
                            end_min=end_min,
                            cap=cap
                        )
                        
                        if cv2_success:
                            print(f"    Class_{class_num} CV2 background saved")
                        else:
                            print(f"    Class_{class_num} CV2 background FAILED")
                        
                        # Generate Moving Average background for this class
                        moving_avg_background_path = os.path.join(class_dir, f"{video_code}_class_{class_num}_background_moving_avg.png")
                        moving_avg_success = generate_moving_average_background(
                            video_path=video_path,
                            output_path=moving_avg_background_path,
                            use_entire_video=False,  # Use specific time range
                            start_min=start_min,
                            end_min=end_min,
                            alpha=0.1,
                            cap=cap
                        )
                        
                        if moving_avg_success:
                            print(f"    Class_{class_num} Moving Avg background saved")
                        else:
                            print(f"    Class_{class_num} Moving Avg background FAILED")
                
                finally:
                    # Always release the video capture
                    cap.release()
            
            print(f"\nClass-specific background generation completed for {len(video_codes)} videos")
            print("Background files saved in class directories (16 backgrounds per video)")
            
            # Now extract class frames using the generated backgrounds
            print("\n" + "="*50)
            print("STEP 2.5: EXTRACTING CLASS FRAMES USING GENERATED BACKGROUNDS")
            print("="*50)
            
            successful_videos = 0
            failed_videos = 0
            
            # Process videos using the mapping we already created
            for i, video_code in enumerate(video_codes):
                file_info = video_code_to_file[video_code]
                video_path = file_info['file_path']
                filename = file_info['filename']
                
                print(f"\nProcessing video {i+1}/{len(video_codes)}: {filename} (Code: {video_code})")
                
                try:
                    # Extract class frames for this video
                    class_result = extract_class_frames(
                        video_path=video_path,
                        processed_dataset_path=PROCESSED_DATASET_PATH,
                        video_number=video_code
                    )
                    
                    if class_result and class_result.get('total_classes', 0) > 0:
                        print(f"Successfully processed {class_result['total_classes']} classes for video {video_code}")
                        successful_videos += 1
                    else:
                        print(f"Failed to process video {video_code}")
                        failed_videos += 1
                        
                except Exception as e:
                    print(f"Error processing video {video_code}: {e}")
                    failed_videos += 1
            
            print(f"\nClass frame extraction summary:")
            print(f"  Successful: {successful_videos}")
            print(f"  Failed: {failed_videos}")
            
            if successful_videos == 0:
                print("ERROR: No videos were processed successfully")
                return False
            
        
        ##################################################################
        # Step 3-5: Add metadata to class JSON files
        # Check for consolidated metadata first, otherwise use individual sources
        ##################################################################
        used_consolidated_metadata = False
        if step_3 or step_4 or step_5:
            print("\n" + "="*50)
            print("STEP 3-5: ADDING METADATA TO CLASS FILES")
            print("="*50)
            
            # Try to load consolidated metadata
            consolidated_metadata = load_consolidated_metadata(CONSOLIDATED_METADATA_PATH)
            
            if consolidated_metadata is not None:
                used_consolidated_metadata = True
                # USE CONSOLIDATED METADATA
                print("\nUsing consolidated metadata file (skipping individual Excel/CSV/JSON sources)")
                print("="*50)
                
                metadata_result = populate_metadata_from_consolidated(
                    PROCESSED_DATASET_PATH,
                    consolidated_metadata,
                    video_codes
                )
                
                if metadata_result['successful_updates'] == 0:
                    print("ERROR: Failed to populate any metadata from consolidated file")
                    return False
                
                print(f"\nMetadata successfully populated from consolidated file!")
                print(f"  {metadata_result['successful_updates']} class JSON files created/updated")
                
            else:
                # FALLBACK TO ORIGINAL METHOD (Individual sources)
                print("\nConsolidated metadata not found, using individual sources...")
                print("="*50)
                
                # Step 3: Add distance descriptors
                if step_3:
                    print("\n" + "-"*40)
                    print("STEP 3: ADDING DISTANCE DESCRIPTORS")
                    print("-"*40)
                    
                    print("Adding distance descriptors to class JSON files...")
                    distance_result = add_distance_descriptors(EXCEL_PATH, PROCESSED_DATASET_PATH)
                    if distance_result is None:
                        print("ERROR: Failed to add distance descriptors")
                        return False
                    print(f"Distance descriptors added for {len(distance_result)} videos")
                
                # Step 4: Add leak rates
                if step_4:
                    print("\n" + "-"*40)
                    print("STEP 4: ADDING LEAK RATES")
                    print("-"*40)
                    
                    print("Adding leak rates to class JSON files...")
                    try:
                        leak_rate_result = add_leak_rates_to_classes(
                            PROCESSED_DATASET_PATH,
                            CLASSES_JSON_PATH
                        )
                        
                        if leak_rate_result:
                            print(f"Leak rate addition completed successfully!")
                            print(f"  Processed {leak_rate_result['total_videos']} videos and {leak_rate_result['total_classes']} classes")
                            print(f"  Successful updates: {leak_rate_result['successful_updates']}")
                            print(f"  Failed updates: {leak_rate_result['failed_updates']}")
                        else:
                            print("Leak rate addition failed!")
                            return False
                            
                    except Exception as e:
                        print(f"Error adding leak rates: {e}")
                        return False
                
                # Step 5: Add PPM data
                if step_5:
                    print("\n" + "-"*40)
                    print("STEP 5: ADDING PPM DATA")
                    print("-"*40)
                    
                    # Check if CSV file exists, convert if needed
                    csv_path = PLUME_MODELING_PATH.replace('.xlsx', '.csv')
                    if not os.path.exists(csv_path):
                        print("CSV file not found, converting Excel to CSV...")
                        if not convert_excel_to_csv(PLUME_MODELING_PATH, csv_path):
                            print("ERROR: Failed to convert Excel to CSV")
                            return False
                        print("CSV conversion completed successfully!")
                    else:
                        print(f"CSV file already exists: {csv_path}")
                    
                    print("Adding PPM data to class JSON files...")
                    print("DEBUG: This step should add PPM values to individual class JSON files")
                    try:
                        ppm_result = add_ppm_data_to_classes(
                            PROCESSED_DATASET_PATH,
                            PLUME_MODELING_PATH
                        )
                        
                        if ppm_result:
                            print(f"PPM data addition completed successfully!")
                            print(f"  Processed {ppm_result['total_videos']} videos and {ppm_result['total_classes']} classes")
                            print(f"  Successful updates: {ppm_result['successful_updates']}")
                            print(f"  Failed updates: {ppm_result['failed_updates']}")
                            print(f"  No matches found: {ppm_result['no_match_found']}")
                        else:
                            print("PPM data addition failed!")
                            return False
                            
                    except Exception as e:
                        print(f"Error adding PPM data: {e}")
                        return False

        ##################################################################
        # Step 6: Create example scaled versions of subtracted class images
        ##################################################################
        if step_6:
        
                # Create example scaled versions of subtracted class images (after PPM data is added)
                print("\n" + "="*50)
                print("STEP 6: CREATING EXAMPLE SCALED SUBTRACTED IMAGES")
                print("="*50)
                
                for i, video_code in enumerate(video_codes):
                    video_dir = os.path.join(PROCESSED_DATASET_PATH, video_code)
                    print(f"\nCreating scaled examples for video {i+1}/{len(video_codes)}: {video_code}")
                    
                    # Import scaling function
                    from image_scaling import scale_jpg_to_ppm
                    import numpy as np
                    import json
                    import cv2
                    
                    # Process each class (0-7)
                    for class_num in range(8):
                        class_dir = os.path.join(video_dir, f"Class_{class_num}")
                        
                        if not os.path.exists(class_dir):
                            print(f"  Class_{class_num}: Directory not found, skipping...")
                            continue
                        
                        # Look for subtracted images in this class directory
                        subtracted_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg') and 'subtracted' in f]
                        
                        if not subtracted_files:
                            print(f"  Class_{class_num}: No subtracted images found, skipping...")
                            continue
                        
                        # Load class metadata to get PPM value
                        class_json_file = os.path.join(class_dir, f"{video_code}_class_{class_num}.json")
                        if not os.path.exists(class_json_file):
                            print(f"  Class_{class_num}: No JSON file found, skipping...")
                            continue
                        
                        try:
                            with open(class_json_file, 'r') as f:
                                class_data = json.load(f)
                            
                            ppm_value = class_data.get('ppm')
                            if ppm_value is None:
                                print(f"  Class_{class_num}: No PPM value found, skipping...")
                                continue
                            
                            print(f"  Class_{class_num}: Found {len(subtracted_files)} subtracted images, PPM={ppm_value}")
                            
                            # Scale the first few subtracted images as examples
                            examples_to_scale = min(3, len(subtracted_files))  # Scale up to 3 examples
                            
                            for idx, subtracted_file in enumerate(subtracted_files[:examples_to_scale]):
                                subtracted_path = os.path.join(class_dir, subtracted_file)
                                
                                # Create scaled version filename
                                base_name = subtracted_file.replace('.jpg', '')
                                scaled_filename = f"{base_name}_scaled.jpg"
                                scaled_path = os.path.join(class_dir, scaled_filename)
                                
                                # Scale the subtracted image to PPM values
                                scaled_array = scale_jpg_to_ppm(
                                    jpg_path=subtracted_path,
                                    ppm_value=ppm_value,
                                    output_path=None,  # Don't save as numpy, we'll save as JPG manually
                                    grayscale=True
                                )
                                
                                if scaled_array is not None:
                                    # Convert scaled array back to JPG format for human viewing
                                    # Map PPM values (0-ppm_value) to display range (0-255)
                                    # This preserves the relative brightness relationships
                                    scaled_display = ((scaled_array / ppm_value) * 255).astype(np.uint8)
                                    cv2.imwrite(scaled_path, scaled_display)
                                    print(f"    {subtracted_file} -> {scaled_filename} (PPM: 0-{ppm_value:.2f} -> Display: 0-255)")
                                else:
                                    print(f"    Failed to scale {subtracted_file}")
                        
                        except Exception as e:
                            print(f"  Class_{class_num}: Error processing - {e}")
                            import traceback
                            print(f"  Full error: {traceback.format_exc()}")
                            continue
                
                print(f"\nExample scaled subtracted image generation completed for {len(video_codes)} videos")
                print("Scaled files saved with '_scaled' suffix in class directories")
        
        # Final summary
        print("\n" + "="*80)
        print("DATASET CREATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Processed {successful_videos} videos")
        print(f"Extracted class frames")
        if used_consolidated_metadata:
            print(f"Added metadata from consolidated_metadata.json")
        else:
            print(f"Added metadata from individual Excel/CSV/JSON sources")
        print("\nDataset base is ready for numpy dataset generation")
        
        return True
        
    except Exception as e:
        print(f"\nCRITICAL ERROR in dataset creation: {e}")
        print("Dataset creation failed!")
        return False


# Run the complete dataset creation pipeline
if __name__ == "__main__":
    # TEST MODE: Set to True to process only specific videos
    test_mode = False
    test_video_codes = ["2583", "2581", "2580"]  # Add your 4-digit video codes here
    
    # STEP CONTROL: Set to False to skip specific steps
    step_1 = True   # Load Excel data and video files
    step_2 = True  # Extract class frames and generate backgrounds
    step_3 = True   # Add distance descriptors
    step_4 = True   # Add leak rates
    step_5 = True   # Add PPM data
    step_6 = True   # Create example scaled subtracted images
    
    if test_mode:
        print(f"RUNNING IN TEST MODE - Processing only videos: {test_video_codes}")
        success = create_dataset_from_scratch(
            test_videos=test_video_codes,
            step_1=step_1, step_2=step_2, step_3=step_3, step_4=step_4, step_5=step_5
        )
    else:
        print("RUNNING IN FULL MODE - Processing all videos")
        success = create_dataset_from_scratch(
            step_1=step_1, step_2=step_2, step_3=step_3, step_4=step_4, step_5=step_5
        )
    
    if success:
        print("\nBase dataset creation completed successfully!")
        print("Ready for numpy dataset creation in 2_numpy_dataset_creation.py")
    else:
        print("\nBase dataset creation failed!")
        print("Check the error messages above for details.")