import os
from verification_functions import check_jpg_pixel_values, check_multiple_jpgs

def verify_jpg_brightness(file_list):
    """
    Verify the brightest pixel values in a list of JPG files.
    
    Args:
        file_list (list): List of paths to JPG files to check
        
    Returns:
        dict: Results with file paths as keys and statistics as values
    """
    print("VERIFYING JPG BRIGHTNESS")
    print("="*50)
    print(f"Checking {len(file_list)} JPG files...")
    print()
    
    # Check all files
    results = check_multiple_jpgs(file_list)
    
    # Summary of brightest pixels
    print("\n" + "="*50)
    print("BRIGHTEST PIXEL SUMMARY")
    print("="*50)
    
    if results:
        successful = sum(1 for stats in results.values() if stats is not None)
        print(f"Successfully processed: {successful}/{len(file_list)} files")
        
        if successful > 0:
            # Get all max values from successful files
            max_values = [stats["max_value"] for stats in results.values() if stats is not None]
            min_values = [stats["min_value"] for stats in results.values() if stats is not None]
            
            print(f"\nBrightest Pixel Statistics:")
            print(f"  Highest pixel value found: {max(max_values)}")
            print(f"  Lowest pixel value found: {min(min_values)}")
            print(f"  Average brightest pixel: {sum(max_values)/len(max_values):.2f}")
            
            # Check for unusually bright images
            very_bright = [v for v in max_values if v > 200]
            if very_bright:
                print(f"\nWARNING: {len(very_bright)} files have very bright pixels (>200)")
                print("These might be overexposed or incorrectly scaled")
            
            # Check for dark images
            very_dark = [v for v in max_values if v < 50]
            if very_dark:
                print(f"\nNOTE: {len(very_dark)} files have relatively dark brightest pixels (<50)")
                print("These might be underexposed or correctly scaled")
    
    return results

if __name__ == "__main__":
    # Example usage - replace with your actual file list
    jpg_files_to_check = [
        # Add your JPG file paths here
        # Example:
        # "Processed_Dataset/1237/Class_0/1237_frame_01-02_class_0_subtracted.jpg",
        # "Processed_Dataset/1237/Class_0/1237_frame_01-02_class_0_scaled.jpg",
        # "Processed_Dataset/1238/Class_1/1238_frame_05-10_class_1_subtracted.jpg",
        # etc...
        "Processed_Dataset/1469/Class_1/1469_class_1_subtracted_cv2_scaled.jpg"
    ]
    
    if not jpg_files_to_check:
        print("No files specified in jpg_files_to_check list")
        print("Please add your JPG file paths to the jpg_files_to_check list")
    else:
        # Verify the brightness of the specified JPG files
        results = verify_jpg_brightness(jpg_files_to_check)
        
        print(f"\nVerification completed for {len(jpg_files_to_check)} files")
