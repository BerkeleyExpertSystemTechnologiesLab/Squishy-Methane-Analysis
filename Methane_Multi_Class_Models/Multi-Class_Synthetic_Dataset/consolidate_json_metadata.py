"""
Script to consolidate metadata from individual JSON files in Processed_Dataset
into a single hierarchical JSON file.
"""

import json
import os
from pathlib import Path


def read_json_file(filepath):
    """
    Reads a JSON file and returns its contents as a dictionary.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary containing the JSON data, or None if file cannot be read
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def extract_specific_keys(data_dict, keys_to_extract):
    """
    Extracts specific keys from a dictionary.
    
    Args:
        data_dict: Source dictionary
        keys_to_extract: List of keys to extract
        
    Returns:
        Dictionary containing only the specified keys
    """
    return {key: data_dict[key] for key in keys_to_extract if key in data_dict}


def find_all_class_json_files(base_directory):
    """
    Finds all XXXX_class_X.json files in the directory structure.
    
    Args:
        base_directory: Path to the Processed_Dataset directory
        
    Returns:
        List of tuples: (video_number, class_number, filepath)
    """
    base_path = Path(base_directory)
    json_files = []
    
    # Iterate through all subdirectories
    for video_dir in sorted(base_path.iterdir()):
        if video_dir.is_dir() and video_dir.name.isdigit():
            video_number = video_dir.name
            
            # Look for Class_X subdirectories
            for class_dir in sorted(video_dir.iterdir()):
                if class_dir.is_dir() and class_dir.name.startswith("Class_"):
                    class_number = class_dir.name.split("_")[1]
                    
                    # Expected JSON filename
                    json_filename = f"{video_number}_class_{class_number}.json"
                    json_filepath = class_dir / json_filename
                    
                    if json_filepath.exists():
                        json_files.append((video_number, class_number, str(json_filepath)))
                    else:
                        print(f"Warning: Expected file not found: {json_filepath}")
    
    return json_files


def organize_data_hierarchically(json_files, keys_to_extract):
    """
    Organizes extracted data into a hierarchical structure.
    
    Args:
        json_files: List of tuples (video_number, class_number, filepath)
        keys_to_extract: List of keys to extract from each JSON
        
    Returns:
        Dictionary organized as {video_number: {class_X: {data}}}
    """
    organized_data = {}
    
    for video_number, class_number, filepath in json_files:
        # Read the JSON file
        data = read_json_file(filepath)
        if data is None:
            continue
        
        # Extract specific keys
        extracted_data = extract_specific_keys(data, keys_to_extract)
        
        # Initialize video_number key if it doesn't exist
        if video_number not in organized_data:
            organized_data[video_number] = {}
        
        # Add data under the appropriate class
        class_key = f"class_{class_number}"
        organized_data[video_number][class_key] = extracted_data
    
    return organized_data


def save_json_to_file(data, output_filepath):
    """
    Saves data to a JSON file with pretty formatting.
    
    Args:
        data: Dictionary to save
        output_filepath: Path where the JSON file will be saved
    """
    try:
        with open(output_filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Successfully saved consolidated data to: {output_filepath}")
    except Exception as e:
        print(f"Error saving to {output_filepath}: {e}")


def main():
    """Main function to consolidate all JSON metadata."""
    # Configuration
    base_directory = "Processed_Dataset"
    output_file = "consolidated_metadata.json"
    
    keys_to_extract = [
        "video_number",
        "class",
        "distance_m",
        "leak_rate_scfh",
        "leak_rate_scfh_std_dev",
        "leak_rate_gh",
        "leak_rate_gh_std_dev",
        "ppm"
    ]
    
    print("Starting metadata consolidation...")
    print(f"Base directory: {base_directory}")
    
    # Find all JSON files
    print("\nSearching for JSON files...")
    json_files = find_all_class_json_files(base_directory)
    print(f"Found {len(json_files)} JSON files")
    
    # Organize data hierarchically
    print("\nExtracting and organizing data...")
    consolidated_data = organize_data_hierarchically(json_files, keys_to_extract)
    print(f"Organized data for {len(consolidated_data)} videos")
    
    # Save to file
    print("\nSaving consolidated data...")
    save_json_to_file(consolidated_data, output_file)
    
    # Print summary
    print("\n--- Summary ---")
    print(f"Total videos: {len(consolidated_data)}")
    for video_num in sorted(consolidated_data.keys())[:3]:  # Show first 3 as examples
        classes = len(consolidated_data[video_num])
        print(f"  Video {video_num}: {classes} classes")
    if len(consolidated_data) > 3:
        print(f"  ... and {len(consolidated_data) - 3} more videos")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

