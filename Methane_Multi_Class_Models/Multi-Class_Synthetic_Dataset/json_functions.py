import pandas as pd
import cv2
import os
import re
import json
from pathlib import Path




def find_class_json_file(class_dir, class_num):
    """
    Find the class JSON file in a class directory.
    
    Args:
        class_dir (str): Path to the class directory
        class_num (int): Class number (0-7)
        
    Returns:
        str or None: Path to the class JSON file, or None if not found
    """
    if not os.path.exists(class_dir):
        return None
    
    for file in os.listdir(class_dir):
        if file.endswith('.json') and f"class_{class_num}" in file:
            return os.path.join(class_dir, file)
    return None


def load_classes_data(classes_json_path):
    """
    Load leak rate data from classes.json file.
    
    Args:
        classes_json_path (str): Path to the classes.json file
        
    Returns:
        dict or None: Dictionary containing leak rate data, or None if failed
    """
    try:
        with open(classes_json_path, 'r') as f:
            classes_data = json.load(f)
        
        leak_rates = classes_data['classes']
        print(f"Loaded leak rate data for {len(leak_rates)} classes")
        return leak_rates
        
    except Exception as e:
        print(f"Error loading classes data: {e}")
        return None


def update_class_with_leak_rates(class_json_file, leak_rates, class_num):
    """
    Update a single class JSON file with leak rate data.
    
    Args:
        class_json_file (str): Path to the class JSON file
        leak_rates (dict): Dictionary containing leak rate data
        class_num (int): Class number for logging
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load existing class data
        with open(class_json_file, 'r') as f:
            class_data = json.load(f)
        
        # Get the class number from the JSON file
        json_class_num = class_data.get('class')
        
        if json_class_num is None:
            print(f"  Class_{class_num}: No class number found in JSON")
            return False
        
        # Get leak rate data for this class
        class_key = str(json_class_num)
        if class_key not in leak_rates:
            print(f"  Class_{class_num}: No leak rate data for class {json_class_num}")
            return False
        
        leak_rate_data = leak_rates[class_key]
        
        # Add leak rate data to class data
        class_data.update({
            "leak_rate_scfh": leak_rate_data["leak_rate_scfh"],
            "leak_rate_scfh_std_dev": leak_rate_data["leak_rate_scfh_std_dev"],
            "leak_rate_gh": leak_rate_data["leak_rate_gh"],
            "leak_rate_gh_std_dev": leak_rate_data["leak_rate_gh_std_dev"]
        })
        
        # Write updated data back to file
        with open(class_json_file, 'w') as f:
            json.dump(class_data, f, indent=2)
        
        print(f"  Class_{class_num}: Added leak rates for class {json_class_num}")
        return True
        
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"  Class_{class_num}: Error processing JSON file - {e}")
        return False


def add_leak_rates_to_classes(processed_dataset_path, classes_json_path):
    """
    Add leak rate data to all class JSON files based on their class number.
    
    Args:
        processed_dataset_path (str): Path to the Processed_Dataset directory
        classes_json_path (str): Path to the classes.json file
        
    Returns:
        dict: Summary of processed files
    """
    try:
        # Load classes data
        leak_rates = load_classes_data(classes_json_path)
        if leak_rates is None:
            return None
        
        processed_summary = {
            'total_videos': 0,
            'total_classes': 0,
            'successful_updates': 0,
            'failed_updates': 0
        }
        
        # Go through each video directory
        for video_dir in os.listdir(processed_dataset_path):
            video_path = os.path.join(processed_dataset_path, video_dir)
            
            # Skip if not a directory
            if not os.path.isdir(video_path):
                continue
            
            # Check if it's a 4-digit video directory
            if not video_dir.isdigit() or len(video_dir) != 4:
                continue
            
            processed_summary['total_videos'] += 1
            print(f"\nProcessing video directory: {video_dir}")
            
            # Go through each class directory (0-7)
            for class_num in range(8):
                class_dir = os.path.join(video_path, f"Class_{class_num}")
                
                # Find the class JSON file
                class_json_file = find_class_json_file(class_dir, class_num)
                
                if not class_json_file:
                    print(f"  Class_{class_num}: No JSON file found")
                    continue
                
                # Update the class file with leak rates
                if update_class_with_leak_rates(class_json_file, leak_rates, class_num):
                    processed_summary['successful_updates'] += 1
                    processed_summary['total_classes'] += 1
                else:
                    processed_summary['failed_updates'] += 1
        
        print(f"\nSummary:")
        print(f"  Total videos processed: {processed_summary['total_videos']}")
        print(f"  Total classes processed: {processed_summary['total_classes']}")
        print(f"  Successful updates: {processed_summary['successful_updates']}")
        print(f"  Failed updates: {processed_summary['failed_updates']}")
        
        return processed_summary
        
    except Exception as e:
        print(f"Error adding leak rates to classes: {e}")
        return None



def add_ppm_data_to_classes(processed_dataset_path, plume_modeling_path):
    """
    Add PPM data to all class JSON files based on class and leak_rate_scfh matching.
    
    Args:
        processed_dataset_path (str): Path to the Processed_Dataset directory
        plume_modeling_path (str): Path to the Gasvid Plume Models.xlsx file (will convert to .csv)
        
    Returns:
        dict: Summary of processed files
    """
    try:
        # Check if CSV file exists, if not convert from Excel
        csv_path = plume_modeling_path.replace('.xlsx', '.csv')
        
        if not os.path.exists(csv_path):
            print("CSV file not found, converting from Excel...")
            if not convert_excel_to_csv(plume_modeling_path, csv_path):
                print("Failed to convert Excel to CSV")
                return None
        
        # Load plume modeling data from CSV (much faster!)
        print(f"Loading plume modeling data from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"Loaded plume modeling data with {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types: {df.dtypes}")
        
        processed_summary = {
            'total_videos': 0,
            'total_classes': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'no_match_found': 0
        }
        
        # Go through each video directory
        for video_dir in os.listdir(processed_dataset_path):
            video_path = os.path.join(processed_dataset_path, video_dir)
            
            # Skip if not a directory
            if not os.path.isdir(video_path):
                continue
            
            # Check if it's a 4-digit video directory
            if not video_dir.isdigit() or len(video_dir) != 4:
                continue
            
            processed_summary['total_videos'] += 1
            print(f"\nProcessing video directory: {video_dir}")
            
            # Go through each class directory (0-7)
            for class_num in range(8):
                class_dir = os.path.join(video_path, f"Class_{class_num}")
                
                if not os.path.exists(class_dir):
                    continue
                
                # Find the class JSON file
                class_json_file = None
                for file in os.listdir(class_dir):
                    if file.endswith('.json') and f"class_{class_num}" in file:
                        class_json_file = os.path.join(class_dir, file)
                        break
                
                if not class_json_file:
                    print(f"  Class_{class_num}: No JSON file found")
                    continue
                
                try:
                    # Load existing class data
                    with open(class_json_file, 'r') as f:
                        class_data = json.load(f)
                    
                    # Get the class number, leak rate, and distance from the JSON file
                    json_class_num = class_data.get('class')
                    leak_rate_scfh = class_data.get('leak_rate_scfh')
                    distance_m = class_data.get('distance_m')
                    
                    if json_class_num is None or leak_rate_scfh is None or distance_m is None:
                        print(f"  Class_{class_num}: Missing class number, leak rate, or distance in JSON")
                        processed_summary['failed_updates'] += 1
                        continue
                    
                    # Search for matching row in Excel data
                    # First, let's see what columns are available
                    print(f"  Class_{class_num}: Available columns: {list(df.columns)}")
                    
                    # Find the correct column names (they might be slightly different)
                    class_col = None
                    leak_rate_col = None
                    distance_col = None
                    ppm_col = None
                    
                    for col in df.columns:
                        if 'class' in str(col).lower():
                            class_col = col
                        elif 'leak_rate_scfh' in str(col).lower():
                            leak_rate_col = col
                        elif 'distance' in str(col).lower():
                            distance_col = col
                        elif 'ppm' in str(col).lower():
                            ppm_col = col
                    
                    print(f"  Class_{class_num}: Found columns - class: {class_col}, leak_rate: {leak_rate_col}, distance: {distance_col}, ppm: {ppm_col}")
                    
                    if not all([class_col, leak_rate_col, distance_col, ppm_col]):
                        print(f"  Class_{class_num}: Missing required columns in Excel data")
                        processed_summary['failed_updates'] += 1
                        continue
                    
                    # Look for rows where class, leak_rate_scfh, and distance all match
                    matching_rows = df[
                        (df[class_col] == json_class_num) & 
                        (abs(df[leak_rate_col] - leak_rate_scfh) < 0.1) &  # Allow small tolerance for leak rate
                        (abs(df[distance_col] - distance_m) < 0.1)  # Allow small tolerance for distance
                    ]
                    
                    if len(matching_rows) == 0:
                        print(f"  Class_{class_num}: No matching PPM data found for class {json_class_num}, leak_rate {leak_rate_scfh}, distance {distance_m}m")
                        processed_summary['no_match_found'] += 1
                        continue
                    
                    # Get the first matching row (in case there are multiple)
                    matching_row = matching_rows.iloc[0]
                    ppm_value = matching_row.get(ppm_col)
                    
                    if pd.isna(ppm_value):
                        print(f"  Class_{class_num}: PPM value is NaN in matching row")
                        processed_summary['failed_updates'] += 1
                        continue
                    
                    # Add PPM data to class data
                    class_data["ppm"] = float(ppm_value)
                    
                    # Write updated data back to file
                    with open(class_json_file, 'w') as f:
                        json.dump(class_data, f, indent=2)
                    
                    print(f"  Class_{class_num}: Added PPM value {ppm_value} for class {json_class_num}, leak_rate {leak_rate_scfh}, distance {distance_m}m")
                    processed_summary['successful_updates'] += 1
                    processed_summary['total_classes'] += 1
                    
                except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                    print(f"  Class_{class_num}: Error processing JSON file - {e}")
                    processed_summary['failed_updates'] += 1
        
        print(f"\nSummary:")
        print(f"  Total videos processed: {processed_summary['total_videos']}")
        print(f"  Total classes processed: {processed_summary['total_classes']}")
        print(f"  Successful updates: {processed_summary['successful_updates']}")
        print(f"  Failed updates: {processed_summary['failed_updates']}")
        print(f"  No match found: {processed_summary['no_match_found']}")
        
        return processed_summary
        
    except Exception as e:
        print(f"Error adding PPM data to classes: {e}")
        return None