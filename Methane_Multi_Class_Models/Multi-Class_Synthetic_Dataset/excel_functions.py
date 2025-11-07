import pandas as pd
import cv2
import os
import re
import json
from pathlib import Path
from json_functions import find_class_json_file



def load_excel_data(excel_file_path):
    """
    Load data from Excel file and return a pandas DataFrame.
    
    Args:
        excel_file_path (str): Path to the Excel file
        
    Returns:
        pandas.DataFrame: DataFrame containing the Excel data
    """
    try:
        # Check if file exists
        if not os.path.exists(excel_file_path):
            raise FileNotFoundError(f"Excel file not found: {excel_file_path}")
        
        # Read Excel file
        # You can specify sheet_name if needed (e.g., sheet_name=0 for first sheet)
        df = pd.read_excel(excel_file_path)
        
        # print(f"Successfully loaded Excel file: {excel_file_path}")
        # print(f"Data shape: {df.shape}")
        # print(f"Columns: {list(df.columns)}")
        # print("\nFirst few rows:")
        # print(df.head())
        
        return df
        
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None



def get_distance_mapping(excel_file_path):
    """
    Retrieve distance mapping from Excel file.
    
    Args:
        excel_file_path (str): Path to the Excel file containing video data
        
    Returns:
        dict: Dictionary mapping video numbers to distances, or None if failed
    """
    try:
        # Load Excel data
        df = load_excel_data(excel_file_path)
        if df is None:
            print("Failed to load Excel data")
            return None
        
        # Get the first two columns (Video No. and Imaging Distance)
        video_data = df.iloc[:, [0, 1]]  # First two columns
        video_data.columns = ['video_number', 'imaging_distance_m']
        
        print(f"Retrieved {len(video_data)} video entries from Excel...")
        
        distance_mapping = {}
        
        # Process each video entry
        for index, row in video_data.iterrows():
            # Skip rows with NaN values
            if pd.isna(row['video_number']) or pd.isna(row['imaging_distance_m']):
                print(f"Skipping row {index}: Missing video number or distance data")
                continue
                
            video_number = str(int(row['video_number']))  # Convert to string, remove decimal
            distance = float(row['imaging_distance_m'])
            distance_mapping[video_number] = distance
        
        print(f"Successfully created distance mapping for {len(distance_mapping)} videos")
        return distance_mapping
        
    except Exception as e:
        print(f"Error retrieving distance mapping: {e}")
        return None



def add_distance_descriptors(excel_file_path, processed_dataset_path):
    """
    Add imaging distance descriptors to each Class_X directory.
    
    Args:
        excel_file_path (str): Path to the Excel file containing video data
        processed_dataset_path (str): Path to the Processed_Dataset directory
        
    Returns:
        dict: Summary of processed directories and their distance values
    """
    try:
        # Get distance mapping from Excel
        distance_mapping = get_distance_mapping(excel_file_path)
        if distance_mapping is None:
            return None
        
        print(f"Processing {len(distance_mapping)} video entries...")
        
        processed_summary = {}
        
        # Process each video entry
        for video_number, distance in distance_mapping.items():
            video_dir = os.path.join(processed_dataset_path, video_number)
            
            if not os.path.exists(video_dir):
                print(f"Video directory not found: {video_dir}")
                processed_summary[video_number] = {
                    'success': False,
                    'distance_m': distance,
                    'video_number': video_number,
                    'error': 'Video directory not found'
                }
                continue
            
            # Create distance data
            distance_data = {"distance_m": distance}
            
            # Add distance to each Class_X directory (0-7)
            class_successes = 0
            class_failures = 0
            
            for class_num in range(8):
                class_dir = os.path.join(video_dir, f"Class_{class_num}")
                
                if not os.path.exists(class_dir):
                    print(f"  Class_{class_num} directory not found for video {video_number}")
                    class_failures += 1
                    continue
                
                # Find the class JSON file
                class_json_file = find_class_json_file(class_dir, class_num)
                
                if not class_json_file:
                    print(f"  Class_{class_num}: No JSON file found for video {video_number}")
                    class_failures += 1
                    continue
                
                # Load existing class data and add distance
                try:
                    with open(class_json_file, 'r') as f:
                        class_data = json.load(f)
                    
                    # Add distance data
                    class_data.update(distance_data)
                    
                    # Write updated data back to file
                    with open(class_json_file, 'w') as f:
                        json.dump(class_data, f, indent=2)
                    
                    print(f"  Class_{class_num}: Added distance {distance}m for video {video_number}")
                    class_successes += 1
                    
                except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                    print(f"  Class_{class_num}: Error updating JSON file for video {video_number}: {e}")
                    class_failures += 1
            
            processed_summary[video_number] = {
                'success': class_successes > 0,
                'distance_m': distance,
                'video_number': video_number,
                'class_successes': class_successes,
                'class_failures': class_failures
            }
            
            if class_successes > 0:
                print(f"Added distance descriptor for video {video_number}: {distance}m ({class_successes} classes updated)")
            else:
                print(f"Failed to add distance for video {video_number} (no classes updated)")
        
        print(f"\nSummary: Processed {len(processed_summary)} video entries")
        return processed_summary
        
    except Exception as e:
        print(f"Error adding distance descriptors: {e}")
        return None




def convert_excel_to_csv(excel_path, csv_path):
    """
    Convert Excel file to CSV format for faster loading.
    
    Args:
        excel_path (str): Path to the Excel file
        csv_path (str): Path to save the CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Converting Excel file to CSV: {excel_path}")
        
        # Load Excel data
        df = load_excel_data(excel_path)
        if df is None:
            print("Failed to load Excel data")
            return False
        
        # Fix column names based on the actual Excel structure
        if len(df) > 1:
            # Use row 1 as column names and drop the header rows
            df.columns = df.iloc[1].values
            df = df.drop([0, 1]).reset_index(drop=True)
            
            # Clean up column names
            df.columns = [str(col).strip() if pd.notna(col) else f"col_{i}" for i, col in enumerate(df.columns)]
        
        # Convert data types for better performance
        numeric_columns = ['Class', 'leak_rate_scfh', 'Distance (m)', 'Theta (deg)', 'ppm']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Save as CSV
        df.to_csv(csv_path, index=False)
        
        print(f"Successfully converted to CSV: {csv_path}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"Error converting Excel to CSV: {e}")
        return False

