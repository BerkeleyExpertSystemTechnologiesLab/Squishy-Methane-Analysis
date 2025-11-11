import os
import re
from pathlib import Path

# Define paths
excel_path = "Original_Dataset/GasVid Logging File.xlsx"
mov_path = "Original_Dataset/Videos-20251002T175522Z-1-001/"
processed_dataset_path = "Processed_Dataset"
classes_json_path = "classes.json"
plume_modeling_path = "Plume_Modeling/Gasvid Plume Models.xlsx"



def extract_video_code(filename):
    """
    Extract the 4-digit code from video filename.
    
    Args:
        filename (str): Video filename (e.g., "MOV_1237.mp4")
        
    Returns:
        str: 4-digit code (e.g., "1237")
    """
    # Extract 4-digit number from filename
    match = re.search(r'MOV_(\d{4})\.mp4', filename)
    if match:
        return match.group(1)
    else:
        # Fallback: extract any 4-digit number
        match = re.search(r'(\d{4})', filename)
        if match:
            return match.group(1)
        else:
            return "0000"  # Default if no code found


        

def get_directory_files(directory_path, file_extensions=None):
    """
    Get all files in a directory and store them in a dictionary structure.
    
    Args:
        directory_path (str): Path to the directory
        file_extensions (list): List of file extensions to filter (e.g., ['.mp4', '.mov'])
        
    Returns:
        dict: Dictionary containing file information
    """
    try:
        # Check if directory exists
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not os.path.isdir(directory_path):
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        files_info = {
            'directory_path': directory_path,
            'total_files': 0,
            'files': []
        }
        
        # Get all files in directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            # Skip if not a file
            if not os.path.isfile(file_path):
                continue
            
            # Filter by file extensions if specified
            if file_extensions:
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext not in file_extensions:
                    continue
            
            # Get file information
            file_info = {
                'filename': filename,
                'file_path': file_path,
                'file_extension': os.path.splitext(filename)[1].lower(),
                'file_size': os.path.getsize(file_path)
            }
            
            files_info['files'].append(file_info)
        
        files_info['total_files'] = len(files_info['files'])
        
        # print(f"Directory: {directory_path}")
        # print(f"Total files found: {files_info['total_files']}")
        # print("\nFiles:")
        # for file_info in files_info['files']:
        #     print(f"  {file_info['filename']} ({file_info['file_size']} bytes)")
        
        return files_info
        
    except Exception as e:
        # print(f"Error reading directory: {e}")
        return None




def create_class_directories(video_codes=None, num_classes=8):
    """
    Create directory structure for each class under each video code in the processed dataset.
    
    Creates directories like:
    Processed_Dataset/
    ├── 1234/
    │   ├── Class_0/
    │   ├── Class_1/
    │   ├── Class_2/
    │   ├── Class_3/
    │   ├── Class_4/
    │   ├── Class_5/
    │   ├── Class_6/
    │   └── Class_7/
    
    Args:
        video_codes (list, optional): List of 4-digit video codes. If None, gets codes from mov_path.
        num_classes (int): Number of classes to create (default: 8)
        
    Returns:
        dict: Dictionary with creation results including:
            - created_dirs: List of successfully created directories
            - failed_dirs: List of directories that failed to create
            - total_created: Number of directories created
    """
    if not video_codes:
        print("Error: No video codes provided")
        return None
    
    created_dirs = []
    failed_dirs = []
    
    # print(f"Creating class directories for {len(video_codes)} videos with {num_classes} classes each...")
    
    try:
        for video_code in video_codes:
            # print(f"\nCreating directories for video {video_code}:")
            
            # Create main video directory
            video_dir = os.path.join(processed_dataset_path, video_code)
            os.makedirs(video_dir, exist_ok=True)
            # print(f"  Created video directory: {video_dir}")
            
            # Create class directories under this video
            for class_num in range(num_classes):
                class_dir = os.path.join(video_dir, f"Class_{class_num}")
                
                try:
                    os.makedirs(class_dir, exist_ok=True)
                    created_dirs.append(class_dir)
                    # print(f"    Created Class_{class_num}: {class_dir}")
                except Exception as e:
                    failed_dirs.append(class_dir)
                    print(f"    Failed to create Class_{class_num}: {e}")
        
        result = {
            'created_dirs': created_dirs,
            'failed_dirs': failed_dirs,
            'total_created': len(created_dirs),
            'total_failed': len(failed_dirs),
            'video_codes': video_codes,
            'num_classes': num_classes
        }
        
        # print(f"\n" + "="*60)
        # print("DIRECTORY CREATION SUMMARY")
        # print("="*60)
        # print(f"Total directories created: {len(created_dirs)}")
        # print(f"Total directories failed: {len(failed_dirs)}")
        # print(f"Videos processed: {len(video_codes)}")
        # print(f"Classes per video: {num_classes}")
        
        if failed_dirs:
            print(f"\nFailed directories:")
            for failed_dir in failed_dirs:
                print(f"  - {failed_dir}")
        
        return result
        
    except Exception as e:
        print(f"Error creating class directories: {e}")
        return None