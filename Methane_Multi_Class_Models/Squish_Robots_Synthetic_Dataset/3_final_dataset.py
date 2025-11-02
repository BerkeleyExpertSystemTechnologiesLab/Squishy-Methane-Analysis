import os
import shutil
import json
import numpy as np
from pathlib import Path

def create_final_dataset_structure():
    """
    Create the final dataset directory structure.
    
    Returns:
        str: Path to the Final_Dataset directory
    """
    final_dataset_path = "Final_Dataset"
    
    # Create main directory
    os.makedirs(final_dataset_path, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(final_dataset_path, "data"), exist_ok=True)
    os.makedirs(os.path.join(final_dataset_path, "metadata"), exist_ok=True)
    os.makedirs(os.path.join(final_dataset_path, "splits"), exist_ok=True)
    
    # Create class directories for data and metadata
    for class_num in range(8):
        data_class_dir = os.path.join(final_dataset_path, "data", f"class_{class_num}")
        metadata_class_dir = os.path.join(final_dataset_path, "metadata", f"class_{class_num}")
        
        os.makedirs(data_class_dir, exist_ok=True)
        os.makedirs(metadata_class_dir, exist_ok=True)
    
    print(f"Created directory structure: {final_dataset_path}")
    return final_dataset_path

def copy_numpy_files(processed_dataset_path, final_dataset_path):
    """
    Copy all numpy files from processed dataset to final dataset structure.
    
    Args:
        processed_dataset_path (str): Path to the processed dataset
        final_dataset_path (str): Path to the final dataset
    """
    print("\nCopying numpy files...")
    
    data_dir = os.path.join(final_dataset_path, "data")
    sample_counts = {f"class_{i}": 0 for i in range(8)}
    
    # Find all video directories
    if not os.path.exists(processed_dataset_path):
        print(f"Error: Processed dataset directory not found: {processed_dataset_path}")
        return
    
    video_dirs = [d for d in os.listdir(processed_dataset_path) 
                  if os.path.isdir(os.path.join(processed_dataset_path, d)) and d.isdigit() and len(d) == 4]
    
    print(f"Found {len(video_dirs)} video directories")
    
    for video_code in sorted(video_dirs):
        video_dir = os.path.join(processed_dataset_path, video_code)
        print(f"Processing video {video_code}...")
        
        # Process each class for this video
        for class_num in range(8):
            class_dir = os.path.join(video_dir, f"Class_{class_num}")
            
            if not os.path.exists(class_dir):
                print(f"  Class_{class_num}: Directory not found, skipping...")
                continue
            
            # Look for processed_data subdirectory
            processed_data_dir = os.path.join(class_dir, "processed_data")
            
            if not os.path.exists(processed_data_dir):
                print(f"  Class_{class_num}: No processed_data directory found, skipping...")
                continue
            
            # Copy all .npy files from processed_data
            npy_files = [f for f in os.listdir(processed_data_dir) if f.endswith('.npy')]
            
            if not npy_files:
                print(f"  Class_{class_num}: No .npy files found, skipping...")
                continue
            
            print(f"  Class_{class_num}: Found {len(npy_files)} .npy files")
            
            # Copy each .npy file with original naming
            for npy_file in npy_files:
                source_path = os.path.join(processed_data_dir, npy_file)
                target_path = os.path.join(data_dir, f"class_{class_num}", npy_file)
                
                try:
                    shutil.copy2(source_path, target_path)
                    sample_counts[f'class_{class_num}'] += 1
                except Exception as e:
                    print(f"    Error copying {npy_file}: {e}")
    
    # Print summary
    print(f"\nNumpy files copied successfully:")
    for class_name, count in sample_counts.items():
        print(f"  {class_name}: {count} samples")

def copy_json_files(processed_dataset_path, final_dataset_path):
    """
    Copy all JSON files from processed dataset to final dataset structure.
    
    Args:
        processed_dataset_path (str): Path to the processed dataset
        final_dataset_path (str): Path to the final dataset
    """
    print("\nCopying JSON files...")
    
    metadata_dir = os.path.join(final_dataset_path, "metadata")
    sample_counts = {f"class_{i}": 0 for i in range(8)}
    
    # Find all video directories
    video_dirs = [d for d in os.listdir(processed_dataset_path) 
                  if os.path.isdir(os.path.join(processed_dataset_path, d)) and d.isdigit() and len(d) == 4]
    
    for video_code in sorted(video_dirs):
        video_dir = os.path.join(processed_dataset_path, video_code)
        print(f"Processing video {video_code}...")
        
        # Process each class for this video
        for class_num in range(8):
            class_dir = os.path.join(video_dir, f"Class_{class_num}")
            
            if not os.path.exists(class_dir):
                print(f"  Class_{class_num}: Directory not found, skipping...")
                continue
            
            # Look for JSON files in the class directory
            json_files = [f for f in os.listdir(class_dir) if f.endswith('.json')]
            
            if not json_files:
                print(f"  Class_{class_num}: No JSON files found, skipping...")
                continue
            
            print(f"  Class_{class_num}: Found {len(json_files)} JSON files")
            
            # Copy each JSON file with original naming
            for json_file in json_files:
                source_path = os.path.join(class_dir, json_file)
                target_path = os.path.join(metadata_dir, f"class_{class_num}", json_file)
                
                try:
                    shutil.copy2(source_path, target_path)
                    sample_counts[f'class_{class_num}'] += 1
                except Exception as e:
                    print(f"    Error copying {json_file}: {e}")
    
    # Print summary
    print(f"\nJSON files copied successfully:")
    for class_name, count in sample_counts.items():
        print(f"  {class_name}: {count} samples")

def create_dataset_info(final_dataset_path):
    """
    Create a dataset_info.json file with summary information.
    
    Args:
        final_dataset_path (str): Path to the final dataset
    """
    print("\nCreating dataset_info.json...")
    
    dataset_info = {
        "dataset_name": "Gas Leak Detection Dataset",
        "description": "Multi-modal dataset with gas leak images and metadata",
        "structure": {
            "data": "Contains .npy files with 2-channel arrays (background + gas leak)",
            "metadata": "Contains .json files with PPM, distance, and leak rate data",
            "splits": "Contains train/val/test split files (to be created)"
        },
        "classes": 8,
        "data_format": {
            "numpy_files": "2-channel arrays: [background, gas_leak_ppm]",
            "json_files": "Metadata with PPM, distance, leak_rate, video_code"
        }
    }
    
    # Count samples per class
    data_dir = os.path.join(final_dataset_path, "data")
    sample_counts = {}
    
    for class_num in range(8):
        class_dir = os.path.join(data_dir, f"class_{class_num}")
        if os.path.exists(class_dir):
            npy_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
            sample_counts[f"class_{class_num}"] = len(npy_files)
        else:
            sample_counts[f"class_{class_num}"] = 0
    
    dataset_info["sample_counts"] = sample_counts
    dataset_info["total_samples"] = sum(sample_counts.values())
    
    # Save dataset info
    info_path = os.path.join(final_dataset_path, "dataset_info.json")
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Dataset info saved to: {info_path}")
    print(f"Total samples: {dataset_info['total_samples']}")
    
    return dataset_info

def create_final_dataset(processed_dataset_path="Processed_Dataset"):
    """
    Create the final exportable dataset from the processed dataset.
    
    Args:
        processed_dataset_path (str): Path to the processed dataset directory
    """
    print("CREATING FINAL DATASET")
    print("="*50)
    
    # Step 1: Create directory structure
    final_dataset_path = create_final_dataset_structure()
    
    # Step 2: Copy numpy files
    copy_numpy_files(processed_dataset_path, final_dataset_path)
    
    # Step 3: Copy JSON files
    copy_json_files(processed_dataset_path, final_dataset_path)
    
    # Step 4: Create dataset info
    dataset_info = create_dataset_info(final_dataset_path)
    
    print("\n" + "="*50)
    print("FINAL DATASET CREATION COMPLETED!")
    print("="*50)
    print(f"Final dataset created at: {final_dataset_path}")
    print(f"Total samples: {dataset_info['total_samples']}")
    print("\nDirectory structure:")
    print("Final_Dataset/")
    print("├── data/")
    print("│   ├── class_0/")
    print("│   ├── class_1/")
    print("│   └── ...")
    print("├── metadata/")
    print("│   ├── class_0/")
    print("│   ├── class_1/")
    print("│   └── ...")
    print("├── splits/")
    print("└── dataset_info.json")
    
    return final_dataset_path

if __name__ == "__main__":
    # Create the final dataset
    final_dataset_path = create_final_dataset()
    
    print(f"\nFinal dataset ready for export at: {final_dataset_path}")
    print("You can now use this dataset with PyTorch DataLoaders!")
