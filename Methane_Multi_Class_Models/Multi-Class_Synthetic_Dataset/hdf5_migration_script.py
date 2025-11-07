import h5py
import numpy as np
import json
import os
from pathlib import Path
import shutil

def migrate_to_hdf5(processed_dataset_path="Processed_Dataset", output_path="HDF5_Dataset"):
    """
    Migrate the current JSON + NumPy format to HDF5 format.
    
    Args:
        processed_dataset_path (str): Path to current processed dataset
        output_path (str): Path to save HDF5 dataset
    """
    print("Starting migration to HDF5 format...")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get all video directories
    video_dirs = [d for d in os.listdir(processed_dataset_path) 
                  if os.path.isdir(os.path.join(processed_dataset_path, d)) and d.isdigit() and len(d) == 4]
    
    total_samples = 0
    successful_migrations = 0
    
    for video_code in sorted(video_dirs):
        print(f"\nProcessing video {video_code}...")
        video_dir = os.path.join(processed_dataset_path, video_code)
        output_video_dir = os.path.join(output_path, video_code)
        os.makedirs(output_video_dir, exist_ok=True)
        
        # Process each class
        for class_num in range(8):
            class_dir = os.path.join(video_dir, f"Class_{class_num}")
            if not os.path.exists(class_dir):
                continue
                
            # Find the combined numpy file and metadata
            combined_npy = None
            metadata_json = None
            
            for file in os.listdir(class_dir):
                if file.endswith('_combined.npy'):
                    combined_npy = os.path.join(class_dir, file)
                elif file.endswith('_metadata.json'):
                    metadata_json = os.path.join(class_dir, file)
            
            if combined_npy and metadata_json:
                try:
                    # Load data
                    image_data = np.load(combined_npy)
                    
                    with open(metadata_json, 'r') as f:
                        metadata = json.load(f)
                    
                    # Create HDF5 file
                    hdf5_filename = f"{video_code}_class_{class_num}.h5"
                    hdf5_path = os.path.join(output_video_dir, hdf5_filename)
                    
                    with h5py.File(hdf5_path, 'w') as f:
                        # Store image data with compression
                        f.create_dataset('image', data=image_data, 
                                       compression='gzip', compression_opts=9)
                        
                        # Store metadata as attributes
                        for key, value in metadata.items():
                            f.attrs[key] = value
                        
                        # Add format version for future compatibility
                        f.attrs['format_version'] = '1.0'
                        f.attrs['created_by'] = 'hdf5_migration_script'
                    
                    print(f"  Class_{class_num}: Migrated successfully")
                    successful_migrations += 1
                    total_samples += 1
                    
                except Exception as e:
                    print(f"  Class_{class_num}: Error - {e}")
            else:
                print(f"  Class_{class_num}: Missing files")
    
    print("\n" + "="*60)
    print("MIGRATION SUMMARY")
    print("="*60)
    print(f"Total samples processed: {total_samples}")
    print(f"Successful migrations: {successful_migrations}")
    print(f"Failed migrations: {total_samples - successful_migrations}")
    print(f"Output directory: {output_path}")
    
    return successful_migrations

def create_pytorch_dataset_class():
    """
    Create a PyTorch Dataset class for the HDF5 format.
    """
    pytorch_code = '''
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os

class GasLeakDataset(Dataset):
    """
    PyTorch Dataset for gas leak detection using HDF5 format.
    """
    
    def __init__(self, dataset_path, transform=None):
        """
        Args:
            dataset_path (str): Path to the HDF5 dataset directory
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.dataset_path = dataset_path
        self.transform = transform
        self.samples = []
        
        # Collect all HDF5 files
        for video_dir in os.listdir(dataset_path):
            video_path = os.path.join(dataset_path, video_dir)
            if os.path.isdir(video_path):
                for file in os.listdir(video_path):
                    if file.endswith('.h5'):
                        self.samples.append(os.path.join(video_path, file))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            dict: Sample containing 'image', 'metadata', and 'target'
        """
        hdf5_path = self.samples[idx]
        
        with h5py.File(hdf5_path, 'r') as f:
            # Load image data
            image = torch.from_numpy(f['image'][:]).float()
            
            # Load metadata
            metadata = {
                'distance_m': f.attrs['distance_m'],
                'leak_rate_scfh': f.attrs['leak_rate_scfh'],
                'ppm_value': f.attrs['ppm_value'],
                'video_id': f.attrs['video_id'],
                'class_id': f.attrs['class_id']
            }
            
            # Create target (you can modify this based on your needs)
            target = torch.tensor(f.attrs['class_id'], dtype=torch.long)
        
        sample = {
            'image': image,
            'metadata': metadata,
            'target': target
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

# Example usage:
if __name__ == "__main__":
    # Create dataset
    dataset = GasLeakDataset("HDF5_Dataset")
    
    # Create data loader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Test loading
    for batch in dataloader:
        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Batch target shape: {batch['target'].shape}")
        print(f"Sample metadata: {batch['metadata'][0]}")
        break
'''
    
    with open("pytorch_dataset.py", "w") as f:
        f.write(pytorch_code)
    
    print("Created pytorch_dataset.py with HDF5 Dataset class")

if __name__ == "__main__":
    # Run migration
    successful = migrate_to_hdf5()
    
    if successful > 0:
        print(f"\nCreating PyTorch Dataset class...")
        create_pytorch_dataset_class()
        print(f"\nMigration completed! You can now use the HDF5 format with PyTorch.")
    else:
        print("Migration failed. Please check your dataset structure.")
