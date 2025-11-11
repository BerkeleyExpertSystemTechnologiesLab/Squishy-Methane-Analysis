import cv2
import numpy as np
import os


def jpg_to_numpy(jpg_path, output_path=None, grayscale=True):
    """
    Convert a JPG image to a numpy array and optionally save it.
    
    Args:
        jpg_path (str): Path to the input JPG file
        output_path (str, optional): Path to save the numpy array (.npy file)
        grayscale (bool): Whether to convert to grayscale (default: True)
        
    Returns:
        numpy.ndarray: The image as a numpy array, or None if failed
    """
    try:
        # Check if file exists
        if not os.path.exists(jpg_path):
            print(f"Error: JPG file not found: {jpg_path}")
            return None
        
        # Read the image
        if grayscale:
            image = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
        
        if image is None:
            print(f"Error: Could not read image from {jpg_path}")
            return None
        
        # Convert to numpy array if not already
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        print(f"Successfully converted JPG to numpy array: {jpg_path}")
        print(f"Array shape: {image.shape}, dtype: {image.dtype}")
        
        # Save to file if output path is provided
        if output_path:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save as numpy array
            np.save(output_path, image)
            print(f"Saved numpy array to: {output_path}")
        
        return image
        
    except Exception as e:
        print(f"Error converting JPG to numpy: {e}")
        return None


def scale_numpy_array(array, new_max, new_min=0, preserve_dtype=True):
    """
    Scale a numpy array to a new range.
    
    Args:
        array (numpy.ndarray): Input numpy array
        new_max (float): New maximum value for the scaled array
        new_min (float): New minimum value for the scaled array (default: 0)
        preserve_dtype (bool): Whether to preserve the original data type (default: True)
        
    Returns:
        numpy.ndarray: Scaled numpy array, or None if failed
    """
    try:
        if not isinstance(array, np.ndarray):
            print("Error: Input must be a numpy array")
            return None
        
        # Get current min and max values
        current_min = np.min(array)
        current_max = np.max(array)
        
        print(f"Original range: [{current_min}, {current_max}]")
        print(f"Target range: [{new_min}, {new_max}]")
        
        # Handle case where all values are the same
        if current_max == current_min:
            print("Warning: All values in array are the same, setting to new_min")
            scaled_array = np.full_like(array, new_min)
        else:
            # Scale the array to new range
            # Formula: new_value = new_min + (old_value - old_min) * (new_max - new_min) / (old_max - old_min)
            scaled_array = new_min + (array - current_min) * (new_max - new_min) / (current_max - current_min)
        
        # Preserve original data type if requested
        if preserve_dtype:
            if array.dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
                # For unsigned integers, clip to valid range and convert
                scaled_array = np.clip(scaled_array, new_min, new_max)
                scaled_array = scaled_array.astype(array.dtype)
            elif array.dtype in [np.int8, np.int16, np.int32, np.int64]:
                # For signed integers, clip to valid range and convert
                scaled_array = np.clip(scaled_array, new_min, new_max)
                scaled_array = scaled_array.astype(array.dtype)
            else:
                # For floats, just convert to same type
                scaled_array = scaled_array.astype(array.dtype)
        else:
            # Clip to target range
            scaled_array = np.clip(scaled_array, new_min, new_max)
        
        # Verify the scaling worked
        actual_min = np.min(scaled_array)
        actual_max = np.max(scaled_array)
        
        print(f"Scaled range: [{actual_min}, {actual_max}]")
        print(f"Array shape: {scaled_array.shape}, dtype: {scaled_array.dtype}")
        
        return scaled_array
        
    except Exception as e:
        print(f"Error scaling numpy array: {e}")
        return None


def scale_and_save_array(array, new_max, new_min=0, output_path=None, preserve_dtype=True):
    """
    Scale a numpy array and optionally save it.
    
    Args:
        array (numpy.ndarray): Input numpy array
        new_max (float): New maximum value for the scaled array
        new_min (float): New minimum value for the scaled array (default: 0)
        output_path (str, optional): Path to save the scaled array
        preserve_dtype (bool): Whether to preserve the original data type (default: True)
        
    Returns:
        numpy.ndarray: Scaled numpy array, or None if failed
    """
    try:
        # Scale the array
        scaled_array = scale_numpy_array(array, new_max, new_min, preserve_dtype)
        
        if scaled_array is None:
            return None
        
        # Save to file if output path is provided
        if output_path:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save as numpy array
            np.save(output_path, scaled_array)
            print(f"Saved scaled array to: {output_path}")
        
        return scaled_array
        
    except Exception as e:
        print(f"Error scaling and saving array: {e}")
        return None


def scale_jpg_to_ppm(jpg_path, ppm_value, output_path=None, grayscale=False):
    """
    Scale a JPG image so that the maximum pixel value becomes the specified PPM value,
    and all other pixels are proportionally scaled between 0 and that PPM value.
    
    Args:
        jpg_path (str): Path to the input JPG file
        ppm_value (float): Target PPM value for the maximum pixel
        output_path (str, optional): Path to save the scaled image as .npy file
        grayscale (bool): Whether to convert to grayscale (default: True)
        
    Returns:
        numpy.ndarray: Scaled array with values between 0 and ppm_value
    """
    try:
        # Load the JPG image
        image = cv2.imread(jpg_path)
        if image is None:
            raise ValueError(f"Could not load image: {jpg_path}")
        
        # Convert to grayscale if requested
        if grayscale:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Convert to float for precise scaling
        image_float = image.astype(np.float64)
        
        # Find current min and max values
        current_min = np.min(image_float)
        current_max = np.max(image_float)
        
        # Handle case where all pixels are the same value
        if current_max == current_min:
            print(f"Warning: All pixels have the same value ({current_max})")
            # Set all pixels to half the PPM value
            scaled_image = np.full_like(image_float, ppm_value / 2.0)
        else:
            # Scale to 0 to ppm_value range
            scaled_image = ((image_float - current_min) / (current_max - current_min)) * ppm_value
        
        # Ensure values are within bounds
        scaled_image = np.clip(scaled_image, 0, ppm_value)
        
        print(f"Scaled image from range [{current_min:.2f}, {current_max:.2f}] to [0, {ppm_value:.2f}]")
        print(f"New min: {np.min(scaled_image):.2f}, New max: {np.max(scaled_image):.2f}")
        
        # Save if output path provided
        if output_path:
            np.save(output_path, scaled_image)
            print(f"Scaled image saved to: {output_path}")
        
        return scaled_image
        
    except Exception as e:
        print(f"Error scaling JPG to PPM: {e}")
        return None



def numpy_to_jpg(array, output_path, normalize=True, quality=95):
    """
    Convert a NumPy array to a JPG image file.
    
    Args:
        array (numpy.ndarray): Input array to convert
        output_path (str): Path to save the JPG file
        normalize (bool): Whether to normalize array to 0-255 range (default: True)
        quality (int): JPG quality (1-100, default: 95)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure array is numpy array
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        
        # Handle different array types
        if array.dtype == np.float64 or array.dtype == np.float32:
            if normalize:
                # Normalize to 0-255 range
                array_min = np.min(array)
                array_max = np.max(array)
                
                if array_max > array_min:
                    array = ((array - array_min) / (array_max - array_min) * 255).astype(np.uint8)
                else:
                    # All values are the same
                    array = np.full_like(array, 128, dtype=np.uint8)
            else:
                # Clip to 0-255 range and convert to uint8
                array = np.clip(array, 0, 255).astype(np.uint8)
        else:
            # Convert to uint8 if not already
            array = array.astype(np.uint8)
        
        # Ensure values are in valid range
        array = np.clip(array, 0, 255)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as JPG
        success = cv2.imwrite(output_path, array, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        if success:
            print(f"Successfully saved array to JPG: {output_path}")
            print(f"Array shape: {array.shape}, dtype: {array.dtype}")
            return True
        else:
            print(f"Error: Failed to save JPG file: {output_path}")
            return False
            
    except Exception as e:
        print(f"Error converting numpy array to JPG: {e}")
        return False


