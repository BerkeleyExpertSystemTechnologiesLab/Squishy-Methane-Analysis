import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Viewer", layout="centered")

st.title("Data Viewer")

def get_original_image_path(annotated_path):
    """
    Given an annotated image path, return the original image path.
    For example: "/path/annotated_image.png" -> "/path/image.png"
    """
    path = Path(annotated_path)
    filename = path.name
    
    # Remove "annotated_" prefix if present
    if filename.startswith("annotated_"):
        original_filename = filename.replace("annotated_", "", 1)
        original_path = path.parent / original_filename
        return original_path
    
    return path

def create_intensity_histogram(image_path):
    """
    Create a histogram of pixel intensity vs number of pixels.
    Assumes grayscale image.
    """
    image = Image.open(image_path)
    
    # Convert to grayscale if not already
    if image.mode != 'L':
        image = image.convert('L')
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Flatten the array to get all pixel values
    pixels = img_array.flatten()
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pixels, bins=256, range=(0, 256), edgecolor='black', alpha=0.7)
    ax.set_xlabel('Pixel Intensity (0-255)')
    ax.set_ylabel('Number of Pixels')
    ax.set_title('Histogram of Pixel Intensity')
    ax.grid(True, alpha=0.3)
    
    return fig

# Input box for image path
image_path = st.text_input("Enter the path to an image:", placeholder="/path/to/image.jpg")

# Display image if path is provided
if image_path:
    try:
        path = Path(image_path)
        if path.exists():
            image = Image.open(path)
            st.image(image, use_container_width=True)
            
            # Get original image path
            original_path = get_original_image_path(image_path)
            
            if original_path.exists() and original_path != path:
                st.success(f"Found original image: {original_path}")
                
                # Create and display histogram
                fig = create_intensity_histogram(original_path)
                st.pyplot(fig)
            elif original_path == path:
                st.info("No 'annotated_' prefix found. Creating histogram for the provided image.")
                fig = create_intensity_histogram(image_path)
                st.pyplot(fig)
            else:
                st.warning(f"Original image not found: {original_path}")
        else:
            st.error(f"File not found: {image_path}")
    except Exception as e:
        st.error(f"Error loading image: {e}")
